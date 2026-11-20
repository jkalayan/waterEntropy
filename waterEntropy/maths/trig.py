"""
Functions for common trigonometric calculations
"""

import os
import weakref

import MDAnalysis
import MDAnalysis.lib.nsgrid
import numpy as np

import waterEntropy.utils.selections as Selections
from waterEntropy.utils.topology_cache import get_positions, get_topology_cache


def get_neighbourlist(
    atom: np.ndarray, neighbours, dimensions: np.ndarray, max_cutoff=9e9
):
    """
    Use MDAnalysis to get distances between an atom and neighbours within
    a given cutoff. Each atom index pair sorted by distance are outputted.

    :param atom: (3,) array of an atom coordinates.
    :param neighbours: MDAnalysis array of heavy atoms in the system,
        not the atom itself and not bonded to the atom.
    :param dimensions: (6,) array of system box dimensions.
    :param max_cutoff: set the maximum cutoff value for finding neighbour distances
    """
    # check atom coords are not in neighbour coords list
    if not (atom == neighbours.positions).all(axis=1).any():
        pairs, distances = MDAnalysis.lib.distances.capped_distance(
            atom,
            neighbours.positions,
            max_cutoff=max_cutoff,
            min_cutoff=None,
            box=dimensions,
            method=None,
            return_distances=True,
        )
        neighbour_indices = neighbours[pairs[:][:, 1]].indices
        sorted_distances, sorted_indices = zip(
            *sorted(
                zip(distances, neighbour_indices, strict=False), key=lambda x: x[0]
            ),
            strict=False,
        )
        return np.array(sorted_indices), np.array(sorted_distances)
    raise ValueError(
        f"Atom coordinates {atom} in neighbour list {neighbours.positions[:10]}"
    )


# Grid shortlists are only ever used to propose a superset of the candidates;
# every distance that reaches the caller is still computed by the original
# kernel on that subset. The margin makes "within max_cutoff by the grid's own
# arithmetic" a strict superset of "within max_cutoff by ours", so no neighbour
# can be lost to a rounding difference between the two.
#
# FastNS rejects a cutoff larger than half the box, and the margin can push a
# cutoff the original path handled over that limit (a 10 A cutoff in an 18 A
# box). Rather than turn a working call into an exception, such frames keep the
# original full scan.
_GRID_MARGIN = 1e-3
_SHORTLIST_CACHE = weakref.WeakKeyDictionary()

# The shortlist path is bitwise identical to the full scan in every gate run so
# far (including a full 1001-frame trajectory), but it is the one component of
# this package that has not had independent review, so it is opt-in. Set
# CODEENTROPY_FAST_NEIGHBOURS=1 for roughly 15% more speed; leaving it unset
# keeps the original per-query scan.
_FAST_NEIGHBOURS_ENV = "CODEENTROPY_FAST_NEIGHBOURS"


def _fast_neighbours_enabled() -> bool:
    """Return whether per-frame neighbour shortlists are enabled."""
    return os.environ.get(_FAST_NEIGHBOURS_ENV, "0") not in ("0", "", "false", "False")


def _heavy_neighbour_shortlists(system, positions, max_cutoff):
    """Return per-heavy-atom candidate shortlists for the current frame.

    Scanning every heavy atom for each query dominates the water stage: ~2,200
    queries per frame, each against ~14,000 heavy atoms, i.e. ~31M distance
    evaluations. One grid search per frame answers all of them at once, in CSR
    form (``starts`` offsets into a neighbour array ordered by query atom).

    A shortlist is only valid for the exact inputs it was built from, so the
    cache is keyed on the positions *and* the periodic box: the same coordinates
    under a different box have different periodic neighbours, and reusing the
    old shortlist would silently miss some. Nothing in waterEntropy mutates
    positions or the box mid-frame today; the guard is here so that correctness
    does not depend on that staying true. The check is a contiguous comparison,
    far cheaper than the per-query scan it replaces.

    Returns ``None`` when no grid can serve this frame (cutoff too large for the
    box), leaving callers on the original full-scan path.

    :param system: mdanalysis instance of atoms in a frame
    :param positions: current full-Universe positions
    :param max_cutoff: neighbour cutoff the shortlists must cover
    """
    universe = getattr(system, "universe", system)
    cache = get_topology_cache(system)
    dimensions = np.asarray(system.dimensions, dtype=np.float32)

    entry = _SHORTLIST_CACHE.get(universe)
    if (
        entry is not None
        and entry["cutoff"] == max_cutoff
        and np.array_equal(entry["dimensions"], dimensions)
        and entry["positions"].shape == positions.shape
        and np.array_equal(entry["positions"], positions)
    ):
        return entry["shortlists"]

    heavy_indices = cache["heavy_indices"]
    heavy_positions = np.ascontiguousarray(positions[heavy_indices], dtype=np.float32)
    try:
        grid = MDAnalysis.lib.nsgrid.FastNS(
            float(max_cutoff) + _GRID_MARGIN,
            heavy_positions,
            box=dimensions,
            pbc=True,
        )
    except ValueError:
        # Cutoff too large for this box: no shortlist, keep the full scan.
        _SHORTLIST_CACHE[universe] = {
            "cutoff": max_cutoff,
            "dimensions": dimensions.copy(),
            "positions": positions.copy(),
            "shortlists": None,
        }
        return None
    pairs = grid.self_search().get_pairs()

    # Symmetrise: self_search reports each pair once.
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.argsort(rows, kind="stable")
    neighbours = cols[order]
    counts = np.bincount(rows, minlength=len(heavy_indices))
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    shortlists = {"neighbours": neighbours, "starts": starts}
    _SHORTLIST_CACHE[universe] = {
        "cutoff": max_cutoff,
        "dimensions": dimensions.copy(),
        "positions": positions.copy(),
        "shortlists": shortlists,
    }
    return shortlists


def get_sorted_neighbours(i_idx: int, system, max_cutoff=10):
    """
    For a given atom, find neighbouring united atoms from closest to furthest
    within a given cutoff.

    :param i_idx: idx of atom i
    :param system: mdanalysis instance of atoms in a frame
    :param max_cutoff: set the maximum cutoff value for finding neighbours
    """
    positions = get_positions(system)
    i_coords = positions[i_idx]
    # 1. get the heavy atom neighbour distances within a given distance cutoff
    # CHECK Find out which of the options below is better for RAD shells
    #       Should the central atom bonded UAs be allowed to block?
    #       This was not done in original code, keep the same here
    cache = get_topology_cache(system)
    heavy_position = cache["heavy_index_positions"].get(int(i_idx))

    shortlists = (
        None
        if heavy_position is None or not _fast_neighbours_enabled()
        else _heavy_neighbour_shortlists(system, positions, max_cutoff)
    )

    if shortlists is None:
        # No shortlist for this atom (light atom) or this frame (cutoff too
        # large for the box): keep the original full scan.
        candidate_indices = np.delete(
            cache["heavy_indices"], cache["exclude_heavy_positions"][i_idx]
        )
    else:
        start, stop = (
            shortlists["starts"][heavy_position],
            shortlists["starts"][heavy_position + 1],
        )
        shortlist = shortlists["neighbours"][start:stop]
        # Ascending atom-index order is what the stable distance sort below
        # relies on to break ties, exactly as the full scan did.
        shortlist = np.sort(shortlist)
        keep = np.ones(len(shortlist), dtype=bool)
        excluded = cache["exclude_heavy_positions"][i_idx]
        if len(excluded):
            keep &= ~np.isin(shortlist, excluded)
        candidate_indices = cache["heavy_indices"][shortlist[keep]]

    candidate_positions = positions[candidate_indices]
    # 2. Get the neighbours sorted from closest to furthest. Distances come from
    # the same kernel as before, on a subset: capped_distance values are
    # per-pair and elementwise, so they are bitwise identical to the full-scan
    # values. "bruteforce" is what the original single-point call resolved to
    # (`_determine_method`: len(reference) < 10), pinned here so the subset
    # cannot pick a different backend.
    pairs, distances = MDAnalysis.lib.distances.capped_distance(
        i_coords,
        candidate_positions,
        max_cutoff=max_cutoff,
        min_cutoff=None,
        box=system.dimensions,
        method="bruteforce",
        return_distances=True,
    )
    # Original Python sorted() is stable: equal distances retain atom order.
    order = np.argsort(distances, kind="stable")
    sorted_indices = candidate_indices[pairs[:, 1]][order]
    sorted_distances = distances[order]
    return sorted_indices, sorted_distances


def get_shell_neighbour_selection(
    shell, donator, system, heavy_atoms=True, max_cutoff=10
):
    """
    get shell neighbours ordered by ascending distance, this is used for
    finding possible hydrogen bonding neighbours.

    :param shell: the instance for class waterEntropy.neighbours.RAD.RAD
        containing coordination shell neighbours
    :param donator: the mdanalysis object for the donator
    :param system: mdanalysis instance of all atoms in current frame
    :param heavy_atoms: consider heavy atoms in a shell as neighbours
    :max_cutoff: set the maximum cutoff value for finding neighbours
    """
    # 1a. Select heavy atoms in shell, can only donate to heavy atoms in the shell
    neighbours = Selections.get_selection(system, "index", shell.UA_shell)
    if not heavy_atoms:
        # 1b. Select all atoms in the shell, included bonded to atoms (Hs included)
        #   Can donate to any atoms in a shell
        all_shell_bonded = neighbours[:].bonds.indices
        all_shell_indices = list(set().union(*all_shell_bonded))
        # can donate to any atom in the shell
        neighbours = Selections.get_selection(system, "index", all_shell_indices)
    # 1c. can donate to any neighbours outside a shell, not used
    # neighbours = system.select_atoms(
    #     f"all and not index {donator.index} and "
    #     f"not bonded index {donator.index}"
    #     )
    # 2. Get the neighbours sorted from closest to furthest
    sorted_indices, sorted_distances = get_neighbourlist(
        donator.position, neighbours, system.dimensions, max_cutoff
    )
    return sorted_indices, sorted_distances


def get_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, dimensions: np.ndarray):
    """
    Get the angle between three atoms, taking into account PBC.

    :param a: (3,) array of atom cooordinates
    :param b: (3,) array of atom cooordinates
    :param c: (3,) array of atom cooordinates
    :param dimensions: (3,) array of system box dimensions.
    """
    ba = np.abs(a - b)
    bc = np.abs(c - b)
    ac = np.abs(c - a)
    ba = np.where(ba > 0.5 * dimensions, ba - dimensions, ba)
    bc = np.where(bc > 0.5 * dimensions, bc - dimensions, bc)
    ac = np.where(ac > 0.5 * dimensions, ac - dimensions, ac)
    dist_ba = np.sqrt((ba**2).sum(axis=-1))
    dist_bc = np.sqrt((bc**2).sum(axis=-1))
    dist_ac = np.sqrt((ac**2).sum(axis=-1))
    cosine_angle = (dist_ac**2 - dist_bc**2 - dist_ba**2) / (-2 * dist_bc * dist_ba)
    return cosine_angle


def get_distance(a: np.ndarray, b: np.ndarray, dimensions: np.ndarray):
    """
    Calculates distance and accounts for PBCs.

    :param a: (3,) array of atom cooordinates
    :param b: (3,) array of atom cooordinates
    :param dimensions: (3,) array of system box dimensions.
    """
    delta = np.abs(b - a)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    distance = np.sqrt((delta**2).sum(axis=-1))
    return distance


def get_vector(a: np.ndarray, b: np.ndarray, dimensions: np.ndarray):
    """
    get vector of two coordinates over PBCs.

    :param a: (3,) array of atom cooordinates
    :param b: (3,) array of atom cooordinates
    :param dimensions: (3,) array of system box dimensions.
    """
    delta = b - a
    delta_wrapped = []
    for delt, box in zip(delta, dimensions, strict=False):
        if delt < 0 and delt < 0.5 * box:
            delt = delt + box
        if delt > 0 and delt > 0.5 * box:
            delt = delt - box
        delta_wrapped.append(delt)
    delta_wrapped = np.array(delta_wrapped)

    return delta_wrapped
