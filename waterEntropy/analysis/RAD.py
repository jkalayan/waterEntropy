"""
These functions calculate coordination shells using RAD the relative
angular distance.
"""

import numpy as np

import waterEntropy.maths.trig as Trig
from waterEntropy.analysis.shells import ShellCollection
from waterEntropy.utils.topology_cache import get_positions

_PAIR_ENUMERATION: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _pair_enumeration(R: int):
    """Return the (y, z) pair indices and per-y offsets for a shell of size R.

    These depend only on R, which is capped at 25, so they are computed once
    and reused instead of being rebuilt on every call.

    :param R: number of candidate neighbours considered
    """
    cached = _PAIR_ENUMERATION.get(R)
    if cached is None:
        y_per_pair = np.concatenate([np.full(y, y) for y in range(1, R)])
        z_per_pair = np.concatenate([np.arange(y) for y in range(1, R)])
        offsets = np.concatenate([[0], np.cumsum(np.arange(1, R))])
        cached = (y_per_pair, z_per_pair, offsets)
        _PAIR_ENUMERATION[R] = cached
    return cached


def get_RAD_neighbours(i_coords, sorted_indices, sorted_distances, system):
    # pylint: disable=too-many-locals
    r"""
    For a given set of atom coordinates, find its RAD shell from the distance
    sorted atom list, truncated to the closests 30 atoms.

    This function calculates coordination shells using RAD the relative
    angular distance, as defined first in DOI:10.1063/1.4961439
    where united atoms (heavy atom + bonded Hydrogens) are defined as neighbours if
    they fulfil the following condition:

    .. math::
        \Bigg(\frac{1}{r_{ij}}\Bigg)^2>\Bigg(\frac{1}{r_{ik}}\Bigg)^2 \cos \theta_{jik}

    For a given particle :math:`i`, neighbour :math:`j` is in its coordination
    shell if :math:`k` is not blocking particle :math:`j`. In this implementation
    of RAD, we enforce symmetry, whereby neighbouring particles must be in each
    others coordination shells.

    :param i_coords: xyz coordinates of atom :math:`i`
    :param sorted_indices: list of atom indices sorted from closest to
        furthest from atom :math:`i`
    :param sorted_distances: list of atom distances sorted from closest to
        furthest from atom :math:`i`
    :param system: mdanalysis instance of atoms in a frame
    """
    # 1. truncate neighbour list to closest 25 united atoms
    range_limit = min(len(sorted_distances), 25)
    if range_limit == 0:
        return []
    # system.atoms.positions copies the position array for *every* atom in
    # the system (tens of thousands of atoms in a solvated box) on each
    # access. Fetch it (and the other per-frame system properties) once.
    all_positions = get_positions(system)
    all_indices = system.atoms.indices
    dims = system.dimensions[:3]

    idx = sorted_indices[:range_limit]
    dist = sorted_distances[:range_limit]
    positions = all_positions[idx]
    R = range_limit
    shell = [int(all_indices[idx[0]])]  # closest neighbour has no z's to check
    if R == 1:
        return shell

    # The original algorithm is an O(R^2) Python loop calling get_angle once
    # per (y, z) candidate pair (y = 1..R-1, z = 0..y-1) - up to ~300 calls
    # for R=25, each paying numpy's per-call dispatch overhead on a single
    # triple. get_angle already reduces over axis=-1, so it works unchanged
    # on batched (M, 3) inputs: build every (y, z) pair once and evaluate
    # them all in a single call instead of one Python call per pair.
    # y_per_pair/z_per_pair enumerate pairs grouped by y, with z ascending
    # within each group - this ordering is what lets the first-trigger scan
    # below reproduce the original's early-exit semantics exactly.
    # The pair enumeration depends only on R (at most 25), while this runs a
    # couple of thousand times per frame, so the arrays are built once per R.
    y_per_pair, z_per_pair, offsets = _pair_enumeration(R)

    j_coords = positions[y_per_pair]
    k_coords = positions[z_per_pair]
    rij = dist[y_per_pair]
    rik = dist[z_per_pair]

    costheta = Trig.get_angle(j_coords, i_coords, k_coords, dims)
    # NaN comparisons are always False in numpy, so block_mask is already
    # False wherever nan_mask is True - matching the original's "isnan halts
    # checking without setting blocked=True".
    nan_mask = np.isnan(costheta)
    block_mask = (1.0 / rij) ** 2 < ((1.0 / rik) ** 2) * costheta
    trigger_mask = nan_mask | block_mask

    # 2. for each y (from closest to furthest), the original breaks out of
    # its z-loop (ascending, closest-to-i first) at the FIRST z that either
    # blocks y (LHS < RHS) or produces a NaN angle (halts checking, not a
    # block). Only that first trigger among each y's z's matters.
    for y in range(1, R):
        start, end = offsets[y - 1], offsets[y]
        group_trigger = trigger_mask[start:end]
        if group_trigger.any():
            first = int(np.argmax(group_trigger))
            if block_mask[start + first]:
                continue  # blocked - not in shell
        shell.append(int(all_indices[idx[y]]))
    return shell


def get_RAD_shell(
    UA, system, shells: ShellCollection, sorted_indices=None, sorted_distances=None
):
    """
    For a given united atom, find its RAD shell, returning the atom indices
    for the heavy atoms that are in its shell.

    :param UA: mdanalysis instance of a united atom in a frame
    :param system: mdanalysis instance of atoms in a frame
    :param shells: ShellCollection instance
    """
    # 1. first check if a shell has already been found for this UA
    shell = shells.find_shell(UA.index)
    if not shell:
        # 2. get the nearest neighbours for the UA, sorted from closest to
        # furthest
        if sorted_indices is None:
            sorted_indices, sorted_distances = Trig.get_sorted_neighbours(
                UA.index, system
            )
        # 3. now find the RAD shell of the UA
        shell_indices = get_RAD_neighbours(
            UA.position, sorted_indices, sorted_distances, system
        )
        # 4. populate the class instance for RAD shells
        shells.add_data(UA.index, shell_indices)
        shell = shells.find_shell(UA.index)
    return shell
