"""
Per-Universe cache of topology-derived data (heavy-atom mask, bonded-atom
adjacency lists) shared by trig.get_sorted_neighbours and
selections.find_bonded_heavy_atom / find_bonded_atoms.

Masses and bonds are static across a trajectory - only positions change
frame to frame - so deriving these once per Universe instead of re-parsing
an MDAnalysis selection string (e.g. "bonded index N") on every atom, every
frame, removes what was measured to be the dominant cost of the interfacial
water search on a real ~40k-atom solvated system.
"""

import weakref

_TOPOLOGY_CACHE = weakref.WeakKeyDictionary()


def _optional_atom_attr(atoms, name):
    try:
        return getattr(atoms, name)
    except (AttributeError, LookupError, ValueError):
        return None


def get_positions(system):
    """Read the current full-Universe positions without copying all atoms.

    This is a live view, used only within a frame; never retained across seeks.
    AtomGroup inputs retain the original indexed-position semantics.
    """
    universe = getattr(system, "universe", system)
    if system.atoms is universe.atoms:
        return universe.trajectory.ts.positions
    return system.atoms.positions


def get_topology_cache(system):
    """Return (and lazily build) the cached topology data for a Universe.

    :param system: mdanalysis Universe (or AtomGroup with a .universe) for
        the frame being processed.
    """
    universe = getattr(system, "universe", system)
    cache = _TOPOLOGY_CACHE.get(universe)
    if cache is None:
        masses = universe.atoms.masses
        heavy_mask = (masses >= 2) & (masses <= 999)
        heavy_indices = heavy_mask.nonzero()[0]
        heavy_index_positions = {}
        for pos, atom_idx in enumerate(heavy_indices):
            heavy_index_positions[int(atom_idx)] = pos
        bonded_lists = [[] for _ in range(universe.atoms.n_atoms)]
        bond_indices = universe.atoms.bonds.to_indices()
        for a, b in bond_indices:
            bonded_lists[a].append(b)
            bonded_lists[b].append(a)
        exclude_heavy_positions = []
        for atom_idx, bonded in enumerate(bonded_lists):
            positions = []
            atom_pos = heavy_index_positions.get(atom_idx)
            if atom_pos is not None:
                positions.append(atom_pos)
            for bonded_idx in bonded:
                bonded_pos = heavy_index_positions.get(bonded_idx)
                if bonded_pos is not None:
                    positions.append(bonded_pos)
            exclude_heavy_positions.append(sorted(set(positions)))
        cache = {
            "heavy_mask": heavy_mask,
            "heavy_indices": heavy_indices,
            # position of each heavy atom within heavy_indices, or absent for
            # light atoms; used to index per-frame neighbour shortlists.
            "heavy_index_positions": heavy_index_positions,
            "exclude_heavy_positions": exclude_heavy_positions,
            "bonded_lists": bonded_lists,
            "masses": masses,
            "charges": _optional_atom_attr(universe.atoms, "charges"),
            "resids": _optional_atom_attr(universe.atoms, "resids"),
            "resnames": _optional_atom_attr(universe.atoms, "resnames"),
            "types": _optional_atom_attr(universe.atoms, "types"),
            "indices": universe.atoms.indices,
        }
        _TOPOLOGY_CACHE[universe] = cache
    return cache
