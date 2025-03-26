"""
Functions for common trigonometric calculations
"""

import MDAnalysis
import numpy as np

import waterEntropy.utils.selections as Selections


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
            *sorted(zip(distances, neighbour_indices), key=lambda x: x[0])
        )
        return np.array(sorted_indices), np.array(sorted_distances)
    raise ValueError(
        f"Atom coordinates {atom} in neighbour list {neighbours.positions[:10]}"
    )


def get_sorted_neighbours(i_idx: int, system, max_cutoff=25):
    """
    For a given atom, find neighbouring united atoms from closest to furthest
    within a given cutoff.

    :param i_idx: idx of atom i
    :param system: mdanalysis instance of atoms in a frame
    :param max_cutoff: set the maximum cutoff value for finding neighbours
    """
    i_coords = system.atoms.positions[i_idx]
    # 1. get the heavy atom neighbour distances within a given distance cutoff
    # CHECK Find out which of the options below is better for RAD shells
    #       Should the central atom bonded UAs be allowed to block?
    #       This was not done in original code, keep the same here
    neighbours = system.select_atoms(
        f"""mass 2 to 999 and not index {i_idx}
                                    and not bonded index {i_idx}"""
        # f"""mass 2 to 999 and not index {i_idx}"""  # bonded UAs can block
    )
    # 2. Get the neighbours sorted from closest to furthest
    sorted_indices, sorted_distances = get_neighbourlist(
        i_coords, neighbours.atoms, system.dimensions, max_cutoff
    )
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
    # neighbours = system.select_atoms(f"all and not index {donator.index} and not bonded index {donator.index}")
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
    for delt, box in zip(delta, dimensions):
        if delt < 0 and delt < 0.5 * box:
            delt = delt + box
        if delt > 0 and delt > 0.5 * box:
            delt = delt - box
        delta_wrapped.append(delt)
    delta_wrapped = np.array(delta_wrapped)

    return delta_wrapped
