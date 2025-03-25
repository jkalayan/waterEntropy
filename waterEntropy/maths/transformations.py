"""
Functions for transforming atomic positions and forces
"""

import numpy as np
from numpy import linalg as LA

import waterEntropy.maths.trig as Trig
import waterEntropy.utils.selections as Selections


def get_torques(
    molecule, center_of_mass: np.ndarray, rotation_axes: np.ndarray, MI_axis: np.ndarray
):
    """
    For a selection of atoms, use their positions and forces to get the
    torque (3,) for that selection of atoms. The positions are first translated
    to the center of mass, then the translated positions and forces are
    rotated to align with the chosen rotation axes. Lastly the torque
    is calculated from the cross product of the transformed positions and
    forces, which is subsequently divided by the sqaure root of the moment
    of inertia axis.

    :param coords: mdanalysis instance of atoms selected for positions
    :param forces: mdanalysis instance of atoms selected for forces
    :param center_of_mass: a (3,) array of the chosen center of mass
    :param rotation_axes: a (3,3) array to rotate forces along
    :param MI_axis: a (3,) array for the moment of inertia axis center
    """
    MI_axis_sqrt = np.sqrt(MI_axis)  # sqrt moi to weight torques
    translated_coords = molecule.positions - center_of_mass
    rotated_coords = np.tensordot(translated_coords, rotation_axes.T, axes=1)
    rotated_forces = np.tensordot(molecule.forces, rotation_axes.T, axes=1)
    cross_prod = np.cross(rotated_coords, rotated_forces)
    torque = np.sum(np.divide(cross_prod, MI_axis_sqrt), axis=0)

    return torque


def get_rotated_sum_forces(molecule, rotation_axes: np.ndarray):
    """
    Rotated the forces for a given seletion of atoms along a particular rotation
    axes (3,3)

    :param molecule: mdanalysis instance of molecule
    :param rotation_axes: a (3,3) array to rotate forces along
    """
    forces_summed = np.sum(molecule.forces, axis=0)
    rotated_sum_forces = np.tensordot(forces_summed, rotation_axes.T, axes=1)
    return rotated_sum_forces


def get_mass_weighted_forces(molecule, rotation_axes: np.ndarray):
    """
    For a given set of atoms, sum their forces and rotate these summed forces
    using the rotation axes (3,3)

    :param molecule: mdanalysis instance of molecule
    :param rotation_axes: a (3,3) array to rotate forces along
    """
    rotated_sum_forces = get_rotated_sum_forces(molecule, rotation_axes)
    mass_sqrt = np.sum(molecule.masses) ** 0.5
    mass_weighted_force = rotated_sum_forces / mass_sqrt
    return mass_weighted_force  # (3,)


def get_covariance_matrix(ft: np.ndarray, halve=0.5):
    """
    Get the outer product of the mass weighted forces or torques (ft) and
    half values if halve=True

    :param ft: (3,) array of either mass weighted forces or torques
    :param halve: Boolean to set weather covariance matrix should be halved
        (i.e. divide by :math:`2^2`)
    """
    cov_matrix = np.outer(ft, ft)
    if halve:
        cov_matrix *= halve**2
    return cov_matrix


def get_UA_masses(molecule):
    """
    For a given molecule, return a list of masses of UAs
    (combination of the heavy atoms + bonded hydrogen atoms. This list is used to
    get the moment of inertia tensor for molecules larger than one UA

    :param molecule: mdanalysis instance of molecule
    """
    UA_masses = []
    for atom in molecule:
        if atom.mass > 1.1:
            UA_mass = atom.mass
            bonded_atoms = molecule.select_atoms(f"bonded index {atom.index}")
            bonded_H_atoms = bonded_atoms.select_atoms("mass 1 to 1.1")
            for H in bonded_H_atoms:
                UA_mass += H.mass
            UA_masses.append(UA_mass)
        else:
            continue
    return UA_masses


def get_axes(molecule, molecule_scale: str):
    """
    From a selection of atoms, get the ordered principal axes (3,3) and
    the ordered moment of inertia axes (3,) for that selection of atoms

    :param molecule: mdanalysis instance of molecule
    :param molecule_scale: the length scale of molecule
    """
    # default moment of inertia
    moment_of_inertia = molecule.moment_of_inertia()
    if molecule_scale == "single_UA":
        pass  # moment_of_inertia = molecule.moment_of_inertia()
    if molecule_scale == "multiple_UAs":
        UAs = Selections.find_molecule_UAs(molecule)
        center_of_mass = molecule.center_of_mass()
        masses = get_UA_masses(molecule)
        moment_of_inertia = MOI(center_of_mass, UAs.positions, masses)
    principal_axes = molecule.principal_axes()
    # diagonalise moment of inertia tensor here
    # pylint: disable=unused-variable
    eigenvalues, _eigenvectors = LA.eig(moment_of_inertia)
    # principal axes = eigenvectors.T[order])
    # comment: could get principal axes from transformed eigenvectors
    #           but would need to sort out directions, so use MDAnalysis
    #           function instead

    # sort eigenvalues of moi tensor by largest to smallest magnitude
    order = abs(eigenvalues).argsort()[::-1]  # decending order
    # principal_axes = principal_axes[order] # PI already ordered correctly
    MOI_axis = eigenvalues[order]

    return principal_axes, MOI_axis


def MOI(CoM: np.ndarray, positions: np.ndarray, masses: list):
    """
    Use this function to calculate moment of inertia for cases where the
    mass list will contain masses of UAs rather than individual atoms and
    the postions will be those for the UAs only (excluding the H atoms
    coordinates).

    :param CoM: a (3,) array of the chosen center of mass
    :param positions: a (N,3) array of point positions
    :param masses: a (N,) list of point masses
    """
    I = np.zeros((3, 3))
    for coord, mass in zip(positions, masses):
        I[0][0] += (abs(coord[1] - CoM[1]) ** 2 + abs(coord[2] - CoM[2]) ** 2) * mass
        I[0][1] -= (coord[0] - CoM[0]) * (coord[1] - CoM[1]) * mass
        I[1][0] -= (coord[0] - CoM[0]) * (coord[1] - CoM[1]) * mass

        I[1][1] += (abs(coord[0] - CoM[0]) ** 2 + abs(coord[2] - CoM[2]) ** 2) * mass
        I[0][2] -= (coord[0] - CoM[0]) * (coord[2] - CoM[2]) * mass
        I[2][0] -= (coord[0] - CoM[0]) * (coord[2] - CoM[2]) * mass

        I[2][2] += (abs(coord[0] - CoM[0]) ** 2 + abs(coord[1] - CoM[1]) ** 2) * mass
        I[1][2] -= (coord[1] - CoM[1]) * (coord[2] - CoM[2]) * mass
        I[2][1] -= (coord[1] - CoM[1]) * (coord[2] - CoM[2]) * mass

    return I


def get_custom_axes(a: np.ndarray, b_list: list, c: np.ndarray, dimensions: np.ndarray):
    r"""
    For atoms a, b_list and c, calculate the axis to rotate forces around:

    - axis1: use the normalised vector ab as axis1. If there is more than one bonded
      heavy atom (HA), average over all the normalised vectors calculated from b_list
      and use this as axis1). b_list contains all the bonded heavy atom
      coordinates.

    - axis2: use cross product of normalised vector ac and axis1 as axis2.
      If there are more than two bonded heavy atoms, then use normalised vector
      b[0]c to cross product with axis1, this gives the axis perpendicular to
      axis1.

    - axis3: the cross product of axis1 and axis2, which is perpendicular to
      axis1 and axis2.

    :param a: central united-atom coordinates (3,)
    :param b_list: list of heavy bonded atom positions (3,N)
    :param c: atom coordinates of either a second heavy atom or a hydrogen atom
        if there are no other bonded heavy atoms in b_list (where N=1 in b_list)
        (3,)
    :param dimensions: dimensions of the simulation box (3,)

    ::

          a          1 = norm_ab
         / \         2 = |_ norm_ab and norm_ac (use bc if more than 2 HAs)
        /   \        3 = |_ 1 and 2
      b       c

    """
    axis1 = np.zeros(3)
    # average of all heavy atom covalent bond vectors for axis1
    for b in b_list:
        ab_vector = Trig.get_vector(a, b, dimensions)
        # scale vector with distance
        ab_dist = np.sqrt((ab_vector**2).sum(axis=-1))
        scaled_vector = np.divide(ab_vector, ab_dist)
        axis1 += scaled_vector  # ab_vector

    if len(b_list) > 2:
        ac_vector = Trig.get_vector(b_list[0], c, dimensions)
    else:
        ac_vector = Trig.get_vector(a, c, dimensions)
    ac_dist = np.sqrt((ac_vector**2).sum(axis=-1))
    ac_vector_norm = np.divide(ac_vector, ac_dist)

    if len(b_list) > 2:
        axis2 = np.cross(ac_vector_norm, axis1)
    else:
        axis2 = np.cross(axis1, ac_vector_norm)
    axis3 = np.cross(axis1, axis2)

    custom_axes = np.array((axis1, axis2, axis3))

    return custom_axes


def get_flipped_axes(positions, custom_axes, center_of_mass, dimensions):
    """
    For a given set of custom axes, ensure the axes are pointing in the
    correct direction wrt the heavy atom position and the chosen center
    of mass.
    """
    # sorting out PIaxes for MoI for UA fragment
    custom_axis = np.sum(custom_axes**2, axis=1)
    PIaxes = custom_axes / custom_axis**0.5

    # get dot product of Paxis1 and CoM->atom1 vect
    # will just be [0,0,0]
    RRaxis = Trig.get_vector(positions[0], center_of_mass, dimensions)
    # flip each Paxis if its pointing out of UA
    for i in range(3):
        dotProd1 = np.dot(PIaxes[i], RRaxis)
        PIaxes[i] = np.where(dotProd1 < 0, -PIaxes[i], PIaxes[i])

    return PIaxes


def get_custom_PI_MOI(molecule, custom_rotation_axes, center_of_mass, dimensions):
    """
    Get MOI tensor (PIaxes) and center point coordinates (custom_MI_axis)
    for UA level, where eigenvalues and vectors are not used.
    Note, positions and masses are provided separately as some cases
    require using the positions of heavy atoms only, but the masses of all
    atoms for a given selection of atoms.

    :param coords: MDAnalysis instance of molecule
    :param custom_rotation_axes: (3,3) arrray of rotation axes
    :param center_of_mass: (3,) center of mass for collection of atoms N
    :param masses: (N,) list of masses for collection of atoms, note this
        should be the same length as coords. If there are no hydrogens in
        the coords array, then the masses of these should be added to the
        heavy atom
    :param dimensions: (3,) array of system box dimensions.
    """
    # sorting out PIaxes for MoI for UA fragment
    custom_rotation_axes = get_flipped_axes(
        molecule.positions, custom_rotation_axes, center_of_mass, dimensions
    )
    translated_coords = molecule.positions - center_of_mass
    custom_MI_axis = np.zeros(3)
    for coord, mass in zip(translated_coords, molecule.masses):
        axis_component = np.sum(
            np.cross(custom_rotation_axes, coord) ** 2 * mass, axis=1
        )
        custom_MI_axis += axis_component

    return custom_rotation_axes, custom_MI_axis


def get_bonded_axes(system, atom, dimensions):
    """
    For a given united atom, find how to select bonded atoms to get the axes
    for rotating forces around. Few cases for choosing united atom axes:

    ::

        X -- H = bonded to one or more light atom

        X -- R = bonded to one heavy atom

        R -- X -- H = bonded to one heavy and one light atom

        R1 -- X -- R2 = bonded to two heavy atoms

        R1 -- X -- R2 = bonded to more than two heavy atoms
              |
              R3

    Note that axis2 is calculated by taking the cross product between axis1 and
    the vector chosen for each case, dependent on bonding:

    - case1: if all the bonded atoms are hydrogens, then just use the moment of
      inertia as all the axes.

    - case2: no axes required to rotate forces.

    - case3: use XR vector as axis1, vector XH to calculate axis2

    - case4: use vector XR1 as axis1, and XR2 to calculate axis2

    - case5: get the sum of all XR normalised vectors as axis1, then use vector
      R1R2 to calculate axis2

    """
    # check atom is a heavy atom
    if not atom.mass > 1.1:
        return None
    position_vector = atom.position
    custom_axes = None
    # find the heavy bonded atoms and light bonded atoms
    heavy_bonded, light_bonded = Selections.find_bonded_atoms(atom.index, system)
    UA_all = atom + heavy_bonded + light_bonded
    # now find which atoms to select to find the axes for rotating forces
    if len(heavy_bonded) == 2:
        custom_axes = get_custom_axes(
            atom.position,
            [heavy_bonded[0].position],
            heavy_bonded[1].position,
            dimensions,
        )
    if len(heavy_bonded) == 1 and len(light_bonded) >= 1:
        custom_axes = get_custom_axes(
            atom.position,
            [heavy_bonded[0].position],
            light_bonded[0].position,
            dimensions,
        )
    if len(heavy_bonded) > 2:
        custom_axes = get_custom_axes(
            atom.position, heavy_bonded.positions, heavy_bonded[1].position, dimensions
        )
    if len(heavy_bonded) == 1 and len(light_bonded) == 1:
        custom_axes = get_custom_axes(
            atom.position, [heavy_bonded[0].position], np.zeros(3), dimensions
        )
    if len(heavy_bonded) == 0:
        # !! Check if this scale is correct
        custom_axes, position_vector = get_axes(UA_all, molecule_scale="single_UA")

    if custom_axes is not None:
        custom_axes, position_vector = get_custom_PI_MOI(
            UA_all, custom_axes, atom.position, dimensions
        )

    return custom_axes, position_vector
