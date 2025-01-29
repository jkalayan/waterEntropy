#!/usr/bin/env python

"""
These functions calculate the force and torque matrices
"""

import numpy as np
from numpy import linalg as LA

from waterEntropy.utils.selections import get_selection
from waterEntropy.utils.trig import get_vector


class Covariance:  # pylint: disable=too-few-public-methods
    """
    Class for labelling coordination shell neighbours
    """


def get_torques(coords, forces, center_of_mass, rotation_axes, MI_axis):
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
    # center_of_mass = coords.center_of_mass()
    # rotation_axes, MI_axis = get_axes(coords)
    MI_axis_sqrt = np.sqrt(MI_axis)  # sqrt moi to weight torques
    translated_coords = coords.positions - center_of_mass
    rotated_coords = np.tensordot(translated_coords, rotation_axes.T, axes=1)
    rotated_forces = np.tensordot(forces.positions, rotation_axes.T, axes=1)
    cross_prod = np.cross(rotated_coords, rotated_forces)
    torque = np.sum(np.divide(cross_prod, MI_axis_sqrt), axis=0)

    return torque


def get_rotated_sum_forces(forces, rotation_axes):
    """
    Rotated the forces for a given seletion of atoms along a particular rotation
    axes (3,3)

    :param forces: mdanalysis instance of atoms selected for forces
    :param rotation_axes: a (3,3) array to rotate forces along
    """
    forces_summed = np.sum(forces.positions, axis=1)
    rotated_sum_forces = np.tensordot(forces_summed, rotation_axes.T, axes=1)
    return rotated_sum_forces


def get_FT_atoms(system, coords):
    """
    Find heavy atoms for atom selection in coords and find all atoms in the
    coords selection. Heavy atoms selection is used for

    :param system: mdanalysis instance of all atoms in a frame
    :param coords: mdanalysis instance of atoms selected for positions
    """
    heavy_atoms = coords.select_atoms("mass 1.1 to 999")
    all_resid_idxs = list(set(coords.resids))  # remove repeats
    all_atoms = get_selection(system, "resid", all_resid_idxs)
    return heavy_atoms, all_atoms


def get_mass_weighted_forces(forces, rotation_axes):
    """
    For a given set of atoms, sum their forces and rotate these summed forces
    using the rotation axes (3,3)

    :param forces: mdanalysis instance of atoms selected for forces
    :param rotation_axes: a (3,3) array to rotate forces along
    """
    rotated_sum_forces = get_rotated_sum_forces(forces, rotation_axes)
    mass_sqrt = np.sum(forces.masses) ** 0.5
    mass_weighted_force = rotated_sum_forces / mass_sqrt
    return mass_weighted_force  # (3,)


def get_covariance_matrix(ft, halve=True):
    """
    Get the outer product of the mass weighted forces or torques (ft) and
    half values if halve=True

    :param ft: (3,) array of either mass weighted forces or torques
    :param halve: Boolean to set weather covariance matrix should be halved
        (i.e. divide by 2**2)
    """
    cov_matrix = np.outer(ft, ft)
    if halve:
        cov_matrix = cov_matrix / 4.0
    return cov_matrix


def get_axes(coords):
    """
    From a selection of atoms, get the ordered principal axes (3,3) and
    the ordered moment of inertia axes (3,) for that selection of atoms

    :param coords: mdanalysis instance of atoms selected for axes calculation
    """
    moment_of_interia = coords.moment_of_inertia()
    principal_axes = coords.principal_axes()
    # diagonalise moment of inertia tensor here
    # pylint: disable=unused-variable
    eigenvalues, eigenvectors = LA.eig(moment_of_interia)
    # principal axes = eigenvectors.T[order])
    # comment: could get principal axes from transformed eigenvectors
    #           but would need to sort of directions, so use MDAnalysis
    #           function instead

    # sort eigenvalues of moi tensor by largest to smallest magnitude
    order = abs(eigenvalues).argsort()[::-1]  # decending order
    principal_axes = principal_axes[order]
    MOI_axis = eigenvalues[order]

    return principal_axes, MOI_axis


def get_custom_axes(a, b_list, c, dimensions):
    r"""
    For atoms a, b and c, use normalised vector ab as axis1,
    the vector perpendicular to ab and ac as axis2 and the vector
    perpendicular to axis1 and axis2 as axis3.
    This case is used when atoms a and b are heavy atoms and c is a
    hydrogen atom:

    ```
    ......a          1 = norm_ab
    ...../ \         2 = |_ norm ab and norm_ac
    ..../   \        3 = |_ 1 and 2
    ..b       c
    ```

    """
    axis1 = np.zeros(3)
    for b in b_list:
        ab_vector = get_vector(a, b, dimensions)
        ab_dist = np.sqrt((ab_vector**2).sum(axis=-1))
        axis1 += np.divide(ab_vector, ab_dist)

    if len(b_list) > 2:
        # the second axis will be the vector
        ac_vector = get_vector(b_list[0], c, dimensions)
    else:
        ac_vector = get_vector(a, c, dimensions)
    ac_dist = np.sqrt((ac_vector**2).sum(axis=-1))
    ac_vector_norm = np.divide(ac_vector, ac_dist)

    if len(b_list) > 2:
        axis2 = np.cross(ac_vector_norm, axis1)
    else:
        axis2 = np.cross(axis1, ac_vector_norm)
    axis3 = np.cross(axis1, axis2)

    custom_axes = np.array((axis1, axis2, axis3))

    return custom_axes


def get_flipped_axes(coords, custom_axes, center_of_mass, dimensions):
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
    RRaxis = get_vector(coords[0], center_of_mass, dimensions)
    # flip each Paxis if its pointing out of UA
    for i in range(3):
        dotProd1 = np.dot(PIaxes[i], RRaxis)
        PIaxes[i] = np.where(dotProd1 < 0, -PIaxes[i], PIaxes[i])

    return PIaxes


def get_custom_PI_MOI(coords, custom_rotation_axes, center_of_mass, masses, dimensions):
    """
    Get MOI tensor (PIaxes) and center point coordinates (custom_MI_axis)
    for UA level, where eigenvalues and vectors are not used.
    Note, positions and masses are provided separately as some cases
    require using the positions of heavy atoms only, but the masses of all
    atoms for a given selection of atoms.

    :param coords: (N,3) array of coordinates for a collection of atoms
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
        coords, custom_rotation_axes, center_of_mass, dimensions
    )
    translated_coords = coords - center_of_mass
    custom_MI_axis = np.zeros(3)
    for coord, mass in zip(translated_coords, masses):
        axis_component = np.sum(
            np.cross(custom_rotation_axes, coord) ** 2 * mass, axis=1
        )
        custom_MI_axis += axis_component

    return custom_rotation_axes, custom_MI_axis


def get_atoms_masses(coords, include_Hs=True):
    """
    Depending on the length scale, either include or exclude hydrogen atom
    coordinates from subsequent axes calculations. If H positions are excluded,
    then add their mass to their bonded heavy atom in the mass list.
    """
    atom_indices = []
    masses = []
    for atom in coords:
        if include_Hs:
            masses.append(atom.mass)
            atom_indices.append(atom.index)
        else:
            if atom.mass > 1.1:
                UA_mass = 0
                atom_indices.append(atom.index)
                UA_mass += atom.mass
                for b in atom.bonds:
                    bonded = b[1]
                    if bonded.mass > 1 and bonded.mass < 1.1:
                        UA_mass += bonded.mass
                masses.append(UA_mass)

    return atom_indices, masses


def get_bonded_axes(system, atom, dimensions, include_Hs=True):
    r"""
    Few scenarios for choosing united atom axes:

    X -- H = bonded to one light atom
    X -- R = bonded to one heavy atom
    R -- X -- H = bonded to one heavy and one light atom
    R -- X -- R = bonded to two heavy atoms
    R -- X -- R = bonded to more than two heavy atoms
    .....|
    .....R

    """
    # check atom is a heavy atom
    position_vector = atom.position
    custom_axes = None
    if not atom.mass > 1.1:
        return None
    atom_indices, masses = get_atoms_masses(atom, include_Hs)
    heavy_bonded = []
    light_bonded = []
    atom_indices = [atom.index]
    for b in atom.bonds:
        bonded_atom = b[1]
        atom_indices.append(bonded_atom.index)
        if bonded_atom.mass > 1.1:
            heavy_bonded.append(bonded_atom)
        if bonded_atom.mass > 1 and bonded_atom.mass < 1.1:
            light_bonded.append(bonded_atom)
    if len(heavy_bonded) == 2:
        custom_axes = get_custom_axes(
            atom[0], [heavy_bonded[0]], heavy_bonded[1], dimensions
        )
    if len(heavy_bonded) == 1 and len(light_bonded) >= 1:
        custom_axes = get_custom_axes(
            atom[0], [heavy_bonded[0]], light_bonded[0], dimensions
        )
    if len(heavy_bonded) > 2:
        custom_axes = get_custom_axes(
            atom[0], heavy_bonded, heavy_bonded[0], dimensions
        )
    if len(heavy_bonded) == 1 and len(light_bonded) == 1:
        custom_axes = get_custom_axes(atom[0], heavy_bonded[0], np.zeros(3), dimensions)
    if len(heavy_bonded) == 0:
        all_bonded = get_selection(system, "index", atom_indices)
        custom_axes, position_vector = get_axes(all_bonded)

    if custom_axes:
        custom_axes, position_vector = get_custom_PI_MOI(
            all_bonded, custom_axes, atom.position, masses, dimensions
        )

    return custom_axes, position_vector


# def get_forces_torques(coords, forces, include_H):
#     """
#     Axes required for various levels:
#     1) Polymer level: trans axes = principle axes, rot axes = principle axes
#         position vector = center of mass
#     2) Residue level: trans axes = principal axes of polymer, rot axes = residue
#         principal axes if not bonded to anything else and position vector is the
#         center of mass of the residue, otherwise if it's bonded to other
#         residues then set the position vector as (between the bonded atoms
#         or center of residue??) and use the average vector for bonded heavy
#         atoms to get the rot axes
#     3) United atom level: trans axes = principal axes of residue,
#         rot axes = ave vector of bonded heavy atoms and position vector is the
#         heavy atom in the UA


#     In my code, I have a few options:
#     - If the molecule is a single UA and bonded 1 or more Hs, then use all atoms
#         in getting trans and rot axes, nothing fancy here
#     - If the molecule contains more that one UA, then don't use the H atom
#         coords to get pricipal axes. Use this principal axes in the torque
#         calculation, but use the coord list including the Hs to calculate the
#         torques. The positional vector is the center of mass of all atoms including
#         Hs.

#     """
