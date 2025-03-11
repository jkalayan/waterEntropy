"""
These functions calculate the force and torque matrices
"""

import numpy as np
from numpy import linalg as LA

from waterEntropy.utils.helpers import nested_dict
from waterEntropy.utils.selections import find_bonded_atoms, find_molecule_UAs
from waterEntropy.utils.trig import get_vector


class Covariance:
    """
    Class for covariance matrices for whole molecules (WM) and the united atoms
    (UA). Molecules are separated based on their name and residue ID.
    The residue ID is replaced with "bulk" to group all molecules with the same
    name.
    """

    def __init__(self):
        self.forces = nested_dict()
        self.torques = nested_dict()
        self.counts = nested_dict()

    def add_data(self, nearest, molecule_name, force, torque):
        """
        Add force, torque covariances to the class dictionaries
        """
        # order is important, update count last
        Covariance.populate_dicts(
            self, nearest, molecule_name, force, self.forces, self.counts
        )
        Covariance.populate_dicts(
            self, nearest, molecule_name, torque, self.torques, self.counts
        )
        Covariance.update_counts(self, nearest, molecule_name, self.counts)

    def populate_dicts(
        self, nearest, molecule_name, variable, variable_dict, count_dict
    ):
        # pylint: disable=too-many-arguments
        """
        For a given molecule, append the summed, weighted and rotated forces,
        and the torques for the whole molecule. Add as a running average by
        keeping count of the number of molecules added.

        :param self: class instance
        :param molecule_name: name of molecule
        :param variable: variable to update a dict
        :param variable_dict: the dictionary where updated variables are added
        """
        # add molecule name to dicts if it doesn't exist
        if (nearest, molecule_name) not in variable_dict:
            variable_dict[(nearest, molecule_name)] = variable
        else:
            # get running average of forces and torques
            stored_variable = variable_dict[(nearest, molecule_name)]
            stored_count = count_dict[(nearest, molecule_name)]
            updated_variable = (stored_variable * stored_count + variable) / (
                stored_count + 1
            )
            # update dictionaries with running averages
            variable_dict[(nearest, molecule_name)] = updated_variable

    def update_counts(self, nearest, molecule_name, count_dict):
        """Update the counts in dictionary"""
        if (nearest, molecule_name) not in count_dict:
            count_dict[(nearest, molecule_name)] = 1
        else:
            count_dict[(nearest, molecule_name)] += 1


def get_torques(molecule, center_of_mass, rotation_axes, MI_axis):
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


def get_rotated_sum_forces(molecule, rotation_axes):
    """
    Rotated the forces for a given seletion of atoms along a particular rotation
    axes (3,3)

    :param molecule: mdanalysis instance of molecule
    :param rotation_axes: a (3,3) array to rotate forces along
    """
    forces_summed = np.sum(molecule.forces, axis=0)
    rotated_sum_forces = np.tensordot(forces_summed, rotation_axes.T, axes=1)
    return rotated_sum_forces


def get_mass_weighted_forces(molecule, rotation_axes):
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


def get_covariance_matrix(ft, halve=0.5):
    """
    Get the outer product of the mass weighted forces or torques (ft) and
    half values if halve=True

    :param ft: (3,) array of either mass weighted forces or torques
    :param halve: Boolean to set weather covariance matrix should be halved
        (i.e. divide by :math:`2^2`)
    """
    cov_matrix = np.outer(ft, ft)
    if halve:
        cov_matrix = cov_matrix * (halve**2)
    return cov_matrix


def get_axes(molecule, molecule_scale):
    """
    From a selection of atoms, get the ordered principal axes (3,3) and
    the ordered moment of inertia axes (3,) for that selection of atoms

    :param molecule: mdanalysis instance of molecule
    """
    # default moment of inertia
    moment_of_inertia = molecule.moment_of_inertia()
    if molecule_scale == "single_UA":
        pass  # moment_of_inertia = molecule.moment_of_inertia()
    if molecule_scale == "multiple_UAs":
        UAs = find_molecule_UAs(molecule)
        center_of_mass = molecule.center_of_mass()
        masses = get_UA_masses(molecule)
        moment_of_inertia = MOI(center_of_mass, UAs.positions, masses)
    principal_axes = molecule.principal_axes()
    # diagonalise moment of inertia tensor here
    # pylint: disable=unused-variable
    eigenvalues, eigenvectors = LA.eig(moment_of_inertia)
    # principal axes = eigenvectors.T[order])
    # comment: could get principal axes from transformed eigenvectors
    #           but would need to sort out directions, so use MDAnalysis
    #           function instead

    # sort eigenvalues of moi tensor by largest to smallest magnitude
    order = abs(eigenvalues).argsort()[::-1]  # decending order
    principal_axes = principal_axes[order]
    MOI_axis = eigenvalues[order]

    return principal_axes, MOI_axis


def get_UA_masses(molecule):
    """
    For a given molecule, return a list of masses of UAs
    (combination of the heavy atoms + bonded hydrogen atoms. This list is used to
    get the moment of inertia tensor for molecules larger than one UA
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


def MOI(CoM, positions, masses):
    """
    Use this function to calculate moment of inertia for cases where the
    mass list will contain masses of UAs rather than individual atoms and
    the postions will be those for the UAs only (excluding the H atoms
    coordinates).
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


def get_custom_axes(a, b_list, c, dimensions):
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
        ab_vector = get_vector(a, b, dimensions)
        # scale vector with distance
        ab_dist = np.sqrt((ab_vector**2).sum(axis=-1))
        scaled_vector = np.divide(ab_vector, ab_dist)
        axis1 += scaled_vector  # ab_vector

    if len(b_list) > 2:
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
    heavy_bonded, light_bonded = find_bonded_atoms(atom.index, system)
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


def guess_length_scale(molecule):
    """Guess what the length scale of the molecule is"""
    molecule_scale = None
    UAs = find_molecule_UAs(molecule)
    if len(UAs) == 1:
        molecule_scale = "single_UA"
    elif len(UAs) > 1:
        if len(molecule.atoms.residues) > 1:
            molecule_scale = "polymer"
        else:
            molecule_scale = "multiple_UAs"
    else:
        molecule_scale = "no_UA"
    return molecule_scale


def get_forces_torques(covariances, molecule, nearest, system):
    # pylint: disable=too-many-locals
    """
    Calculate the covariance matrices of molecules and populate these
    in the covariances instance.
    """
    # 1. get the length scale of the molecule
    molecule_scale = guess_length_scale(molecule)
    # 1b. get the value to scale the forces and torque matrices with
    if molecule_scale == "single_UA":
        scale_covariance = 0.5
    else:
        # don't scale larger molecules automatically as there may be longer
        # lengthscales in the hierarchy
        scale_covariance = 1
    # 2. Get the axes of the molecule
    principal_axes, MOI_axis = get_axes(molecule, molecule_scale)
    # 3. Get the center point of the molecule
    center_of_mass = molecule.center_of_mass()
    # 4. calculate the torque from the forces and axes
    torque = get_torques(molecule, center_of_mass, principal_axes, MOI_axis)
    # 5. calculate the mass weighted forces
    mass_weighted_force = get_mass_weighted_forces(molecule, principal_axes)
    # 6. calculate the covariance matrices
    F_cov_matrix = get_covariance_matrix(mass_weighted_force, scale_covariance)
    T_cov_matrix = get_covariance_matrix(torque, scale_covariance)
    # add the covariances to the class instance
    covariances.add_data(nearest, molecule.resnames[0], F_cov_matrix, T_cov_matrix)

    # not applicable to water
    if molecule_scale == "multiple_UAs":
        UAs = find_molecule_UAs(molecule)
        F_cov_matrices, T_cov_matrices = [], []
        for UA in UAs:
            # find the axes based on what the UA is bonded to
            custom_axes, position_vector = get_bonded_axes(
                system, UA, system.dimensions[:3]
            )
            if custom_axes is not None:  # ignore if UA is only bonded to one other UA
                # set the center of mass as the coordinates of the UA
                center_of_mass = UA.position
                # calcuate the torques using the custom axes based on bonds
                torque = get_torques(
                    molecule, center_of_mass, custom_axes, position_vector
                )
                # calcuate the mass weighted forces using the custom axes based on bonds
                mass_weighted_force = get_mass_weighted_forces(molecule, custom_axes)
                # calculate and append the covariances matrices
                F_cov_matrix = get_covariance_matrix(
                    mass_weighted_force, scale_covariance
                )
                T_cov_matrix = get_covariance_matrix(torque, scale_covariance)
                F_cov_matrices.append([F_cov_matrix])
                T_cov_matrices.append([T_cov_matrix])
        # populate the covariances to the class instance
        # TO-DO: set a class instance specifically for the UA length scale of molecules > 1UA
        covariances.add_data(
            molecule,
            molecule,
            np.concatenate(F_cov_matrices, axis=0),
            np.concatenate(T_cov_matrices, axis=0),
        )
