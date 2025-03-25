"""
Get the rotated mass-weighted forces and inertia-weighted torques from
transformed atom positons and forces and save these as covariance matrices
in a Covariances class instance.
"""

import numpy as np

import waterEntropy.maths.transformations as Transformation
from waterEntropy.utils.helpers import nested_dict
from waterEntropy.utils.selections import find_molecule_UAs


class Covariances:
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
        Covariances.populate_dicts(
            self, nearest, molecule_name, force, self.forces, self.counts
        )
        Covariances.populate_dicts(
            self, nearest, molecule_name, torque, self.torques, self.counts
        )
        Covariances.update_counts(self, nearest, molecule_name, self.counts)

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
    principal_axes, MOI_axis = Transformation.get_axes(molecule, molecule_scale)
    # 3. Get the center point of the molecule
    center_of_mass = molecule.center_of_mass()
    # 4. calculate the torque from the forces and axes
    torque = Transformation.get_torques(
        molecule, center_of_mass, principal_axes, MOI_axis
    )
    # 5. calculate the mass weighted forces
    mass_weighted_force = Transformation.get_mass_weighted_forces(
        molecule, principal_axes
    )
    # 6. calculate the covariance matrices
    F_cov_matrix = Transformation.get_covariance_matrix(
        mass_weighted_force, scale_covariance
    )
    T_cov_matrix = Transformation.get_covariance_matrix(torque, scale_covariance)
    # add the covariances to the class instance
    covariances.add_data(nearest, molecule.resnames[0], F_cov_matrix, T_cov_matrix)

    # not applicable to water
    if molecule_scale == "multiple_UAs":
        UAs = find_molecule_UAs(molecule)
        F_cov_matrices, T_cov_matrices = [], []
        for UA in UAs:
            # find the axes based on what the UA is bonded to
            custom_axes, position_vector = Transformation.get_bonded_axes(
                system, UA, system.dimensions[:3]
            )
            if custom_axes is not None:  # ignore if UA is only bonded to one other UA
                # set the center of mass as the coordinates of the UA
                center_of_mass = UA.position
                # calcuate the torques using the custom axes based on bonds
                torque = Transformation.get_torques(
                    molecule, center_of_mass, custom_axes, position_vector
                )
                # calcuate the mass weighted forces using the custom axes based on bonds
                mass_weighted_force = Transformation.get_mass_weighted_forces(
                    molecule, custom_axes
                )
                # calculate and append the covariances matrices
                F_cov_matrix = Transformation.get_covariance_matrix(
                    mass_weighted_force, scale_covariance
                )
                T_cov_matrix = Transformation.get_covariance_matrix(
                    torque, scale_covariance
                )
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
