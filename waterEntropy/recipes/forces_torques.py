"""
Get the rotated mass-weighted forces and inertia-weighted torques from
transformed atom positons and forces and save these as covariance matrices
in a Covariances class instance.
"""

import numpy as np

from waterEntropy.entropy.convariances import CovarianceCollection
import waterEntropy.maths.transformations as Transformation
import waterEntropy.utils.selections as Select


def get_forces_torques(
    covariances: CovarianceCollection, molecule, nearest: str, system
):
    # pylint: disable=too-many-locals
    """
    Calculate the covariance matrices of molecules and populate these
    in the covariances instance. Molecules are grouped based on the molecule
    name and the name of the molecule that is nearest to it.

    :param covariances: instance of CovarianceCollection class
    :param molecule: MDAnalysis instance of a molecule
    :param nearest: name of nearest molecule
    :param system: MDAnalysis instance of whole system
    """
    # 1. get the length scale of the molecule
    molecule_scale = Select.guess_length_scale(molecule)
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
        UAs = Select.find_molecule_UAs(molecule)
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
