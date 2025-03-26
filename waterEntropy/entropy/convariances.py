"""
Store the rotated mass-weighted forces and inertia-weighted torques in this class.
"""

import numpy as np

from waterEntropy.utils.helpers import nested_dict


class CovarianceCollection:
    """
    Class for covariance matrices molecules, grouped by what their closest
    non-like molecule is.
    """

    def __init__(self):
        self.forces = nested_dict()
        self.torques = nested_dict()
        self.counts = nested_dict()

    def add_data(
        self, nearest, molecule_name: str, force: np.ndarray, torque: np.ndarray
    ):
        """
        Add force, torque covariances to the class dictionaries

        :param nearest: the value for grouping molecules into a class. e.g. the
            name of the nearest molecule
        :param molecule_name: the name of the molecule being populated into the
            dictionaries
        :param force: array of the mass-weighted forces of the molecule
        :param torque: array of the inertia-weighted torques of the molecule
        """
        # order is important, update count last
        CovarianceCollection.populate_dicts(
            self, nearest, molecule_name, force, self.forces, self.counts
        )
        CovarianceCollection.populate_dicts(
            self, nearest, molecule_name, torque, self.torques, self.counts
        )
        CovarianceCollection.update_counts(self, nearest, molecule_name, self.counts)

    def populate_dicts(
        self, nearest, molecule_name, variable, variable_dict, count_dict
    ):
        # pylint: disable=too-many-arguments
        """
        For a given molecule, append the summed, weighted and rotated forces,
        and the torques for the whole molecule. Add as a running average by
        keeping count of the number of molecules added.

        :param self: class instance
        :param nearest: the value for grouping molecules into a class. e.g. the
            name of the nearest molecule
        :param molecule_name: name of molecule
        :param variable: variable to update a dict
        :param variable_dict: the dictionary where updated variables are added
        :param count_dict: the dictionary where counts are updated
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
        """Update the counts in dictionary

        :param nearest: the value for grouping molecules into a class. e.g. the
            name of the nearest molecule
        :param molecule_name: name of molecule
        :param count_dict: the dictionary where counts are updated
        """
        if (nearest, molecule_name) not in count_dict:
            count_dict[(nearest, molecule_name)] = 1
        else:
            count_dict[(nearest, molecule_name)] += 1
