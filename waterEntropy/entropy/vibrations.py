"""
Store vibrational entropy from covariance matrices
"""

import numpy as np
from numpy import linalg as LA

from waterEntropy.entropy.convariances import CovarianceCollection
from waterEntropy.utils.helpers import nested_dict


class Vibrations:
    r"""
    Store molecule vibrational entropy information here, where

    .. math::
        S^{\mathrm{vib}} = S^{\mathrm{trans}} + S^{\mathrm{rot}}

    Entropies are calculated from the covariance matrices inputted in the
    add_data method.
    """

    # constants
    BOLTZMANN = 1.38064852e-23  # (J / K)
    HBAR = 1.0545718e-34  # (J s)
    AVOGADRO = 6.022e23
    GAS_CONSTANT = 8.314  # (J/ (K mol)
    # conversions
    CALORIE_TO_JOULE = 4184
    KG_TO_AMU = 6.02214086e26
    PER_MOLE_TO_PER_MOLECULE = 6.02214086e23
    ANGSTROM_TO_METRE = 1e-10
    KJ_TO_J = 1000
    NANOMETRE_TO_METRE = 1e-9

    def __init__(self, temperature, force_units):
        self.temperature = temperature
        self.force_units = force_units
        self.translational_freq = nested_dict()
        self.rotational_freq = nested_dict()
        self.translational_S = nested_dict()
        self.rotational_S = nested_dict()

    def add_data(self, covariances: CovarianceCollection, diagonalise=True):
        """
        Calculate and add the frequencies and vibrational entropies to class
        instance

        :param covariances: instance of CovarianceCollection class
        :param diagonalise: diagonalise the covariance matrix rather than getting
            the eigenvalues and eigenvectors
        """
        for (nearest, molecule_name), force_covariance in covariances.forces.items():
            torque_covariance = covariances.torques[(nearest, molecule_name)]
            force_Svib, force_frequency = Vibrations.get_data(
                self, force_covariance, diagonalise
            )
            torque_Svib, torque_frequency = Vibrations.get_data(
                self, torque_covariance, diagonalise
            )
            Vibrations.populate_dicts(
                self, nearest, molecule_name, force_Svib, self.translational_S
            )
            Vibrations.populate_dicts(
                self, nearest, molecule_name, torque_Svib, self.rotational_S
            )
            Vibrations.populate_dicts(
                self, nearest, molecule_name, force_frequency, self.translational_freq
            )
            Vibrations.populate_dicts(
                self, nearest, molecule_name, torque_frequency, self.rotational_freq
            )

    def get_data(self, covariance: CovarianceCollection, diagonalise: bool):
        """
        Get the frequencies and vibrational entropies from the covariance matrix

        :param covariances: instance of CovarianceCollection class
        :param diagonalise: whether to diagonalise the covariance matrix
        """
        # convert the covariance matrix to the correct units
        if self.force_units == "gromacs":
            covariance *= self.kJ_conversion()
        else:
            covariance *= self.kcal_conversion()
        Svib, frequency = Vibrations.calculate_entropies(self, covariance, diagonalise)

        return Svib, frequency

    def populate_dicts(self, nearest, molecule_name: str, variable, variable_dict):
        """
        Add frequencies / entropies to class instance dictionaries

        :param nearest: the value for grouping molecules into a class. e.g. the
            name of the nearest molecule
        :param molecule_name: the name of the molecule being populated into the
            dictionaries
        :param variable: the variable being added to the variable dictionary
        :param variable_dict: the dictionary being updated with a variable
        """
        if (nearest, molecule_name) not in variable_dict:
            variable_dict[(nearest, molecule_name)] = variable
        else:
            np.append(variable_dict[(nearest, molecule_name)], variable)

    def kJ_conversion(self):
        """
        Convert force covariance matrix from kJ/(mol nm m^2) to J/mol
        SI: kg.m/s^2 = N
        The forces are squared, so do the same for the constant
        """
        numerator = (self.KJ_TO_J**2) * self.KG_TO_AMU
        denominator = (self.AVOGADRO**2) * (self.NANOMETRE_TO_METRE**2)
        conversion = numerator / denominator

        return conversion

    def kcal_conversion(self):
        """
        Convert force covariance matrix from Kcal/(mol Ang m^2) to J/mol
        SI: kg.m/s^2 = N
        The forces are squared, so do the same for the constant
        """
        numerator = (self.CALORIE_TO_JOULE**2) * self.KG_TO_AMU
        denominator = (self.AVOGADRO**2) * (self.ANGSTROM_TO_METRE**2)
        conversion = numerator / denominator

        return conversion

    def calculate_entropies(self, covariance: CovarianceCollection, diagonalise: bool):
        r"""
        Calculate the vibrational entropies from the equation of a quantum
        harmonic oscillator:

        .. math::
            S^{\mathrm{vib}} = k_{\mathrm{B}} \sum_{i=1}^{N_\mathrm{vib}} \\
            \Bigg( \frac{h\nu_i/k_{\mathrm{B}}T}{e^{h\nu_i/k_{\mathrm{B}}T}-1} \\
            - \ln(1 - e^{-h\nu_i/k_{\mathrm{B}}T} ) \Bigg)

        where:

        - :math:`k_{\mathrm{B}}` is the Boltzmann constant :math:`\mathrm{J/K}` (Joule per Kelvin),

        :param covariances: instance of CovarianceCollection class
        :param diagonalise: whether to diagonalise the covariance matrix
        """
        frequency, eigenvalues = Vibrations.calculate_frequencies(
            self, covariance, diagonalise
        )
        a = (self.HBAR * frequency) / (self.BOLTZMANN * self.temperature)
        b = np.where(a != 0, np.exp(a) - 1, 0.0)
        c = np.where(a != 0, np.log(1 - np.exp(-a)), 0.0)
        Svib = np.where((b != 0) & (c != 0), a / b - c, 0)

        return Svib * self.GAS_CONSTANT, eigenvalues

    def calculate_frequencies(
        self, covariance: CovarianceCollection, diagonalise: bool
    ):
        """
        Calculate the frequencies for each molecule from the force and torque
        covariance matrices.

        :param covariances: instance of CovarianceCollection class
        :param diagonalise: whether to diagonalise the covariance matrix
        """
        if diagonalise:
            eigenvalues = covariance.diagonal()  # get matrix diagonals only
        else:
            eigenvalues, _eigenvectors = LA.eig(covariance)
        # Replace eigenvalues numbers with 0
        filtered_eigenvalues = np.where(np.isreal(eigenvalues), eigenvalues.real, 0)
        # Replace negative eigenvalues with 0
        filtered_eigenvalues = np.where(
            filtered_eigenvalues >= 0, filtered_eigenvalues, 0
        )
        # calculate frequencies from filtered eigenvalues
        frequency = (filtered_eigenvalues / (self.BOLTZMANN * self.temperature)) ** 0.5

        return frequency, eigenvalues


def print_Svib_data(vibrations: Vibrations, covariances: CovarianceCollection):
    """
    Print the orientational entropies of interfacial solvent

    :param vibrations: instance of Vibrations class
    :param covariances: instance of CovarianceCollection class
    """
    for near_solvent_name, Strans in vibrations.translational_S.items():
        near = near_solvent_name[0]
        solvent_name = near_solvent_name[1]
        # forces = covariances.forces[near_solvent_name]
        # torques = covariances.torques[near_solvent_name]
        Strans = vibrations.translational_S[near_solvent_name]
        Srot = vibrations.rotational_S[near_solvent_name]
        # trans_freqs = vibrations.translational_freq[near_solvent_name]
        # rot_freqs = vibrations.rotational_freq[near_solvent_name]
        counts = covariances.counts[near_solvent_name]
        print(near, solvent_name, sum(Strans), sum(Srot), counts)
