"""Tests for vibrational entropy helpers."""

import numpy as np
import pytest

from waterEntropy.entropy.convariances import CovarianceCollection
from waterEntropy.entropy.vibrations import Vibrations, print_Svib_data


def test_populate_dicts_existing_key():
    """Exercise the existing-key population path."""
    vibrations = Vibrations(300)
    values = {("SOL", "WAT"): np.array([1.0, 2.0])}

    vibrations.populate_dicts("SOL", "WAT", np.array([3.0]), values)

    np.testing.assert_array_equal(values[("SOL", "WAT")], [1.0, 2.0])


def test_conversion_factors():
    """Return positive conversion factors for supported force units."""
    vibrations = Vibrations(300)

    assert vibrations.kJ_conversion() > 0
    assert vibrations.mda_conversion() > 0
    assert vibrations.kcal_conversion() > 0
    assert vibrations.kcal_conversion() == pytest.approx(
        vibrations.mda_conversion()
        * (vibrations.CALORIE_TO_JOULE / vibrations.KJ_TO_J) ** 2
    )


def test_add_data_calculates_entropy_and_frequencies():
    """Calculate and store translational and rotational vibration data."""
    covariances = CovarianceCollection()
    covariance = np.diag([1.0, 2.0, 3.0])
    covariances.add_data("SOL", "WAT", covariance.copy(), covariance.copy())

    vibrations = Vibrations(300)
    vibrations.add_data(covariances, diagonalise=False)

    assert len(vibrations.translational_S[("SOL", "WAT")]) == 3
    assert len(vibrations.rotational_S[("SOL", "WAT")]) == 3
    assert np.all(vibrations.translational_freq[("SOL", "WAT")] >= 0)


def test_calculate_frequencies_diagonalises_and_filters_negative_values():
    """Use covariance diagonals and clamp negative eigenvalues."""
    vibrations = Vibrations(300)
    covariance = np.diag([4.0, -1.0, 0.0])

    frequencies, eigenvalues = vibrations.calculate_frequencies(covariance, True)

    np.testing.assert_array_equal(eigenvalues, [4.0, -1.0, 0.0])
    assert np.all(frequencies >= 0)
    assert frequencies[1] == frequencies[2] == 0


def test_print_s_vib_data(capsys):
    """Print formatted entropy data for named and unnamed nearest solutes."""
    vibrations = Vibrations(300)
    vibrations.translational_S[("ACE_1", "WAT")] = np.array([1.23456, 2.0])
    vibrations.rotational_S[("ACE_1", "WAT")] = np.array([3.0])
    vibrations.translational_S[("WAT", "WAT")] = np.array([4.0])
    vibrations.rotational_S[("WAT", "WAT")] = np.array([5.0])
    covariances = CovarianceCollection()
    covariances.counts[("ACE_1", "WAT")] = 2
    covariances.counts[("WAT", "WAT")] = 3

    print_Svib_data(vibrations, covariances)

    output = capsys.readouterr().out
    assert "1 ACE WAT 3.2346 3.0000 6.2346 2" in output
    assert "N/A WAT WAT 4.0000 5.0000 9.0000 3" in output
