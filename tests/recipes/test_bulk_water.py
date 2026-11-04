"""Tests for waterEntropy bulk water functions in neighbours."""

import numpy as np
import pytest

import waterEntropy.recipes.bulk_water as GetBulk
from tests.input_files import load_inputs


def bulk_water_entropy():
    """Return the entropy dictionaries calcalated via serial process"""
    system = load_inputs.get_amber_arginine_soln_universe()
    Sorient_dict, covariances, vibrations = GetBulk.get_bulk_water_orient_entropy(
        system, start=0, end=1, step=1
    )
    return (
        Sorient_dict,
        covariances,
        vibrations,
    )


BULK_WATER_ENTROPY_DICTS = pytest.mark.parametrize(
    "bulk_water_entropy_dicts",
    [bulk_water_entropy()],
)


@BULK_WATER_ENTROPY_DICTS
def test_Sorient_dict(bulk_water_entropy_dicts):
    """Test outputted orientational entropy values of solvent molecules around a given solute molecule"""
    # resid: {resname = [Sorient, count]}
    Sorient_dict = bulk_water_entropy_dicts[0]
    assert Sorient_dict["WAT"]["WAT"]["Sorient_ave"] == pytest.approx(
        11.295694179188464
    )
    assert Sorient_dict["WAT"]["WAT"]["tot_count"] == pytest.approx(867)
    assert Sorient_dict["WAT"]["WAT"]["N_c_ave"] == pytest.approx(6.8788927335640135)
    assert Sorient_dict["WAT"]["WAT"]["N_w_ave"] == pytest.approx(6.8788927335640135)
    assert Sorient_dict["WAT"]["WAT"]["Nc_eff_ave"] == pytest.approx(6.8788927335640135)
    assert Sorient_dict["WAT"]["WAT"]["pbias_ave"] == pytest.approx(0.25)
    assert Sorient_dict["WAT"]["WAT"]["Sorient_Nc_ave"] == pytest.approx(
        11.295694179188464
    )
    assert Sorient_dict["WAT"]["WAT"]["Sorient_Nw_ave"] == pytest.approx(
        11.295694179188464
    )
    assert Sorient_dict["WAT"]["WAT"]["Sorient_HB_ave2"] == pytest.approx(
        11.295694179188464
    )
    assert Sorient_dict["WAT"]["WAT"]["Sorient_Nc_ave2"] == pytest.approx(
        11.295694179188464
    )
    assert Sorient_dict["WAT"]["WAT"]["Sorient_Nw_ave2"] == pytest.approx(
        11.295694179188464
    )


@BULK_WATER_ENTROPY_DICTS
def test_covariances(bulk_water_entropy_dicts):
    """Test the covariance matrices"""

    covariances = bulk_water_entropy_dicts[1]
    forces = covariances.forces[("WAT", "WAT")]
    torques = covariances.torques[("WAT", "WAT")]
    count = covariances.counts[("WAT", "WAT")]

    np.testing.assert_allclose(
        np.linalg.eigvalsh(forces),
        np.array([673788.02213212, 948440.87510386, 1504472.28032419]),
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        np.linalg.eigvalsh(torques),
        np.array([6423718.64945265, 8479586.95316266, 14067779.02454107]),
        rtol=1e-5,
        atol=1e-5,
    )

    assert count == 867


@BULK_WATER_ENTROPY_DICTS
def test_vibrations(bulk_water_entropy_dicts):
    "Test the vibrational entropies"
    vibrations = bulk_water_entropy_dicts[2]
    Strans = vibrations.translational_S[("WAT", "WAT")]
    Srot = vibrations.rotational_S[("WAT", "WAT")]
    trans_freqs = vibrations.translational_freq[("WAT", "WAT")]
    rot_freqs = vibrations.rotational_freq[("WAT", "WAT")]

    assert np.allclose(Strans, np.array([14.32595231, 17.61982376, 16.21354637]))
    assert np.allclose(sum(Strans), 48.159322433317016)
    assert np.allclose(Srot, np.array([5.68693308, 7.5082148, 8.55525875]))
    assert np.allclose(sum(Srot), 21.75040663028019)
    assert np.allclose(trans_freqs, np.array([1504472, 673788, 948440]))
    assert np.allclose(rot_freqs, np.array([14067779, 8479586, 6423718]))
