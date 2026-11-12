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
    """
    Test outputted orientational entropy values of solvent molecules around a
    given solute molecule
    """
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
        np.array([670238.35897205, 956264.48039086, 1500198.33819726]),
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        np.linalg.eigvalsh(torques),
        np.array([6429679.41993371, 8474535.92813576, 14066869.27908691]),
        rtol=1e-5,
        atol=1e-5,
    )

    assert np.allclose(
        forces,
        np.array(
            [
                [677734.94716348, -1664.33742052, 45818.66526039],
                [-1664.33742052, 1497942.10124106, -34999.54642979],
                [45818.66526039, -34999.54642979, 951024.12915563],
            ]
        ),
    )
    assert np.allclose(
        torques,
        np.array(
            [
                [8527024.55153785, -97775.14870001, 567575.66039858],
                [-97775.14870001, 6436426.29332659, 101015.90117385],
                [567575.66039858, 101015.90117385, 14007633.78229195],
            ]
        ),
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

    assert np.allclose(
        sorted(Strans), np.array([14.33754546, 16.17982393, 17.64158753])
    )
    assert np.allclose(sum(Strans), 48.158956923942)
    assert np.allclose(
        sorted(Srot),
        np.array([5.687156789928, 7.5104303887, 8.55171640995]),
    )
    assert np.allclose(sum(Srot), 21.749303588578)
    assert np.allclose(
        sorted(trans_freqs), np.array([670238.36, 956264.48, 1500198.34])
    )
    assert np.allclose(
        sorted(rot_freqs), np.array([6429679.42, 8474535.93, 14066869.28])
    )
