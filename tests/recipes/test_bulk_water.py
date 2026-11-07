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
        np.array([672039.16362729, 942109.73348218, 1512552.2804507]),
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        np.linalg.eigvalsh(torques),
        np.array([6407943.77132577, 8467981.43061079, 14095159.42521983]),
        rtol=1e-5,
        atol=1e-5,
    )

    assert np.allclose(
        forces,
        np.array(
            [
                [677734.94716348, 15739.92772357, -39821.31124937],
                [15739.92772357, 1497942.10124106, -88748.89476841],
                [-39821.31124937, -88748.89476841, 951024.12915563],
            ]
        ),
    )
    assert np.allclose(
        torques,
        np.array(
            [
                [8527024.55153785, -245284.12233727, 694323.21516024],
                [-245284.12233727, 6436426.29332659, -54491.9337198],
                [694323.21516024, -54491.9337198, 14007633.78229195],
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

    assert np.allclose(sorted(Strans), np.array([14.30412666, 16.24104262, 17.6305319]))
    assert np.allclose(sum(Strans), 48.175701175567795)
    assert np.allclose(
        sorted(Srot),
        np.array([5.680208099397993, 7.513307654068938, 8.564650609536658]),
    )
    assert np.allclose(sum(Srot), 21.75816636300359)
    assert np.allclose(trans_freqs, np.array([672039, 942109, 1512552]))
    assert np.allclose(rot_freqs, np.array([14095159, 8467981, 6407943]))
