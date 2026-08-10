""" Tests for waterEntropy bulk water functions in neighbours."""

import numpy as np
import pytest

from tests.input_files import load_inputs
import waterEntropy.recipes.bulk_water as GetBulk


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
    assert Sorient_dict["WAT"]["WAT"] == pytest.approx(
        [
            11.295694179188464,
            867,
            6.8788927335640135,
            6.8788927335640135,
            6.8788927335640135,
            0.25,
            11.295694179188464,
            11.295694179188464,
            11.295694179188464,
            11.295694179188464,
            11.295694179188464,
        ]
    )


@BULK_WATER_ENTROPY_DICTS
def test_covariances(bulk_water_entropy_dicts):
    """Test the covariance matrices"""

    covariances = bulk_water_entropy_dicts[1]
    forces = covariances.forces[("WAT", "WAT")]
    torques = covariances.torques[("WAT", "WAT")]
    count = covariances.counts[("WAT", "WAT")]

    print("forces", forces)
    print("torques", torques)
    print("count", count)

    assert np.allclose(
        forces,
        np.array(
            [
                [677734.94716348, 57012.65622729, 1713.67763604],
                [57012.65622729, 1497942.10124106, 37804.65424194],
                [1713.67763604, 37804.65424194, 951024.12915563],
            ]
        ),
    )
    assert np.allclose(
        torques,
        np.array(
            [
                [8527024.55153786, -154805.72447386, 576257.37509633],
                [-154805.72447386, 6436426.29332658, 56359.58996768],
                [576257.37509633, 56359.58996768, 14007633.78229195],
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

    assert np.allclose(Strans, np.array([14.32595231, 17.61982376, 16.21354637]))
    assert np.allclose(sum(Strans), 48.159322433317016)
    assert np.allclose(Srot, np.array([5.68693308, 7.5082148, 8.55525875]))
    assert np.allclose(sum(Srot), 21.75040663028019)
    assert np.allclose(trans_freqs, np.array([1504472, 673788, 948440]))
    assert np.allclose(rot_freqs, np.array([14067779, 8479586, 6423718]))
