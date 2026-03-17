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
            6.053509907760105,
            6.053509907760105,
            6.053509907760105,
            0.2291068586055006,
        ]
    )


@BULK_WATER_ENTROPY_DICTS
def test_covariances(bulk_water_entropy_dicts):
    """Test the covariance matrices"""

    covariances = bulk_water_entropy_dicts[1]
    forces = covariances.forces[("WAT", "WAT")]
    torques = covariances.torques[("WAT", "WAT")]
    count = covariances.counts[("WAT", "WAT")]

    assert np.allclose(
        forces,
        np.array(
            [
                [677766.79725371, 57015.33553172, 1713.75817016],
                [57015.33553172, 1498012.49688953, 37806.43086811],
                [1713.75817016, 37806.43086811, 951068.82244533],
            ]
        ),
    )
    assert np.allclose(
        torques,
        np.array(
            [
                [8.82222234e06, -1.57822726e05, 5.92784745e05],
                [-1.57822726e05, 6.49223541e06, 6.08836668e03],
                [5.92784745e05, 6.08836668e03, 1.41024728e07],
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

    assert np.allclose(Strans, np.array([17.59459292, 14.34269432, 16.2012916]))
    assert np.allclose(sum(Strans), 48.1385788374531)
    assert np.allclose(Srot, np.array([7.36084486, 8.51428332, 5.6781004]))
    assert np.allclose(sum(Srot), 21.553228577722663)
    assert np.allclose(trans_freqs, np.array([[677766, 1498012, 951068]]))
    assert np.allclose(rot_freqs, np.array([[8822222, 6492235, 14102472]]))
