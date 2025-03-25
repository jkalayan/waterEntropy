""" Tests for waterEntropy interfacial solvent functions in neighbours."""

import numpy as np
import pytest

from tests.input_files import load_inputs
import waterEntropy.recipes.interfacial_solvent as GetSolvent

# get mda universe for arginine in solution
system = load_inputs.get_amber_arginine_soln_universe()
Sorient_dict, covariances, vibrations, frame_solvent_indices = (
    GetSolvent.get_interfacial_water_orient_entropy(system, start=0, end=4, step=2)
)


def test_Sorient_dict():
    """Test outputted orientational entropy values"""
    # resid: {resname = [Sorient, count]}
    assert Sorient_dict[1]["ACE"] == pytest.approx([3.0489096643431974, 17])
    assert Sorient_dict[2]["ARG"] == pytest.approx([1.4543131445108897, 31])
    assert Sorient_dict[3]["NME"] == pytest.approx([2.038000764549012, 13])


def test_covariances():
    "Test the covariance matrices"

    forces = covariances.forces[("ACE_1", "WAT")]
    torques = covariances.torques[("ACE_1", "WAT")]
    count = covariances.counts[("ACE_1", "WAT")]

    assert np.allclose(
        forces,
        np.array(
            [[475329, -9871, 89940], [-9871, 1546614, 283755], [89940, 283755, 1053114]]
        ),
    )
    assert np.allclose(
        torques,
        np.array(
            [
                [58104052, 11723947, 3183189],
                [11723947, 30729540, 84751],
                [3183189, 84751, 51102359],
            ]
        ),
    )
    assert count == 17


def test_vibrations():
    "Test the vibrational entropies"
    Strans = vibrations.translational_S[("ACE_1", "WAT")]
    Srot = vibrations.rotational_S[("ACE_1", "WAT")]
    trans_freqs = vibrations.translational_freq[("ACE_1", "WAT")]
    rot_freqs = vibrations.rotational_freq[("ACE_1", "WAT")]

    assert np.allclose(Strans, np.array([19.05833289, 14.212621, 15.78320831]))
    assert np.allclose(Srot, np.array([1.66928353, 3.21696628, 1.94000706]))
    assert np.allclose(trans_freqs, np.array([[475329, 1546614, 1053114]]))
    assert np.allclose(rot_freqs, np.array([[58104052, 30729540, 51102359]]))


def test_frame_solvent_indices():
    """Test the get interfacial water orient entropy function"""
    # frame: {resname: {resid = [shell indices]}}
    assert frame_solvent_indices[0].get("ACE").get(1) == [
        621,
        888,
        1038,
        1143,
        1413,
        1737,
        1800,
    ]
    assert frame_solvent_indices[0].get("ARG").get(2) == [
        54,
        168,
        237,
        369,
        747,
        1797,
        2004,
        2019,
        2130,
        2262,
        2640,
        2646,
        2688,
    ]
    assert frame_solvent_indices[0].get("NME").get(3) == [486, 489, 834, 879, 984, 1698]
    assert frame_solvent_indices[2].get("ACE").get(1) == [
        324,
        621,
        888,
        1800,
        2085,
        2130,
        2565,
        2652,
        2694,
        2721,
    ]
    assert frame_solvent_indices[2].get("ARG").get(2) == [
        54,
        168,
        237,
        243,
        642,
        726,
        843,
        849,
        1479,
        1698,
        2019,
        2136,
        2244,
        2253,
        2262,
        2265,
        2640,
        2646,
    ]
    assert frame_solvent_indices[2].get("NME").get(3) == [
        282,
        807,
        834,
        879,
        1413,
        2190,
        2688,
    ]
