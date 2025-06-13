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
frame_solvent_shells = GetSolvent.get_interfacial_shells(system, start=0, end=4, step=2)


def test_frame_solvent_shells():
    """Test outputted shell indices outputted in frame_solvent_shells dictionary
    from a first shell solvent"""
    # frame: {atom_idx: [shell_indices]}
    assert len(frame_solvent_shells[0].keys()) == 32
    assert len(frame_solvent_shells[2].keys()) == 36
    assert frame_solvent_shells[0][1024] == [415, 931, 1318, 1618, 25, 1474, 1282]
    assert frame_solvent_shells[2][1024] == [1318, 460, 580, 25, 1714, 19, 2497]


def test_Sorient_dict():
    """Test outputted orientational entropy values of solvent molecules around a given solute molecule"""
    # resid: {resname = [Sorient, count]}
    assert Sorient_dict[1]["ACE"] == pytest.approx([2.2473807716251804, 13])
    assert Sorient_dict[2]["ARG"] == pytest.approx([2.6481382191024245, 35])
    assert Sorient_dict[3]["NME"] == pytest.approx([0.9503950365967891, 12])


def test_covariances():
    "Test the covariance matrices"

    forces = covariances.forces[("ACE_1", "WAT")]
    torques = covariances.torques[("ACE_1", "WAT")]
    count = covariances.counts[("ACE_1", "WAT")]

    assert np.allclose(
        forces,
        np.array(
            [
                [14436849, 712686, -2141242],
                [712686, 19792305, 8921010],
                [-2141242, 8921010, 9880023],
            ]
        ),
    )
    assert np.allclose(
        torques,
        np.array(
            [
                [2.89828793e08, 5.06914732e07, -2.92085655e07],
                [5.06914732e07, 5.83455839e07, -1.13204566e06],
                [-2.92085655e07, -1.13204566e06, 1.48596049e08],
            ]
        ),
    )
    assert count == 13


def test_vibrations():
    "Test the vibrational entropies"
    Strans = vibrations.translational_S[("ACE_1", "WAT")]
    Srot = vibrations.rotational_S[("ACE_1", "WAT")]
    trans_freqs = vibrations.translational_freq[("ACE_1", "WAT")]
    rot_freqs = vibrations.rotational_freq[("ACE_1", "WAT")]

    assert np.allclose(Strans, np.array([5.59725172, 4.54547502, 6.94449285]))
    assert np.allclose(Srot, np.array([0.07211633, 1.6609092, 0.37677331]))
    assert np.allclose(trans_freqs, np.array([[14436849, 19792305, 9880023]]))
    assert np.allclose(
        rot_freqs, np.array([[2.89828793e08, 5.83455839e07, 1.48596049e08]])
    )


def test_frame_solvent_indices():
    """Test the get interfacial water orient entropy function"""
    # frame: {resname: {resid = [shell indices]}}
    assert frame_solvent_indices[0].get("ACE").get(1) == [
        121,
        235,
        481,
        505,
        1639,
        2260,
        2314,
    ]
    assert frame_solvent_indices[0].get("ARG").get(2) == [
        55,
        244,
        274,
        862,
        931,
        1024,
        1165,
        1282,
        1474,
        1855,
        1912,
        2005,
        2056,
        2077,
        2245,
        2497,
    ]
    assert frame_solvent_indices[0].get("NME").get(3) == [
        85,
        205,
        265,
        1057,
        1246,
        1753,
    ]
    assert frame_solvent_indices[0].get("Cl-").get(4) == [460, 1669, 2041]
    assert frame_solvent_indices[2].get("ACE").get(1) == [
        505,
        664,
        1036,
        1147,
        1165,
        2260,
    ]
    assert frame_solvent_indices[2].get("ARG").get(2) == [
        49,
        235,
        274,
        346,
        409,
        475,
        580,
        862,
        931,
        1024,
        1048,
        1228,
        1855,
        1858,
        1891,
        2056,
        2404,
        2497,
        2605,
    ]
    assert frame_solvent_indices[2].get("NME").get(3) == [
        43,
        85,
        205,
        763,
        1246,
        1753,
    ]
    assert frame_solvent_indices[2].get("Cl-").get(4) == [
        460,
        1669,
        1945,
        2005,
        2041,
    ]
