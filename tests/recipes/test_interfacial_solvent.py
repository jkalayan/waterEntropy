""" Tests for waterEntropy interfacial solvent functions in neighbours."""

import numpy as np
import pytest

from tests.input_files import load_inputs
import waterEntropy.recipes.interfacial_solvent as GetSolvent


def serial_interfacial_entropy():
    """Return the entropy dictionaries calcalated via serial process"""
    system = load_inputs.get_amber_arginine_soln_universe()
    Sorient_dict, covariances, vibrations, frame_solvent_indices, n_frames = (
        GetSolvent.get_interfacial_water_orient_entropy(system, start=0, end=4, step=2)
    )
    frame_solvent_shells = GetSolvent.get_interfacial_shells(
        system, start=0, end=4, step=2
    )
    return (
        Sorient_dict,
        covariances,
        vibrations,
        frame_solvent_indices,
        n_frames,
        frame_solvent_shells,
    )


def parallel_interfacial_entropy():
    """Return the entropy dictionaries calcalated via serial process"""
    system = load_inputs.get_amber_arginine_soln_universe()
    Sorient_dict, covariances, vibrations, frame_solvent_indices, n_frames = (
        GetSolvent.get_interfacial_water_orient_entropy(
            system, start=0, end=4, step=2, parallel=True
        )
    )
    return Sorient_dict, covariances, vibrations, frame_solvent_indices, n_frames


INTERFACIAL_ENTROPY_DICTS = pytest.mark.parametrize(
    "interfacial_entropy_dicts",
    [serial_interfacial_entropy(), parallel_interfacial_entropy()],
)


def test_frame_solvent_shells():
    """Test outputted shell indices outputted in frame_solvent_shells dictionary
    from a first shell solvent"""
    frame_solvent_shells = serial_interfacial_entropy()[5]
    # frame: {atom_idx: [shell_indices]}
    assert len(frame_solvent_shells[0].keys()) == 32
    assert len(frame_solvent_shells[2].keys()) == 36
    assert frame_solvent_shells[0][1024] == [415, 931, 1318, 1618, 25, 1474, 1282]
    assert frame_solvent_shells[2][1024] == [1318, 460, 580, 25, 1714, 19, 2497]


@INTERFACIAL_ENTROPY_DICTS
def test_Sorient_dict(interfacial_entropy_dicts):
    """Test outputted orientational entropy values of solvent molecules around a given solute molecule"""
    # resid: {resname = [Sorient, count]}
    Sorient_dict = interfacial_entropy_dicts[0]
    assert Sorient_dict[1]["ACE"] == pytest.approx(
        [
            2.2473807716251804,
            13,
            7.0,
            5.6923076923076925,
            3.2245457772507913,
            0.116959182573051,
            11.549772583950544,
            8.866118517437595,
            1.0246793134504353,
            11.549772583950546,
            8.866118517437595,
        ]
    )
    assert Sorient_dict[2]["ARG"] == pytest.approx(
        [
            2.6481382191024245,
            35,
            7.514285714285714,
            6.0,
            3.464361256421488,
            0.114223362162831,
            12.394889769242496,
            9.388627866311323,
            0.47562453963817075,
            12.394889769242495,
            9.388627866311323,
        ]
    )
    assert Sorient_dict[3]["NME"] == pytest.approx(
        [
            0.9503950365967891,
            12,
            7.333333333333333,
            5.833333333333333,
            2.314863945578231,
            0.0798514739229025,
            12.077905047118842,
            9.183582113622649,
            0.0,
            12.077905047118845,
            9.18358211362265,
        ]
    )


@INTERFACIAL_ENTROPY_DICTS
def test_covariances(interfacial_entropy_dicts):
    """Test the covariance matrices"""

    covariances = interfacial_entropy_dicts[1]
    forces = covariances.forces[("ACE_1", "WAT")]
    torques = covariances.torques[("ACE_1", "WAT")]
    count = covariances.counts[("ACE_1", "WAT")]

    assert np.allclose(
        forces,
        np.array(
            [
                [824648.12, 40709.41, -122310.016],
                [40709.41, 1130557.41, 509577.56],
                [-122310.016, 509577.56, 564357.39],
            ]
        ),
    )
    assert np.allclose(
        torques,
        np.array(
            [
                [16555327.18, 2895550.56, -1668424.15],
                [2895550.56, 3332761.45, -64663.65],
                [-1668424.15, -64663.65, 8487963.46],
            ]
        ),
    )
    assert count == 13


@INTERFACIAL_ENTROPY_DICTS
def test_vibrations(interfacial_entropy_dicts):
    "Test the vibrational entropies"
    vibrations = interfacial_entropy_dicts[2]
    Strans = vibrations.translational_S[("ACE_1", "WAT")]
    Srot = vibrations.rotational_S[("ACE_1", "WAT")]
    trans_freqs = vibrations.translational_freq[("ACE_1", "WAT")]
    rot_freqs = vibrations.rotational_freq[("ACE_1", "WAT")]

    assert np.allclose(Strans, np.array([21.93192349, 16.6612125, 14.52832366]))
    assert np.allclose(sum(Strans), 53.12145964157872)
    assert np.allclose(Srot, np.array([4.95436837, 11.94115717, 7.63322808]))
    assert np.allclose(sum(Srot), 24.528753622255945)
    assert np.allclose(trans_freqs, np.array([[237430, 850514, 1431617]]))
    assert np.allclose(rot_freqs, np.array([[17463732, 2712583, 8199736]]))


@INTERFACIAL_ENTROPY_DICTS
def test_frame_solvent_indices(interfacial_entropy_dicts):
    """Test the get interfacial water orient entropy function"""
    # frame: {resname: {resid = [shell indices]}}
    frame_solvent_indices = interfacial_entropy_dicts[3]
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


@INTERFACIAL_ENTROPY_DICTS
def test_n_frames(interfacial_entropy_dicts):
    """Test the get interfacial water orient entropy function"""
    n_frames = interfacial_entropy_dicts[4]
    assert n_frames == 2
