"""Tests for waterEntropy interfacial solvent functions in neighbours."""

import numpy as np
import pytest

import waterEntropy.recipes.interfacial_solvent as GetSolvent
from tests.input_files import load_inputs


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

    assert Sorient_dict[1]["ACE"]["Sorient_ave"] == pytest.approx(2.2473807716251804)
    assert Sorient_dict[1]["ACE"]["tot_count"] == pytest.approx(13)
    assert Sorient_dict[1]["ACE"]["N_c_ave"] == pytest.approx(7.0)
    assert Sorient_dict[1]["ACE"]["N_w_ave"] == pytest.approx(5.6923076923076925)
    assert Sorient_dict[1]["ACE"]["Nc_eff_ave"] == pytest.approx(3.2245457772507913)
    assert Sorient_dict[1]["ACE"]["pbias_ave"] == pytest.approx(0.116959182573051)
    assert Sorient_dict[1]["ACE"]["Sorient_Nc_ave"] == pytest.approx(11.549772583950544)
    assert Sorient_dict[1]["ACE"]["Sorient_Nw_ave"] == pytest.approx(8.866118517437595)
    assert Sorient_dict[1]["ACE"]["Sorient_HB_ave2"] == pytest.approx(
        1.0246793134504353
    )
    assert Sorient_dict[1]["ACE"]["Sorient_Nc_ave2"] == pytest.approx(
        11.549772583950546
    )
    assert Sorient_dict[1]["ACE"]["Sorient_Nw_ave2"] == pytest.approx(8.866118517437595)

    assert Sorient_dict[2]["ARG"]["Sorient_ave"] == pytest.approx(2.6481382191024245)
    assert Sorient_dict[2]["ARG"]["tot_count"] == pytest.approx(35)
    assert Sorient_dict[2]["ARG"]["N_c_ave"] == pytest.approx(7.514285714285714)
    assert Sorient_dict[2]["ARG"]["N_w_ave"] == pytest.approx(6.0)
    assert Sorient_dict[2]["ARG"]["Nc_eff_ave"] == pytest.approx(3.464361256421488)
    assert Sorient_dict[2]["ARG"]["pbias_ave"] == pytest.approx(0.114223362162831)
    assert Sorient_dict[2]["ARG"]["Sorient_Nc_ave"] == pytest.approx(12.394889769242496)
    assert Sorient_dict[2]["ARG"]["Sorient_Nw_ave"] == pytest.approx(9.388627866311323)
    assert Sorient_dict[2]["ARG"]["Sorient_HB_ave2"] == pytest.approx(
        0.47562453963817075
    )
    assert Sorient_dict[2]["ARG"]["Sorient_Nc_ave2"] == pytest.approx(
        12.394889769242495
    )
    assert Sorient_dict[2]["ARG"]["Sorient_Nw_ave2"] == pytest.approx(9.388627866311323)

    assert Sorient_dict[3]["NME"]["Sorient_ave"] == pytest.approx(0.9503950365967891)
    assert Sorient_dict[3]["NME"]["tot_count"] == pytest.approx(12)
    assert Sorient_dict[3]["NME"]["N_c_ave"] == pytest.approx(7.333333333333333)
    assert Sorient_dict[3]["NME"]["N_w_ave"] == pytest.approx(5.833333333333333)
    assert Sorient_dict[3]["NME"]["Nc_eff_ave"] == pytest.approx(2.314863945578231)
    assert Sorient_dict[3]["NME"]["pbias_ave"] == pytest.approx(0.0798514739229025)
    assert Sorient_dict[3]["NME"]["Sorient_Nc_ave"] == pytest.approx(12.077905047118842)
    assert Sorient_dict[3]["NME"]["Sorient_Nw_ave"] == pytest.approx(9.183582113622649)
    assert Sorient_dict[3]["NME"]["Sorient_HB_ave2"] == pytest.approx(0.0)
    assert Sorient_dict[3]["NME"]["Sorient_Nc_ave2"] == pytest.approx(
        12.077905047118845
    )
    assert Sorient_dict[3]["NME"]["Sorient_Nw_ave2"] == pytest.approx(9.18358211362265)


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
                [824648.12302947, -1099.32151405, -44598.64004836],
                [-1099.32151405, 1130557.41192537, 63512.95489094],
                [-44598.64004836, 63512.95489094, 564357.38949483],
            ]
        ),
    )
    assert np.allclose(
        torques,
        np.array(
            [
                [16555327.17944721, 4032510.71077588, -2984683.56005148],
                [4032510.71077588, 3332761.45484771, -1083918.62436743],
                [-2984683.56005148, -1083918.62436743, 8487963.45839714],
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

    assert np.allclose(
        sorted(Strans), np.array([15.46750522, 16.75340504, 18.45542769])
    )
    assert np.allclose(sum(Strans), 50.676337954279)
    assert np.allclose(sorted(Srot), np.array([4.73429158, 7.96112985, 12.80223863]))
    assert np.allclose(sum(Srot), 25.497660051181)
    assert np.allclose(
        sorted(trans_freqs), np.array([550198.33, 831653.74, 1137710.85])
    )
    assert np.allclose(
        sorted(rot_freqs), np.array([2190115.53, 7512997.23, 18672939.34])
    )


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
