""" Tests for waterEntropy interfacial solvent functions in neighbours."""

import pytest

from tests.input_files import load_inputs
import waterEntropy.neighbours.interfacial_solvent as GetSolvent

# get mda universe for arginine in solution
system = load_inputs.get_amber_arginine_soln_universe()


def test_get_interfacial_water_orient_entropy():
    """Test the get interfacial water orient entropy function"""
    Sorient_dict, frame_solvent_indices = (
        GetSolvent.get_interfacial_water_orient_entropy(system, 0, 4, 2)
    )

    # resid, resname, Sorient, count
    assert Sorient_dict[1]["ACE"] == pytest.approx([0.28667395887678, 17])
    assert Sorient_dict[2]["ARG"] == pytest.approx([0.09037301291929652, 30])
    assert Sorient_dict[3]["NME"] == pytest.approx([0.40187190769635084, 12])
    # frame, resname, resid, shell
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
    assert frame_solvent_indices[0].get("NME").get(3) == [486, 489, 834, 879, 984]
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
