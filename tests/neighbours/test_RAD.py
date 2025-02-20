""" Tests for waterEntropy RAD functions in neighbours."""

import pytest

from tests.input_files import load_inputs
import waterEntropy.neighbours.RAD as RADShell
import waterEntropy.utils.selections as Select

# get mda universe for arginine in solution
system = load_inputs.get_amber_arginine_soln_universe()
resid_list = [1, 2, 3]
# select all molecules above 1 UA in size
system_solutes = Select.get_selection(system, "resid", resid_list)
solvent_UA = system.select_atoms("index 621")[0]

# find neighours around solvent UA, closest to furthest
sorted_indices, sorted_distances = RADShell.get_sorted_neighbours(
    solvent_UA.index, system
)


def test_get_sorted_neighbours():
    """Test the get sorted neighbours function"""

    assert list(sorted_indices[:10]) == [
        2517,
        1353,
        1800,
        1,
        5,
        741,
        1038,
        768,
        1464,
        4,
    ]
    assert list(sorted_distances[:10]) == pytest.approx(
        [
            2.7088637555843293,
            2.875420900468947,
            3.36660473389714,
            3.5774782395107163,
            3.5832420456549348,
            3.6107700733207855,
            3.6559141986837203,
            3.709932576820671,
            3.926065109380465,
            3.9808174224132804,
        ]
    )


def test_get_RAD_neighbours():
    """Test the get RAD neighbours function"""
    shell = RADShell.get_RAD_neighbours(
        solvent_UA.position, sorted_indices, sorted_distances, system
    )
    assert shell == [2517, 1353, 1800, 1, 5, 1038, 1464, 888, 1017]


def test_get_RAD_shell():
    """Test the get RAD shell function"""
    # pylint: disable=pointless-statement
    # got to first frame of trajectory
    system.trajectory[0]
    # get the shell of a solvent UA
    shell = RADShell.get_RAD_shell(solvent_UA, system)
    # add shell to the RAD class
    shell = RADShell.RAD(solvent_UA.index, shell)
    # get the shell labels
    shell = RADShell.get_shell_labels(solvent_UA.index, system, shell)

    assert shell.UA_shell == [2517, 1353, 1800, 1, 5, 1038, 1464, 888, 1017]
    assert shell.labels == [
        "2_WAT",
        "2_WAT",
        "1_WAT",
        "0_ACE",
        "ACE",
        "X_WAT",
        "2_WAT",
        "X_WAT",
        "2_WAT",
    ]


def test_find_interfacial_solvent():
    """Test the find interfacial solvent function"""
    solvent_indices = RADShell.find_interfacial_solvent(system_solutes, system)

    assert solvent_indices == [
        621,
        1143,
        1800,
        1737,
        1413,
        888,
        1038,
        834,
        237,
        2004,
        1878,
        2688,
        2640,
        1797,
        747,
        369,
        2646,
        2019,
        168,
        2262,
        54,
        2130,
        486,
        984,
        489,
        879,
    ]
