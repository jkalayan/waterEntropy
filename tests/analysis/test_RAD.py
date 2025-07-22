""" Tests for waterEntropy RAD functions in neighbours."""

import pytest

from tests.input_files import load_inputs
import waterEntropy.analysis.RAD as RADShell
import waterEntropy.analysis.shell_labels as RADLabels
from waterEntropy.analysis.shells import ShellCollection
import waterEntropy.maths.trig as Trig
from waterEntropy.recipes.interfacial_solvent import find_interfacial_solvent
import waterEntropy.utils.selections as Selections

# get mda universe for arginine in solution
system = load_inputs.get_amber_arginine_soln_universe()
resid_list = [1, 2, 3]
# select all molecules above 1 UA in size
system_solutes = Selections.get_selection(system, "resid", resid_list)
solvent_UA = system.select_atoms("index 621")[0]

# find neighours around solvent UA, closest to furthest
sorted_indices, sorted_distances = Trig.get_sorted_neighbours(solvent_UA.index, system)


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
    # create ShellCollection instance
    shells = ShellCollection()
    # get the shell of a solvent UA
    shell_indices = RADShell.get_RAD_shell(solvent_UA, system, shells)
    # add shell to the RAD class
    shells.add_data(solvent_UA.index, shell_indices)
    # get the shell back
    shell = shells.find_shell(solvent_UA.index)
    # get the shell labels
    shell = RADLabels.get_shell_labels(solvent_UA.index, system, shell, shells)

    assert shell.UA_shell == [2517, 1353, 1800, 1, 5, 1038, 1464, 888, 1017]
    assert shell.labels == [
        "2_WAT",
        "2_WAT",
        "1_WAT",
        "0_ACE",
        "ACE",
        "1_WAT",
        "2_WAT",
        "1_WAT",
        "2_WAT",
    ]


def test_find_interfacial_solvent():
    """Test the find interfacial solvent function"""
    shells = ShellCollection()
    solvent_indices = find_interfacial_solvent(system_solutes, system, shells)

    assert sorted(solvent_indices) == sorted(
        [
            621,
            1143,
            1737,
            1800,
            888,
            1038,
            1413,
            834,
            2004,
            2688,
            237,
            1878,
            2640,
            168,
            747,
            1797,
            2646,
            369,
            2019,
            2262,
            54,
            2130,
            486,
            879,
            1698,
            489,
            984,
        ]
    )
