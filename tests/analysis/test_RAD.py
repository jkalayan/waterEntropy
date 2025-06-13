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
solvent_UA = system.select_atoms("index 274")[0]

# find neighours around solvent UA, closest to furthest
sorted_indices, sorted_distances = Trig.get_sorted_neighbours(solvent_UA.index, system)


def test_get_sorted_neighbours():
    """Test the get sorted neighbours function"""

    assert list(sorted_indices[:10]) == [
        29,
        1228,
        1507,
        244,
        28,
        76,
        322,
        580,
        1855,
        30,
    ]
    assert list(sorted_distances[:10]) == pytest.approx(
        [
            2.79362936,
            2.86005522,
            3.0090825,
            3.03406432,
            3.493638,
            3.60498544,
            3.74683653,
            3.82608146,
            4.07706006,
            4.08403076,
        ]
    )


def test_get_RAD_neighbours():
    """Test the get RAD neighbours function"""
    shell = RADShell.get_RAD_neighbours(
        solvent_UA.position, sorted_indices, sorted_distances, system
    )
    assert shell == [29, 1228, 1507, 244, 76, 322]


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

    assert shell.UA_shell == [29, 1228, 1507, 244, 76, 322]
    assert shell.labels == [
        "0_ARG",
        "2_WAT",
        "2_WAT",
        "1_WAT",
        "2_WAT",
        "2_WAT",
    ]


def test_find_interfacial_solvent():
    """Test the find interfacial solvent function"""
    shells = ShellCollection()
    solvent_indices = find_interfacial_solvent(system_solutes, system, shells)
    # print([int(x) for x in solvent_indices])

    assert sorted(solvent_indices) == sorted(
        [
            1024,
            1282,
            2056,
            265,
            2314,
            1165,
            274,
            2077,
            1057,
            931,
            55,
            1855,
            2497,
            1474,
            2245,
            205,
            2260,
            85,
            2005,
            1753,
            862,
            1246,
            481,
            121,
            1507,
            1639,
            235,
            2413,
            244,
            1912,
            505,
        ]
    )
