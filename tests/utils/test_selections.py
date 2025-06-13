""" Tests for waterEntropy selections functions in utils."""

from tests.input_files import load_inputs
import waterEntropy.utils.selections as Select

# get mda universe for arginine in solution
system = load_inputs.get_amber_arginine_soln_universe()
resid_list = [1, 2, 3, 4]
# select all molecules above 1 UA in size
system_solutes = Select.get_selection(system, "resid", resid_list)


def test_find_solute_molecules():
    """Test the find solute molecules function"""
    # find all molecule resids larger than 1 UA in size
    resid_list_solutes = Select.find_solute_molecules(system)

    assert resid_list_solutes == [1, 2, 3, 4]


def test_get_selection():
    """Test the get selection function"""
    # select all molecules above 1 UA in size
    solutes = Select.get_selection(system, "resid", resid_list)

    assert list(solutes.names) == [
        "H1",
        "CH3",
        "H2",
        "H3",
        "C",
        "O",
        "N",
        "H",
        "CA",
        "HA",
        "CB",
        "HB2",
        "HB3",
        "CG",
        "HG2",
        "HG3",
        "CD",
        "HD2",
        "HD3",
        "NE",
        "HE",
        "CZ",
        "NH1",
        "HH11",
        "HH12",
        "NH2",
        "HH21",
        "HH22",
        "C",
        "O",
        "N",
        "H",
        "C",
        "H1",
        "H2",
        "H3",
        "Cl-",
    ]


def test_find_bonded_heavy_atom():
    """Test the find bonded heavy atom function"""
    # pre-defined list of solvent atom numbers
    solvent_indices = [
        1024,
        1282,
        1669,
        2056,
        265,
        2314,
        1165,
        274,
        2077,
        2041,
        1057,
        931,
        55,
        1855,
        2497,
        1474,
        2245,
        460,
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
    # find the resids from the atom numbers above
    first_shell_solvent = Select.get_selection(system, "index", solvent_indices)
    # find all heavy atoms bonded to a hydrogen
    bonded_to_H = Select.find_bonded_heavy_atom(0, system_solutes)

    assert list(first_shell_solvent.resids) == [
        11,
        21,
        33,
        61,
        71,
        74,
        81,
        84,
        146,
        153,
        161,
        280,
        303,
        334,
        345,
        381,
        408,
        420,
        484,
        495,
        539,
        549,
        577,
        611,
        630,
        661,
        673,
        678,
        685,
        741,
        746,
        764,
        797,
        825,
    ]
    assert bonded_to_H.name == "CH3"


def test_find_molecule_UAs():
    """Test find molecule UAs function"""
    # find all UAs in a selection of molecules
    UAs = Select.find_molecule_UAs(system_solutes)

    assert list(UAs.names) == [
        "CH3",
        "C",
        "O",
        "N",
        "CA",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "C",
        "O",
        "N",
        "C",
        "Cl-",
    ]
