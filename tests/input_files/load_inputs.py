"""Load input files into MDAnalysis universes for testing"""

import os

from MDAnalysis import Universe

from .. import TEST_DIR


def get_amber_arginine_soln_universe():
    """Create a MDAnalysis universe from the arginine simulation"""

    topology = os.path.join(
        TEST_DIR, "input_files", "amber", "arginine_solution", "system.prmtop"
    )
    coordinates = os.path.join(
        TEST_DIR, "input_files", "amber", "arginine_solution", "system.nc"
    )

    u = Universe(topology, coordinates)
    return u


def get_gmx_aspirin_soln_universe():
    """Create a MDAnalysis universe from the arginine simulation"""

    topology = os.path.join(
        TEST_DIR, "input_files", "gromacs", "aspirin_solution", "system.gro"
    )
    coordinates = os.path.join(
        TEST_DIR, "input_files", "gromacs", "aspirin_solution", "system.trr"
    )

    u = Universe(topology, coordinates)
    return u
