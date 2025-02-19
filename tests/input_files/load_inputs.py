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
        TEST_DIR, "input_files", "amber", "arginine_solution", "system.mdcrd"
    )
    forces = os.path.join(
        TEST_DIR, "input_files", "amber", "arginine_solution", "system.frc"
    )

    u = Universe(topology, coordinates, format="MDCRD")
    # seperate universe where forces are loaded trajectory
    uf = Universe(topology, forces, format="MDCRD")
    # set the has_forces flag on the Timestep first
    u.trajectory.ts.has_forces = True
    # add the forces (which are saved as positions be default) from uf to u
    u.atoms.forces = uf.atoms.positions
    return u


def get_amber_nacl_soln_universe():
    """Create a MDAnalysis universe from the arginine simulation"""

    topology = os.path.join(
        TEST_DIR, "input_files", "amber", "nacl_solution", "system.prmtop"
    )
    coordinates = os.path.join(
        TEST_DIR, "input_files", "amber", "nacl_solution", "system.mdcrd"
    )
    forces = os.path.join(
        TEST_DIR, "input_files", "amber", "nacl_solution", "system.frc"
    )

    u = Universe(topology, coordinates, format="MDCRD")
    # seperate universe where forces are loaded trajectory
    uf = Universe(topology, forces, format="MDCRD")
    # set the has_forces flag on the Timestep first
    u.trajectory.ts.has_forces = True
    # add the forces (which are saved as positions be default) from uf to u
    u.atoms.forces = uf.atoms.positions
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


def get_lammps_arginine_soln_universe():
    """Create a MDAnalysis universe from the arginine simulation"""

    topology = os.path.join(
        TEST_DIR, "input_files", "lammps", "arginine_solution", "system.prmtop"
    )
    coordinates = os.path.join(
        TEST_DIR,
        "input_files",
        "lammps",
        "arginine_solution",
        "Trajectory_npt_1.data.gz",
    )
    # forces = os.path.join(
    #     TEST_DIR, "input_files", "lammps", "arginine_solution", "Forces_npt_1.data.gz"
    # )
    # energies = os.path.join(
    #     TEST_DIR, "input_files", "lammps", "arginine_solution", "Energy_npt_1.data.gz"
    # )

    uc = Universe(
        topology, coordinates, atom_style="id type x y z", format="LAMMPSDUMP"
    )
    # Loading forces and energies not working with MDAnalysis
    # uf = Universe(topology, forces, atom_style='id type x y z', format='LAMMPSDUMP')
    # ue = Universe(topology, energies, atom_style='id type x y z', format='LAMMPSDUMP')
    return uc
