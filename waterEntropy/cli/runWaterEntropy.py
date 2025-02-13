#!/usr/bin/env python

"""
"""

import argparse
from collections import defaultdict
from datetime import datetime
import logging
import sys

# import MDAnalysis
from MDAnalysis import Universe
from MDAnalysis.coordinates.LAMMPS import DumpReader

import waterEntropy.neighbours.HB as HBond
import waterEntropy.neighbours.RAD as RADShell
from waterEntropy.neighbours.force_torque import (
    get_atoms_masses,
    get_axes,
    get_bonded_axes,
    get_covariance_matrix,
    get_custom_axes,
    get_custom_PI_MOI,
    get_FT_atoms,
    get_mass_weighted_forces,
    get_torques,
)
import waterEntropy.neighbours.interfacial_solvent as GetSolvent
import waterEntropy.statistics.orientations as Orient
from waterEntropy.utils.helpers import nested_dict
import waterEntropy.utils.selections as Select

# nested_dict = lambda: defaultdict(nested_dict)  # create nested dict in one go


# from MDAnalysis.topology.LAMMPSParser import LammpsDumpParser


def run_waterEntropy(
    file_topology="file_topology",
    file_coords="file_coords",
    file_forces="file_forces",
    file_energies="file_energies",
    list_files="list_files",
):
    # pylint: disable=all
    """
    Functions required:
    [] Force torque for whole molecule principal axes
    [] Force torque for custom axes
    [] Covariance matrices
    [X] Nearest non-like
    [X] RAD shell
    [X] HB in shell
    [X] neighbour list
    [X] label RAD shell
    [] molecule level RAD shell
    [] save stats into multiple dicts
    [] running averages
    [] Dict for
    """

    startTime = datetime.now()
    print(startTime)

    # help(MDAnalysis.coordinates.LAMMPS)
    # topology = Universe(file_topology)
    # print(topology.atoms[0].name)
    # trajectory = DumpReader("na_cl_2/Trajectory_npt_1.data.gz")
    # # print([int(ts.time) for ts in trajectory])
    # for ts in trajectory[:1]:
    #     print(ts.positions[:10])
    # #forces = DumpReader("na_cl_2/Forces_npt_1.data.gz").convert_forces_from_native()
    # sys.exit()

    # coords = Universe(file_topology, file_coords, format="TRJ")
    coords = Universe(file_topology, file_coords)
    start, end, step = 0, 10, 1
    print(coords.trajectory)

    GetSolvent.get_interfacial_water_orient_entropy(coords, start, end, step)

    sys.exit()
    # forces = Universe(file_topology, file_forces, format="TRJ")
    forces2 = Universe(file_topology, file_coords, format="TRJ")
    # print(forces.trajectory[0].positions)
    # print(forces.atoms)
    frame_solvent_indices = nested_dict()
    for ts1, ts2 in zip(coords.trajectory[:], forces2.trajectory[:]):
        print(ts1, ts2.frame)
        # print("\n", coords.trajectory.time)
        atom = coords.select_atoms("all")

        # 1. find > 1 UA molecules in system, these are the solutes
        resid_list = Select.find_large_molecules(coords)
        solutes = Select.get_selection(coords, "resid", resid_list)

        # 2. find the interfacial solvent molecules that are 1 UA in size
        # and are in the RAD shell of any solute
        solvent_indices = RADShell.find_interfacial_solvent(solutes, coords)
        first_shell_solvent = Select.get_selection(coords, "index", solvent_indices)

        # 3. iterate through first shell solvent and find their RAD shells,
        # HBing in the shells and shell labels
        for solvent in first_shell_solvent:
            # 3a. find RAD shell of interfacial solvent
            shell = RADShell.get_RAD_shell(solvent, coords)
            shell = RADShell.RAD(solvent.index, shell)
            # print(">", solvent.index, shell.UA_shell)
            # 3b. find HBing in the shell
            HBond.get_shell_HBs(shell, coords)
            # 3c. find RAD shell labels
            shell = RADShell.get_shell_labels(solvent.index, coords, shell)
            # 3d. find HB labels
            HBond.get_HB_labels(solvent.index, coords)
            # print(shell.labels)
            # print(shell.donates_to_labels)
            # print(shell.accepts_from_labels)
            if shell.nearest_nonlike_idx:
                # 3e. populate the labels into a dictionary for stats
                # only if a different atom is in the RAD shell
                nearest_resid = coords.atoms[shell.nearest_nonlike_idx].resid
                nearest_resname = coords.atoms[shell.nearest_nonlike_idx].resname
                Orient.Labels(
                    nearest_resid,
                    nearest_resname,
                    shell.labels,
                    shell.donates_to_labels,
                    shell.accepts_from_labels,
                )

        dict1 = {
            ("labelled_shell"): {
                "shell_count": 0,
                "donates_to": {
                    "labelled_donators": 0,
                },
                "accepts_from": {
                    "labelled_acceptors": 0,
                },
            }
        }
        dict2 = {
            "nearest_resid": {
                ("labelled_shell"): {
                    "shell_count": 0,
                    "donates_to": {
                        "labelled_donators": 0,
                    },
                    "accepts_from": {
                        "labelled_acceptors": 0,
                    },
                }
            }
        }
        # clear each shell and HB dictionary ready for next frame.
        RADShell.RAD.shells.clear()
        HBond.HB.donating_to.clear()
        HBond.HB.accepting_from.clear()

        # # select heavy atoms that are not in water
        # # need a better way to do to this, where we can specify water based
        # # on what it is bonded to. So if we find O bonded to two Hs
        # # (check masses)
        # # atom = coords.select_atoms("mass 2 to 999 and not resname WAT")
        # atom = coords.select_atoms("resid 2")
        # atomF = forces.select_atoms("resid 2")
        # atom2 = atom.select_atoms("mass 1.1 to 999")
        # print(atom.positions)
        # heavy_atoms, all_atoms = get_FT_atoms(coords, atom)
        # print(heavy_atoms.indices)
        # print(all_atoms.indices)

        # print("\n" * 2)
        # center_of_mass = atom.center_of_mass()
        # principal_axes, MOI_axis = get_axes(atom)
        # # print(principal_axes, MOI_axis)

        # atom_indices, masses = get_atoms_masses(all_atoms, include_Hs=False)
        # print(atom_indices, masses)

        # torque = get_torques(atom, atomF, center_of_mass, principal_axes, MOI_axis)
        # cov_torque = get_covariance_matrix(torque, halve=True)
        # mass_weighted_force = get_mass_weighted_forces(atomF, principal_axes)
        # # print(mass_weighted_force)
        # custom_axis = get_custom_axes(
        #     atom.positions[0],
        #     [atom.positions[1]],
        #     atom.positions[2],
        #     coords.dimensions[:3],
        # )

        # get_custom_PI_MOI(
        #     atom.positions,
        #     principal_axes,
        #     atom.positions[0],
        #     atom.masses,
        #     atom.dimensions[:3],
        # )

    Sorient_dict = Orient.get_resid_orientational_entropy_from_dict(
        Orient.Labels.resid_labelled_shell_counts
    )

    # d = {1: {"CL": {("0_CL", "1_WAT", "1_WAT", "1_WAT", "1_WAT", "1_WAT", "2_WAT", "2_WAT"):
    #                             {"shell_count": 2,
    #                             "accepts_from": {"2_WAT": 3, "1_WAT": 1},
    #                             "donates_to": {"0_CL": 1, "1_WAT": 1}
    #                             }}}}
    # Sorient_dict = Orient.get_resid_orientational_entropy_from_dict(d)

    print("\n" * 3)
    for resid, resname_key in sorted(list(Sorient_dict.items())):
        for resname, Sor_count in sorted(list(resname_key.items())):
            print(resid, resname, Sor_count)

    sys.stdout.flush()
    print("end")
    print(datetime.now() - startTime)


def main():
    """ """
    try:
        usage = "runWaterEntropy.py [-h]"
        parser = argparse.ArgumentParser(
            description="Program for reading "
            "in molecule forces, coordinates and energies for "
            "entropy calculations.",
            usage=usage,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument_group("Options")
        parser.add_argument(
            "-t",
            "--file_topology",
            metavar="file",
            default=None,
            help="name of file containing system topology.",
        )
        parser.add_argument(
            "-c",
            "--file_coords",
            metavar="file",
            default=None,
            help="name of file containing coordinates.",
        )
        parser.add_argument(
            "-f",
            "--file_forces",
            metavar="file",
            default=None,
            help="name of file containing forces.",
        )
        parser.add_argument(
            "-e",
            "--file_energies",
            metavar="file",
            default=None,
            help="name of file containing energies.",
        )
        parser.add_argument(
            "-l",
            "--list_files",
            action="store",
            metavar="file",
            default=False,
            help="file containing list of file paths.",
        )
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error(
            "Command line arguments are ill-defined, please check the arguments."
        )
        raise
        sys.exit(1)

    run_waterEntropy(
        file_topology=op.file_topology,
        file_coords=op.file_coords,
        file_forces=op.file_forces,
        file_energies=op.file_energies,
        list_files=op.list_files,
    )


if __name__ == "__main__":
    main()
