#!/usr/bin/env python

"""
"""

import argparse
from datetime import datetime
import logging
import sys

from MDAnalysis import Universe

import waterEntropy.recipes.interfacial_solvent as GetSolvent
import waterEntropy.analysis.vibrations as GetVibrations

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
    [X] Force torque for whole molecule principal axes
    [X] Force torque for custom axes
    [X] Covariance matrices
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

    # load topology and coordinates
    u = Universe(file_topology, file_coords, format="MDCRD")
    # seperate universe where forces are the loaded trajectory
    uf = Universe(file_topology, file_forces, format="MDCRD")
    # set the has_forces flag on the Timestep first
    u.trajectory.ts.has_forces = True
    # add the forces (which are saved as positions be default) from uf to u
    u.atoms.forces = uf.atoms.positions
    # set the frames to be analysed
    start, end, step = 0, 4, 2
    print(u.trajectory)
    # u.trajectory[frame] # move to a particular frame using this

    Sorient_dict, covariances, vibrations, frame_solvent_indices = GetSolvent.get_interfacial_water_orient_entropy(u, start, end, step)
    GetSolvent.print_Sorient_dicts(Sorient_dict)
    # GetSolvent.print_frame_solvent_dicts(frame_solvent_indices)
    GetVibrations.print_Svib_data(vibrations, covariances)


    sys.stdout.flush()
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
