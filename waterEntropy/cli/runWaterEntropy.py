#!/usr/bin/env python

"""
"""

import argparse
from datetime import datetime
import logging
import sys
import numpy as np

from MDAnalysis import Universe

import waterEntropy.recipes.interfacial_solvent as GetSolvent
import waterEntropy.recipes.bulk_water as GetBulkSolvent
import waterEntropy.entropy.vibrations as VIB
import waterEntropy.entropy.orientations as OR

def run_waterEntropy(
    file_topology="file_topology",
    file_coords="file_coords",
    start="start",
    end="end",
    step="step",
):
    """
    """

    startTime = datetime.now()
    print(startTime)

    # load topology and coordinates
    u = Universe(file_topology, file_coords)

    Sorient_dict, covariances, vibrations, frame_solvent_indices = GetSolvent.get_interfacial_water_orient_entropy(u, start, end, step, temperature=298)
    OR.print_Sorient_dicts(Sorient_dict)
    # GetSolvent.print_frame_solvent_dicts(frame_solvent_indices)
    VIB.print_Svib_data(vibrations, covariances)
    bulk_Sorient_dict, bulk_covariances, bulk_vibrations = GetBulkSolvent.get_bulk_water_orient_entropy(u, start, end, step, temperature=298)
    OR.print_Sorient_dicts(bulk_Sorient_dict)
    VIB.print_Svib_data(bulk_vibrations, bulk_covariances)


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
            "-top",
            "--file_topology",
            metavar="file",
            default=None,
            help="name of file containing system topology.",
        )
        parser.add_argument(
            "-crd",
            "--file_coords",
            metavar="file",
            default=None,
            help="name of file containing positions and forces in a single file.",
        )
        parser.add_argument(
            "-s",
            "--start",
            action="store",
            type=int,
            default=0,
            help="frame number to start analysis from.",
        )
        parser.add_argument(
            "-e",
            "--end",
            action="store",
            type=int,
            default=1,
            help="frame number to end analysis at.",
        )
        parser.add_argument(
            "-dt",
            "--step",
            action="store",
            type=int,
            default=1,
            help="steps to take between start and end frame selections.",
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
        start=op.start, 
        end=op.end, 
        step=op.step,
    )


if __name__ == "__main__":
    main()
