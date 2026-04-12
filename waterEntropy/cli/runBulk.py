#!/usr/bin/env python

"""
Script to run bulk water entropy
"""

import argparse
from datetime import datetime
import logging
import os
import sys

from MDAnalysis import Universe

import waterEntropy.entropy.orientations as OR
import waterEntropy.entropy.vibrations as VIB
import waterEntropy.recipes.bulk_water as GetBulkSolvent


def run_waterBulk(args):
    """
    Run entropy calculations for bulk water.
    """

    startTime = datetime.now()
    print(startTime)

    # load topology and coordinates
    u = Universe(args.file_topology, args.file_coords)

    # bulk waters
    bulk_Sorient_dict, bulk_covariances, bulk_vibrations = (
        GetBulkSolvent.get_bulk_water_orient_entropy(
            u, args.start, args.end, args.step, args.temperature
        )
    )
    OR.print_Sorient_dicts(bulk_Sorient_dict)
    VIB.print_Svib_data(bulk_vibrations, bulk_covariances)

    print(datetime.now() - startTime)


def _conda_env():
    """Determine the activated conda/mamba environment."""
    try:
        return os.environ["CONDA_DEFAULT_ENV"]
    except KeyError:
        logging.error("Please activate your conda/mamba environment")
        sys.exit(1)


def main():
    """Entrypoint for running the WaterEntropy for interfacial water calculation."""

    try:
        usage = "waterBulk [-h]"
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
            "--file-topology",
            metavar="file",
            default=None,
            help="name of file containing system topology.",
        )
        parser.add_argument(
            "-crd",
            "--file-coords",
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
        parser.add_argument(
            "-temp",
            "--temperature",
            action="store",
            type=float,
            default=298,
            help="Target temperature the simulation was performed at in Kelvin.",
        )
        args = parser.parse_args()
        # if args.hpc is True: args.parallel = True # No need to set both on CLI.
    except argparse.ArgumentError:
        logging.error(
            "Command line arguments are ill-defined, please check the arguments."
        )
        raise

    run_waterBulk(args)


if __name__ == "__main__":
    main()
