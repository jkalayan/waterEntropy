#!/usr/bin/env python

"""
"""

import argparse
from datetime import datetime
import logging
import os
import shutil
import psutil
import sys
import numpy as np

from MDAnalysis import Universe

import waterEntropy.recipes.interfacial_solvent as GetSolvent
import waterEntropy.recipes.bulk_water as GetBulkSolvent
import waterEntropy.entropy.vibrations as VIB
import waterEntropy.entropy.orientations as OR
from waterEntropy.utils.dask_clusters import slurm_configure_cluster, slurm_submit_master


def run_waterEntropy(args):
    """
    """

    startTime = datetime.now()
    print(startTime)

    if args.hpc is True:
        client = slurm_configure_cluster(args)
    else:
        client = None

    # load topology and coordinates
    u = Universe(args.file_topology, args.file_coords)
    # interfacial waters
    Sorient_dict, covariances, vibrations, frame_solvent_indices, n_frames = GetSolvent.get_interfacial_water_orient_entropy(u, args.start, args.end, args.step, args.temperature, args.parallel, client)
    print(f"Number of frames analysed: {n_frames}")
    OR.print_Sorient_dicts(Sorient_dict)
    # GetSolvent.print_frame_solvent_dicts(frame_solvent_indices)
    VIB.print_Svib_data(vibrations, covariances)

    # bulk waters
    # bulk_Sorient_dict, bulk_covariances, bulk_vibrations = GetBulkSolvent.get_bulk_water_orient_entropy(u, start, end, step, temperature)
    # OR.print_Sorient_dicts(bulk_Sorient_dict)
    # VIB.print_Svib_data(bulk_vibrations, bulk_covariances)

    dask_tmp = os.path.join(os.getcwd(), "dask-scratch-space")
    if os.path.exists(dask_tmp) and os.path.isdir(dask_tmp):
        shutil.rmtree(dask_tmp)
    sys.stdout.flush()
    print(datetime.now() - startTime)


def _conda_env():
    """Determine the activated conda/mamba environment."""
    try:
        return os.environ["CONDA_DEFAULT_ENV"]
    except KeyError:
        logging.error("Please activate your conda/mamba environment")
        sys.exit(1)


def _conda_exec():
    """Determine the conda/mamba executable."""
    try:
        os.environ['MAMBA_EXE']
        return "mamba"
    except KeyError:
        try:
            os.environ['CONDA_EXE']
            return "conda"
        except KeyError:
            logging.error("Cannot determine your conda executable, make sure they are initialised.")
            sys.exit(1)


def _conda_path():
    """Determine the conda path"""
    try:
        return os.environ["CONDA_EXE"]
    except KeyError:
        logging.error("Please make sure you have conda/mamba set up correctly.")
        sys.exit(1)


def main():
    """Entrypoint for running the WaterEntropy for interfacial water calculation."""

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
        parser.add_argument(
            "-p",
            "--parallel",
            action="store_true",
            help="Whether to perform the interfacial water calculations in parallel.",
        )
        parser.add_argument(
            "--conda-exec",
            action="store",
            type=str,
            default=_conda_exec(),
            help="conda/mamba executable to use for HPC runs.",
        )
        parser.add_argument(
            "--conda-env",
            action="store",
            type=str,
            default=_conda_env(),
            help="Name of the conda/mamba environment to activate for HPC runs.",
        )
        parser.add_argument(
            "--conda-path",
            action="store",
            type=str,
            default=_conda_path(),
            help="Path to conda executable on HPC machine.",
        )
        parser.add_argument(
            "--hpc",
            action="store_true",
            help="Whether to perform the interfacial water calculations on a slurm cluster.",
        )
        parser.add_argument(
            "--hpc-account",
            action="store",
            type=str,
            default="",
            help="Which account budget to submit with?",
        )
        parser.add_argument(
            "--hpc-constraint",
            action="store",
            type=str,
            default="",
            help="Constraints to apply to job allocation, such as hardware generation.",
        )
        parser.add_argument(
            "--hpc-cores",
            action="store",
            type=int,
            default=psutil.cpu_count(logical=False),
            help="How many cores per node?",
        )
        parser.add_argument(
            "--hpc-memory",
            action="store",
            type=str,
            default=f"{psutil.virtual_memory().total / 1024.0 ** 3}GB",
            help="How memory per node?",
        )
        parser.add_argument(
            "--hpc-nodes",
            action="store",
            type=int,
            default=1,
            help="How many HPC nodes?",
        )
        parser.add_argument(
            "--hpc-processes",
            action="store",
            type=int,
            default=psutil.cpu_count(logical=False),
            help="How many dask processes per node?",
        )
        parser.add_argument(
            "--hpc-queue",
            action="store",
            type=str,
            default="standard",
            help="Which SLURM queue to submit to?",
        )
        parser.add_argument(
            "--hpc-qos",
            action="store",
            type=str,
            default="",
            help="QoS to apply to job.",
        )
        parser.add_argument(
            "--hpc-walltime",
            action="store",
            type=str,
            default="24:00:00",
            help="How long to request cluster for?",
        )
        parser.add_argument(
            "--submit",
            action="store_true",
            help="Whether to self submit on HPC.",
        )
        args = parser.parse_args()
        if args.hpc is True: args.parallel = True # No need to set both on CLI.
    except argparse.ArgumentError:
        logging.error(
            "Command line arguments are ill-defined, please check the arguments."
        )
        raise
        sys.exit(1)

    if args.submit:
      slurm_submit_master(args)
      sys.exit(0)
    run_waterEntropy(args)


if __name__ == "__main__":
    main()
