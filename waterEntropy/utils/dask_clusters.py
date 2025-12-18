"""
Helpers for setting up dask clusters on HPC.
"""

import sys

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import psutil


def configure_slurm_cluster(args):
    """Configure a SLURM HPC cluster to run bigger jobs on."""

    cpus = psutil.cpu_count(logical=False)
    interfaces = psutil.net_if_addrs()
    cli = sys.argv
    print(f"cpus = {cpus}")
    print(f"interfaces = {interfaces}")
    print(f"argv = {cli}")
    prologue = []
    # conda init hook
    prologue.append(f'eval "$({args.conda_path}/conda shell.bash hook)"')
    # mamba init hook.
    if args.conda_exec == "mamba":
        prologue.append(f'eval "$({args.conda_exec} shell hook --shell bash)"')
    # activate environment.
    prologue.append(f"{args.conda_exec} activate {args.conda_env}")
    # turn on maximum boost on ARCHER2.
    prologue.append("export SLURM_CPU_FREQ_REQ=2250000")

    # Define a slurm cluster.
    cluster = SLURMCluster(
        cores=args.slurm_cores,
        processes=args.slurm_processes,
        memory=args.slurm_memory,
        queue=args.slurm_queue,
        job_directives_skip=["--mem"],
        job_extra_directives=['--qos="standard"'],
        python="srun python",
        account=args.slurm_account,
        walltime=args.slurm_walltime,
        shebang="#!/bin/bash --login",
        local_directory="$PWD",
        interface="hsn0",
        job_script_prologue=prologue,
    )

    cluster.scale(jobs=args.slurm_nodes)
    client = Client(cluster)
    print("The job script that will be submitted to slurm is:")
    print(cluster.job_script())

    return client
