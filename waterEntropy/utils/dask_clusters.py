"""
Helpers for setting up dask clusters on HPC.
"""

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import psutil


def configure_hpc_cluster(args):
    """Configure a SLURM HPC cluster to run bigger jobs on."""

    # Extra directives.
    extra = []
    if args.qos != "":
        extra.append(f'--qos="{args.hpc_qos}"')
    if args.constraint != "":
        extra.append(f'--constraint="{args.hpc_constraint}"')

    # Skip directives.
    skip = []
    skip = ["--mem"]

    # Prologues for appending env vars into the job submission script.
    prologue = []
    prologue.append(f'eval "$({args.conda_path}/conda shell.bash hook)"')
    if args.conda_exec == "mamba":
        prologue.append(f'eval "$({args.conda_exec} shell hook --shell bash)"')
    prologue.append(f"{args.conda_exec} activate {args.conda_env}")
    prologue.append("export SLURM_CPU_FREQ_REQ=2250000")

    # Interfaces.
    hpc_nics = ["ib0", "hsn0", "eth0"]
    interfaces = psutil.net_if_addrs()
    for iface in hpc_nics:
        if iface in interfaces:
            break

    # Define a slurm cluster.
    cluster = SLURMCluster(
        cores=args.hpc_cores,
        processes=args.hpc_processes,
        memory=args.hpc_memory,
        queue=args.hpc_queue,
        job_directives_skip=skip,
        job_extra_directives=extra,
        python="srun python",
        account=args.hpc_account,
        walltime=args.hpc_walltime,
        shebang="#!/bin/bash --login",
        local_directory="$PWD",
        interface=iface,
        job_script_prologue=prologue,
    )

    cluster.scale(jobs=args.hpc_nodes)
    client = Client(cluster)
    print("The job script that will be submitted to slurm is:")
    print(cluster.job_script())

    return client
