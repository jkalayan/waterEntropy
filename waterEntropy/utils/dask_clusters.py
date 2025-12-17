"""
Helpers for setting up dask clusters on HPC.
"""

from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def configure_slurm_cluster(args):
    """Configure a SLURM HPC cluster to run bigger jobs on."""
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
        job_script_prologue=[
            'eval "$(/mnt/lustre/a2fs-nvme/work/c01/c01/jtg2/miniforge3/bin/conda shell.bash hook)"',
            'eval "$(mamba shell hook --shell bash)"',
            "mamba activate waterentropy",
            "export SLURM_CPU_FREQ_REQ=2250000",
        ],
    )

    cluster.scale(jobs=args.slurm.nodes)
    client = Client(cluster)

    return client
