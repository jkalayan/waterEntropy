"""
Helpers for setting up dask clusters on HPC.
"""

import subprocess
import sys

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import psutil


def slurm_submit_master(args):
    """Submit a master worker process for coordinating dask cluster setup,
    orchestration and shutdown."""
    # Get original CLI and remove --submit and exec.
    cli = sys.argv
    cli.pop(0)
    cli.remove("--submit")
    # Form a submit script
    with open("WE-master-submit.sh", "w", encoding="utf-8") as f:
        f.write("#!/bin/bash --login\n\n")
        f.write("#SBATCH --job-name=waterentropy-master\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks=1\n")
        f.write("#SBATCH --cpus-per-task=2\n")
        f.write(f"#SBATCH --time={args.hpc_walltime}\n")
        if args.hpc_account != "":
            f.write(f"#SBATCH --account={args.hpc_account}\n")
        f.write(f"#SBATCH --partition={args.hpc_queue}\n")
        if args.hpc_qos != "":
            f.write(f"#SBATCH --qos={args.hpc_qos}\n")
        f.write("\n")
        f.write(f'eval "$({args.conda_path} shell.bash hook)"\n')
        if args.conda_exec == "mamba":
            f.write(f'eval "$({args.conda_exec} shell hook --shell bash)"\n')
        f.write(f"{args.conda_exec} activate {args.conda_env}\n\n")
        f.write(f"srun waterEntropy {' '.join(cli)}")

    try:
        sub = subprocess.check_output(["bash", "-c", "sbatch WE-master-submit.sh"])
        print(sub.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print(e.output)


def slurm_configure_cluster(args):
    """Configure a SLURM HPC cluster to run bigger jobs on."""

    # Extra directives.
    extra = []
    if args.hpc_account != "":
        extra.append(f'--account="{args.hpc_account}"')
    if args.hpc_qos != "":
        extra.append(f'--qos="{args.hpc_qos}"')
    if args.hpc_constraint != "":
        extra.append(f'--constraint="{args.hpc_constraint}"')

    # Skip directives.
    skip = []
    skip = ["--mem"]

    # Prologues for appending env vars into the job submission script.
    prologue = []
    prologue.append(f'eval "$({args.conda_path} shell.bash hook)"')
    if args.conda_exec == "mamba":
        prologue.append(f'eval "$({args.conda_exec} shell hook --shell bash)"')
    prologue.append(f"{args.conda_exec} activate {args.conda_env}")
    prologue.append("export SLURM_CPU_FREQ_REQ=2250000")

    # Interfaces.
    hpc_nics = ["bond0", "ib0", "hsn0", "eth0"]
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
        walltime=args.hpc_walltime,
        shebang="#!/bin/bash --login",
        local_directory="$PWD",
        interface=iface,
        job_script_prologue=prologue,
    )

    cluster.scale(jobs=args.hpc_nodes)
    client = Client(cluster)
    with open("dask-cluster-submit.sh", "w", encoding="utf-8") as f:
        f.writelines(cluster.job_script())

    return client
