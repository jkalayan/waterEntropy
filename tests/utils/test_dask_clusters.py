""" Tests for waterEntropy dask_clusters functions in utils."""

import argparse
import os
import sys
from unittest import mock

import pytest

import waterEntropy.utils.dask_clusters as dc

SUBMITFILE_TESTCASE1 = """#!/bin/bash --login

#SBATCH --job-name=waterentropy-master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --account=c01-bio
#SBATCH --partition=standard
#SBATCH --qos=standard

eval "$(/path/to/conda shell.bash hook)"
conda activate waterentropy

srun waterEntropy --file-topology box.prmtop --file-coords frames.nc --start 0 --end 512 --step 1 --hpc --hpc-nodes 4 --hpc-account c01-bio --hpc-qos standard"""  # pylint: disable=line-too-long

SUBMITFILE_TESTCASE2 = """#!/bin/bash --login

#SBATCH --job-name=waterentropy-master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --account=c01-bio
#SBATCH --partition=standard
#SBATCH --qos=standard

eval "$(/path/to/conda shell.bash hook)"
eval "$(mamba shell hook --shell bash)"
mamba activate waterentropy

srun waterEntropy --file-topology box.prmtop --file-coords frames.nc --start 0 --end 512 --step 1 --hpc --hpc-nodes 4 --hpc-account c01-bio --hpc-qos standard"""  # pylint: disable=line-too-long


def args_helper_directives(args):
    """helper to setup the CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpc-account", type=str, default="")
    parser.add_argument("--hpc-constraint", type=str, default="")
    parser.add_argument("--hpc-qos", type=str, default="")
    args = parser.parse_args(args)
    return args


def args_helper_prologues(args):
    """helper to setup the CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", type=str, default="")
    parser.add_argument("--conda-exec", type=str, default="")
    parser.add_argument("--conda-path", type=str, default="")
    args = parser.parse_args(args)
    return args


def args_helper_submitfile(args):
    """helper to setup the CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", type=str, default="")
    parser.add_argument("--conda-exec", type=str, default="")
    parser.add_argument("--conda-path", type=str, default="")
    parser.add_argument("--file-topology", type=str, default="")
    parser.add_argument("--file-coords", type=str, default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--hpc-nodes", default="")
    parser.add_argument("--hpc-account", type=str, default="")
    parser.add_argument("--hpc-qos", type=str, default="")
    parser.add_argument("--hpc-queue", type=str, default="standard")
    parser.add_argument("--hpc-walltime", type=str, default="24:00:00")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(args)
    return args


def test_slurm_envfix1():
    """Test that calling function at all executes ok"""
    dc.check_slurm_env()


def test_slurm_envfix2():
    """Create target env var and see if it gets deleted."""
    # Set the environment variable to some value and check it is set.
    os.environ["SLURM_CPU_BIND"] = "1"
    assert os.environ["SLURM_CPU_BIND"] == "1"
    dc.check_slurm_env()
    # Check we now get a keyerror exception.
    with pytest.raises(KeyError):
        assert os.environ["SLURM_CPU_BIND"] == "1"


def test_slurm_directives_account():
    """Test that account gets set"""
    args = args_helper_directives(["--hpc-account", "c01"])
    directives = dc.slurm_directives(args)[0]
    assert directives == ['--account="c01"']


def test_slurm_directives_constraint():
    """Test that constraints get set"""
    args = args_helper_directives(["--hpc-constraint", "intel25"])
    directives = dc.slurm_directives(args)[0]
    assert directives == ['--constraint="intel25"']


def test_slurm_directives_qos():
    """Test that qos gets set"""
    args = args_helper_directives(["--hpc-qos", "standard"])
    directives = dc.slurm_directives(args)[0]
    assert directives == ['--qos="standard"']


def test_slurm_directives_all():
    """Test multiple values get set."""
    args = args_helper_directives(
        ["--hpc-account", "c01", "--hpc-constraint", "intel25", "--hpc-qos", "standard"]
    )
    directives = dc.slurm_directives(args)[0]
    assert directives == [
        '--account="c01"',
        '--qos="standard"',
        '--constraint="intel25"',
    ]


def test_slurm_directives_skip():
    """Test that skipped values work."""
    args = args_helper_directives(["--hpc-account", "c01"])
    skip = dc.slurm_directives(args)[1]
    assert skip == ["--mem"]


def test_slurm_prologues_conda():
    """Test that given plausable values that the prologue for conda is correctly assembled"""
    args = args_helper_prologues(
        [
            "--conda-env",
            "waterentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
        ]
    )
    prologue = dc.slurm_prologues(args)
    assert prologue == [
        'eval "$(/path/to/conda shell.bash hook)"',
        "conda activate waterentropy",
        "export SLURM_CPU_FREQ_REQ=2250000",
    ]


def test_slurm_prologues_mamba():
    """Test that given plausable values that the prologue for mamba is correctly assembled"""
    args = args_helper_prologues(
        [
            "--conda-env",
            "waterentropy",
            "--conda-exec",
            "mamba",
            "--conda-path",
            "/path/to/conda",
        ]
    )
    prologue = dc.slurm_prologues(args)
    assert prologue == [
        'eval "$(/path/to/conda shell.bash hook)"',
        'eval "$(mamba shell hook --shell bash)"',
        "mamba activate waterentropy",
        "export SLURM_CPU_FREQ_REQ=2250000",
    ]


@mock.patch("psutil.net_if_addrs")
def test_interface_selection(net_if_addrs):
    """Test interface selection"""
    net_if_addrs.return_value = ["ib0", "eth0"]
    iface = dc.system_network_interface()
    assert iface == "ib0"


@mock.patch("subprocess.check_output")
def test_submit_master(checkoutput):
    """Test master submit file creation"""
    mock_stdout = mock.MagicMock()
    mock_stdout.configure_mock(
        **{"stdout.decode.return_value": "test job submitted id xxxxxx"}
    )
    checkoutput.return_value = mock_stdout
    cli = [
        "waterEntropy",
        "--file-topology",
        "box.prmtop",
        "--file-coords",
        "frames.nc",
        "--start",
        "0",
        "--end",
        "512",
        "--step",
        "1",
        "--hpc",
        "--hpc-nodes",
        "4",
        "--hpc-account",
        "c01-bio",
        "--hpc-qos",
        "standard",
        "--submit",
    ]
    args = args_helper_submitfile(
        [
            "--conda-env",
            "waterentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
            "--file-topology",
            "box.prmtop",
            "--file-coords",
            "frames.nc",
            "--start",
            "0",
            "--end",
            "512",
            "--step",
            "1",
            "--hpc",
            "--hpc-nodes",
            "4",
            "--hpc-account",
            "c01-bio",
            "--hpc-qos",
            "standard",
            "--submit",
        ]
    )
    with mock.patch.object(sys, "argv", cli):
        dc.slurm_submit_master(args)
    with open("WE-master-submit.sh", encoding="utf-8") as file:
        submitfile = file.read()
        assert submitfile == SUBMITFILE_TESTCASE1
    os.remove("WE-master-submit.sh")


@mock.patch("subprocess.check_output")
def test_submit_master_mamba(checkoutput):
    """Test master submit file creation"""
    mock_stdout = mock.MagicMock()
    mock_stdout.configure_mock(
        **{"stdout.decode.return_value": "test job submitted id xxxxxx"}
    )
    checkoutput.return_value = mock_stdout
    cli = [
        "waterEntropy",
        "--file-topology",
        "box.prmtop",
        "--file-coords",
        "frames.nc",
        "--start",
        "0",
        "--end",
        "512",
        "--step",
        "1",
        "--hpc",
        "--hpc-nodes",
        "4",
        "--hpc-account",
        "c01-bio",
        "--hpc-qos",
        "standard",
        "--submit",
    ]
    args = args_helper_submitfile(
        [
            "--conda-env",
            "waterentropy",
            "--conda-exec",
            "mamba",
            "--conda-path",
            "/path/to/conda",
            "--file-topology",
            "box.prmtop",
            "--file-coords",
            "frames.nc",
            "--start",
            "0",
            "--end",
            "512",
            "--step",
            "1",
            "--hpc",
            "--hpc-nodes",
            "4",
            "--hpc-account",
            "c01-bio",
            "--hpc-qos",
            "standard",
            "--submit",
        ]
    )
    with mock.patch.object(sys, "argv", cli):
        dc.slurm_submit_master(args)
    with open("WE-master-submit.sh", encoding="utf-8") as file:
        submitfile = file.read()
        assert submitfile == SUBMITFILE_TESTCASE2
    os.remove("WE-master-submit.sh")
