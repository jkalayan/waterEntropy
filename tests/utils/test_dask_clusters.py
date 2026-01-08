""" Tests for waterEntropy dask_clusters functions in utils."""

import argparse
import os
from unittest import mock

import pytest

import waterEntropy.utils.dask_clusters as dc


def args_helper_directives(args):
    """helper to setup the CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpc-account", default="")
    parser.add_argument("--hpc-constraint", default="")
    parser.add_argument("--hpc-qos", default="")
    args = parser.parse_args(args)
    return args


def args_helper_prologues(args):
    """helper to setup the CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", default="")
    parser.add_argument("--conda-exec", default="")
    parser.add_argument("--conda-path", default="")
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
