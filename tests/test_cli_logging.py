"""Tests for CLI error logging."""

import argparse
import logging

import pytest

from waterEntropy.cli import runBulk, runWaterEntropy


@pytest.mark.parametrize(
    ("module", "function", "variables", "message"),
    [
        (runBulk, "_conda_env", ("CONDA_DEFAULT_ENV",), "activate your conda"),
        (runWaterEntropy, "_conda_env", ("CONDA_DEFAULT_ENV",), "activate your conda"),
        (
            runWaterEntropy,
            "_conda_exec",
            ("MAMBA_EXE", "CONDA_EXE"),
            "Cannot determine your conda executable",
        ),
        (runWaterEntropy, "_conda_path", ("CONDA_EXE",), "conda/mamba set up"),
    ],
)
def test_missing_conda_settings_are_logged(
    module, function, variables, message, monkeypatch, caplog
):
    """Report missing environment settings before exiting."""
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
        getattr(module, function)()

    assert exc_info.value.code == 1
    assert any(
        record.name == module.__name__ and message in record.message
        for record in caplog.records
    )


def test_bulk_argument_errors_are_logged_and_reraised(monkeypatch, caplog):
    """Report argument errors without swallowing the original exception."""
    error = argparse.ArgumentError(None, "invalid argument")

    def fail_parse(_parser):
        raise error

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fail_parse)
    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(argparse.ArgumentError) as exc_info,
    ):
        runBulk.main()

    assert exc_info.value is error
    assert any(
        record.name == runBulk.__name__
        and "Command line arguments are ill-defined" in record.message
        for record in caplog.records
    )
