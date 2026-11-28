"""Tests for force and torque covariance recipes."""

from types import SimpleNamespace

import numpy as np

from waterEntropy.entropy.convariances import CovarianceCollection
from waterEntropy.recipes import forces_torques


def test_get_forces_torques_single_ua(monkeypatch):
    """Scale and store force and torque covariances for one united atom."""
    molecule = SimpleNamespace(resnames=["WAT"], center_of_mass=lambda: np.zeros(3))
    calls = []

    monkeypatch.setattr(
        forces_torques.Select, "guess_length_scale", lambda _molecule: "single_UA"
    )
    monkeypatch.setattr(
        forces_torques.Transformation,
        "get_axes",
        lambda _molecule, scale: calls.append(("axes", scale)) or ("axes", "moi"),
    )
    monkeypatch.setattr(
        forces_torques.Transformation,
        "get_torques",
        lambda *_args: calls.append(("torques",)) or "torque",
    )
    monkeypatch.setattr(
        forces_torques.Transformation,
        "get_mass_weighted_forces",
        lambda *_args: calls.append(("forces",)) or "force",
    )
    monkeypatch.setattr(
        forces_torques.Transformation,
        "get_covariance_matrix",
        lambda value, scale: np.full((3, 3), scale if value else -1),
    )

    covariances = CovarianceCollection()
    forces_torques.get_forces_torques(covariances, molecule, "SOL", object())

    np.testing.assert_array_equal(covariances.forces[("SOL", "WAT")], 0.5)
    np.testing.assert_array_equal(covariances.torques[("SOL", "WAT")], 0.5)
    assert covariances.counts[("SOL", "WAT")] == 1
    assert calls == [("axes", "single_UA"), ("torques",), ("forces",)]


def test_get_forces_torques_multiple_uas_skips_unbonded(monkeypatch):
    """Aggregate per-UA covariances while skipping unavailable bond axes."""
    uas = [
        SimpleNamespace(position=np.array([1.0, 0, 0])),
        SimpleNamespace(position=np.array([2.0, 0, 0])),
    ]
    molecule = type("Molecule", (), {})()
    molecule.resnames = ["SOL"]
    molecule.center_of_mass = lambda: np.zeros(3)
    system = SimpleNamespace(dimensions=np.array([10.0, 11, 12, 90, 90, 90]))
    calls = []
    covariance_values = iter([1, 1, 2, 2])

    monkeypatch.setattr(
        forces_torques.Select, "guess_length_scale", lambda _molecule: "multiple_UAs"
    )
    monkeypatch.setattr(
        forces_torques.Select, "find_molecule_UAs", lambda _molecule: uas
    )
    monkeypatch.setattr(
        forces_torques.Transformation,
        "get_axes",
        lambda *_args: ("principal", "moi"),
    )
    monkeypatch.setattr(
        forces_torques.Transformation,
        "get_torques",
        lambda *_args: np.eye(3),
    )
    monkeypatch.setattr(
        forces_torques.Transformation,
        "get_mass_weighted_forces",
        lambda *_args: np.eye(3),
    )
    monkeypatch.setattr(
        forces_torques.Transformation,
        "get_covariance_matrix",
        lambda value, scale: np.full((3, 3), next(covariance_values)),
    )

    def bonded_axes(_system, ua, _dimensions):
        calls.append(ua)
        return (None, None) if ua is uas[0] else (("custom",), ("position",))

    monkeypatch.setattr(
        forces_torques.Transformation,
        "get_bonded_axes",
        bonded_axes,
    )

    covariances = CovarianceCollection()
    forces_torques.get_forces_torques(covariances, molecule, "WAT", system)

    assert calls == [uas[0], uas[1]]
    np.testing.assert_array_equal(covariances.forces[(molecule, molecule)], 2)
    np.testing.assert_array_equal(covariances.torques[(molecule, molecule)], 2)
    assert covariances.forces[(molecule, molecule)].shape == (1, 3, 3)
