"""Tests for coordinate and force transformations."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.input_files import load_inputs
from waterEntropy.entropy.convariances import CovarianceCollection
from waterEntropy.entropy.vibrations import Vibrations
from waterEntropy.maths import transformations as trans
from waterEntropy.recipes.forces_torques import get_forces_torques


class FakeAtom:
    # pylint: disable=too-few-public-methods
    """Atom stub that can be combined with bonded selections."""

    mass = 12.0
    index = 0
    position = np.array([0.0, 0, 0])

    def __add__(self, other):
        return FakeSelection([self, *other])


class FakeAtoms(list):
    # pylint: disable=too-few-public-methods
    """Minimal atom collection for mass aggregation tests."""

    def __init__(self, atoms=(), bonded=()):
        super().__init__(atoms)
        self.bonded = bonded

    def select_atoms(self, selection):
        """Return bonded atoms or hydrogens matching the selection."""
        if selection.startswith("bonded index"):
            return FakeAtoms(self.bonded)
        return FakeAtoms(atom for atom in self if 1 <= atom.mass <= 1.1)


def test_get_ua_masses_adds_bonded_hydrogens_and_skips_hydrogens():
    """Aggregate hydrogen masses into heavy-atom united atoms."""
    heavy = SimpleNamespace(mass=12.0, index=0)
    hydrogen = SimpleNamespace(mass=1.0, index=1)
    heavy.bonded = [hydrogen]
    atoms = FakeAtoms([heavy, hydrogen])
    atoms.bonded = [hydrogen]

    assert trans.get_UA_masses(atoms) == [13.0]


def test_get_axes_for_multiple_uas_uses_ua_moi(monkeypatch):
    """Build and sort the moment of inertia axes for multiple united atoms."""
    molecule = SimpleNamespace(
        moment_of_inertia=lambda unwrap: np.diag([1.0, 3.0, 2.0]),
        atoms=object(),
        positions=np.zeros((1, 3)),
        principal_axes=lambda: np.eye(3),
        center_of_mass=lambda: np.zeros(3),
    )
    monkeypatch.setattr(
        trans.Selections,
        "find_molecule_UAs",
        lambda _molecule: SimpleNamespace(positions=np.array([[1.0, 0, 0]])),
    )
    monkeypatch.setattr(trans, "get_UA_masses", lambda _molecule: [2.0])
    monkeypatch.setattr(trans, "MOI", lambda *_args: np.diag([1.0, 3.0, 2.0]))
    monkeypatch.setattr(trans, "make_whole", lambda _atoms: None)

    axes, moi = trans.get_axes(molecule, "multiple_UAs", np.ones(3))

    np.testing.assert_array_equal(axes, np.eye(3))
    np.testing.assert_array_equal(moi, [3.0, 2.0, 1.0])


@pytest.mark.parametrize(
    "signs",
    [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, -1]],
)
def test_single_ua_axes_have_geometry_defined_signs(monkeypatch, signs):
    """Orient every principal axis consistently despite eigenvector sign changes."""
    positions = np.array([[0.0, 0, 0], [1.0, 1, 0], [1.0, -1, 0]])
    principal_axes = np.array([[0.0, 0, 1], [0, 1, 0], [1, 0, 0]])
    molecule = SimpleNamespace(
        moment_of_inertia=lambda unwrap: np.diag([1.0, 2.0, 3.0]),
        atoms=object(),
        positions=positions,
        principal_axes=lambda: principal_axes * np.array(signs)[:, None],
    )
    monkeypatch.setattr(trans, "make_whole", lambda _atoms: None)
    monkeypatch.setattr(
        trans.Selections,
        "find_molecule_UAs",
        lambda _molecule: [SimpleNamespace(position=positions[0])],
    )

    axes, _ = trans.get_axes(molecule, "single_UA", np.ones(3) * 10)

    np.testing.assert_allclose(axes, principal_axes * np.array([-1, 1, 1])[:, None])


def test_single_ua_axis_orientation_rotates_with_molecule():
    """The chosen frame follows rigid rotations and translations."""
    positions = np.array([[0.0, 0, 0], [1.0, 1, 0], [1.0, -1, 0]])
    axes = np.array([[0.0, 0, 1], [0, 1, 0], [1, 0, 0]])
    rotation = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
    translation = np.array([3.0, 4, 5])
    expected = trans.orient_single_ua_axes(positions, axes, positions[0])
    rotated_positions = positions @ rotation.T + translation

    actual = trans.orient_single_ua_axes(
        rotated_positions, axes @ rotation.T, rotated_positions[0]
    )

    np.testing.assert_allclose(actual, expected @ rotation.T, atol=1e-14)


def test_single_ua_axes_ignore_near_orthogonal_bond_projections():
    """Tiny numerical noise must not override the molecular plane normal."""
    positions = np.array([[0.0, 0, 0], [1.0, 1, 0], [1.0, -1, 0]])
    axes = np.array([[1e-10, 0, 1], [0, 1, 0], [1, 0, 0]])

    oriented = trans.orient_single_ua_axes(positions, axes, positions[0])

    np.testing.assert_allclose(oriented[0], -axes[0])


def test_single_ua_axes_reject_unoriented_geometry():
    """Reject a geometry that cannot provide a direction for every axis."""
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])

    with pytest.raises(ValueError, match="molecular geometry"):
        trans.orient_single_ua_axes(positions, np.eye(3), positions[0])


def test_water_covariances_ignore_individual_principal_axis_signs(monkeypatch):
    """Sign choices for different waters must not change averaged observables."""
    system = load_inputs.get_amber_arginine_soln_universe()
    waters = system.select_atoms("resname WAT and name O")[:20]

    def calculate():
        covariance = CovarianceCollection()
        for water in waters:
            get_forces_torques(covariance, water.fragment, "WAT", system)
        vibration = Vibrations(298)
        vibration.add_data(covariance, diagonalise=False)
        return covariance, vibration

    original_covariance, original_vibration = calculate()
    atom_group = type(waters)
    original_axes = atom_group.principal_axes

    def flipped_axes(molecule, *args, **kwargs):
        axes = original_axes(molecule, *args, **kwargs)
        signs = np.array(
            [-1 if (int(molecule.indices[0]) // 3) & (1 << i) else 1 for i in range(3)]
        )
        return axes * signs[:, None]

    monkeypatch.setattr(atom_group, "principal_axes", flipped_axes)
    flipped_covariance, flipped_vibration = calculate()
    key = ("WAT", "WAT")

    assert original_covariance.counts[key] == flipped_covariance.counts[key] == 20
    for field in ("forces", "torques"):
        np.testing.assert_allclose(
            getattr(original_covariance, field)[key],
            getattr(flipped_covariance, field)[key],
            rtol=1e-12,
            atol=1e-12,
        )
    for field in ("translational_S", "rotational_S"):
        np.testing.assert_allclose(
            np.sort(getattr(original_vibration, field)[key]),
            np.sort(getattr(flipped_vibration, field)[key]),
            rtol=1e-12,
            atol=1e-12,
        )


def test_single_ua_axes_reject_degenerate_moments(monkeypatch):
    """A degenerate inertia tensor cannot define a unique principal frame."""
    molecule = SimpleNamespace(
        moment_of_inertia=lambda unwrap: np.diag([1.0, 1.0, 2.0]),
        atoms=object(),
        positions=np.array([[0.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]]),
        principal_axes=lambda: np.eye(3),
    )
    monkeypatch.setattr(trans, "make_whole", lambda _atoms: None)
    monkeypatch.setattr(
        trans.Selections,
        "find_molecule_UAs",
        lambda _molecule: [SimpleNamespace(position=molecule.positions[0])],
    )

    with pytest.raises(ValueError, match="degenerate"):
        trans.get_axes(molecule, "single_UA", np.ones(3) * 10)


def test_moi_returns_symmetric_tensor():
    """Calculate diagonal and off-diagonal inertia terms."""
    result = trans.MOI(
        np.zeros(3),
        np.array([[1.0, 2.0, 3.0]]),
        [2.0],
    )

    np.testing.assert_array_equal(
        result,
        [[26.0, -4.0, -6.0], [-4.0, 20.0, -12.0], [-6.0, -12.0, 10.0]],
    )


def test_get_custom_axes_uses_vector_between_bonded_atoms_for_three_bonds():
    """Construct axes using the first pair of bonded heavy atoms."""
    axes = trans.get_custom_axes(
        np.zeros(3),
        [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])],
        np.array([1.0, 1.0, 0]),
        np.array([10.0, 10.0, 10.0]),
    )

    np.testing.assert_allclose(axes[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(axes[1], [1.0, 0.0, -1.0])
    np.testing.assert_allclose(axes[2], [-1.0, 2.0, -1.0])


def test_get_custom_pi_moi_flips_axes_and_sums_atom_contributions():
    """Return oriented axes and their mass-weighted inertia components."""
    molecule = SimpleNamespace(
        positions=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        masses=np.array([2.0, 1.0]),
    )

    axes, moi = trans.get_custom_PI_MOI(
        molecule, np.eye(3), np.array([1.0, 0, 0]), np.ones(3) * 10
    )

    np.testing.assert_allclose(axes, np.eye(3))
    np.testing.assert_allclose(moi, [0.0, 2.0, 2.0])


def test_get_bonded_axes_returns_none_for_light_atom():
    """Skip bonded-axis calculations for hydrogen atoms."""
    atom = SimpleNamespace(mass=1.0)

    assert trans.get_bonded_axes(None, atom, np.ones(3)) is None


@pytest.mark.parametrize(
    ("heavy_count", "light_count", "expected_b_list", "expected_c"),
    [
        (2, 0, "first_heavy", "second_heavy"),
        (1, 2, "first_heavy", "first_light"),
        (1, 1, "first_heavy", "zero"),
        (3, 0, "all_heavy", "second_heavy"),
    ],
)
def test_get_bonded_axes_selects_bond_vectors(
    monkeypatch, heavy_count, light_count, expected_b_list, expected_c
):
    """Choose the bond vectors for each supported heavy-atom arrangement."""
    atom = FakeAtom()
    heavy = [
        SimpleNamespace(position=np.array([float(i + 1), 0, 0]))
        for i in range(heavy_count)
    ]
    light = [
        SimpleNamespace(position=np.array([0.0, float(i + 1), 0]))
        for i in range(light_count)
    ]
    heavy_atoms = SimpleNamespace(
        __len__=lambda self: len(heavy),
        positions=np.array([a.position for a in heavy]),
    )
    # len() must be supported by the selection object.
    heavy_atoms = FakeSelection(heavy)
    light_atoms = FakeSelection(light)
    monkeypatch.setattr(
        trans.Selections, "find_bonded_atoms", lambda *_args: (heavy_atoms, light_atoms)
    )
    monkeypatch.setattr(
        trans, "get_custom_PI_MOI", lambda *_args: ("oriented_axes", "moi")
    )
    captured = {}

    def fake_custom_axes(_a, b_list, c, _dimensions):
        captured["b"] = b_list
        captured["c"] = c
        return "custom_axes"

    monkeypatch.setattr(trans, "get_custom_axes", fake_custom_axes)

    axes, moi = trans.get_bonded_axes(object(), atom, np.ones(3))

    assert axes == "oriented_axes"
    assert moi == "moi"
    if expected_b_list == "all_heavy":
        assert len(captured["b"]) == heavy_count
    else:
        assert len(captured["b"]) == 1
    if expected_c == "zero":
        np.testing.assert_array_equal(captured["c"], np.zeros(3))
    elif expected_c == "first_light":
        np.testing.assert_array_equal(captured["c"], light[0].position)
    elif expected_c == "second_heavy":
        np.testing.assert_array_equal(captured["c"], heavy[1].position)


class FakeSelection(list):
    """List-like atom selection with a positions array."""

    @property
    def positions(self):
        """Return the selected atoms’ positions as an array."""
        return np.array([atom.position for atom in self])

    def __add__(self, other):
        """Combine this selection with another atom or selection."""
        return FakeSelection([*self, *other])
