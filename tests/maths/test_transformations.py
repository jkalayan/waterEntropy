"""Tests for coordinate and force transformations."""

from types import SimpleNamespace

import numpy as np
import pytest

from waterEntropy.maths import transformations as trans


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
        np.testing.assert_array_equal(
            captured["c"], heavy[1].position if heavy_count == 2 else heavy[1].position
        )


class FakeSelection(list):
    """List-like atom selection with a positions array."""

    @property
    def positions(self):
        """Return the selected atoms’ positions as an array."""
        return np.array([atom.position for atom in self])

    def __add__(self, other):
        """Combine this selection with another atom or selection."""
        return FakeSelection([*self, *other])
