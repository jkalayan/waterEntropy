"""Tests for waterEntropy trig functions in utils."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.input_files import load_inputs
from waterEntropy.maths import trig

# get mda universe for arginine in solution
system = load_inputs.get_amber_arginine_soln_universe()
# some dummy atom positions
a = np.array([0, 0, 1])
b = np.array([0, 1, 0])
c = np.array([1, 0, 0])
d = np.array([0, 1, 1])
e = np.array([0, 11, 11])
dimensions = np.array([10, 10, 10])


def test_get_neighbourlist():
    """Test the get neighbourList function"""
    i_idx = 1
    i_coords = system.atoms.positions[i_idx]
    # get the heavy atom neighbour distances within a given distance cutoff
    neighbours = system.select_atoms(
        f"""mass 2 to 999 and not index {i_idx}"""  # bonded UAs can block
    )

    sorted_indices, sorted_distances = trig.get_neighbourlist(
        i_coords, neighbours, system.dimensions, max_cutoff=9e9
    )

    # check atom_i coords are not in the neighbour positions
    assert not (i_coords == neighbours.positions).all(axis=1).any()
    assert list(sorted_indices[:10]) == [4, 5, 6, 121, 1639, 1912, 8, 2260, 2413, 2314]
    assert list(sorted_distances[:10]) == pytest.approx(
        [
            1.5144727,
            2.37901456,
            2.44541458,
            3.4388837,
            3.52790124,
            3.67029653,
            3.78243397,
            3.85329979,
            3.99867795,
            4.07845752,
        ]
    )


def test_get_neighbourlist_rejects_duplicate_atom():
    """Reject a neighbour selection containing the queried atom."""
    with pytest.raises(ValueError, match="Atom coordinates"):
        trig.get_neighbourlist(
            system.atoms[1].position, system.atoms, system.dimensions
        )


def test_heavy_neighbour_shortlists_cache_and_fallback():
    """Build, reuse, and skip a shortlist when the cutoff is too large."""
    positions = trig.get_positions(system)
    shortlists = trig._heavy_neighbour_shortlists(system, positions, 5)

    assert shortlists is not None
    assert trig._heavy_neighbour_shortlists(system, positions, 5) is shortlists
    assert trig._heavy_neighbour_shortlists(system, positions, 1e9) is None


def test_get_sorted_neighbours_fast_path(monkeypatch):
    """Use the shortlist path for a heavy atom query."""
    monkeypatch.setenv("CODEENTROPY_FAST_NEIGHBOURS", "1")
    indices, distances = trig.get_sorted_neighbours(274, system, max_cutoff=5)
    light_indices, light_distances = trig.get_sorted_neighbours(0, system, max_cutoff=5)

    assert len(indices) == len(distances)
    assert np.all(distances[:-1] <= distances[1:])
    assert len(light_indices) == len(light_distances)


def test_get_shell_neighbour_selection_includes_bonded_atoms():
    """Include all atoms in a shell when heavy-atom filtering is disabled."""
    shell = SimpleNamespace(UA_shell=[29, 1228])
    indices, distances = trig.get_shell_neighbour_selection(
        shell, system.atoms[274], system, heavy_atoms=False
    )

    assert len(indices) == len(distances)
    assert len(indices) > 0


def test_get_angle():
    """Test the get angle function"""
    cosine_angle1 = trig.get_angle(a, b, c, dimensions)
    cosine_angle2 = trig.get_angle(a, b, d, dimensions)
    cosine_angle3 = trig.get_angle(a, b, e, dimensions)
    assert cosine_angle1 == 0.5
    assert cosine_angle2 == pytest.approx(0.7071067811865477)
    assert cosine_angle3 == pytest.approx(0.7071067811865477)


def test_get_distance():
    """Test the get distance function"""
    distance1 = trig.get_distance(a, b, dimensions)
    distance2 = trig.get_distance(a, d, dimensions)
    distance3 = trig.get_distance(c, d, dimensions)
    distance4 = trig.get_distance(c, e, dimensions)
    assert distance1 == pytest.approx(1.4142135623730951)
    assert distance2 == 1.0
    assert distance3 == pytest.approx(1.7320508075688772)
    assert distance4 == pytest.approx(1.7320508075688772)


def test_get_vector():
    """Test the get vector function"""
    delta_wrapped1 = trig.get_vector(a, b, dimensions)
    delta_wrapped2 = trig.get_vector(a, d, dimensions)
    delta_wrapped3 = trig.get_vector(c, d, dimensions)
    delta_wrapped4 = trig.get_vector(c, e, dimensions)
    assert list(delta_wrapped1) == [0, 1, -1]
    assert list(delta_wrapped2) == [0, 1, 0]
    assert list(delta_wrapped3) == [-1, 1, 1]
    assert list(delta_wrapped4) == [-1, 1, 1]
