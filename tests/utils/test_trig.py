""" Tests for waterEntropy trig functions in utils."""

import numpy as np
import pytest

from tests.input_files import load_inputs
from waterEntropy.utils import trig

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
    assert (i_coords == neighbours.positions).all(axis=1).any() is False
    assert list(sorted_indices[:10]) == [4, 6, 5, 621, 1143, 8, 1800, 1413, 888, 10]
    assert list(sorted_distances[:10]) == pytest.approx(
        [
            1.5091673359341684,
            2.4051885846072225,
            2.4432441098341773,
            3.5774782395107163,
            3.6060799363818425,
            3.799589247428219,
            3.9103016915217963,
            4.296909612890527,
            4.322326226845404,
            4.453261937356677,
        ]
    )


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
