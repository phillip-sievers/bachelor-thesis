"""
UTs for the geometry operations adn transformations. Because its super tricky
"""

import pytest
from pyproj import Geod
from shapely.geometry import Point, Polygon

from thesis.geometry.signed_distance import (
    is_inside,
    signed_distance_m,
    to_utm,
    utm_epsg,
)


@pytest.mark.parametrize(
    "longitude, latitude, expected",
    [
        (-103.4, 44.7, 26913),
        (-73.98, 40.75, 26918),
        (-124.0, 45.0, 26910),
        (-67.0, 45.0, 26919),
    ],
)
def test_utm_espg_return_correct_zone(longitude, latitude, expected):
    assert utm_epsg(longitude, latitude) == expected


@pytest.fixture
def square():
    return Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])


@pytest.fixture
def el_chapo():
    """L-shape"""
    return Polygon([(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)])


@pytest.mark.parametrize(
    "point, expected",
    [
        ((50, 50), -50.0),
        ((99, 50), -1.0),
        ((50, 0), 0.0),
        ((150, 50), 50.0),
        ((200, 200), 141.4213562),
    ],
)
def test_signed_distance_square(square, point, expected):
    assert signed_distance_m(Point(point), square) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((20, 20), -20.0),
        ((90, 60), 20.0),
    ],
)
def test_signed_distance_lchapo(el_chapo, point, expected):
    assert signed_distance_m(Point(point), el_chapo) == pytest.approx(expected, abs=1e-6)


def test_utm_distance_matches_geodesic():
    lon1, lat1 = -103.45, 44.72
    geod = Geod(ellps="GRS80")
    lon2, lat2, _ = geod.fwd(lon1, lat1, 90, 100)

    epsg = utm_epsg(lon1, lat1)
    p1 = to_utm(Point(lon1, lat1), epsg)
    p2 = to_utm(Point(lon2, lat2), epsg)

    assert p1.distance(p2) == pytest.approx(100, abs=0.05)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((50, 50), True),
        ((99, 50), True),
        ((50, 0), True),
        ((150, 50), False),
        ((200, 200), False),
    ],
)
def test_is_inside_square(square, point, expected):
    assert is_inside(point, square) is expected


@pytest.mark.parametrize(
    "point, expected",
    [
        ((20, 20), True),
        ((90, 60), False),
    ],
)
def test_is_inside_l(el_chapo, point, expected):
    assert is_inside(point, el_chapo) is expected
