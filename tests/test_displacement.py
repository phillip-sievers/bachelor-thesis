from math import atan2, pi, tau

import pytest
from shapely.geometry import Point, box

from thesis.geometry.displacement import (
    generate_displacements,
    point_at_signed_distance,
)
from thesis.geometry.signed_distance import signed_distance_m


@pytest.mark.parametrize(
    "target_m, expected_x",
    [
        (-5, 15),
        (-3, 17),
        (-1, 19),
        (1, 21),
        (3, 23),
        (5, 25),
        (10, 30),
        (20, 40),
        (40, 60),
        (80, 100),
    ],
)
def test_configured_levels_are_edge_distances_along_one_ray(target_m, expected_x):
    footprint = box(0, 0, 20, 20)
    anchor = Point(10, 10)

    pin = point_at_signed_distance(footprint, anchor, target_m, bearing_radians=0, tolerance_m=0.1)
    assert pin
    assert pin.equals_exact(Point(expected_x, 10), tolerance=0.1)
    assert signed_distance_m(pin, footprint) == pytest.approx(target_m, abs=0.1)


def test_inside_level_unreachable_on_the_fixed_bearing_is_unavailable():
    footprint = box(0, 0, 100, 20)

    pin = point_at_signed_distance(footprint, Point(5, 5), -8, bearing_radians=pi, tolerance_m=0.1)

    assert pin is None


def test_numerical_miss_is_not_reported_as_structurally_unavailable():
    footprint = box(0, 0, 20, 20)

    with pytest.raises(RuntimeError, match="tolerance"):
        point_at_signed_distance(footprint, Point(10, 10), 80, bearing_radians=0.3, tolerance_m=0)


def test_generator_reuses_one_reproducible_bearing_across_levels():
    footprint = box(0, 0, 20, 20)
    anchor = Point(10, 10)
    first = generate_displacements(footprint, anchor, [-5, -1, 1, 80], seed=42, building_id="house")
    repeated = generate_displacements(footprint, anchor, [-5, -1, 1, 80], seed=42, building_id="house")
    fixed_levels = [result for result in first if result.target_signed_distance_m is not None]
    bearing = fixed_levels[0].bearing_radians

    assert len({result.bearing_radians for result in fixed_levels}) == 1
    assert bearing is not None
    assert all(
        atan2(result.point.y - anchor.y, result.point.x - anchor.x) % tau == pytest.approx(bearing)
        for result in fixed_levels
    )
    assert [result.point.wkt for result in first] == [result.point.wkt for result in repeated]


def test_generator_records_an_unreachable_level():
    result = generate_displacements(box(0, 0, 20, 20), Point(10, 10), [-15], seed=42, building_id="house")[0]

    assert result.target_signed_distance_m == -15
    assert result.actual_signed_distance_m is None
    assert result.point is None


def test_generator_adds_maximum_inscribed_circle_control():
    footprint = box(0, 0, 100, 20)

    control = generate_displacements(
        footprint,
        Point(5, 5),
        [-8],
        seed=42,
        building_id="house",
    )[-1]

    assert control.target_signed_distance_m is None
    assert control.actual_signed_distance_m == pytest.approx(-10, abs=0.1)
    assert control.bearing_radians is None
    assert footprint.covers(control.point)
