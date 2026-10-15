"""Replacement tests for the removed v1 ray-displacement generator."""

import math

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import Point, box

from thesis.geometry.bins import DistanceBin, distance_bins
from thesis.geometry.scene_sampling import (
    GridInvariantError,
    SceneGrid,
    accept_exact,
    assert_crop_transform,
    build_scene_grid,
    exact_label,
    sample_source_bins,
)


def _footprints(records):
    return gpd.GeoDataFrame(
        {
            "building_id": [record[0] for record in records],
            "provenance": [record[1] for record in records],
            "geometry": [record[2] for record in records],
        },
        crs="EPSG:5070",
    )


def test_outside_label_uses_nearest_polygon_not_source_polygon():
    footprints = _footprints(
        [
            ("source", "overture-a", box(0, 0, 4, 4)),
            ("near", "overture-b", box(6, 0, 10, 4)),
        ]
    )

    label = exact_label(Point(5.5, 2), footprints)

    assert label.signed_distance_m == pytest.approx(0.5)
    assert label.nearest_polygon_id == "near"
    assert label.nearest_polygon_provenance == "overture-b"
    assert label.bearing_radians == pytest.approx(0.0)


def test_inside_label_is_negative_distance_to_union_edge():
    footprints = _footprints([("source", "overture-a", box(0, 0, 10, 10))])

    label = exact_label(Point(5, 4), footprints)

    assert label.signed_distance_m == pytest.approx(-4)
    assert label.nearest_polygon_id == "source"
    assert label.bearing_radians == pytest.approx(3 * math.pi / 2)


def test_inside_nearest_polygon_comes_from_the_union_edge_not_an_occluded_overlap_edge():
    footprints = _footprints(
        [
            ("wide", "a", box(0, 0, 10, 10)),
            ("overlap", "b", box(4, 4, 8, 8)),
        ]
    )

    label = exact_label(Point(5, 5), footprints)

    assert label.signed_distance_m == pytest.approx(-5)
    assert label.nearest_polygon_id == "wide"


def test_scene_grid_uses_exact_affine_and_nearest_ownership():
    transform = from_origin(0, 12, 1, 1)
    footprints = _footprints(
        [
            ("source", "a", box(2, 4, 6, 8)),
            ("near", "b", box(8, 4, 11, 8)),
        ]
    )

    grid = build_scene_grid(footprints, "source", (12, 12), transform)

    assert grid.transform == transform
    assert grid.signed_distance_m[5, 6] == pytest.approx(1)
    assert grid.owner_at(5, 6) == "source"
    assert grid.owner_at(5, 7) == "near"
    assert grid.signed_distance_m[5, 3] < 0


def test_partial_source_is_kept_and_missing_inside_bins_are_size_failures():
    transform = from_origin(0, 20, 1, 1)
    footprints = _footprints([("source", "a", box(8, 8, 12, 12))])
    grid = build_scene_grid(footprints, "source", (20, 20), transform)
    requested = (
        DistanceBin(-5, -5, -math.sqrt(15)),
        DistanceBin(-3, -math.sqrt(15), -math.sqrt(3)),
        DistanceBin(1, 0.3, math.sqrt(3)),
    )

    result = sample_source_bins(
        grid,
        footprints,
        "source",
        requested,
        seed=11,
        crop_size=1,
        tip_row=0,
        tip_col=0,
    )

    assert result.usable
    assert set(result.available_bins) == {1}
    assert result.unavailable_reasons == {-5: "size", -3: "size"}
    assert result.innermost_signed_distance_m < 0
    assert result.outermost_signed_distance_m > 0


def test_crop_reason_is_recorded_when_owned_slots_cannot_center_a_crop():
    transform = from_origin(0, 20, 1, 1)
    footprints = _footprints([("source", "a", box(9, 9, 11, 11))])
    grid = build_scene_grid(footprints, "source", (20, 20), transform)
    far = (DistanceBin(10, math.sqrt(50), math.sqrt(200)),)

    result = sample_source_bins(
        grid,
        footprints,
        "source",
        far,
        seed=11,
        crop_size=18,
        tip_row=9,
        tip_col=9,
    )

    assert result.unavailable_reasons == {10: "crop"}


def test_ownership_reason_is_recorded_when_another_building_owns_the_band():
    grid = SceneGrid(
        signed_distance_m=np.array([[1.0, 3.0], [3.0, 3.0]], dtype=np.float32),
        owner_codes=np.array([[1, 2], [2, 2]], dtype=np.int32),
        transform=from_origin(0, 2, 1, 1),
        source_code=1,
        id_by_code=(None, "source", "neighbor"),
    )
    footprints = _footprints(
        [
            ("source", "a", box(0, 1, 1, 2)),
            ("neighbor", "b", box(10, 10, 12, 12)),
        ]
    )

    result = sample_source_bins(
        grid,
        footprints,
        "source",
        (DistanceBin(3, math.sqrt(3), math.sqrt(15)),),
        seed=11,
        crop_size=1,
        tip_row=0,
        tip_col=0,
    )

    assert result.unavailable_reasons == {3: "ownership"}


def test_inside_ownership_is_not_misreported_as_size():
    grid = SceneGrid(
        signed_distance_m=np.array([[-2.0]], dtype=np.float32),
        owner_codes=np.array([[2]], dtype=np.int32),
        transform=from_origin(0, 1, 1, 1),
        source_code=1,
        id_by_code=(None, "source", "neighbor"),
        source_footprint_mask=np.array([[True]]),
    )
    footprints = _footprints(
        [
            ("source", "a", box(0, 0, 1, 1)),
            ("neighbor", "b", box(0, 0, 1, 1)),
        ]
    )

    result = sample_source_bins(
        grid,
        footprints,
        "source",
        (DistanceBin(-3, -math.sqrt(15), -math.sqrt(3)),),
        seed=11,
        crop_size=1,
        tip_row=0,
        tip_col=0,
    )

    assert result.unavailable_reasons == {-3: "ownership"}


def test_exact_nearest_polygon_may_differ_from_raster_owner():
    grid = SceneGrid(
        signed_distance_m=np.array([[2.0]], dtype=np.float32),
        owner_codes=np.array([[1]], dtype=np.int32),
        transform=from_origin(0, 2, 1, 1),
        source_code=1,
        id_by_code=(None, "source", "neighbor"),
    )
    footprints = _footprints(
        [
            ("source", "a", box(10, 10, 11, 11)),
            ("neighbor", "b", box(2.5, 1, 3, 2)),
        ]
    )

    result = sample_source_bins(
        grid,
        footprints,
        "source",
        (DistanceBin(3, math.sqrt(3), math.sqrt(15)),),
        seed=11,
        crop_size=1,
        tip_row=0,
        tip_col=0,
    )

    pin = result.pins_by_bin[3][0]
    assert pin.nearest_polygon_id == "neighbor"
    assert pin.exact_signed_distance_m == pytest.approx(2.0)


def test_raster_candidate_without_valid_exact_label_is_other():
    transform = from_origin(100, 104, 1, 1)
    grid = SceneGrid(
        signed_distance_m=np.full((4, 4), 1.0),
        owner_codes=np.ones((4, 4), dtype=np.int32),
        transform=transform,
        source_code=1,
        id_by_code=(None, "source"),
    )
    footprints = _footprints([("source", "a", box(0, 0, 2, 2))])

    result = sample_source_bins(
        grid,
        footprints,
        "source",
        (DistanceBin(1, 0.3, math.sqrt(3)),),
        seed=11,
        crop_size=1,
        tip_row=0,
        tip_col=0,
    )

    assert not result.usable
    assert result.unavailable_reasons == {1: "other"}


def test_negative_raster_candidate_without_valid_exact_label_is_other():
    transform = from_origin(100, 104, 1, 1)
    grid = SceneGrid(
        signed_distance_m=np.full((4, 4), -1.0),
        owner_codes=np.ones((4, 4), dtype=np.int32),
        transform=transform,
        source_code=1,
        id_by_code=(None, "source"),
    )
    footprints = _footprints([("source", "a", box(0, 0, 2, 2))])

    result = sample_source_bins(
        grid,
        footprints,
        "source",
        (DistanceBin(-1, -math.sqrt(3), -0.3, include_upper=True),),
        seed=11,
        crop_size=1,
        tip_row=0,
        tip_col=0,
    )

    assert result.unavailable_reasons == {-1: "other"}


def test_exact_distance_must_remain_in_bin_and_outside_gap():
    by_id = {item.bin_id: item for item in distance_bins()}

    assert accept_exact(-0.3, by_id[-1])
    assert not accept_exact(-0.299999, by_id[-1])
    assert not accept_exact(1.8, by_id[1])
    assert accept_exact(80, by_id[80])


def test_crop_transform_must_be_integer_slice_of_scene():
    transform = from_origin(0, 100, 0.6, 0.6)
    window = Window(10, 20, 512, 512)
    expected = window_transform(window, transform)

    assert_crop_transform(transform, expected, window)
    shifted = expected * Affine.translation(0.1, 0)
    with pytest.raises(GridInvariantError):
        assert_crop_transform(transform, shifted, window)
