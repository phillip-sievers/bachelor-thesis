import warnings
from shutil import make_archive

import geopandas as gpd
import pytest
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon, box

from thesis.data.sampling import (
    filter_footprints,
    image_extent,
    image_is_rural,
    load_urban_areas,
    pin_hits_other_footprint,
    target_fits_image,
)


def test_load_urban_areas_reads_zip_and_projects_to_metric_crs(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    gpd.GeoDataFrame(
        {"UACE20": ["12345"], "geometry": [box(-103.5, 44.7, -103.4, 44.8)]},
        crs="EPSG:4269",
    ).to_file(source_dir / "urban.shp")
    archive = make_archive(str(tmp_path / "urban"), "zip", root_dir=source_dir)

    urban = load_urban_areas(archive)

    assert len(urban) == 1
    assert urban.columns.tolist() == ["geometry"]
    assert urban.crs == "EPSG:5070"


def test_footprint_filters_do_not_require_isolation():
    footprints = gpd.GeoDataFrame(
        {
            "building_id": ["target", "neighbor", "small", "thin", "invalid", "line"],
            "geometry": [
                box(0, 0, 12, 12),
                box(15, 0, 27, 12),
                box(100, 0, 108, 8),
                box(200, 0, 240, 3),
                Polygon([(300, 0), (310, 10), (300, 10), (310, 0), (300, 0)]),
                LineString([(400, 0), (420, 0)]),
            ],
        },
        crs="EPSG:5070",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        retained, stages = filter_footprints(footprints, min_area_m2=100, max_aspect_ratio=6)

    assert retained["building_id"].tolist() == ["target", "neighbor"]
    assert [(stage.name, stage.count) for stage in stages] == [
        ("input", 6),
        ("valid_polygon", 4),
        ("min_area_m2", 3),
        ("max_aspect_ratio", 2),
    ]
    assert stages[-1].fraction_of_input == pytest.approx(2 / 6)


def test_image_extent_is_centered_on_moved_pin():
    extent = image_extent(Point(40, 80), width_m=20)

    assert extent.bounds == pytest.approx((30, 70, 50, 90))


def test_image_intersecting_urban_area_is_not_rural():
    urban_index = STRtree([box(20, 20, 40, 40)])

    assert not image_is_rural(box(0, 0, 30, 30), urban_index, buffer_m=0)


def test_census_buffer_rejects_image_near_urban_boundary():
    urban_index = STRtree([box(20, 0, 30, 10)])
    extent = box(0, 0, 10, 10)

    assert image_is_rural(extent, urban_index, buffer_m=9.9)
    assert not image_is_rural(extent, urban_index, buffer_m=10)


def test_target_and_anchor_must_fit_inside_image():
    target = box(-4, -4, 4, 4)
    anchor = Point(0, 0)

    assert target_fits_image(target, anchor, box(-5, -5, 5, 5))
    assert not target_fits_image(target, anchor, box(-3, -5, 5, 5))


def test_pin_on_non_target_footprint_is_ambiguous():
    footprints = STRtree([box(0, 0, 10, 10), box(20, 0, 30, 10)])

    assert not pin_hits_other_footprint(Point(5, 5), target_index=0, footprints=footprints)
    assert pin_hits_other_footprint(Point(25, 5), target_index=0, footprints=footprints)
    assert not pin_hits_other_footprint(Point(15, 5), target_index=0, footprints=footprints)
