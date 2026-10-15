import geopandas as gpd
import pytest
from shapely.geometry import box

from thesis.data.eligibility import (
    CropReach,
    add_scene_geometries,
    add_spatial_covariates,
    assign_building_strata,
    content_pool,
    context_for_scene,
    filter_scene_rural,
    load_box_geometries,
    mechanically_eligible,
    polygon_adjacency,
    scene_bounds_for_source,
    seeded_screen_subsample,
)


def test_content_pool_then_rural_pool_assigns_strata_before_mechanical_eligibility():
    footprints = gpd.GeoDataFrame(
        {
            "building_id": ["small", "medium", "large", "filtered"],
            "box_name": ["box-a"] * 4,
            "geometry": [
                box(0, 0, 10, 10),
                box(20, 0, 34, 12),
                box(40, 0, 60, 15),
                box(70, 0, 78, 8),
            ],
        },
        crs="EPSG:5070",
    )

    retained, _ = content_pool(
        footprints,
        min_area_m2=100,
        max_aspect_ratio=6,
    )
    retained, cuts = assign_building_strata(retained)

    assert retained["building_id"].tolist() == ["small", "medium", "large"]
    assert retained["building_stratum"].tolist() == ["small", "medium", "large"]
    assert retained["footprint_area_m2"].tolist() == pytest.approx([100, 168, 300])
    assert cuts[0] < cuts[1]


def test_scene_rurality_checks_the_entire_directional_scene():
    sources = gpd.GeoDataFrame(
        {"building_id": ["rural", "edge"], "geometry": [box(0, 0, 10, 10), box(100, 0, 110, 10)]},
        crs="EPSG:5070",
    )
    sources = add_scene_geometries(
        sources,
        max_distance_m=5,
        crop_reach=CropReach(north_m=5, west_m=5, south_m=5, east_m=5),
        slack_m=0,
    )
    urban = gpd.GeoDataFrame({"geometry": [box(115, 0, 120, 10)]}, crs=sources.crs)

    rural, mask = filter_scene_rural(sources, urban)

    assert mask.tolist() == [True, False]
    assert rural["building_id"].tolist() == ["rural"]


def test_enlarged_scene_uses_directional_crop_reaches_and_slack():
    source = box(0, 0, 10, 10)

    bounds = scene_bounds_for_source(
        source,
        max_distance_m=80,
        crop_reach=CropReach(north_m=153.9, west_m=153.9, south_m=153.3, east_m=153.3),
        slack_m=1.2,
    )

    assert bounds.bounds == pytest.approx((-235.1, -234.5, 244.5, 245.1))


def test_mechanical_eligibility_requires_both_box_and_one_item():
    scene = box(-235.1, -234.5, 244.5, 245.1)

    assert mechanically_eligible(scene, box(-300, -300, 300, 300), box(-300, -300, 300, 300))
    assert not mechanically_eligible(scene, box(-200, -300, 300, 300), box(-300, -300, 300, 300))
    assert not mechanically_eligible(scene, box(-300, -300, 300, 300), box(-300, -200, 300, 300))


def test_context_query_keeps_intersecting_buildings_and_provenance():
    footprints = gpd.GeoDataFrame(
        {
            "building_id": ["inside", "crossing", "outside"],
            "provenance": ["osm", "msft", "osm"],
            "geometry": [box(1, 1, 2, 2), box(9, 1, 11, 2), box(20, 20, 21, 21)],
        },
        crs="EPSG:5070",
    )

    rows = context_for_scene(footprints, box(0, 0, 10, 10))

    assert rows["building_id"].tolist() == ["inside", "crossing"]
    assert rows.loc[rows["building_id"] == "crossing", "provenance"].item() == "msft"


def test_box_geometries_map_grid_coordinates_to_configured_ids(tmp_path):
    path = tmp_path / "boxes.parquet"
    gpd.GeoDataFrame(
        {"grid_x": [1], "grid_y": [2], "geometry": [box(0, 0, 25_000, 25_000)]},
        crs="EPSG:5070",
    ).to_parquet(path)

    result = load_box_geometries(path, [{"name": "eco_box_01", "grid_x": 1, "grid_y": 2}])

    assert result["box_id"].tolist() == ["eco_box_01"]
    assert result.crs.to_epsg() == 5070


def test_spatial_covariates_use_footprint_distance_and_80m_neighbors():
    footprints = gpd.GeoDataFrame(
        {
            "building_id": ["a", "b", "c"],
            "box_id": ["box-a"] * 3,
            "geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10), box(200, 0, 210, 10)],
        },
        crs="EPSG:5070",
    )

    result = add_spatial_covariates(footprints)

    assert result.set_index("building_id").loc["a", "nearest_neighbor_distance_m"] == pytest.approx(10)
    assert result.set_index("building_id").loc["a", "neighbors_within_80m"] == 1
    assert result.set_index("building_id").loc["c", "neighbors_within_80m"] == 0


def test_screen_cap_is_seeded_per_box_and_weights_screened_boxes_equally():
    frame = gpd.GeoDataFrame(
        {
            "building_id": [f"a-{index}" for index in range(5)] + [f"b-{index}" for index in range(2)],
            "box_id": ["a"] * 5 + ["b"] * 2,
            "geometry": [box(index, 0, index + 0.5, 0.5) for index in range(7)],
        },
        crs="EPSG:5070",
    )

    sampled = seeded_screen_subsample(frame, per_box_cap=3, seed=7)
    repeated = seeded_screen_subsample(frame.iloc[::-1], per_box_cap=3, seed=7)

    assert sampled.groupby("box_id").size().to_dict() == {"a": 3, "b": 2}
    assert sampled.groupby("box_id")["audit_weight"].sum().to_dict() == pytest.approx({"a": 1, "b": 1})
    assert sampled["building_id"].tolist() == repeated["building_id"].tolist()


def test_ecoregion_adjacency_requires_a_shared_polygon_boundary():
    regions = gpd.GeoDataFrame(
        {
            "code": ["a", "b", "c", "d"],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 0, 4, 1), box(1, 1, 2, 2)],
        },
        crs="EPSG:5070",
    )

    assert polygon_adjacency(regions, code_column="code", allowed_codes={"a", "b", "c", "d"}) == {
        "a": ("b",),
        "b": ("a", "d"),
        "c": (),
        "d": ("b",),
    }
