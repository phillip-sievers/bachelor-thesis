import json

import geopandas as gpd
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from thesis.geometry.scene_sampling import PinCandidate, SourceBinResult
from thesis.run import (
    CheckpointMismatch,
    StageNotReady,
    assign_naip_eligibility,
    build_source_and_pin_rows,
    require_passing_gate,
    run_boxwise,
    unavailability_report,
    zero_eligible_reason,
)


def test_naip_eligibility_checks_overlapping_quarter_quads_and_native_window():
    sources = gpd.GeoDataFrame(
        {
            "source_id": ["source"],
            "scene_geometry_wkb": [box(-10, -10, 10, 10).wkb],
            "geometry": [box(-1, -1, 1, 1)],
        },
        crs="EPSG:5070",
    )
    items = [
        {"id": "edge", "year": 2024, "gsd": 0.6, "geometry": box(-180, -90, -100, 90)},
        {"id": "overlap", "year": 2024, "gsd": 0.6, "geometry": box(-180, -90, 180, 90)},
    ]
    transform = from_origin(-100, 100, 0.6, 0.6)

    result = assign_naip_eligibility(
        sources,
        items,
        gsd_m=0.6,
        gsd_tolerance=0.01,
        grid_loader=lambda item: {
            "transform": tuple(transform[:6]),
            "src_crs": "EPSG:5070",
            "width": 400,
            "height": 400,
            "acquired_date": "2024-07-01",
            "season": "summer",
        },
    )

    assert result.rejections.empty
    assert result.eligible["naip_item_id"].tolist() == ["overlap"]
    assert result.eligible.iloc[0]["scene_window"] == (150, 150, 34, 34)


def test_boxwise_screen_checkpoints_and_resumes(tmp_path):
    calls = []

    def first_pass(box_id, box_output):
        calls.append(box_id)
        if box_id == "b":
            raise RuntimeError("synthetic interruption")
        box_output.write_text(box_id)

    with pytest.raises(RuntimeError, match="synthetic"):
        run_boxwise(("a", "b", "c"), first_pass, tmp_path, config_hash="a" * 64, code_version="b" * 64)

    checkpoint = json.loads((tmp_path / "screen-checkpoint.json").read_text())
    assert checkpoint["completed_boxes"] == ["a"]

    def second_pass(box_id, box_output):
        calls.append(box_id)
        box_output.write_text(box_id)

    run_boxwise(("a", "b", "c"), second_pass, tmp_path, config_hash="a" * 64, code_version="b" * 64)

    assert calls == ["a", "b", "b", "c"]
    assert json.loads((tmp_path / "screen-checkpoint.json").read_text())["completed_boxes"] == ["a", "b", "c"]


def test_boxwise_resume_rejects_config_or_code_drift(tmp_path):
    run_boxwise(("a",), lambda box_id, path: path.write_text(box_id), tmp_path, "a" * 64, "b" * 40)

    with pytest.raises(CheckpointMismatch):
        run_boxwise(("a",), lambda *_: None, tmp_path, "c" * 64, "b" * 40)


def _pin(bin_id, repeat_index, distance):
    return PinCandidate(
        source_id="source",
        bin_id=bin_id,
        repeat_index=repeat_index,
        target_signed_distance_m=distance,
        row=300,
        col=301,
        x=100.5,
        y=200.5,
        exact_signed_distance_m=distance,
        nearest_polygon_id="source",
        nearest_polygon_provenance="msft",
        bearing_radians=1.0,
    )


def test_partial_scan_becomes_one_source_record_with_availability_and_pin_rows():
    scan = SourceBinResult(
        source_id="source",
        pins_by_bin={1: (_pin(1, 0, 0.8), _pin(1, 1, 1.0), _pin(1, 2, 1.2))},
        unavailable_reasons={-5: "size"},
        innermost_signed_distance_m=-1.5,
        outermost_signed_distance_m=80.0,
    )

    source, pins = build_source_and_pin_rows(
        scan,
        all_bin_ids=(-5, 1),
        scene_window=(10, 20, 800, 800),
        scene_item_id="item",
    )

    assert source["status"] == "eligible"
    assert source["available_bin_count"] == 1
    assert source["bin_availability"] == {"-5": False, "1": True}
    assert source["unavailability_reasons"] == {"-5": "size"}
    assert source["complete_source"] is False
    assert source["scene_id"] == "4a60aca0462556af"
    assert len(pins) == 3
    assert all(row["scene_item_id"] == "item" for row in pins)


def test_zero_bin_scan_remains_eligible_for_replacement():
    scan = SourceBinResult(
        source_id="zero",
        pins_by_bin={},
        unavailable_reasons={-5: "size", 1: "ownership"},
        innermost_signed_distance_m=-0.5,
        outermost_signed_distance_m=0.5,
    )

    source, pins = build_source_and_pin_rows(scan, (-5, 1), (0, 0, 800, 800), "item")

    assert source["status"] == "eligible_zero_bins"
    assert pins == []


def test_unavailability_report_crosses_bin_reason_stratum_and_ecoregion():
    sources = pd.DataFrame(
        {
            "source_id": ["a", "b"],
            "building_stratum": ["small", "large"],
            "ecoregion_code": ["9", "9"],
            "unavailability_reasons": [{"-5": "size"}, {"80": "ownership"}],
        }
    )

    report = unavailability_report(sources)

    assert report.to_dict("records") == [
        {"bin_id": -5, "reason": "size", "building_stratum": "small", "ecoregion_code": "9", "count": 1},
        {"bin_id": 80, "reason": "ownership", "building_stratum": "large", "ecoregion_code": "9", "count": 1},
    ]


@pytest.mark.parametrize(
    ("inventory", "screen", "expected"),
    [
        (
            {"full_pool_count": 0, "content_pool_count": 0, "rural_content_pool_count": 0, "box_contained_count": 0},
            {"eligible_count": 0, "tile_rejections": {}},
            "no_overture_buildings",
        ),
        (
            {"full_pool_count": 4, "content_pool_count": 0, "rural_content_pool_count": 0, "box_contained_count": 0},
            {"eligible_count": 0, "tile_rejections": {}},
            "content_filters",
        ),
        (
            {"full_pool_count": 4, "content_pool_count": 3, "rural_content_pool_count": 0, "box_contained_count": 0},
            {"eligible_count": 0, "tile_rejections": {}},
            "scene_rurality",
        ),
        (
            {"full_pool_count": 4, "content_pool_count": 3, "rural_content_pool_count": 2, "box_contained_count": 0},
            {"eligible_count": 0, "tile_rejections": {}},
            "box_containment",
        ),
        (
            {"full_pool_count": 4, "content_pool_count": 3, "rural_content_pool_count": 2, "box_contained_count": 2},
            {"eligible_count": 0, "tile_rejections": {"non_native_gsd": 2}},
            "tile_containment:non_native_gsd",
        ),
    ],
)
def test_zero_eligible_reason_records_the_exact_loss_stage(inventory, screen, expected):
    assert zero_eligible_reason(inventory, pd.Series(screen)) == expected


def test_render_stage_requires_a_passing_gate(tmp_path):
    with pytest.raises(StageNotReady, match="gate report"):
        require_passing_gate(tmp_path)
    (tmp_path / "gate.json").write_text(json.dumps({"passed": False, "failure": "coverage"}))
    with pytest.raises(StageNotReady, match="coverage"):
        require_passing_gate(tmp_path)
    (tmp_path / "gate.json").write_text(json.dumps({"passed": True}))
    require_passing_gate(tmp_path)
