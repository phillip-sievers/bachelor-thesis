from datetime import UTC, date, datetime

import duckdb
import pytest
from pydantic import ValidationError

from thesis.data.manifest import (
    BoxRecord,
    DeviationRecord,
    ManifestBundle,
    PinRecord,
    RenderRecord,
    RunRecord,
    SourceRecord,
    write_manifest,
)

BIN_IDS = (-5, -3, -1, 1, 3, 5, 10, 20, 40, 80)


def _run(**overrides):
    row = {
        "schema_version": 2,
        "run_id": "run-v2",
        "created_at": datetime(2026, 9, 10, 10, 0, tzinfo=UTC),
        "config_hash": "a" * 64,
        "git_commit": "b" * 40,
        "code_version": "c" * 64,
        "overture_release": "2026-08-19.0",
        "area_log_cuts": (5.0452346355471915, 5.518237342793738),
        "coverage_threshold": 0.5,
        "screen_per_box_cap": 200,
        "spill_revisit_fraction": 0.05,
        "delta_m": None,
        "census_urban_vintage": "2020 UA / TIGER 2025",
        "ecoregion_artifact": "na_cec_eco_l1.zip",
        "seeds": {
            "box_quota": 1,
            "screen_subsample": 2,
            "candidate_order": 3,
            "pin_sampling": 4,
            "ablation_assignment": 5,
            "deficit_spill": 6,
            "delta_reduction": 7,
            "sanity_render": 8,
        },
        "ecoregion_merge_mapping": {str(value): str(value) for value in (5, 6, 7, 8, 9, 10, 11, 12, 13, 15)},
    }
    row.update(overrides)
    return row


def _box(**overrides):
    row = {
        "run_id": "run-v2",
        "box_id": "eco-9-box-1",
        "ecoregion_code": "9",
        "ecoregion_name": "GREAT PLAINS",
        "state": "KS",
        "center_rural": True,
        "full_pool_count": 1_000,
        "content_pool_count": 800,
        "rural_content_pool_count": 750,
        "eligible_count": 700,
        "screened_count": 700,
        "scene_rural_loss": 50,
        "box_containment_loss": 25,
        "tile_containment_loss": 25,
        "spacing_loss": 1,
        "main_quota": 17,
        "realized_main": 17,
        "spill_in": 0,
        "spill_out": 0,
        "zero_eligible_reason": None,
        "box_selection_probability": 0.01,
        "imagery_dates": (date(2024, 7, 1),),
        "imagery_seasons": ("summer",),
        "footprint_provenance_mix": {"msft": 0.7, "osm": 0.3},
    }
    row.update(overrides)
    return row


def _source(**overrides):
    availability = {str(bin_id): True for bin_id in BIN_IDS}
    row = {
        "run_id": "run-v2",
        "source_id": "source-1",
        "box_id": "eco-9-box-1",
        "quota_box_id": "eco-9-box-1",
        "ecoregion_code": "9",
        "building_stratum": "medium",
        "assignment": "main",
        "state": "KS",
        "source_footprint_provenance": "msft",
        "footprint_area_m2": 180.0,
        "source_anchor_lat": 38.5,
        "source_anchor_lon": -98.0,
        "nearest_neighbor_distance_m": 50.0,
        "neighbors_within_80m": 2,
        "scene_id": "scene-1",
        "naip_item_id": "naip-item",
        "naip_source_crs": "EPSG:26914",
        "naip_item_transform": (0.6, 0.0, 50.0, 0.0, -0.6, 250.0),
        "imagery_date": date(2024, 7, 1),
        "imagery_season": "summer",
        "bin_availability": availability,
        "unavailability_reasons": {},
        "innermost_signed_distance_m": -6.0,
        "outermost_signed_distance_m": 120.0,
        "complete_source": True,
        "complete_inside": True,
        "scene_rural": True,
        "rejection_provenance": (),
        "scene_window": (10, 20, 800, 800),
        "scene_transform": (0.6, 0.0, 100.0, 0.0, -0.6, 200.0),
    }
    row.update(overrides)
    return row


def _pin(**overrides):
    row = {
        "run_id": "run-v2",
        "pin_id": "pin-1",
        "source_id": "source-1",
        "bin_id": 1,
        "bin_lower_m": 0.3,
        "bin_upper_m": 3**0.5,
        "repeat_index": 0,
        "target_signed_distance_m": 1.1,
        "exact_signed_distance_m": 1.05,
        "ground_truth_inside": False,
        "tip_row": 256,
        "tip_col": 256,
        "tip_x": 123.3,
        "tip_y": 456.7,
        "tip_lat": 38.5,
        "tip_lon": -98.0,
        "owning_polygon_id": "source-1",
        "nearest_polygon_id": "source-1",
        "nearest_polygon_provenance": "msft",
        "nearest_edge_bearing_radians": 1.2,
        "radial_displacement_m": 4.0,
        "source_to_pin_bearing_radians": 2.0,
        "crop_window": (100, 200, 512, 512),
        "crop_transform": (0.6, 0.0, 100.0, 0.0, -0.6, 200.0),
    }
    row.update(overrides)
    return row


def _render(**overrides):
    row = {
        "run_id": "run-v2",
        "render_id": "render-1",
        "pin_id": "pin-1",
        "preset_id": "google-red",
        "marker_shape": "google_pin",
        "marker_color": "red",
        "marker_size_px": 15,
        "marker_opacity": 1.0,
        "prompt_variant": "main",
        "image_path": "images/render-1.png",
        "image_sha256": "d" * 64,
    }
    row.update(overrides)
    return row


def _bundle():
    return ManifestBundle(
        run=RunRecord.model_validate(_run()),
        boxes=(BoxRecord.model_validate(_box()),),
        sources=(SourceRecord.model_validate(_source()),),
        pins=(PinRecord.model_validate(_pin()),),
        renders=(RenderRecord.model_validate(_render()),),
        deviations=(),
    )


def test_v2_manifest_writes_all_normalized_tables(tmp_path):
    path = tmp_path / "manifest.duckdb"

    write_manifest(path, _bundle())

    with duckdb.connect(str(path), read_only=True) as connection:
        assert {name for (name,) in connection.execute("SHOW TABLES").fetchall()} == {
            "runs",
            "boxes",
            "sources",
            "pins",
            "renders",
            "deviations",
            "samples",
        }
        source = connection.execute("SELECT assignment, complete_source, bin_availability FROM sources").fetchone()
    assert source[0:2] == ("main", True)
    assert source[2]["-5"] is True
    with duckdb.connect(str(path), read_only=True) as connection:
        compatible = connection.execute(
            """
            SELECT sample_id, building_id, target_signed_edge_distance_m,
                   actual_signed_edge_distance_m, naip_source_crs
            FROM samples
            """
        ).fetchone()
    assert compatible == ("render-1", "source-1", 1.1, 1.05, "EPSG:26914")


def test_partial_source_requires_one_reason_per_unavailable_bin():
    availability = {str(bin_id): True for bin_id in BIN_IDS}
    availability["-5"] = False

    with pytest.raises(ValidationError, match="unavailability"):
        SourceRecord.model_validate(
            _source(
                bin_availability=availability,
                unavailability_reasons={},
                complete_source=False,
                complete_inside=False,
            )
        )

    source = SourceRecord.model_validate(
        _source(
            bin_availability=availability,
            unavailability_reasons={"-5": "size"},
            complete_source=False,
            complete_inside=False,
        )
    )
    assert source.unavailability_reasons == {"-5": "size"}


def test_complete_source_flag_must_match_availability_vector():
    availability = {str(bin_id): True for bin_id in BIN_IDS}
    availability["80"] = False

    with pytest.raises(ValidationError, match="complete_source"):
        SourceRecord.model_validate(
            _source(
                bin_availability=availability,
                unavailability_reasons={"80": "ownership"},
                complete_source=True,
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"tip_row": 255},
        {"tip_col": 255},
        {"exact_signed_distance_m": 0.2},
        {"exact_signed_distance_m": 1.8},
        {"exact_signed_distance_m": -1.0, "ground_truth_inside": False},
        {"nearest_edge_bearing_radians": 6.283185307179586},
    ],
)
def test_pin_enforces_tip_gap_bin_sign_and_bearing(overrides):
    with pytest.raises(ValidationError):
        PinRecord.model_validate(_pin(**overrides))


def test_run_requires_locked_gate_and_all_named_seeds():
    with pytest.raises(ValidationError):
        RunRecord.model_validate(_run(coverage_threshold=0.6))
    with pytest.raises(ValidationError):
        RunRecord.model_validate(_run(seeds={"candidate_order": 2}))


def test_spill_deviation_requires_origin_and_destination_boxes():
    with pytest.raises(ValidationError):
        DeviationRecord.model_validate(
            {
                "run_id": "run-v2",
                "deviation_id": "spill-1",
                "deviation_type": "spill",
                "source_id": "source-1",
                "origin_box_id": "eco-9-box-1",
                "destination_box_id": None,
                "details": {},
            }
        )


def test_writer_rejects_broken_foreign_keys(tmp_path):
    bundle = _bundle()
    broken = ManifestBundle(
        run=bundle.run,
        boxes=bundle.boxes,
        sources=bundle.sources,
        pins=(PinRecord.model_validate(_pin(source_id="missing")),),
        renders=bundle.renders,
        deviations=bundle.deviations,
    )

    with pytest.raises(ValueError, match="unknown source"):
        write_manifest(tmp_path / "manifest.duckdb", broken)
