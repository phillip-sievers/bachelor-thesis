import argparse
import json
import runpy
import sys

import duckdb
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

import thesis.run as run
from thesis.data import naip
from thesis.data.sampling import RetentionStage
from thesis.geometry.displacement import DisplacedPin


def _configs(tmp_path):
    return {
        "bins": {"fixed_levels_m": {"inside": [], "outside": [1]}, "target_tolerance_m": 0.1},
        "imagery": {"cache_dir": tmp_path / "cache", "gsd_m": 1.0, "image_px": 10},
        "overture": {"out": "footprints.parquet", "release": "release"},
        "sampling": {"require_target_fully_inside": True, "reject_pin_on_other_footprint": True},
    }


def _candidate(building_id, geometry, anchor, **overrides):
    row = {
        "building_id": building_id,
        "geometry": geometry,
        "anchor_lon": anchor.x,
        "anchor_lat": anchor.y,
        "box_name": "box",
        "state": "KS",
        "ecoregion_code": "9.2",
        "cell_selection_probability": 0.5,
        "eligible_buildings_in_bbox": np.int64(10),
        "sampled_buildings_in_bbox": np.int64(2),
        "sampling_weight": 5.0,
        "footprint_m2_area": 100.0,
    }
    row.update(overrides)
    return row


def test_module_entrypoint_invokes_main(monkeypatch):
    class MainInvoked(Exception):
        pass

    def stop_at_argument_parsing(self):
        raise MainInvoked

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop_at_argument_parsing)
    monkeypatch.setattr(run.overture, "extract", lambda config: pytest.fail("debug candidate build was invoked"))

    with pytest.raises(MainInvoked):
        runpy.run_module("thesis.run", run_name="__main__")


def test_verify_against_accepts_a_dataset_directory(tmp_path, monkeypatch, capsys):
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    for directory in (previous, current):
        directory.mkdir()
        with duckdb.connect(str(directory / "manifest.duckdb")) as connection:
            connection.execute("CREATE TABLE samples (sample_id VARCHAR PRIMARY KEY)")
            connection.execute("INSERT INTO samples VALUES ('same-sample')")

    monkeypatch.setattr(sys, "argv", ["thesis", "--out", str(current), "--verify-against", str(previous)])
    monkeypatch.setattr(run, "build_core", lambda out_dir: [])
    monkeypatch.setattr(run, "split_ablation_holdout", lambda tiles, holdout_size, seed: ([], []))
    monkeypatch.setattr(run, "render_and_manifest", lambda tiles, out_dir: [])

    run.main()

    assert capsys.readouterr().out == "manifests identical\n"


def test_displacement_rejects_targets_outside_image_and_pins_on_neighbors(tmp_path, monkeypatch):
    configs = _configs(tmp_path)
    candidates = gpd.GeoDataFrame(
        [
            _candidate("large", box(-6, -1, 6, 1), Point(0, 0)),
            _candidate("target", box(19, -1, 21, 1), Point(20, 0)),
        ],
        crs="EPSG:5070",
    )
    all_footprints = gpd.GeoDataFrame(
        {
            "building_id": ["large", "target", "neighbor"],
            "geometry": [box(-6, -1, 6, 1), box(19, -1, 21, 1), box(23, -1, 25, 1)],
        },
        crs="EPSG:5070",
    )

    monkeypatch.setattr(run, "load_config", lambda key: configs[key])
    monkeypatch.setattr(run.gpd, "read_parquet", lambda *args, **kwargs: all_footprints)
    monkeypatch.setattr(run, "utm_epsg", lambda lon, lat: 5070)
    monkeypatch.setattr(run, "to_utm", lambda geometry, epsg: geometry)
    monkeypatch.setattr(run, "to_wgs84", lambda geometry, epsg: geometry)
    monkeypatch.setattr(
        run.displacement,
        "generate_displacements",
        lambda footprint, anchor, *args, **kwargs: (
            DisplacedPin(1.0, 1.0, 0.0, Point(0, 0) if anchor.x == 0 else Point(24, 0)),
        ),
    )

    displaced, failures = run.displace_candidates(candidates)

    assert displaced == []
    assert [(failure["candidate_id"], failure["reason"]) for failure in failures] == [
        ("large", "target_or_anchor_outside_image"),
        ("target", "pin_on_other_footprint"),
    ]


def test_displacement_normalizes_numpy_counts_for_strict_manifest(tmp_path, monkeypatch):
    configs = _configs(tmp_path)
    configs["sampling"] = {"require_target_fully_inside": False, "reject_pin_on_other_footprint": False}
    target = box(-1, -1, 1, 1)
    candidates = gpd.GeoDataFrame([_candidate("target", target, Point(0, 0))], crs="EPSG:5070")

    monkeypatch.setattr(run, "load_config", lambda key: configs[key])
    monkeypatch.setattr(run.gpd, "read_parquet", lambda *args, **kwargs: candidates[["building_id", "geometry"]])
    monkeypatch.setattr(run, "utm_epsg", lambda lon, lat: 5070)
    monkeypatch.setattr(run, "to_utm", lambda geometry, epsg: geometry)
    monkeypatch.setattr(run, "to_wgs84", lambda geometry, epsg: geometry)
    monkeypatch.setattr(
        run.displacement,
        "generate_displacements",
        lambda *args, **kwargs: (DisplacedPin(1.0, 1.0, 0.0, Point(2, 0)),),
    )

    displaced, _ = run.displace_candidates(candidates)

    assert type(displaced[0]["eligible_buildings_in_bbox"]) is int
    assert type(displaced[0]["sampled_buildings_in_bbox"]) is int


def test_build_core_persists_failures_and_resumes_completed_imagery(tmp_path, monkeypatch):
    out_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    pin = DisplacedPin(1.0, 1.0, 0.0, Point(1, 0))
    entries = [
        {"building_id": "ok", "pin_lat": 1.0, "pin_lon": 1.0, "pin": pin},
        {"building_id": "no-imagery", "pin_lat": 2.0, "pin_lon": 2.0, "pin": pin},
    ]
    stage = RetentionStage("input", 2, 1.0, 1.0)
    displacement_failure = {"candidate_id": "bad", "stage": "displacement", "reason": "unreachable"}
    calls = {"fetch": 0}

    monkeypatch.setattr(run, "compute_config_hash", lambda: "a" * 64)
    monkeypatch.setattr(run, "build_candidates", lambda: (object(), [stage]))
    monkeypatch.setattr(run, "displace_candidates", lambda candidates: (entries, [displacement_failure]))
    monkeypatch.setattr(
        run,
        "load_config",
        lambda key: {"cache_dir": cache_dir, "gsd_m": 1.0, "image_px": 2} if key == "imagery" else {},
    )

    def fetch_tile(*, lat, lon, gsd, size, config):
        calls["fetch"] += 1
        if lon == 2.0:
            raise naip.NoValidItem("none")
        metadata = {"item_id": "item", "col_off": 3, "row_off": 4, "size": size}
        tile = naip.Tile(np.zeros((3, size, size), dtype=np.uint8), metadata)
        naip.store_atomic(cache_dir, naip.cache_key("item", 3, 4, size), tile.array, tile.metadata)
        return tile

    monkeypatch.setattr(run.naip, "fetch_tile", fetch_tile)

    first = run.build_core(out_dir)

    assert len(first) == 1
    checkpoint = json.loads((out_dir / "checkpoint.json").read_text())
    failures = json.loads((out_dir / "failures.json").read_text())
    assert checkpoint["stage"] == "imagery_complete"
    assert checkpoint["sampling_stages"] == [
        {"name": "input", "count": 2, "fraction_of_input": 1.0, "fraction_of_previous": 1.0}
    ]
    assert [failure["candidate_id"] for failure in failures] == ["bad", "no-imagery"]

    monkeypatch.setattr(run, "build_candidates", lambda: pytest.fail("candidate stage was recomputed"))
    monkeypatch.setattr(run, "displace_candidates", lambda candidates: pytest.fail("displacement was recomputed"))
    monkeypatch.setattr(run.naip, "fetch_tile", lambda **kwargs: pytest.fail("completed pin was fetched again"))

    second = run.build_core(out_dir)

    assert len(second) == 1
    assert calls["fetch"] == 2
