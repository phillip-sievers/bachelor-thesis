from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
import rasterio
from affine import Affine
from rasterio.transform import xy
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely import from_wkb
from shapely.geometry import Point
from shapely.ops import transform as transform_geometry

from thesis.config import REPO_ROOT, load_config, load_run_config
from thesis.data import naip, overture
from thesis.data.eligibility import (
    CropReach,
    add_scene_geometries,
    add_spatial_covariates,
    assign_building_strata,
    content_pool,
    context_for_scene,
    filter_scene_rural,
    load_box_geometries,
    polygon_adjacency,
    seeded_screen_subsample,
)
from thesis.data.manifest import (
    BoxRecord,
    DeviationRecord,
    ManifestBundle,
    PinRecord,
    RenderRecord,
    RunRecord,
    SourceRecord,
    append_renders,
    write_manifest,
)
from thesis.data.sampling import (
    assign_ablation,
    fill_deficits_by_ecoregion,
    load_urban_areas,
    main_box_quotas,
    select_ablation_boxes,
    select_joint_sources,
    stable_seed_key,
    usable_source_pool,
)
from thesis.data.statistics import (
    GateFailure,
    availability_smd_report,
    coverage_report,
    enforce_gate,
    selection_smd_report,
)
from thesis.geometry.bins import distance_bins
from thesis.geometry.scene_sampling import SourceBinResult, assert_crop_transform, build_scene_grid, sample_source_bins
from thesis.render.presets import load_presets


class CheckpointMismatch(RuntimeError):
    pass


class StageNotReady(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NaipEligibility:
    eligible: gpd.GeoDataFrame
    rejections: pd.DataFrame


def _configuration_bundle() -> dict[str, object]:
    run = load_run_config()
    return {
        "run": run,
        **{key: load_config(key) for key in ("bins", "imagery", "overture", "render", "sampling")},
    }


def configuration_hash() -> str:
    encoded = json.dumps(_configuration_bundle(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def code_version() -> str:
    """Hash the local construction code/config, including uncommitted files."""
    digest = hashlib.sha256()
    roots = (REPO_ROOT / "src", REPO_ROOT / "configs")
    paths = [path for root in roots for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    paths.extend((REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _json_object(value):
    return json.loads(value) if isinstance(value, str) else value


def _affine(values) -> Affine:
    if isinstance(values, str):
        values = json.loads(values)
    return Affine(*[float(value) for value in values])


def _geometry_to_crs(geometry, source_crs, target_crs):
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transform_geometry(transformer.transform, geometry)


def assign_naip_eligibility(
    sources: gpd.GeoDataFrame,
    items: Iterable[dict],
    *,
    gsd_m: float,
    gsd_tolerance: float,
    grid_loader: Callable[[dict], dict],
) -> NaipEligibility:
    """Apply D04/D17 containment across every overlapping quarter-quad."""
    item_rows = tuple(items)
    grid_cache: dict[str, dict] = {}
    accepted: list[dict] = []
    rejected: list[dict[str, str]] = []
    for row in sources.to_dict("records"):
        source_id = str(row["source_id"])
        scene_5070 = from_wkb(row["scene_geometry_wkb"])
        scene_wgs84 = _geometry_to_crs(scene_5070, sources.crs, "EPSG:4326")
        try:
            candidates = naip.containing_native_items(item_rows, scene_wgs84, gsd_m, gsd_tolerance)
        except naip.NoValidItem as error:
            rejected.append(
                {
                    "source_id": source_id,
                    "stage": "tile_containment",
                    "reason": "non_native_gsd",
                    "detail": str(error),
                }
            )
            continue
        except naip.TileOutOfBounds as error:
            rejected.append(
                {
                    "source_id": source_id,
                    "stage": "tile_containment",
                    "reason": "tile_edge",
                    "detail": str(error),
                }
            )
            continue

        chosen = None
        alignment_errors = []
        for item in candidates:
            item_id = str(item["id"])
            try:
                if item_id not in grid_cache:
                    grid_cache[item_id] = grid_loader(item)
                metadata = grid_cache[item_id]
                transform = _affine(metadata["transform"])
                scene_native = _geometry_to_crs(scene_5070, sources.crs, metadata["src_crs"])
                window = naip.aligned_scene_window(
                    transform,
                    int(metadata["width"]),
                    int(metadata["height"]),
                    scene_native.bounds,
                )
            except (naip.ResolutionMismatch, naip.TileOutOfBounds) as error:
                alignment_errors.append(str(error))
                continue
            chosen = (item, metadata, window)
            break
        if chosen is None:
            rejected.append(
                {
                    "source_id": source_id,
                    "stage": "tile_containment",
                    "reason": "tile_edge",
                    "detail": " | ".join(alignment_errors),
                }
            )
            continue
        item, metadata, window = chosen
        row.update(
            {
                "naip_item_id": str(item["id"]),
                "imagery_date": metadata["acquired_date"],
                "imagery_season": metadata["season"],
                "naip_source_crs": metadata["src_crs"],
                "naip_item_transform": tuple(metadata["transform"]),
                "naip_item_width": int(metadata["width"]),
                "naip_item_height": int(metadata["height"]),
                "scene_window": (int(window.col_off), int(window.row_off), int(window.width), int(window.height)),
                "scene_transform": tuple(window_transform(window, transform)[:6]),
            }
        )
        accepted.append(row)
    if accepted:
        eligible = gpd.GeoDataFrame(accepted, geometry="geometry", crs=sources.crs)
    else:
        eligible = sources.iloc[0:0].copy()
    failures = pd.DataFrame(rejected, columns=["source_id", "stage", "reason", "detail"])
    return NaipEligibility(eligible=eligible, rejections=failures)


def _valid_context_footprints(footprints: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mask = (
        footprints.geometry.notna()
        & footprints.geometry.is_valid
        & footprints.geometry.geom_type.isin(("Polygon", "MultiPolygon"))
    )
    result = footprints.loc[mask].copy()
    if "provenance" not in result:
        result["provenance"] = "unknown"
    else:
        result["provenance"] = result["provenance"].fillna("unknown").astype(str)
    return result


def _prepare_inventory(out_dir: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    inventory_dir = out_dir / "inventory"
    sources_path = inventory_dir / "pre_naip_sources.parquet"
    boxes_path = inventory_dir / "boxes.parquet"
    counts_path = inventory_dir / "counts.json"

    overture_config = load_config("overture")
    footprints_path = REPO_ROOT / overture_config["out"]
    if not footprints_path.exists() or "provenance" not in pq.read_schema(footprints_path).names:
        footprints_path = overture.extract(overture_config)
    footprints = gpd.read_parquet(footprints_path)
    context = _valid_context_footprints(footprints)
    sampling_config = load_config("sampling")
    v2_config = sampling_config["dataset_v2"]
    boxes = load_box_geometries(REPO_ROOT / v2_config["box_geometry_artifact"], overture_config["bboxes"])

    if sources_path.exists() and boxes_path.exists() and counts_path.exists():
        cached_counts = json.loads(counts_path.read_text())
        cache_matches = (
            cached_counts.get("config_hash") == configuration_hash()
            and cached_counts.get("code_version") == code_version()
        )
        if cache_matches:
            return (
                gpd.read_parquet(sources_path),
                context,
                gpd.read_parquet(boxes_path),
                cached_counts,
            )

    imagery_config = load_config("imagery")
    gsd = float(imagery_config["gsd_m"])
    image_px = int(imagery_config["image_px"])
    tip_row = int(imagery_config["tip_row"])
    tip_col = int(imagery_config["tip_col"])
    reach = CropReach(
        north_m=(tip_row + 0.5) * gsd,
        west_m=(tip_col + 0.5) * gsd,
        south_m=(image_px - tip_row - 0.5) * gsd,
        east_m=(image_px - tip_col - 0.5) * gsd,
    )
    content, stages = content_pool(
        footprints,
        min_area_m2=float(sampling_config["min_footprint_area_m2"]),
        max_aspect_ratio=float(sampling_config["max_aspect_ratio"]),
    )
    metric = add_scene_geometries(
        content.to_crs("EPSG:5070"),
        max_distance_m=float(imagery_config["scene_max_distance_m"]),
        crop_reach=reach,
        slack_m=float(imagery_config["scene_slack_m"]),
    )
    urban_areas = load_urban_areas(REPO_ROOT / sampling_config["census_urban"]["path"])
    rural, rural_mask = filter_scene_rural(metric, urban_areas)
    frozen_cuts = tuple(float(value) for value in v2_config["building_strata"]["frozen_log_area_cuts"])
    rural, measured_cuts = assign_building_strata(rural, frozen_log_area_cuts=frozen_cuts)
    rural["box_id"] = rural["box_id"].astype(str)
    rural["source_id"] = rural["building_id"].astype(str)
    rural["ecoregion"] = rural["ecoregion_code"].astype(str)
    rural = add_spatial_covariates(rural)
    rural["scene_geometry_wkb"] = rural["scene_geometry"].map(lambda geometry: geometry.wkb)
    box_geometry = boxes.set_index("box_id").geometry.to_dict()
    rural["inside_overture_box"] = [
        bool(box_geometry[str(box_id)].covers(scene))
        for box_id, scene in zip(rural["box_id"], rural["scene_geometry"], strict=True)
    ]
    pre_naip = rural.loc[rural["inside_overture_box"]].drop(columns="scene_geometry").copy()

    center_pairs = urban_areas.sindex.query(boxes.geometry.centroid, predicate="intersects")
    center_rural = np.ones(len(boxes), dtype=bool)
    if center_pairs.size:
        center_rural[np.unique(center_pairs[0])] = False
    boxes["center_rural"] = center_rural
    if not center_rural.all():
        raise RuntimeError("configured v1 box selection contains a non-rural center")

    full_counts = footprints.groupby("box_name").size()
    content_counts = content.groupby("box_id").size()
    rural_counts = rural.groupby("box_id").size()
    box_contained_counts = pre_naip.groupby("box_id").size()
    counts = {
        "config_hash": configuration_hash(),
        "code_version": code_version(),
        "measured_log_area_cuts": list(measured_cuts),
        "retention_stages": [stage.__dict__ for stage in stages],
        "boxes": {
            str(box_id): {
                "full_pool_count": int(full_counts.get(box_id, 0)),
                "content_pool_count": int(content_counts.get(box_id, 0)),
                "rural_content_pool_count": int(rural_counts.get(box_id, 0)),
                "box_contained_count": int(box_contained_counts.get(box_id, 0)),
            }
            for box_id in boxes["box_id"]
        },
    }
    inventory_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(pre_naip, sources_path)
    _write_parquet_atomic(boxes, boxes_path)
    _write_json_atomic(counts_path, counts)
    return pre_naip, context, boxes, counts


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def run_boxwise(
    box_ids: Iterable[str],
    process_box: Callable[[str, Path], None],
    out_dir: str | Path,
    config_hash: str,
    code_version: str,
) -> None:
    out_dir = Path(out_dir)
    checkpoint_path = out_dir / "screen-checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("config_hash") != config_hash or checkpoint.get("code_version") != code_version:
            raise CheckpointMismatch("screen checkpoint config or code provenance does not match")
    else:
        checkpoint = {
            "config_hash": config_hash,
            "code_version": code_version,
            "completed_boxes": [],
        }
    completed = set(checkpoint["completed_boxes"])
    for box_id in box_ids:
        if box_id in completed:
            continue
        output = out_dir / "screen" / f"{box_id}.complete"
        output.parent.mkdir(parents=True, exist_ok=True)
        process_box(box_id, output)
        checkpoint["completed_boxes"].append(box_id)
        completed.add(box_id)
        _write_json_atomic(checkpoint_path, checkpoint)


def _pin_id(source_id: str, bin_id: int, repeat_index: int) -> str:
    return hashlib.sha256(f"{source_id}:{bin_id}:{repeat_index}".encode()).hexdigest()


def build_source_and_pin_rows(
    scan: SourceBinResult,
    all_bin_ids: Iterable[int],
    scene_window: tuple[int, int, int, int],
    scene_item_id: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    bin_ids = tuple(int(bin_id) for bin_id in all_bin_ids)
    availability = {str(bin_id): bin_id in scan.pins_by_bin for bin_id in bin_ids}
    source = {
        "source_id": scan.source_id,
        "scene_id": naip.window_cache_key(scene_item_id, Window(*scene_window)),
        "status": "eligible" if scan.usable else "eligible_zero_bins",
        "available_bin_count": len(scan.pins_by_bin),
        "bin_availability": availability,
        "unavailability_reasons": {str(key): value for key, value in scan.unavailable_reasons.items()},
        "innermost_signed_distance_m": scan.innermost_signed_distance_m,
        "outermost_signed_distance_m": scan.outermost_signed_distance_m,
        "complete_source": all(availability.values()),
        "complete_inside": all(availability.get(str(bin_id), False) for bin_id in (-5, -3, -1)),
        "scene_window": tuple(int(value) for value in scene_window),
        "scene_item_id": scene_item_id,
    }
    pins = []
    for bin_id in bin_ids:
        for pin in scan.pins_by_bin.get(bin_id, ()):
            pins.append(
                {
                    "pin_id": _pin_id(scan.source_id, bin_id, pin.repeat_index),
                    "source_id": scan.source_id,
                    "scene_item_id": scene_item_id,
                    "bin_id": bin_id,
                    "repeat_index": pin.repeat_index,
                    "target_signed_distance_m": pin.target_signed_distance_m,
                    "exact_signed_distance_m": pin.exact_signed_distance_m,
                    "row": pin.row,
                    "col": pin.col,
                    "tip_x": pin.x,
                    "tip_y": pin.y,
                    "owning_polygon_id": scan.source_id,
                    "nearest_polygon_id": pin.nearest_polygon_id,
                    "nearest_polygon_provenance": pin.nearest_polygon_provenance,
                    "nearest_edge_bearing_radians": pin.bearing_radians,
                }
            )
    return source, pins


def screen_source(
    row: pd.Series,
    context_wgs84: gpd.GeoDataFrame,
    *,
    seed: int,
    slot_tolerance_m: float,
    crop_size: int,
    tip_row: int,
    tip_col: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run D06/D07/D11 geometry for one mechanically eligible source without reading image pixels."""
    scene_5070 = from_wkb(row["scene_geometry_wkb"])
    scene_wgs84 = _geometry_to_crs(scene_5070, "EPSG:5070", "EPSG:4326")
    context = context_for_scene(context_wgs84, scene_wgs84).to_crs(str(row["naip_source_crs"]))
    scene_transform = _affine(row["scene_transform"])
    scene_window = tuple(row["scene_window"])
    scene_shape = (int(scene_window[3]), int(scene_window[2]))
    scan = sample_source_bins(
        build_scene_grid(context, str(row["source_id"]), scene_shape, scene_transform),
        context,
        str(row["source_id"]),
        distance_bins(),
        seed,
        slot_tolerance_m=slot_tolerance_m,
        crop_size=crop_size,
        tip_row=tip_row,
        tip_col=tip_col,
    )
    source, pins = build_source_and_pin_rows(
        scan,
        (distance_bin.bin_id for distance_bin in distance_bins()),
        tuple(int(value) for value in scene_window),
        str(row["naip_item_id"]),
    )
    source.update(row.to_dict())
    source["bin_availability"] = json.dumps(source["bin_availability"], sort_keys=True)
    source["unavailability_reasons"] = json.dumps(source["unavailability_reasons"], sort_keys=True)

    anchor = _geometry_to_crs(
        Point(float(row["anchor_lon"]), float(row["anchor_lat"])),
        "EPSG:4326",
        str(row["naip_source_crs"]),
    )
    for pin in pins:
        crop = naip.crop_window_for_slot(
            int(pin["row"]),
            int(pin["col"]),
            scene_shape,
            crop_size,
            tip_row,
            tip_col,
        )
        crop_transform = window_transform(crop, scene_transform)
        assert_crop_transform(scene_transform, crop_transform, crop)
        label_x, label_y = xy(crop_transform, tip_row, tip_col, offset="center")
        if not math.isclose(float(label_x), float(pin["tip_x"]), abs_tol=1e-9) or not math.isclose(
            float(label_y), float(pin["tip_y"]), abs_tol=1e-9
        ):
            raise RuntimeError("rendered tip center does not equal the sampled native slot center")
        tip_wgs84 = _geometry_to_crs(
            Point(float(pin["tip_x"]), float(pin["tip_y"])),
            str(row["naip_source_crs"]),
            "EPSG:4326",
        )
        dx = float(pin["tip_x"]) - anchor.x
        dy = float(pin["tip_y"]) - anchor.y
        pin.update(
            {
                "tip_row": tip_row,
                "tip_col": tip_col,
                "tip_lat": tip_wgs84.y,
                "tip_lon": tip_wgs84.x,
                "radial_displacement_m": math.hypot(dx, dy),
                "source_to_pin_bearing_radians": math.atan2(dy, dx) % math.tau,
                "crop_window": (int(crop.col_off), int(crop.row_off), crop_size, crop_size),
                "crop_transform": tuple(float(value) for value in crop_transform[:6]),
            }
        )
    return source, pins


def unavailability_report(sources: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in sources.itertuples(index=False):
        reasons = source.unavailability_reasons
        if isinstance(reasons, str):
            reasons = json.loads(reasons)
        for bin_id, reason in reasons.items():
            rows.append(
                {
                    "bin_id": int(bin_id),
                    "reason": str(reason),
                    "building_stratum": str(source.building_stratum),
                    "ecoregion_code": str(source.ecoregion_code),
                }
            )
    columns = ["bin_id", "reason", "building_stratum", "ecoregion_code", "count"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .groupby(columns[:-1], as_index=False, sort=True)
        .size()
        .rename(columns={"size": "count"})[columns]
    )


def _screen_box(
    box_id: str,
    marker_path: Path,
    sources: gpd.GeoDataFrame,
    context_wgs84: gpd.GeoDataFrame,
    box_geometry,
) -> None:
    imagery = load_config("imagery")
    sampling_config = load_config("sampling")
    v2 = sampling_config["dataset_v2"]
    box_sources = sources.loc[sources["box_id"] == box_id].copy()
    box_wgs84 = _geometry_to_crs(box_geometry, sources.crs, "EPSG:4326")
    items = naip.search_items_for_geometry(box_wgs84, imagery) if not box_sources.empty else []
    eligibility = assign_naip_eligibility(
        box_sources,
        items,
        gsd_m=float(imagery["gsd_m"]),
        gsd_tolerance=float(imagery["gsd_tolerance"]),
        grid_loader=lambda item: naip.item_grid_metadata(
            item,
            float(imagery["gsd_m"]),
            float(imagery["gsd_tolerance"]),
        ),
    )
    screened = seeded_screen_subsample(
        eligibility.eligible,
        per_box_cap=v2["screen"]["per_box_cap"],
        seed=int(v2["seeds"]["screen_subsample"]),
    )
    source_rows: list[dict[str, object]] = []
    pin_rows: list[dict[str, object]] = []
    for _, source_row in screened.iterrows():
        source, pins = screen_source(
            source_row,
            context_wgs84,
            seed=int(v2["seeds"]["pin_sampling"]),
            slot_tolerance_m=float(load_config("bins")["slot_tolerance_m"]),
            crop_size=int(imagery["image_px"]),
            tip_row=int(imagery["tip_row"]),
            tip_col=int(imagery["tip_col"]),
        )
        source_rows.append(source)
        pin_rows.extend(pins)

    box_dir = marker_path.parent / box_id
    box_dir.mkdir(parents=True, exist_ok=True)
    source_frame = pd.DataFrame(source_rows)
    if source_frame.empty:
        source_frame = pd.DataFrame(columns=["source_id", "scene_geometry_wkb"])
    if "geometry" in source_frame:
        source_frame["source_geometry_wkb"] = source_frame["geometry"].map(lambda geometry: geometry.wkb)
        source_frame = source_frame.drop(columns="geometry")
    _write_parquet_atomic(source_frame, box_dir / "sources.parquet")
    pin_frame = pd.DataFrame(pin_rows)
    if pin_frame.empty:
        pin_frame = pd.DataFrame(columns=["pin_id", "source_id"])
    _write_parquet_atomic(pin_frame, box_dir / "pins.parquet")
    _write_parquet_atomic(eligibility.rejections, box_dir / "tile_rejections.parquet")
    _write_json_atomic(
        box_dir / "counts.json",
        {
            "box_id": box_id,
            "pre_naip_count": len(box_sources),
            "eligible_count": len(eligibility.eligible),
            "screened_count": len(screened),
            "tile_rejections": eligibility.rejections["reason"].value_counts().to_dict(),
            "imagery_dates": sorted(set(eligibility.eligible.get("imagery_date", pd.Series(dtype=str)))),
            "imagery_seasons": sorted(set(eligibility.eligible.get("imagery_season", pd.Series(dtype=str)))),
            "footprint_provenance_mix": _provenance_mix(
                eligibility.eligible.get("provenance", pd.Series(dtype=object))
            ),
        },
    )
    marker_path.write_text(box_id)


def _load_screen_outputs(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    box_dirs = sorted(path for path in (out_dir / "screen").iterdir() if path.is_dir())
    sources = pd.concat((pd.read_parquet(path / "sources.parquet") for path in box_dirs), ignore_index=True)
    pins = pd.concat((pd.read_parquet(path / "pins.parquet") for path in box_dirs), ignore_index=True)
    counts = pd.DataFrame([json.loads((path / "counts.json").read_text()) for path in box_dirs])
    sources["scene_geometry"] = sources["scene_geometry_wkb"].map(from_wkb)
    return sources, pins, counts


def _write_audit_reports(out_dir: Path, sources: pd.DataFrame, selected: pd.DataFrame, pins: pd.DataFrame) -> None:
    report_dir = out_dir / "audit"
    report_dir.mkdir(parents=True, exist_ok=True)
    bin_ids = tuple(distance_bin.bin_id for distance_bin in distance_bins())
    availability_smd_report(sources, bin_ids).to_csv(report_dir / "availability_smd.csv", index=False)
    selection_smd_report(selected, sources).to_csv(report_dir / "spacing_smd.csv", index=False)
    coverage_report(selected, bin_ids).to_csv(report_dir / "coverage.csv", index=False)
    unavailability_report(sources).to_csv(report_dir / "unavailability.csv", index=False)
    selected_ids = set(selected["source_id"].astype(str))
    selected_pins = pins.loc[pins["source_id"].astype(str).isin(selected_ids)]
    selected_pins.to_parquet(report_dir / "selected_pins.parquet", index=False)


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _provenance_category(value: object) -> str:
    try:
        records = json.loads(value) if isinstance(value, str) else value
        if isinstance(records, list) and records:
            record = records[0]
            return str(record.get("dataset") or record.get("provider") or record.get("resource") or "unknown")
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return "unknown"


def _provenance_mix(values: Iterable[object]) -> dict[str, float]:
    counts = Counter(_provenance_category(value) for value in values)
    total = sum(counts.values())
    return {key: count / total for key, count in sorted(counts.items())} if total else {}


def zero_eligible_reason(inventory_counts: dict, screen_counts: pd.Series) -> str | None:
    if int(screen_counts["eligible_count"]) > 0:
        return None
    for field, reason in (
        ("full_pool_count", "no_overture_buildings"),
        ("content_pool_count", "content_filters"),
        ("rural_content_pool_count", "scene_rurality"),
        ("box_contained_count", "box_containment"),
    ):
        if int(inventory_counts[field]) == 0:
            return reason
    rejections = _json_object(screen_counts.get("tile_rejections", {}))
    rejection_reasons = sorted(str(reason) for reason, count in rejections.items() if int(count) > 0)
    suffix = ",".join(rejection_reasons) if rejection_reasons else "unknown"
    return f"tile_containment:{suffix}"


def _write_construction_manifest(
    out_dir: Path,
    *,
    sources: pd.DataFrame,
    pins: pd.DataFrame,
    selected: pd.DataFrame,
    boxes: gpd.GeoDataFrame,
    inventory_counts: dict,
    screen_counts: pd.DataFrame,
    quotas: dict[str, int],
    spacing_rejections: pd.DataFrame,
    spills: list[dict[str, str]],
) -> None:
    sampling_config = load_config("sampling")
    v2 = sampling_config["dataset_v2"]
    run_id = configuration_hash()[:16]
    run_record = RunRecord(
        schema_version=2,
        run_id=run_id,
        created_at=datetime.now(UTC),
        config_hash=configuration_hash(),
        git_commit=git_commit(),
        code_version=code_version(),
        overture_release=str(load_config("overture")["release"]),
        area_log_cuts=tuple(v2["building_strata"]["frozen_log_area_cuts"]),
        coverage_threshold=float(v2["gate"]["minimum_bin_coverage"]),
        screen_per_box_cap=v2["screen"]["per_box_cap"],
        spill_revisit_fraction=float(v2["spill"]["revisit_fraction_of_main"]),
        delta_m=None,
        census_urban_vintage=(
            f"{sampling_config['census_urban']['definition_year']} UA / "
            f"TIGER {sampling_config['census_urban']['tiger_release']}"
        ),
        ecoregion_artifact=str(sampling_config["ecoregions"]["artifact"]),
        seeds={key: int(value) for key, value in v2["seeds"].items()},
        ecoregion_merge_mapping={str(key): str(value) for key, value in v2["ecoregion_merge_mapping"].items()},
    )

    selected_main = selected.loc[selected["assignment"] == "main"]
    actual_main = selected_main.groupby("box_id").size()
    spill_in = Counter(record["destination_box_id"] for record in spills)
    spill_out = Counter(record["origin_box_id"] for record in spills)
    spacing_loss = spacing_rejections.groupby("box_id").size() if not spacing_rejections.empty else pd.Series(dtype=int)
    screen_lookup = screen_counts.set_index("box_id")
    box_records = []
    for box in boxes.itertuples(index=False):
        box_id = str(box.box_id)
        counts = inventory_counts["boxes"][box_id]
        screen_row = screen_lookup.loc[box_id]
        eligible_count = int(screen_row["eligible_count"])
        dates = tuple(_as_date(value) for value in screen_lookup.loc[box_id, "imagery_dates"])
        seasons = tuple(str(value) for value in screen_lookup.loc[box_id, "imagery_seasons"])
        provenance_mix = _json_object(screen_lookup.loc[box_id, "footprint_provenance_mix"])
        box_records.append(
            BoxRecord(
                run_id=run_id,
                box_id=box_id,
                ecoregion_code=str(box.ecoregion_code),
                ecoregion_name=str(box.ecoregion_name),
                state=str(box.state),
                center_rural=bool(box.center_rural),
                full_pool_count=int(counts["full_pool_count"]),
                content_pool_count=int(counts["content_pool_count"]),
                rural_content_pool_count=int(counts["rural_content_pool_count"]),
                eligible_count=eligible_count,
                screened_count=int(screen_lookup.loc[box_id, "screened_count"]),
                scene_rural_loss=int(counts["content_pool_count"] - counts["rural_content_pool_count"]),
                box_containment_loss=int(counts["rural_content_pool_count"] - counts["box_contained_count"]),
                tile_containment_loss=int(counts["box_contained_count"] - eligible_count),
                spacing_loss=int(spacing_loss.get(box_id, 0)),
                main_quota=int(quotas[box_id]),
                realized_main=int(actual_main.get(box_id, 0)),
                spill_in=int(spill_in[box_id]),
                spill_out=int(spill_out[box_id]),
                zero_eligible_reason=zero_eligible_reason(counts, screen_row),
                box_selection_probability=float(box.cell_selection_probability),
                imagery_dates=dates,
                imagery_seasons=seasons,
                footprint_provenance_mix=provenance_mix,
            )
        )

    selected_ids = set(selected["source_id"].astype(str))
    selected_by_id = selected.set_index(selected["source_id"].astype(str), drop=False)
    source_records = []
    for source_id, row in selected_by_id.iterrows():
        availability = _json_object(row["bin_availability"])
        reasons = _json_object(row["unavailability_reasons"])
        source_records.append(
            SourceRecord(
                run_id=run_id,
                source_id=str(source_id),
                box_id=str(row["box_id"]),
                quota_box_id=str(row["quota_box_id"]),
                ecoregion_code=str(row["ecoregion_code"]),
                building_stratum=str(row["building_stratum"]),
                assignment=str(row["assignment"]),
                state=str(row["state"]),
                source_footprint_provenance=str(row.get("provenance", "unknown")),
                footprint_area_m2=float(row["footprint_area_m2"]),
                source_anchor_lat=float(row["anchor_lat"]),
                source_anchor_lon=float(row["anchor_lon"]),
                nearest_neighbor_distance_m=float(row["nearest_neighbor_distance_m"]),
                neighbors_within_80m=int(row["neighbors_within_80m"]),
                scene_id=str(row["scene_id"]),
                naip_item_id=str(row["naip_item_id"]),
                naip_source_crs=str(row["naip_source_crs"]),
                naip_item_transform=tuple(float(value) for value in row["naip_item_transform"]),
                imagery_date=_as_date(row["imagery_date"]),
                imagery_season=str(row["imagery_season"]),
                bin_availability={str(key): bool(value) for key, value in availability.items()},
                unavailability_reasons={str(key): str(value) for key, value in reasons.items()},
                innermost_signed_distance_m=float(row["innermost_signed_distance_m"]),
                outermost_signed_distance_m=float(row["outermost_signed_distance_m"]),
                complete_source=bool(row["complete_source"]),
                complete_inside=bool(row["complete_inside"]),
                scene_rural=True,
                rejection_provenance=(
                    (f"spilled_from:{row['quota_box_id']}",) if str(row["quota_box_id"]) != str(row["box_id"]) else ()
                ),
                scene_window=tuple(int(value) for value in row["scene_window"]),
                scene_transform=tuple(float(value) for value in row["scene_transform"]),
            )
        )

    bin_by_id = {distance_bin.bin_id: distance_bin for distance_bin in distance_bins()}
    pin_records = []
    for row in pins.loc[pins["source_id"].astype(str).isin(selected_ids)].itertuples(index=False):
        distance_bin = bin_by_id[int(row.bin_id)]
        pin_records.append(
            PinRecord(
                run_id=run_id,
                pin_id=str(row.pin_id),
                source_id=str(row.source_id),
                bin_id=int(row.bin_id),
                bin_lower_m=float(distance_bin.lower_m),
                bin_upper_m=float(distance_bin.upper_m),
                repeat_index=int(row.repeat_index),
                target_signed_distance_m=float(row.target_signed_distance_m),
                exact_signed_distance_m=float(row.exact_signed_distance_m),
                ground_truth_inside=float(row.exact_signed_distance_m) < 0,
                tip_row=int(row.tip_row),
                tip_col=int(row.tip_col),
                tip_x=float(row.tip_x),
                tip_y=float(row.tip_y),
                tip_lat=float(row.tip_lat),
                tip_lon=float(row.tip_lon),
                owning_polygon_id=str(row.owning_polygon_id),
                nearest_polygon_id=str(row.nearest_polygon_id),
                nearest_polygon_provenance=str(row.nearest_polygon_provenance),
                nearest_edge_bearing_radians=float(row.nearest_edge_bearing_radians),
                radial_displacement_m=float(row.radial_displacement_m),
                source_to_pin_bearing_radians=float(row.source_to_pin_bearing_radians),
                crop_window=tuple(int(value) for value in row.crop_window),
                crop_transform=tuple(float(value) for value in row.crop_transform),
            )
        )
    deviations = tuple(
        DeviationRecord(
            run_id=run_id,
            deviation_id=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
            deviation_type="spill",
            source_id=record["source_id"],
            origin_box_id=record["origin_box_id"],
            destination_box_id=record["destination_box_id"],
            details={"scope": record["scope"]},
        )
        for record in spills
    )
    write_manifest(
        out_dir / "manifest.duckdb",
        ManifestBundle(
            run=run_record,
            boxes=tuple(box_records),
            sources=tuple(source_records),
            pins=tuple(pin_records),
            renders=tuple(),
            deviations=deviations,
        ),
    )


def _audit_and_gate(
    out_dir: Path,
    inventory_boxes: gpd.GeoDataFrame,
    inventory_counts: dict,
) -> None:
    sources, pins, screen_counts = _load_screen_outputs(out_dir)
    v2 = load_config("sampling")["dataset_v2"]
    usable, zero_bins = usable_source_pool(sources)
    quotas = main_box_quotas(inventory_boxes["box_id"], seed=int(v2["seeds"]["box_quota"]))
    ablation_boxes = select_ablation_boxes(
        usable,
        quotas,
        seed=int(v2["seeds"]["ablation_assignment"]),
        target=int(v2["ablation_sources"]),
    )
    combined_capacity = {box_id: main_quota + int(box_id in ablation_boxes) for box_id, main_quota in quotas.items()}
    initial = select_joint_sources(
        usable,
        seed=int(v2["seeds"]["candidate_order"]),
        capacity_by_box=combined_capacity,
    )

    ecoregion_by_box = inventory_boxes.set_index("box_id")["ecoregion_code"].astype(str).to_dict()
    ecoregion_config = load_config("sampling")["ecoregions"]
    ecoregions = gpd.read_file(f"zip://{REPO_ROOT / ecoregion_config['path']}")
    adjacency = polygon_adjacency(
        ecoregions,
        code_column=ecoregion_config["code_field"],
        allowed_codes=set(ecoregion_by_box.values()),
    )
    combined_result = fill_deficits_by_ecoregion(
        initial,
        usable,
        ecoregion_by_box=ecoregion_by_box,
        adjacent_ecoregions=adjacency,
        seed=int(v2["seeds"]["deficit_spill"]),
    )
    assigned = assign_ablation(
        combined_result.selected,
        seed=int(v2["seeds"]["ablation_assignment"]),
        target=int(v2["ablation_sources"]),
        required_box_ids=ablation_boxes,
    )
    ablation_sources = assigned.loc[assigned["assignment"] == "ablation"].copy()
    main_sources = assigned.loc[assigned["assignment"] == "main"].copy()
    selected = assigned.reset_index(drop=True)

    spacing_rejections = combined_result.spacing_rejections.drop_duplicates()
    selection_dir = out_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selected.drop(columns="scene_geometry").to_parquet(selection_dir / "sources.parquet", index=False)
    spacing_rejections.to_csv(selection_dir / "spacing_rejections.csv", index=False)
    zero_bin_output = zero_bins.drop(columns="scene_geometry", errors="ignore")
    zero_bin_output.to_parquet(selection_dir / "zero_bin_sources.parquet", index=False)
    _write_json_atomic(selection_dir / "ablation_boxes.json", sorted(ablation_boxes))
    _write_json_atomic(selection_dir / "spills.json", combined_result.spills)
    _write_audit_reports(out_dir, sources, selected, pins)

    count_lookup = screen_counts.set_index("box_id")
    box_rows = []
    for box_id in inventory_boxes["box_id"].astype(str):
        inventory = inventory_counts["boxes"][box_id]
        screen_row = count_lookup.loc[box_id]
        eligible_count = int(count_lookup.loc[box_id, "eligible_count"])
        box_rows.append(
            {
                "box_id": box_id,
                "eligible_count": eligible_count,
                "zero_eligible_reason": zero_eligible_reason(inventory, screen_row),
            }
        )
    gate_boxes = pd.DataFrame(box_rows)
    _write_construction_manifest(
        out_dir,
        sources=sources,
        pins=pins,
        selected=selected,
        boxes=inventory_boxes,
        inventory_counts=inventory_counts,
        screen_counts=screen_counts,
        quotas=quotas,
        spacing_rejections=spacing_rejections,
        spills=combined_result.spills,
    )
    coverage = coverage_report(selected, (distance_bin.bin_id for distance_bin in distance_bins()))
    gate = {"passed": False}
    try:
        enforce_gate(
            coverage,
            selected,
            gate_boxes,
            expected_main_total=int(v2["main_sources"]),
            coverage_threshold=float(v2["gate"]["minimum_bin_coverage"]),
        )
        if len(ablation_sources) != int(v2["ablation_sources"]):
            raise GateFailure(f"ablation total {len(ablation_sources)} does not equal {v2['ablation_sources']}")
        realized_quota = main_sources.groupby("quota_box_id").size().to_dict()
        quota_failures = {
            box_id: {"required": quota, "realized": int(realized_quota.get(box_id, 0))}
            for box_id, quota in quotas.items()
            if int(realized_quota.get(box_id, 0)) != quota
        }
        if quota_failures:
            raise GateFailure(f"main quota failures after assignment: {quota_failures}")
        gate["passed"] = True
    except (GateFailure, ValueError) as error:
        gate["failure"] = str(error)
    gate.update(
        {
            "main_sources": len(main_sources),
            "ablation_sources": len(ablation_sources),
            "spill_count": len(combined_result.spills),
            "spill_revisit_threshold": int(float(v2["spill"]["revisit_fraction_of_main"]) * int(v2["main_sources"])),
        }
    )
    _write_json_atomic(out_dir / "gate.json", gate)
    if not gate["passed"]:
        raise GateFailure(str(gate["failure"]))


def run_screen(out_dir: Path) -> None:
    sources, context, boxes, counts = _prepare_inventory(out_dir)
    box_geometries = boxes.set_index("box_id").geometry.to_dict()
    run_boxwise(
        boxes["box_id"].astype(str),
        lambda box_id, marker: _screen_box(box_id, marker, sources, context, box_geometries[box_id]),
        out_dir,
        configuration_hash(),
        code_version(),
    )
    _audit_and_gate(out_dir, boxes, counts)


def _save_png(array: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.png")
    bands, height, width = array.shape
    with rasterio.open(
        temporary,
        "w",
        driver="PNG",
        height=height,
        width=width,
        count=bands,
        dtype=array.dtype,
    ) as destination:
        destination.write(array)
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_sanity_render(out_dir: Path, source_limit: int = 200) -> None:
    """Render D23 from the same screened pin records; this is not a statistical gate."""
    from thesis.render.marker import render_at_tip

    require_passing_gate(out_dir)
    if source_limit <= 0:
        raise ValueError("sanity source limit must be positive")
    sampling_config = load_config("sampling")
    imagery = load_config("imagery")
    seed = int(sampling_config["dataset_v2"]["seeds"]["sanity_render"])
    selected = pd.read_parquet(out_dir / "selection" / "sources.parquet")
    selected["_order"] = selected["source_id"].map(lambda value: stable_seed_key(seed, value))
    selected = selected.sort_values(["_order", "source_id"], kind="stable").head(source_limit)
    pins = pd.read_parquet(out_dir / "audit" / "selected_pins.parquet")
    pins = pins.loc[pins["source_id"].astype(str).isin(selected["source_id"].astype(str))]
    _, _, boxes, _ = _prepare_inventory(out_dir)
    box_geometries = boxes.set_index("box_id").geometry.to_dict()
    item_records: dict[str, dict[str, dict]] = {}
    preset_id = "google_pin_red"
    preset = load_presets(load_config("render"))[preset_id]
    scene_cache = REPO_ROOT / imagery["scene_cache_dir"]
    render_records: list[RenderRecord] = []

    with duckdb.connect(str(out_dir / "manifest.duckdb"), read_only=True) as connection:
        run_id = connection.execute("SELECT run_id FROM runs").fetchone()[0]
        existing_render_ids = {row[0] for row in connection.execute("SELECT render_id FROM renders").fetchall()}

    for source in selected.itertuples(index=False):
        box_id = str(source.box_id)
        if box_id not in item_records:
            box_wgs84 = _geometry_to_crs(box_geometries[box_id], "EPSG:5070", "EPSG:4326")
            item_records[box_id] = {
                str(item["id"]): item for item in naip.search_items_for_geometry(box_wgs84, imagery)
            }
        try:
            item = item_records[box_id][str(source.naip_item_id)]
        except KeyError as error:
            raise StageNotReady(f"NAIP item {source.naip_item_id} is no longer returned for {box_id}") from error
        scene_window = Window(*[int(value) for value in source.scene_window])
        cache_key = naip.window_cache_key(str(source.naip_item_id), scene_window)
        tile = naip.load_cached(scene_cache, cache_key)
        if tile is None:
            tile = naip.read_item_window(
                item,
                scene_window,
                float(imagery["gsd_m"]),
                float(imagery["gsd_tolerance"]),
            )
            naip.store_atomic(scene_cache, cache_key, tile.array, tile.metadata)
        stored_scene_transform = _affine(source.scene_transform)
        loaded_scene_transform = _affine(tile.metadata["transform"])
        if not stored_scene_transform.almost_equals(loaded_scene_transform, precision=1e-12):
            raise StageNotReady(f"native scene transform drifted for {source.source_id}")

        for pin in pins.loc[pins["source_id"].astype(str) == str(source.source_id)].itertuples(index=False):
            crop_window = Window(*[int(value) for value in pin.crop_window])
            row_start = int(crop_window.row_off)
            col_start = int(crop_window.col_off)
            size = int(imagery["image_px"])
            crop = tile.array[:, row_start : row_start + size, col_start : col_start + size]
            if crop.shape != (3, size, size):
                raise StageNotReady(f"invalid crop shape for {pin.pin_id}: {crop.shape}")
            crop_transform = window_transform(crop_window, loaded_scene_transform)
            assert_crop_transform(loaded_scene_transform, crop_transform, crop_window)
            if not crop_transform.almost_equals(_affine(pin.crop_transform), precision=1e-12):
                raise StageNotReady(f"crop transform drifted for {pin.pin_id}")
            rendered = render_at_tip(crop, preset)
            render_id = hashlib.sha256(f"{pin.pin_id}:{preset_id}:unchanged".encode()).hexdigest()
            if render_id in existing_render_ids:
                continue
            relative_path = f"sanity/images/{render_id}.png"
            image_hash = _save_png(rendered, out_dir / relative_path)
            render_records.append(
                RenderRecord(
                    run_id=str(run_id),
                    render_id=render_id,
                    pin_id=str(pin.pin_id),
                    preset_id=preset_id,
                    marker_shape=str(preset["shape"]),
                    marker_color=str(preset["color_name"]),
                    marker_size_px=int(preset["size_px"]),
                    marker_opacity=float(preset["opacity"]),
                    prompt_variant="unchanged",
                    image_path=relative_path,
                    image_sha256=image_hash,
                )
            )
    append_renders(out_dir / "manifest.duckdb", render_records)
    _write_json_atomic(
        out_dir / "sanity" / "summary.json",
        {"sources": len(selected), "pins": len(pins), "new_renders": len(render_records), "preset": preset_id},
    )


def require_passing_gate(out_dir: str | Path) -> None:
    path = Path(out_dir) / "gate.json"
    if not path.exists():
        raise StageNotReady("gate report does not exist; run the audit stage first")
    report = json.loads(path.read_text())
    if report.get("passed") is not True:
        raise StageNotReady(f"dataset gate did not pass: {report.get('failure', 'unknown failure')}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Construct the corrected dataset v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("screen", "audit", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--out", type=Path, required=True)
    render = subparsers.add_parser("sanity-render")
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--sources", type=int, default=200)
    args = parser.parse_args(argv)
    if args.command == "screen":
        run_screen(args.out)
        return
    if args.command == "audit":
        _, _, boxes, counts = _prepare_inventory(args.out)
        _audit_and_gate(args.out, boxes, counts)
        return
    if args.command == "sanity-render":
        run_sanity_render(args.out, source_limit=args.sources)
        return
    if args.command == "verify":
        require_passing_gate(args.out)
        return


if __name__ == "__main__":
    main()
