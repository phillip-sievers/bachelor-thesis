import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from shapely import STRtree
from shapely.geometry import Point, Polygon

import thesis.data.naip as naip
import thesis.data.overture as overture
import thesis.data.sampling as sampling
import thesis.geometry.displacement as displacement
from thesis.config import REPO_ROOT, load_config, load_run_config
from thesis.data import manifest
from thesis.geometry.signed_distance import to_utm, to_wgs84, utm_epsg
from thesis.render.marker import draw_marker
from thesis.render.presets import load_presets

DATASET_CONF_KEYS = ("bins", "imagery", "overture", "render", "sampling")

RUN_CONFIG = load_run_config()


def compute_config_hash() -> str:
    bundle = {"seed": RUN_CONFIG["seed"]}
    for key in DATASET_CONF_KEYS:
        bundle[key] = load_config(key)

    encoded = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def get_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def build_candidates() -> tuple[gpd.GeoDataFrame, list[sampling.RetentionStage]]:
    footprints_path = overture.extract(load_config("overture"))
    footprints = gpd.read_parquet(footprints_path)

    sampling_config = load_config("sampling")
    candidates, stages = sampling.filter_footprints(
        footprints,
        min_area_m2=sampling_config["min_footprint_area_m2"],
        max_aspect_ratio=sampling_config["max_aspect_ratio"],
    )

    imagery_config = load_config("imagery")
    bins_config = load_config("bins")
    max_outside_m = max(bins_config["fixed_levels_m"]["outside"])
    half_image_width_m = imagery_config["gsd_m"] * imagery_config["image_px"] / 2
    rural_buffer_m = max_outside_m + half_image_width_m + sampling_config["census_urban"]["buffer_m"]

    urban_areas = sampling.load_urban_areas(REPO_ROOT / sampling_config["census_urban"]["path"])
    assert isinstance(urban_areas.crs, pyproj.CRS)
    urban_index = STRtree(urban_areas.geometry.to_numpy())
    metric_geometry = candidates.geometry.to_crs(urban_areas.crs)

    is_rural = metric_geometry.apply(
        lambda geometry: sampling.image_is_rural(geometry, urban_index, buffer_m=rural_buffer_m)
    )
    previous_count = len(candidates)
    candidates = candidates.loc[is_rural.to_numpy()]
    stages.append(sampling._record_stage("census_rural", len(candidates), stages[0].count, previous_count))
    building_config = sampling_config["sampling_frame"]["building_sampling"]
    sampled = sampling.sample_buildings(
        candidates,
        target=building_config["target_unique_buildings"],
        max_per_box=building_config["maximum_buildings_per_bbox"],
        seed=RUN_CONFIG["seed"],
    )
    stages.append(sampling._record_stage("stage2_building_sample", len(sampled), stages[0].count, len(candidates)))
    return sampled, stages


def displace_candidates(candidates: gpd.GeoDataFrame) -> tuple[list[dict], list[dict]]:
    bins_config = load_config("bins")
    imagery_config = load_config("imagery")
    sampling_config = load_config("sampling")
    target_distances = bins_config["fixed_levels_m"]["inside"] + bins_config["fixed_levels_m"]["outside"]
    overture_config = load_config("overture")
    overture_release = overture_config["release"]
    image_width_m = imagery_config["gsd_m"] * imagery_config["image_px"]

    all_footprints = gpd.read_parquet(
        REPO_ROOT / overture_config["out"],
        columns=["building_id", "geometry"],
    )
    footprint_index = STRtree(all_footprints.geometry.to_numpy())
    footprint_indices = {str(building_id): index for index, building_id in enumerate(all_footprints["building_id"])}

    displaced, rejections = [], []
    for _, row in candidates.iterrows():
        building_id = row["building_id"]
        anchor = Point(row["anchor_lon"], row["anchor_lat"])
        epsg = utm_epsg(anchor.x, anchor.y)

        anchor_utm = to_utm(anchor, epsg)
        assert isinstance(anchor_utm, Point)

        footprint_utm = to_utm(row["geometry"], epsg)
        assert isinstance(footprint_utm, Polygon)

        try:
            pins = displacement.generate_displacements(
                footprint_utm,
                anchor_utm,
                target_distances,
                seed=RUN_CONFIG["seed"],
                building_id=str(building_id),
                tolerance_m=bins_config["target_tolerance_m"],
            )
        except RuntimeError as e:
            rejections.append({"candidate_id": building_id, "stage": "displacement", "reason": str(e)})
            continue

        for pin in pins:
            if pin.point is None:
                rejections.append(
                    {
                        "candidate_id": building_id,
                        "stage": "displacement",
                        "reason": "unreachable_negative_level",
                        "target_signed_edge_distance_m": pin.target_signed_distance_m,
                    }
                )
                continue

            extent = sampling.image_extent(pin.point, image_width_m)
            if sampling_config["require_target_fully_inside"] and not sampling.target_fits_image(
                footprint_utm, anchor_utm, extent
            ):
                rejections.append(
                    {
                        "candidate_id": building_id,
                        "stage": "displacement",
                        "reason": "target_or_anchor_outside_image",
                        "target_signed_edge_distance_m": pin.target_signed_distance_m,
                    }
                )
                continue

            lon_lat_point = to_wgs84(pin.point, epsg)
            assert isinstance(lon_lat_point, Point)
            if sampling_config["reject_pin_on_other_footprint"] and sampling.pin_hits_other_footprint(
                lon_lat_point,
                target_index=footprint_indices[str(building_id)],
                footprints=footprint_index,
            ):
                rejections.append(
                    {
                        "candidate_id": building_id,
                        "stage": "displacement",
                        "reason": "pin_on_other_footprint",
                        "target_signed_edge_distance_m": pin.target_signed_distance_m,
                    }
                )
                continue

            displaced.append(
                {
                    "building_id": building_id,
                    "pin_lat": lon_lat_point.y,
                    "pin_lon": lon_lat_point.x,
                    "pin": pin,
                    "radial_displacement_m": anchor_utm.distance(pin.point),
                    "bbox_name": row["box_name"],
                    "state": row["state"],
                    "ecoregion_code": row["ecoregion_code"],
                    "bbox_selection_probability": row["cell_selection_probability"],
                    "eligible_buildings_in_bbox": int(row["eligible_buildings_in_bbox"]),
                    "sampled_buildings_in_bbox": int(row["sampled_buildings_in_bbox"]),
                    "sampling_weight": row["sampling_weight"],
                    "footprint_area_m2": row["footprint_m2_area"],
                    "target_anchor_lat": row["anchor_lat"],
                    "target_anchor_lon": row["anchor_lon"],
                    "overture_release": overture_release,
                    "footprint_source": "overture",
                }
            )
    return displaced, rejections


def _pin_key(entry: dict) -> str:
    return json.dumps(
        [str(entry["building_id"]), entry["pin"].target_signed_distance_m],
        separators=(",", ":"),
    )


def _serialize_entry(entry: dict) -> dict:
    serialized = {
        key: value.item() if isinstance(value, np.generic) else value for key, value in entry.items() if key != "pin"
    }
    pin = entry["pin"]
    serialized["pin"] = {
        "target_signed_distance_m": pin.target_signed_distance_m,
        "actual_signed_distance_m": pin.actual_signed_distance_m,
        "bearing_radians": pin.bearing_radians,
    }
    return serialized


def _deserialize_entry(entry: dict) -> dict:
    restored = dict(entry)
    pin = restored["pin"]
    target = pin["target_signed_distance_m"]
    restored["pin"] = displacement.DisplacedPin(
        target_signed_distance_m=float(target) if target is not None else None,
        actual_signed_distance_m=pin["actual_signed_distance_m"],
        bearing_radians=pin["bearing_radians"],
        point=None,
    )
    return restored


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _save_progress(path: Path, checkpoint: dict) -> None:
    _write_json(path, checkpoint)
    _write_json(path.with_name("failures.json"), checkpoint["rejections"])


def fetch_imagery(
    pins: list[dict],
    checkpoint: dict | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[list[tuple[dict, naip.Tile]], list[dict]]:
    imagery_config = load_config("imagery")
    imagery_config["cache_dir"] = REPO_ROOT / imagery_config["cache_dir"]

    progress = checkpoint if checkpoint is not None else {"completed_tiles": {}, "rejections": []}
    completed_tiles = progress.setdefault("completed_tiles", {})
    rejected_keys = {
        failure["pin_key"]
        for failure in progress["rejections"]
        if failure["stage"] == "naip_fetch" and "pin_key" in failure
    }
    tiles, rejections = [], []
    for entry in pins:
        pin_key = _pin_key(entry)
        if pin_key in completed_tiles:
            cached = naip.load_cached(Path(imagery_config["cache_dir"]), completed_tiles[pin_key])
            if cached is not None:
                tiles.append((entry, cached))
                continue
        if pin_key in rejected_keys:
            continue

        try:
            tile = naip.fetch_tile(
                lat=entry["pin_lat"],
                lon=entry["pin_lon"],
                gsd=imagery_config["gsd_m"],
                size=imagery_config["image_px"],
                config=imagery_config,
            )
            tiles.append((entry, tile))
            completed_tiles[pin_key] = naip.cache_key(
                tile.metadata["item_id"],
                tile.metadata["col_off"],
                tile.metadata["row_off"],
                tile.metadata["size"],
            )
        except (naip.TileOutOfBounds, naip.ResolutionMismatch, naip.NoValidItem) as e:
            failure = {
                "candidate_id": entry["building_id"],
                "stage": "naip_fetch",
                "reason": str(e),
                "target_signed_edge_distance_m": entry["pin"].target_signed_distance_m,
                "pin_key": pin_key,
            }
            rejections.append(failure)
            progress["rejections"].append(failure)
        if checkpoint_path is not None:
            progress["stage"] = "imagery"
            _save_progress(checkpoint_path, progress)
    return tiles, rejections


def build_core(out_dir: Path) -> list[tuple[dict, naip.Tile]]:
    checkpoint_path = out_dir / "checkpoint.json"
    config_hash = compute_config_hash()
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint["config_hash"] != config_hash:
            raise RuntimeError("checkpoint config hash does not match this run")
        displaced = [_deserialize_entry(entry) for entry in checkpoint["displaced"]]
    else:
        candidates, stages = build_candidates()
        displaced, rejections = displace_candidates(candidates)
        checkpoint = {
            "config_hash": config_hash,
            "stage": "displacement_complete",
            "sampling_stages": [asdict(stage) for stage in stages],
            "displaced": [_serialize_entry(entry) for entry in displaced],
            "completed_tiles": {},
            "rejections": rejections,
        }
        _save_progress(checkpoint_path, checkpoint)

    tiles, _ = fetch_imagery(displaced, checkpoint=checkpoint, checkpoint_path=checkpoint_path)
    checkpoint["stage"] = "imagery_complete"
    _save_progress(checkpoint_path, checkpoint)
    return tiles


def split_ablation_holdout(tiles: list[tuple[dict, naip.Tile]], holdout_size: int, seed: int) -> tuple[list, list]:
    building_ids = sorted({entry["building_id"] for entry, _ in tiles})
    rng = random.Random(seed)
    holdout_ids = set(rng.sample(building_ids, k=holdout_size))
    ablation = [(entry, tile) for entry, tile in tiles if entry["building_id"] in holdout_ids]
    main = [(entry, tile) for entry, tile in tiles if entry["building_id"] not in holdout_ids]
    return main, ablation


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bands, height, width = array.shape
    with rasterio.open(path, "w", driver="PNG", height=height, width=width, count=bands, dtype=array.dtype) as dst:
        dst.write(array)


def render_and_manifest(tiles: list[tuple[dict, naip.Tile]], out_dir: Path) -> list[dict]:
    presets = load_presets(load_config("render"))
    imagery_config = load_config("imagery")
    sampling_config = load_config("sampling")
    config_hash = compute_config_hash()
    git_commit = get_git_commit()

    rows = []
    for entry, tile in tiles:
        pin = entry["pin"]
        is_control = pin.target_signed_distance_m is None
        for preset in presets.values():
            image = draw_marker(
                tile.array,
                tile.metadata["px"],
                tile.metadata["py"],
                preset["shape"],
                preset["color"],
                preset["size_px"],
                preset["opacity"],
            )
            sample_id = manifest.sample_id_for(
                building_id=str(entry["building_id"]),
                displacement_role="deep_inside_control" if is_control else "fixed_level",
                target_signed_edge_distance_m=pin.target_signed_distance_m,
                marker_shape=preset["shape"],
                marker_color=str(preset["color"]),
            )
            image_path = f"images/{sample_id}.png"
            save_image(image, out_dir / image_path)

            rows.append(
                {
                    "manifest_schema_version": 1,
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "config_hash": config_hash,
                    "git_commit": git_commit,
                    "building_id": str(entry["building_id"]),
                    "bbox_name": entry["bbox_name"],
                    "state": entry["state"],
                    "ecoregion_code": entry["ecoregion_code"],
                    "bbox_selection_probability": entry["bbox_selection_probability"],
                    "eligible_buildings_in_bbox": entry["eligible_buildings_in_bbox"],
                    "sampled_buildings_in_bbox": entry["sampled_buildings_in_bbox"],
                    "sampling_weight": entry["sampling_weight"],
                    "overture_release": entry["overture_release"],
                    "footprint_source": entry["footprint_source"],
                    "footprint_area_m2": entry["footprint_area_m2"],
                    "target_anchor_lat": entry["target_anchor_lat"],
                    "target_anchor_lon": entry["target_anchor_lon"],
                    "displacement_role": "deep_inside_control" if is_control else "fixed_level",
                    "target_signed_edge_distance_m": pin.target_signed_distance_m,
                    "actual_signed_edge_distance_m": pin.actual_signed_distance_m,
                    "radial_displacement_m": entry["radial_displacement_m"],
                    "bearing_radians": pin.bearing_radians,
                    "pin_lat": entry["pin_lat"],
                    "pin_lon": entry["pin_lon"],
                    "ground_truth_inside": pin.actual_signed_distance_m <= 0,
                    "include_in_distance_curve": not is_control,
                    "naip_item_id": tile.metadata["item_id"],
                    "naip_year": tile.metadata["year"],
                    "naip_state": tile.metadata["state"],
                    "naip_native_gsd_x_m": tile.metadata["native_gsd"][0],
                    "naip_native_gsd_y_m": tile.metadata["native_gsd"][1],
                    "naip_source_crs": tile.metadata["src_crs"],
                    "rendered_gsd_m": imagery_config["gsd_m"],
                    "image_px": imagery_config["image_px"],
                    "marker_shape": preset["shape"],
                    "marker_color": str(preset["color"]),
                    "marker_size_px": preset["size_px"],
                    "marker_opacity": preset["opacity"],
                    "census_urban_artifact": sampling_config["census_urban"]["artifact"],
                    "census_buffer_m": sampling_config["census_urban"]["buffer_m"],
                }
            )
    manifest.write_manifest(out_dir / "manifest.duckdb", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--verify-against", type=Path)
    args = parser.parse_args()

    tiles = build_core(args.out)
    main_set, holdout = split_ablation_holdout(tiles, holdout_size=100, seed=RUN_CONFIG["seed"])
    render_and_manifest(holdout if args.holdout else main_set, args.out)
    if args.verify_against:
        differences = manifest.diff_manifests(
            args.verify_against / "manifest.duckdb",
            args.out / "manifest.duckdb",
        )
        if differences:
            raise RuntimeError("manifests differ:\n" + "\n".join(differences))
        print("manifests identical")


if __name__ == "__main__":
    main()
