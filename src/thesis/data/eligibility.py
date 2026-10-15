"""Source-pool and mechanical-eligibility helpers (D16--D18)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from thesis.data.sampling import RetentionStage, filter_footprints
from thesis.data.statistics import area_tertile_cuts, assign_area_stratum


@dataclass(frozen=True, slots=True)
class CropReach:
    north_m: float
    west_m: float
    south_m: float
    east_m: float


def content_pool(
    footprints: gpd.GeoDataFrame,
    *,
    min_area_m2: float,
    max_aspect_ratio: float,
) -> tuple[gpd.GeoDataFrame, list[RetentionStage]]:
    retained, stages = filter_footprints(footprints, min_area_m2, max_aspect_ratio)
    retained = retained.rename(columns={"footprint_m2_area": "footprint_area_m2"}).copy()
    retained["box_id"] = retained.get("box_id", retained.get("box_name"))
    retained["log_footprint_area"] = np.log(retained["footprint_area_m2"].to_numpy(dtype=float))
    return retained, stages


def assign_building_strata(
    rural_content_pool: gpd.GeoDataFrame,
    frozen_log_area_cuts: tuple[float, float] | None = None,
) -> tuple[gpd.GeoDataFrame, tuple[float, float]]:
    """Assign D18 strata after scene rurality and before mechanical eligibility."""
    measured = area_tertile_cuts(rural_content_pool)
    if frozen_log_area_cuts is not None and not np.allclose(measured, frozen_log_area_cuts, rtol=0, atol=1e-12):
        raise RuntimeError(f"weighted area tertiles drifted: measured={measured}, frozen={frozen_log_area_cuts}")
    cuts = measured if frozen_log_area_cuts is None else frozen_log_area_cuts
    result = rural_content_pool.copy()
    result["building_stratum"] = result["footprint_area_m2"].map(lambda area: assign_area_stratum(float(area), cuts))
    return result, cuts


def scene_bounds_for_source(
    source_geometry: BaseGeometry,
    max_distance_m: float,
    crop_reach: CropReach,
    slack_m: float,
) -> BaseGeometry:
    if max_distance_m < 0 or slack_m < 0:
        raise ValueError("scene distance and slack must be non-negative")
    min_x, min_y, max_x, max_y = source_geometry.bounds
    return box(
        min_x - max_distance_m - crop_reach.west_m - slack_m,
        min_y - max_distance_m - crop_reach.south_m - slack_m,
        max_x + max_distance_m + crop_reach.east_m + slack_m,
        max_y + max_distance_m + crop_reach.north_m + slack_m,
    )


def add_scene_geometries(
    sources: gpd.GeoDataFrame,
    *,
    max_distance_m: float,
    crop_reach: CropReach,
    slack_m: float,
) -> gpd.GeoDataFrame:
    if sources.crs is None or sources.crs.is_geographic:
        raise ValueError("enlarged scenes require a projected metric CRS")
    result = sources.copy()
    result["scene_geometry"] = [
        scene_bounds_for_source(geometry, max_distance_m, crop_reach, slack_m) for geometry in result.geometry
    ]
    return result


def filter_scene_rural(
    sources: gpd.GeoDataFrame,
    urban_areas: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Keep sources whose entire D17 scene does not intersect a Census UA."""
    if sources.crs != urban_areas.crs:
        raise ValueError("sources and Urban Areas must use the same CRS")
    if "scene_geometry" not in sources:
        raise ValueError("sources must contain enlarged scene geometries")
    scenes = gpd.GeoSeries(sources["scene_geometry"], crs=sources.crs)
    pairs = urban_areas.sindex.query(scenes, predicate="intersects")
    rural = np.ones(len(sources), dtype=bool)
    if pairs.size:
        rural[np.unique(pairs[0])] = False
    return sources.iloc[np.flatnonzero(rural)].copy(), rural


def mechanically_eligible(
    scene_geometry: BaseGeometry,
    overture_box_geometry: BaseGeometry,
    naip_item_geometry: BaseGeometry,
) -> bool:
    return bool(overture_box_geometry.covers(scene_geometry) and naip_item_geometry.covers(scene_geometry))


def context_for_scene(footprints: gpd.GeoDataFrame, scene_geometry: BaseGeometry) -> gpd.GeoDataFrame:
    if footprints.empty:
        return footprints.copy()
    indices = footprints.sindex.query(scene_geometry, predicate="intersects")
    return footprints.iloc[np.sort(indices)].copy()


def load_box_geometries(path: str | Path, configured_boxes: list[dict]) -> gpd.GeoDataFrame:
    boxes = gpd.read_parquet(path)
    if boxes.crs is None or boxes.crs.to_epsg() != 5070:
        raise ValueError("box geometry artifact must use EPSG:5070")
    if boxes.duplicated(["grid_x", "grid_y"]).any():
        raise ValueError("box geometry artifact contains duplicate grid coordinates")

    names = {(int(item["grid_x"]), int(item["grid_y"])): str(item["name"]) for item in configured_boxes}
    if len(names) != len(configured_boxes):
        raise ValueError("configured boxes contain duplicate grid coordinates")
    result = boxes.copy()
    result["box_id"] = [names.get((int(x), int(y))) for x, y in zip(result.grid_x, result.grid_y, strict=True)]
    result = result.loc[result["box_id"].notna()].copy()
    if len(result) != len(configured_boxes):
        missing = sorted(set(names.values()) - set(result["box_id"]))
        raise ValueError(f"missing exact geometries for configured boxes: {missing}")

    bounds = result.geometry.bounds
    widths = bounds.maxx - bounds.minx
    heights = bounds.maxy - bounds.miny
    if not np.allclose(widths, 25_000) or not np.allclose(heights, 25_000):
        raise ValueError("configured Overture boxes must be exactly 25 km square")
    return result


def add_spatial_covariates(
    footprints: gpd.GeoDataFrame,
    *,
    box_column: str = "box_id",
    neighbor_distance_m: float = 80.0,
) -> gpd.GeoDataFrame:
    if footprints.crs is None or footprints.crs.is_geographic:
        raise ValueError("spatial covariates require a projected metric CRS")
    result = footprints.copy()
    result["nearest_neighbor_distance_m"] = np.nan
    result["neighbors_within_80m"] = 0

    for _, positions in result.groupby(box_column, sort=False).indices.items():
        position_array = np.asarray(positions, dtype=int)
        geometries = result.geometry.iloc[position_array].to_numpy()
        tree = STRtree(geometries)
        for local_index, geometry in enumerate(geometries):
            nearest_indices, nearest_distances = tree.query_nearest(
                geometry,
                return_distance=True,
                exclusive=True,
                all_matches=True,
            )
            nearest = float(np.min(nearest_distances)) if len(nearest_distances) else math.inf
            nearby = tree.query(geometry, predicate="dwithin", distance=neighbor_distance_m)
            nearby_count = int(np.count_nonzero(nearby != local_index))
            row_position = position_array[local_index]
            result.iloc[row_position, result.columns.get_loc("nearest_neighbor_distance_m")] = nearest
            result.iloc[row_position, result.columns.get_loc("neighbors_within_80m")] = nearby_count
    result["neighbors_within_80m"] = result["neighbors_within_80m"].astype(int)
    return result


def seeded_screen_subsample(
    eligible: gpd.GeoDataFrame,
    *,
    per_box_cap: int | None,
    seed: int,
) -> gpd.GeoDataFrame:
    """Take a seeded SRS per box for D22; retain full-pool counts separately."""
    if per_box_cap is not None and per_box_cap <= 0:
        raise ValueError("per-box screen cap must be positive or null")
    result = eligible.copy()
    result["_screen_order"] = [
        hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest() for source_id in result["building_id"].astype(str)
    ]
    result = result.sort_values(["box_id", "_screen_order", "building_id"], kind="stable")
    if per_box_cap is not None:
        result = result.groupby("box_id", sort=False, group_keys=False).head(per_box_cap)
    result = result.drop(columns="_screen_order").copy()
    result["screened_in_box"] = result.groupby("box_id")["box_id"].transform("size").astype(int)
    result["audit_weight"] = 1.0 / result["screened_in_box"]
    return result.reset_index(drop=True)


def polygon_adjacency(
    regions: gpd.GeoDataFrame,
    *,
    code_column: str,
    allowed_codes: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Derive D19/D21 adjacency from shared Level-I polygon boundaries."""
    allowed = {str(value) for value in allowed_codes}
    subset = regions.loc[regions[code_column].astype(str).isin(allowed)].copy()
    subset[code_column] = subset[code_column].astype(str)
    dissolved = subset.dissolve(by=code_column).geometry
    result: dict[str, tuple[str, ...]] = {}
    for code, geometry in dissolved.items():
        neighbors = [
            str(other_code)
            for other_code, other_geometry in dissolved.items()
            if other_code != code and geometry.boundary.intersection(other_geometry.boundary).length > 0
        ]
        result[str(code)] = tuple(sorted(neighbors))
    missing = allowed - set(result)
    if missing:
        raise ValueError(f"ecoregion artifact lacks configured codes: {sorted(missing)}")
    return result
