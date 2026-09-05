import logging
import random
import warnings
from dataclasses import dataclass
from math import hypot
from pathlib import Path

import geopandas as gpd
from shapely import STRtree, minimum_rotated_rectangle
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionStage:
    name: str
    count: int
    fraction_of_input: float
    fraction_of_previous: float


def _record_stage(name: str, count: int, input_count: int, previous_count: int) -> RetentionStage:
    stage = RetentionStage(
        name=name,
        count=count,
        fraction_of_input=count / input_count if input_count else 0.0,
        fraction_of_previous=count / previous_count if previous_count else 0.0,
    )
    logger.info(
        "sampling stage=%s count=%d fraction_input=%.4f fraction_previous=%.4f",
        stage.name,
        stage.count,
        stage.fraction_of_input,
        stage.fraction_of_previous,
    )
    return stage


def _aspect_ratio(geometry: Polygon) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"shapely\.constructive")
        rectangle = minimum_rotated_rectangle(geometry)
    coordinates = list(rectangle.exterior.coords)
    lengths = [hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(coordinates, coordinates[1:], strict=False)]
    shortest = min(length for length in lengths if length > 0)
    return max(lengths) / shortest


def filter_footprints(
    footprints: gpd.GeoDataFrame,
    min_area_m2: float,
    max_aspect_ratio: float,
) -> tuple[gpd.GeoDataFrame, list[RetentionStage]]:
    if footprints.crs is None:
        raise ValueError("footprints must have a CRS")

    input_count = len(footprints)
    stages = [_record_stage("input", input_count, input_count, input_count)]

    valid_mask = (
        footprints.geometry.notna() & (footprints.geometry.geom_type == "Polygon") & footprints.geometry.is_valid
    )
    retained = footprints.loc[valid_mask].copy()
    stages.append(_record_stage("valid_polygon", len(retained), input_count, input_count))

    metric = retained.to_crs("EPSG:5070")
    retained["footprint_m2_area"] = metric.geometry.area.to_numpy()
    previous_count = len(retained)
    retained = retained.loc[retained["footprint_m2_area"] >= min_area_m2].copy()
    metric = metric.loc[retained.index]
    stages.append(_record_stage("min_area_m2", len(retained), input_count, previous_count))

    retained["aspect_ratio"] = metric.geometry.map(_aspect_ratio).to_numpy()
    previous_count = len(retained)
    retained = retained.loc[retained["aspect_ratio"] <= max_aspect_ratio].copy()
    stages.append(_record_stage("max_aspect_ratio", len(retained), input_count, previous_count))
    return retained, stages


def image_extent(center: Point, width_m: float) -> Polygon:
    if width_m <= 0:
        raise ValueError("image width must be positive")
    half_width = width_m / 2
    return box(center.x - half_width, center.y - half_width, center.x + half_width, center.y + half_width)


def image_is_rural(image_extent: BaseGeometry, urban_index: STRtree, buffer_m: float) -> bool:
    if buffer_m < 0:
        raise ValueError("Census buffer must be non-negative")
    return len(urban_index.query(image_extent.buffer(buffer_m), predicate="intersects")) == 0


def target_fits_image(target: BaseGeometry, anchor: Point, image_extent: BaseGeometry) -> bool:
    return image_extent.covers(target) and image_extent.covers(anchor)


def pin_hits_other_footprint(pin: Point, target_index: int, footprints: STRtree) -> bool:
    return any(index != target_index for index in footprints.query(pin, predicate="intersects"))


def load_urban_areas(path: str | Path, target_crs: str = "EPSG:5070") -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return gpd.read_file(f"zip://{path}")[["geometry"]].to_crs(target_crs)


def _largest_remainder(shares: dict, total: int) -> dict:
    floors = {k: int(v) for k, v in shares.items()}
    remainder = total - sum(floors.values())
    order = sorted(shares, key=lambda k: (shares[k] - floors[k], k), reverse=True)
    for key in order[:remainder]:
        floors[key] += 1
    return floors


def allocate_building_quota(eligible_per_box, target: int, max_per_box: int) -> dict:
    n_boxes = len(eligible_per_box)
    base_share = target / n_boxes
    quota = _largest_remainder({box_name: base_share for box_name in eligible_per_box.index}, target)

    allocation = {box_name: min(q, eligible_per_box[box_name]) for box_name, q in quota.items()}
    sparse = {box_name for box_name, q in quota.items() if eligible_per_box[box_name] < q}
    shortfall = sum(quota[box_name] - allocation[box_name] for box_name in sparse)

    headroom = {
        box_name: min(max_per_box, eligible_per_box[box_name]) - quota[box_name]
        for box_name in quota
        if box_name not in sparse
    }
    total_headroom = sum(headroom.values())
    if shortfall > 0 and total_headroom > 0:
        redistributed = min(shortfall, total_headroom)
        shares = {box_name: h / total_headroom * redistributed for box_name, h in headroom.items()}
        for box_name, extra in _largest_remainder(shares, redistributed).items():
            allocation[box_name] += extra
    return allocation


def sample_buildings(candidates: gpd.GeoDataFrame, target: int, max_per_box: int, seed: int) -> gpd.GeoDataFrame:
    eligible_per_box = candidates.groupby("box_name").size()
    quota = allocate_building_quota(eligible_per_box, target, max_per_box)

    rng = random.Random(seed)
    selected_ids = []
    for box_name, n in quota.items():
        pool = candidates.index[candidates["box_name"] == box_name].tolist()
        selected_ids.extend(rng.sample(pool, k=n))

    sampled = candidates.loc[selected_ids].copy()
    sampled["eligible_buildings_in_bbox"] = sampled["box_name"].map(eligible_per_box)
    sampled["sampled_buildings_in_bbox"] = sampled["box_name"].map(quota)
    sampled["sampling_weight"] = (sampled["candidate_cells_in_ecoregion"] / sampled["selected_cells_in_ecoregion"]) * (
        sampled["eligible_buildings_in_bbox"] / sampled["sampled_buildings_in_bbox"]
    )
    return sampled
