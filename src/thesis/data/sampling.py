"""Source filtering and seeded joint allocation (D16, D18--D21)."""

import hashlib
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import hypot
from pathlib import Path

import geopandas as gpd
import pandas as pd
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


class SelectionFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SpacingSelection:
    selected: pd.DataFrame
    spacing_rejections: pd.DataFrame
    deficits: dict[str, int]


@dataclass(frozen=True, slots=True)
class SpillSelection:
    selected: pd.DataFrame
    spacing_rejections: pd.DataFrame
    spills: list[dict[str, str]]


def usable_source_pool(screened: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep partial sources, separating only D15 zero-bin replacement candidates."""
    if "available_bin_count" not in screened:
        raise ValueError("screened sources must contain available_bin_count")
    usable = screened.loc[screened["available_bin_count"] > 0].copy()
    zero = screened.loc[screened["available_bin_count"] == 0].copy()
    if not zero.empty:
        zero["reason"] = "zero_usable_bins"
    return usable, zero


def stable_seed_key(seed: int, *parts: object) -> str:
    return hashlib.sha256(":".join((str(seed), *(str(part) for part in parts))).encode()).hexdigest()


def main_box_quotas(box_ids: Iterable[str], seed: int) -> dict[str, int]:
    unique = sorted(set(str(box_id) for box_id in box_ids))
    if len(unique) != 120:
        raise ValueError(f"main allocation requires 120 boxes, got {len(unique)}")
    ordered = sorted(unique, key=lambda box_id: stable_seed_key(seed, box_id))
    seventeen = set(ordered[:80])
    return {box_id: 17 if box_id in seventeen else 16 for box_id in unique}


def select_ablation_boxes(
    candidates: pd.DataFrame,
    main_capacity_by_box: Mapping[str, int],
    *,
    seed: int,
    target: int,
) -> set[str]:
    """Reserve one feasible extra slot per box, balanced over Level-I ecoregions (D19/D20)."""
    if target <= 0:
        raise ValueError("ablation target must be positive")
    required = {"source_id", "box_id", "ecoregion"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate columns missing: {sorted(missing)}")
    box_rows = candidates[["box_id", "ecoregion"]].copy()
    if box_rows.groupby("box_id")["ecoregion"].nunique().gt(1).any():
        raise ValueError("a box cannot belong to more than one ecoregion")
    box_rows = box_rows.drop_duplicates("box_id")
    candidate_counts = candidates.groupby("box_id").size()
    box_rows["candidate_count"] = box_rows["box_id"].map(candidate_counts).astype(int)
    box_rows["main_capacity"] = box_rows["box_id"].astype(str).map(main_capacity_by_box)
    if box_rows["main_capacity"].isna().any():
        missing_boxes = sorted(box_rows.loc[box_rows["main_capacity"].isna(), "box_id"].astype(str).unique())
        raise ValueError(f"main capacities missing for boxes: {missing_boxes}")
    feasible = box_rows.loc[box_rows["candidate_count"] >= box_rows["main_capacity"].astype(int) + 1].copy()
    if len(feasible) < target:
        raise SelectionFailure(f"only {len(feasible)} boxes can reserve an ablation slot; need {target}")
    feasible["_order"] = feasible["box_id"].map(lambda value: stable_seed_key(seed, value))

    selected: set[str] = set()
    ecoregion_counts: dict[str, int] = defaultdict(int)
    while len(selected) < target:
        choices = feasible.loc[~feasible["box_id"].astype(str).isin(selected)].copy()
        choices["_ecoregion_count"] = choices["ecoregion"].astype(str).map(lambda value: ecoregion_counts[value])
        choice = choices.sort_values(["_ecoregion_count", "_order", "box_id"], kind="stable").iloc[0]
        selected.add(str(choice.box_id))
        ecoregion_counts[str(choice.ecoregion)] += 1
    return selected


class _SpatialHash:
    def __init__(self, cell_size: float = 1_000.0):
        self.cell_size = cell_size
        self.geometries: list[BaseGeometry] = []
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)

    def _keys(self, geometry: BaseGeometry):
        min_x, min_y, max_x, max_y = geometry.bounds
        for x in range(math.floor(min_x / self.cell_size), math.floor(max_x / self.cell_size) + 1):
            for y in range(math.floor(min_y / self.cell_size), math.floor(max_y / self.cell_size) + 1):
                yield x, y

    def intersects(self, geometry: BaseGeometry) -> bool:
        candidates = {index for key in self._keys(geometry) for index in self.cells.get(key, ())}
        return any(self.geometries[index].intersects(geometry) for index in candidates)

    def add(self, geometry: BaseGeometry) -> None:
        index = len(self.geometries)
        self.geometries.append(geometry)
        for key in self._keys(geometry):
            self.cells[key].append(index)


def select_joint_sources(
    candidates: pd.DataFrame,
    *,
    seed: int,
    capacity_by_box: Mapping[str, int],
    existing_geometries: Iterable[BaseGeometry] = (),
) -> SpacingSelection:
    """Greedily fill fixed box capacities in one seeded global order."""
    required = {"source_id", "box_id", "scene_geometry"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate columns missing: {sorted(missing)}")
    ordered = candidates.copy()
    ordered["_order"] = ordered["source_id"].map(lambda source_id: stable_seed_key(seed, source_id))
    ordered = ordered.sort_values(["_order", "source_id"], kind="stable")

    spatial = _SpatialHash()
    for geometry in existing_geometries:
        spatial.add(geometry)
    accepted_indices: list[int] = []
    rejected_rows: list[dict[str, str]] = []
    per_box: dict[str, int] = defaultdict(int)
    for index, row in ordered.iterrows():
        box_id = str(row["box_id"])
        if per_box[box_id] >= int(capacity_by_box.get(box_id, 0)):
            continue
        if spatial.intersects(row["scene_geometry"]):
            rejected_rows.append({"source_id": str(row["source_id"]), "box_id": box_id, "reason": "scene_overlap"})
            continue
        accepted_indices.append(index)
        per_box[box_id] += 1
        spatial.add(row["scene_geometry"])

    selected = candidates.loc[accepted_indices].copy().reset_index(drop=True)
    selected["quota_box_id"] = selected["box_id"].astype(str)
    deficits = {
        box_id: int(capacity) - per_box.get(box_id, 0)
        for box_id, capacity in capacity_by_box.items()
        if per_box.get(box_id, 0) < int(capacity)
    }
    rejections = pd.DataFrame(rejected_rows, columns=["source_id", "box_id", "reason"])
    return SpacingSelection(selected=selected, spacing_rejections=rejections, deficits=deficits)


def fill_deficits_by_ecoregion(
    initial: SpacingSelection,
    candidates: pd.DataFrame,
    *,
    ecoregion_by_box: Mapping[str, str],
    adjacent_ecoregions: Mapping[str, Iterable[str]],
    seed: int,
) -> SpillSelection:
    """Fill D21 deficits in-region, then in an adjacent region, with seeded round-robin receivers."""
    selected = initial.selected.copy()
    selected_ids = set(selected["source_id"].astype(str))
    spatial = _SpatialHash()
    for geometry in selected["scene_geometry"]:
        spatial.add(geometry)
    remaining = candidates.loc[~candidates["source_id"].astype(str).isin(selected_ids)].copy()
    remaining["_order"] = remaining["source_id"].map(lambda value: stable_seed_key(seed, value))
    spacing_rows = initial.spacing_rejections.to_dict("records")
    spills: list[dict[str, str]] = []
    received: dict[str, int] = defaultdict(int)

    for origin_box in sorted(initial.deficits, key=lambda value: stable_seed_key(seed, value)):
        origin_ecoregion = str(ecoregion_by_box[origin_box])
        for _ in range(initial.deficits[origin_box]):
            chosen = None
            scope = None
            for candidate_scope, allowed_ecoregions in (
                ("same_ecoregion", {origin_ecoregion}),
                ("adjacent_ecoregion", {str(value) for value in adjacent_ecoregions.get(origin_ecoregion, ())}),
            ):
                choices = remaining.loc[
                    (remaining["box_id"].astype(str) != origin_box)
                    & remaining["ecoregion"].astype(str).isin(allowed_ecoregions)
                ].copy()
                choices["_received"] = choices["box_id"].astype(str).map(lambda value: received[value])
                choices["_box_order"] = (
                    choices["box_id"]
                    .astype(str)
                    .map(lambda value, current_origin=origin_box: stable_seed_key(seed, current_origin, value))
                )
                choices = choices.sort_values(["_received", "_box_order", "_order", "source_id"], kind="stable")
                for index, row in choices.iterrows():
                    if spatial.intersects(row["scene_geometry"]):
                        spacing_rows.append(
                            {
                                "source_id": str(row["source_id"]),
                                "box_id": str(row["box_id"]),
                                "reason": "scene_overlap_during_spill",
                            }
                        )
                        remaining = remaining.drop(index)
                        continue
                    chosen = (index, row)
                    scope = candidate_scope
                    break
                if chosen is not None:
                    break
            if chosen is None or scope is None:
                raise SelectionFailure(f"cannot spill deficit from {origin_box}")
            index, row = chosen
            record = row[candidates.columns].to_dict()
            record["quota_box_id"] = origin_box
            selected = pd.concat([selected, pd.DataFrame([record])], ignore_index=True)
            selected_ids.add(str(row["source_id"]))
            spatial.add(row["scene_geometry"])
            destination = str(row["box_id"])
            received[destination] += 1
            remaining = remaining.drop(index)
            spills.append(
                {
                    "deviation_type": "spill",
                    "scope": scope,
                    "origin_box_id": origin_box,
                    "destination_box_id": destination,
                    "source_id": str(row["source_id"]),
                }
            )
    spacing = pd.DataFrame(spacing_rows, columns=["source_id", "box_id", "reason"])
    return SpillSelection(selected=selected.reset_index(drop=True), spacing_rejections=spacing, spills=spills)


def assign_ablation(
    selected: pd.DataFrame,
    *,
    seed: int,
    target: int = 100,
    required_box_ids: set[str] | None = None,
) -> pd.DataFrame:
    required = {"source_id", "box_id", "ecoregion", "building_stratum"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"selected source columns missing: {sorted(missing)}")
    required_boxes = {str(box_id) for box_id in required_box_ids} if required_box_ids is not None else None
    if required_boxes is not None and len(required_boxes) != target:
        raise ValueError("required ablation box count must equal the ablation target")
    allowed = selected
    if required_boxes is not None:
        allowed = allowed.loc[allowed["box_id"].astype(str).isin(required_boxes)]
        if "quota_box_id" in allowed:
            allowed = allowed.loc[allowed["quota_box_id"].astype(str) == allowed["box_id"].astype(str)]
    if allowed["box_id"].nunique() < target:
        raise SelectionFailure(f"ablation needs {target} distinct boxes")

    available = allowed.copy()
    available["_order"] = [
        stable_seed_key(seed, source_id, box_id, ecoregion, stratum)
        for source_id, box_id, ecoregion, stratum in zip(
            available.source_id,
            available.box_id,
            available.ecoregion,
            available.building_stratum,
            strict=True,
        )
    ]
    used_boxes: set[str] = set()
    selected_indices: list[int] = []
    cell_counts: dict[tuple[str, str], int] = defaultdict(int)
    while len(selected_indices) < target:
        choices = available.loc[~available.index.isin(selected_indices) & ~available["box_id"].isin(used_boxes)].copy()
        if choices.empty:
            raise SelectionFailure("could not satisfy one-ablation-source-per-box constraint")
        choices["_cell_count"] = [
            cell_counts[(str(ecoregion), str(stratum))]
            for ecoregion, stratum in zip(choices.ecoregion, choices.building_stratum, strict=True)
        ]
        choice = choices.sort_values(["_cell_count", "_order", "source_id"], kind="stable").iloc[0]
        selected_indices.append(int(choice.name))
        used_boxes.add(str(choice.box_id))
        cell_counts[(str(choice.ecoregion), str(choice.building_stratum))] += 1

    result = selected.copy()
    result["assignment"] = "main"
    result.loc[selected_indices, "assignment"] = "ablation"
    return result


def reduce_primary_pins(pins: pd.DataFrame, delta_m: float | tuple[float, float], seed: int) -> pd.DataFrame:
    """Apply D12 after a scalar or asymmetric (inside, outside) D13 gap."""
    if isinstance(delta_m, tuple):
        if len(delta_m) != 2 or min(delta_m) < 0:
            raise ValueError("asymmetric delta must be (inside, outside), both non-negative")
        inside_delta, outside_delta = delta_m
    else:
        if delta_m < 0:
            raise ValueError("delta must be non-negative")
        inside_delta = outside_delta = delta_m
    required = {"pin_id", "source_id", "bin_id", "exact_signed_distance_m"}
    missing = required - set(pins.columns)
    if missing:
        raise ValueError(f"pin columns missing: {sorted(missing)}")

    result = pins.copy().sort_values(["source_id", "bin_id", "pin_id"], kind="stable").reset_index(drop=True)
    distances = result["exact_signed_distance_m"]
    result["survives_delta"] = ((distances < 0) & (-distances >= inside_delta)) | (
        (distances > 0) & (distances >= outside_delta)
    )
    result["include_in_curve"] = result["survives_delta"]
    result["primary"] = result["survives_delta"]

    near = result.loc[result["survives_delta"] & result["bin_id"].isin((-1, 1))]
    for (_, _), group in near.groupby(["source_id", "bin_id"], sort=True):
        winner = min(
            group.index,
            key=lambda index: stable_seed_key(
                seed,
                result.at[index, "source_id"],
                result.at[index, "bin_id"],
                result.at[index, "pin_id"],
            ),
        )
        result.loc[group.index, "primary"] = False
        result.at[winner, "primary"] = True
    return result
