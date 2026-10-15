"""Native-grid union labels, ownership, and target-first sampling (D06, D07, D11--D15)."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import xy
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy import ndimage
from shapely import STRtree, union_all
from shapely.geometry import Point
from shapely.ops import nearest_points

from thesis.geometry.bins import DistanceBin, repeats_for_bin, sample_log_uniform_target

UnavailabilityReason = Literal["size", "ownership", "crop", "other"]


class GridInvariantError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExactLabel:
    signed_distance_m: float
    nearest_polygon_id: str
    nearest_polygon_provenance: str
    bearing_radians: float
    nearest_edge_x: float
    nearest_edge_y: float


@dataclass(frozen=True, slots=True)
class PinCandidate:
    source_id: str
    bin_id: int
    repeat_index: int
    target_signed_distance_m: float
    row: int
    col: int
    x: float
    y: float
    exact_signed_distance_m: float
    nearest_polygon_id: str
    nearest_polygon_provenance: str
    bearing_radians: float


@dataclass(frozen=True, slots=True)
class SourceBinResult:
    source_id: str
    pins_by_bin: dict[int, tuple[PinCandidate, ...]]
    unavailable_reasons: dict[int, UnavailabilityReason]
    innermost_signed_distance_m: float
    outermost_signed_distance_m: float

    @property
    def available_bins(self) -> tuple[int, ...]:
        return tuple(self.pins_by_bin)

    @property
    def usable(self) -> bool:
        return bool(self.pins_by_bin)

    @property
    def complete_source(self) -> bool:
        return len(self.pins_by_bin) == 10


@dataclass(frozen=True, slots=True)
class SceneGrid:
    signed_distance_m: np.ndarray
    owner_codes: np.ndarray
    transform: Affine
    source_code: int
    id_by_code: tuple[str | None, ...]
    source_footprint_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.signed_distance_m.shape != self.owner_codes.shape:
            raise GridInvariantError("SDF and ownership rasters must share a shape")
        if self.source_footprint_mask is not None and self.source_footprint_mask.shape != self.signed_distance_m.shape:
            raise GridInvariantError("source-footprint and SDF rasters must share a shape")
        _assert_native_transform(self.transform)

    @property
    def shape(self) -> tuple[int, int]:
        return self.signed_distance_m.shape

    def owner_at(self, row: int, col: int) -> str | None:
        code = int(self.owner_codes[row, col])
        return self.id_by_code[code] if 0 <= code < len(self.id_by_code) else None


def _assert_native_transform(transform: Affine) -> None:
    if transform.b != 0 or transform.d != 0:
        raise GridInvariantError("native scene grid must not be rotated or sheared")
    if transform.a <= 0 or transform.e >= 0 or not math.isclose(transform.a, abs(transform.e), abs_tol=1e-12):
        raise GridInvariantError("native scene pixels must be square with north-up orientation")


def assert_crop_transform(scene_transform: Affine, crop_transform: Affine, window: Window) -> None:
    _assert_native_transform(scene_transform)
    expected = window_transform(window, scene_transform)
    if not expected.almost_equals(crop_transform, precision=1e-12):
        raise GridInvariantError("crop transform is not an integer-offset slice of the scene grid")


def _provenance_for(row) -> str:
    for column in ("provenance", "footprint_provenance", "sources"):
        if hasattr(row, column):
            value = getattr(row, column)
            if value is not None:
                return str(value)
    return "unknown"


class _VectorLabelIndex:
    def __init__(self, footprints: gpd.GeoDataFrame):
        if footprints.empty:
            raise ValueError("exact labels require at least one footprint")
        if footprints.crs is not None and getattr(footprints.crs, "is_geographic", False):
            raise ValueError("exact labels require a metric projected CRS")
        self.footprints = footprints.reset_index(drop=True)
        self.geometries = self.footprints.geometry.to_numpy()
        self.union = union_all(self.geometries)
        self.tree = STRtree(self.geometries)

    def label(self, point: Point) -> ExactLabel:
        inside = bool(self.union.covers(point))
        union_edge_distance = float(point.distance(self.union.boundary))
        if inside:
            edge_point = nearest_points(point, self.union.boundary)[1]
            indices = np.asarray(self.tree.query(edge_point, predicate="intersects"), dtype=int)
            if len(indices) == 0:
                indices = np.atleast_1d(self.tree.query_nearest(edge_point, all_matches=True)).astype(int)
        else:
            indices = np.atleast_1d(self.tree.query_nearest(point, all_matches=True)).astype(int)
        if len(indices) == 0:
            raise RuntimeError("nearest-polygon lookup returned no candidates")
        candidates = []
        for index in indices:
            row = self.footprints.iloc[int(index)]
            geometry = row.geometry
            distance = float(edge_point.distance(geometry.boundary)) if inside else float(point.distance(geometry))
            candidates.append((distance, str(row.building_id), int(index)))
        _, building_id, nearest_index = min(candidates, key=lambda item: (item[0], item[1]))
        nearest_row = self.footprints.iloc[nearest_index]
        if not inside:
            edge_point = nearest_points(point, nearest_row.geometry.boundary)[1]
        bearing = math.atan2(edge_point.y - point.y, edge_point.x - point.x) % math.tau
        signed_distance = -union_edge_distance if inside else union_edge_distance
        return ExactLabel(
            signed_distance_m=signed_distance,
            nearest_polygon_id=building_id,
            nearest_polygon_provenance=_provenance_for(nearest_row),
            bearing_radians=bearing,
            nearest_edge_x=float(edge_point.x),
            nearest_edge_y=float(edge_point.y),
        )


def exact_label(point: Point, footprints: gpd.GeoDataFrame) -> ExactLabel:
    return _VectorLabelIndex(footprints).label(point)


def build_scene_grid(
    footprints: gpd.GeoDataFrame,
    source_id: str,
    shape: tuple[int, int],
    transform: Affine,
) -> SceneGrid:
    _assert_native_transform(transform)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("scene shape must contain two positive dimensions")
    if footprints.empty:
        raise ValueError("scene requires footprints")

    ordered = footprints.assign(_building_id=footprints["building_id"].astype(str)).sort_values(
        "_building_id", kind="stable"
    )
    building_ids = ordered["_building_id"].tolist()
    if source_id not in building_ids:
        raise ValueError(f"source {source_id!r} is absent from the scene footprints")
    id_by_code: tuple[str | None, ...] = (None, *building_ids)
    code_by_id = {building_id: code for code, building_id in enumerate(id_by_code) if building_id is not None}
    shapes = [
        (geometry, code_by_id[building_id])
        for building_id, geometry in zip(ordered["_building_id"], ordered.geometry, strict=True)
        if geometry is not None and not geometry.is_empty
    ]
    labels = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")
    source_geometry = ordered.loc[ordered["_building_id"] == source_id].geometry.iloc[0]
    source_footprint_mask = rasterize(
        [(source_geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    inside = labels > 0
    if not inside.any():
        raise GridInvariantError("no footprint covers a native scene pixel center")

    sampling = (abs(transform.e), transform.a)
    outside_distance, nearest_inside = ndimage.distance_transform_edt(
        ~inside,
        sampling=sampling,
        return_indices=True,
    )
    inside_distance = ndimage.distance_transform_edt(inside, sampling=sampling)
    signed_distance = outside_distance.astype(np.float32)
    signed_distance[inside] = -inside_distance[inside]

    owners = labels.copy()
    outside_rows, outside_cols = np.nonzero(~inside)
    owners[outside_rows, outside_cols] = labels[
        nearest_inside[0, outside_rows, outside_cols],
        nearest_inside[1, outside_rows, outside_cols],
    ]
    return SceneGrid(
        signed_distance_m=signed_distance,
        owner_codes=owners,
        transform=transform,
        source_code=code_by_id[source_id],
        id_by_code=id_by_code,
        source_footprint_mask=source_footprint_mask,
    )


def accept_exact(distance_m: float, distance_bin: DistanceBin) -> bool:
    return math.isfinite(distance_m) and distance_bin.contains(distance_m)


def _bin_mask(values: np.ndarray, distance_bin: DistanceBin) -> np.ndarray:
    upper = values <= distance_bin.upper_m if distance_bin.include_upper else values < distance_bin.upper_m
    return (values >= distance_bin.lower_m) & upper & (np.abs(values) >= 0.3)


def _crop_valid_mask(
    shape: tuple[int, int],
    crop_size: int,
    tip_row: int,
    tip_col: int,
) -> np.ndarray:
    if not 0 <= tip_row < crop_size or not 0 <= tip_col < crop_size:
        raise ValueError("tip must lie inside the output crop")
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return (
        (rows >= tip_row)
        & (rows <= shape[0] - crop_size + tip_row)
        & (cols >= tip_col)
        & (cols <= shape[1] - crop_size + tip_col)
    )


def _rng_for(seed: int, source_id: str, bin_id: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{source_id}:{bin_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _point_for_slot(grid: SceneGrid, row: int, col: int) -> Point:
    x, y = xy(grid.transform, row, col, offset="center")
    return Point(float(x), float(y))


def _exact_extreme(grid: SceneGrid, labels: _VectorLabelIndex, mask: np.ndarray, *, maximum: bool) -> float:
    values = np.where(mask, grid.signed_distance_m, -np.inf if maximum else np.inf)
    flat_index = int(np.argmax(values) if maximum else np.argmin(values))
    row, col = np.unravel_index(flat_index, grid.shape)
    return labels.label(_point_for_slot(grid, int(row), int(col))).signed_distance_m


def sample_source_bins(
    grid: SceneGrid,
    footprints: gpd.GeoDataFrame,
    source_id: str,
    bins: tuple[DistanceBin, ...],
    seed: int,
    *,
    slot_tolerance_m: float = 0.3,
    crop_size: int = 512,
    tip_row: int = 256,
    tip_col: int = 256,
    maximum_target_draws: int = 128,
) -> SourceBinResult:
    if source_id != grid.id_by_code[grid.source_code]:
        raise GridInvariantError("source ID does not match the ownership raster lookup")
    if slot_tolerance_m < 0:
        raise ValueError("slot tolerance must be non-negative")

    source_owned = grid.owner_codes == grid.source_code
    crop_valid = _crop_valid_mask(grid.shape, crop_size, tip_row, tip_col)
    vector_labels = _VectorLabelIndex(footprints)
    achievable = source_owned & crop_valid
    if achievable.any():
        innermost = _exact_extreme(grid, vector_labels, achievable, maximum=False)
        outermost = _exact_extreme(grid, vector_labels, achievable, maximum=True)
    else:
        innermost = math.nan
        outermost = math.nan

    pins_by_bin: dict[int, tuple[PinCandidate, ...]] = {}
    unavailable: dict[int, UnavailabilityReason] = {}
    for distance_bin in bins:
        all_bin_slots = _bin_mask(grid.signed_distance_m, distance_bin)
        owned_bin_slots = all_bin_slots & source_owned
        crop_bin_slots = owned_bin_slots & crop_valid
        if not owned_bin_slots.any() and distance_bin.bin_id < 0:
            source_inside_slots = (
                all_bin_slots & grid.source_footprint_mask
                if grid.source_footprint_mask is not None
                else owned_bin_slots
            )
            unavailable[distance_bin.bin_id] = "ownership" if source_inside_slots.any() else "size"
            continue
        if not all_bin_slots.any():
            unavailable[distance_bin.bin_id] = "crop"
            continue
        if not owned_bin_slots.any():
            unavailable[distance_bin.bin_id] = "ownership"
            continue
        if not crop_bin_slots.any():
            unavailable[distance_bin.bin_id] = "crop"
            continue

        rng = _rng_for(seed, source_id, distance_bin.bin_id)
        chosen_slots: set[tuple[int, int]] = set()
        pins: list[PinCandidate] = []
        required = repeats_for_bin(distance_bin.bin_id)
        for repeat_index in range(required):
            accepted: PinCandidate | None = None
            for _ in range(maximum_target_draws):
                target = sample_log_uniform_target(distance_bin, rng)
                matches = np.argwhere(crop_bin_slots & (np.abs(grid.signed_distance_m - target) <= slot_tolerance_m))
                choices = [(int(row), int(col)) for row, col in matches if (int(row), int(col)) not in chosen_slots]
                rng.shuffle(choices)
                for row, col in choices:
                    point = _point_for_slot(grid, row, col)
                    label = vector_labels.label(point)
                    if not accept_exact(label.signed_distance_m, distance_bin):
                        continue
                    accepted = PinCandidate(
                        source_id=source_id,
                        bin_id=distance_bin.bin_id,
                        repeat_index=repeat_index,
                        target_signed_distance_m=target,
                        row=row,
                        col=col,
                        x=point.x,
                        y=point.y,
                        exact_signed_distance_m=label.signed_distance_m,
                        nearest_polygon_id=label.nearest_polygon_id,
                        nearest_polygon_provenance=label.nearest_polygon_provenance,
                        bearing_radians=label.bearing_radians,
                    )
                    chosen_slots.add((row, col))
                    break
                if accepted is not None:
                    break
            if accepted is None:
                pins = []
                break
            pins.append(accepted)

        if pins:
            pins_by_bin[distance_bin.bin_id] = tuple(pins)
        else:
            unavailable[distance_bin.bin_id] = (
                "size"
                if distance_bin.bin_id < 0
                and grid.source_footprint_mask is not None
                and math.isfinite(innermost)
                and innermost > distance_bin.upper_m
                else "other"
            )

    return SourceBinResult(
        source_id=source_id,
        pins_by_bin=pins_by_bin,
        unavailable_reasons=unavailable,
        innermost_signed_distance_m=innermost,
        outermost_signed_distance_m=outermost,
    )
