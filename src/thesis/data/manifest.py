"""Normalized construction manifest with a v1-compatible sample view (D25)."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

import duckdb
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from thesis.geometry.bins import distance_bins, repeats_for_bin

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Latitude = Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
Longitude = Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]
Season = Literal["winter", "spring", "summer", "autumn"]
Reason = Literal["size", "ownership", "crop", "other"]
Assignment = Literal["main", "ablation"]
Stratum = Literal["small", "medium", "large"]

BIN_IDS = tuple(distance_bin.bin_id for distance_bin in distance_bins())
EXPECTED_SEEDS = {
    "box_quota",
    "screen_subsample",
    "candidate_order",
    "pin_sampling",
    "ablation_assignment",
    "deficit_spill",
    "delta_reduction",
    "sanity_render",
}


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RunRecord(StrictRecord):
    schema_version: Literal[2]
    run_id: NonEmptyStr
    created_at: datetime
    config_hash: Sha256
    git_commit: GitCommit
    code_version: Sha256
    overture_release: NonEmptyStr
    area_log_cuts: tuple[FiniteFloat, FiniteFloat]
    coverage_threshold: Literal[0.5]
    screen_per_box_cap: PositiveInt | None
    spill_revisit_fraction: Literal[0.05]
    delta_m: FiniteFloat | tuple[FiniteFloat, FiniteFloat] | None
    census_urban_vintage: NonEmptyStr
    ecoregion_artifact: NonEmptyStr
    seeds: dict[str, int]
    ecoregion_merge_mapping: dict[str, str]

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) != EXPECTED_SEEDS:
            raise ValueError(f"seeds must be exactly {sorted(EXPECTED_SEEDS)}")
        return value

    @model_validator(mode="after")
    def validate_cuts(self) -> Self:
        if self.area_log_cuts[0] >= self.area_log_cuts[1]:
            raise ValueError("area_log_cuts must be strictly increasing")
        return self


class BoxRecord(StrictRecord):
    run_id: NonEmptyStr
    box_id: NonEmptyStr
    ecoregion_code: NonEmptyStr
    ecoregion_name: NonEmptyStr
    state: NonEmptyStr
    center_rural: bool
    full_pool_count: NonNegativeInt
    content_pool_count: NonNegativeInt
    rural_content_pool_count: NonNegativeInt
    eligible_count: NonNegativeInt
    screened_count: NonNegativeInt
    scene_rural_loss: NonNegativeInt
    box_containment_loss: NonNegativeInt
    tile_containment_loss: NonNegativeInt
    spacing_loss: NonNegativeInt
    main_quota: NonNegativeInt
    realized_main: NonNegativeInt
    spill_in: NonNegativeInt
    spill_out: NonNegativeInt
    zero_eligible_reason: str | None
    box_selection_probability: PositiveFloat
    imagery_dates: tuple[date, ...]
    imagery_seasons: tuple[Season, ...]
    footprint_provenance_mix: dict[str, NonNegativeFloat]

    @model_validator(mode="after")
    def validate_counts_and_mix(self) -> Self:
        if not self.eligible_count <= self.rural_content_pool_count <= self.content_pool_count <= self.full_pool_count:
            raise ValueError("box pool counts must be non-increasing")
        if self.screened_count > self.eligible_count:
            raise ValueError("screened count exceeds eligible count")
        if self.eligible_count == 0 and not self.zero_eligible_reason:
            raise ValueError("a zero-eligible box requires a reason")
        if self.eligible_count > 0 and self.zero_eligible_reason is not None:
            raise ValueError("a nonempty box cannot have a zero-eligible reason")
        if self.footprint_provenance_mix and not math.isclose(
            sum(self.footprint_provenance_mix.values()), 1.0, abs_tol=1e-9
        ):
            raise ValueError("footprint provenance mixture must sum to one")
        if set(self.imagery_seasons) != {self._season(value) for value in self.imagery_dates}:
            raise ValueError("imagery seasons must match imagery dates")
        return self

    @staticmethod
    def _season(value: date) -> str:
        if value.month in (12, 1, 2):
            return "winter"
        if value.month in (3, 4, 5):
            return "spring"
        if value.month in (6, 7, 8):
            return "summer"
        return "autumn"


class SourceRecord(StrictRecord):
    run_id: NonEmptyStr
    source_id: NonEmptyStr
    box_id: NonEmptyStr
    quota_box_id: NonEmptyStr
    ecoregion_code: NonEmptyStr
    building_stratum: Stratum
    assignment: Assignment
    state: NonEmptyStr
    source_footprint_provenance: NonEmptyStr
    footprint_area_m2: PositiveFloat
    source_anchor_lat: Latitude
    source_anchor_lon: Longitude
    nearest_neighbor_distance_m: NonNegativeFloat
    neighbors_within_80m: NonNegativeInt
    scene_id: NonEmptyStr
    naip_item_id: NonEmptyStr
    naip_source_crs: NonEmptyStr
    naip_item_transform: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    imagery_date: date
    imagery_season: Season
    bin_availability: dict[str, bool]
    unavailability_reasons: dict[str, Reason]
    innermost_signed_distance_m: FiniteFloat
    outermost_signed_distance_m: FiniteFloat
    complete_source: bool
    complete_inside: bool
    scene_rural: bool
    rejection_provenance: tuple[NonEmptyStr, ...]
    scene_window: tuple[int, int, PositiveInt, PositiveInt]
    scene_transform: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        expected_keys = {str(bin_id) for bin_id in BIN_IDS}
        if set(self.bin_availability) != expected_keys:
            raise ValueError("bin_availability must contain all ten canonical bin IDs")
        unavailable = {key for key, available in self.bin_availability.items() if not available}
        if set(self.unavailability_reasons) != unavailable:
            raise ValueError("unavailability reasons must match unavailable bins exactly")
        if not any(self.bin_availability.values()):
            raise ValueError("selected manifest sources must have at least one available bin")
        if self.complete_source != all(self.bin_availability.values()):
            raise ValueError("complete_source must equal all(bin_availability)")
        inside_keys = {"-5", "-3", "-1"}
        if self.complete_inside != all(self.bin_availability[key] for key in inside_keys):
            raise ValueError("complete_inside must equal availability of all inside bins")
        if self.innermost_signed_distance_m > self.outermost_signed_distance_m:
            raise ValueError("achievable distance extrema are reversed")
        expected_season = BoxRecord._season(self.imagery_date)
        if self.imagery_season != expected_season:
            raise ValueError("imagery season must match imagery date")
        return self


class PinRecord(StrictRecord):
    run_id: NonEmptyStr
    pin_id: NonEmptyStr
    source_id: NonEmptyStr
    bin_id: int
    bin_lower_m: FiniteFloat
    bin_upper_m: FiniteFloat
    repeat_index: NonNegativeInt
    target_signed_distance_m: FiniteFloat
    exact_signed_distance_m: FiniteFloat
    ground_truth_inside: bool
    tip_row: Literal[256]
    tip_col: Literal[256]
    tip_x: FiniteFloat
    tip_y: FiniteFloat
    tip_lat: Latitude
    tip_lon: Longitude
    owning_polygon_id: NonEmptyStr
    nearest_polygon_id: NonEmptyStr
    nearest_polygon_provenance: NonEmptyStr
    nearest_edge_bearing_radians: Annotated[float, Field(ge=0, lt=math.tau, allow_inf_nan=False)]
    radial_displacement_m: NonNegativeFloat
    source_to_pin_bearing_radians: Annotated[float, Field(ge=0, lt=math.tau, allow_inf_nan=False)]
    crop_window: tuple[NonNegativeInt, NonNegativeInt, Literal[512], Literal[512]]
    crop_transform: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

    @model_validator(mode="after")
    def validate_distance(self) -> Self:
        canonical = {distance_bin.bin_id: distance_bin for distance_bin in distance_bins()}
        if self.bin_id not in canonical:
            raise ValueError("unknown bin ID")
        distance_bin = canonical[self.bin_id]
        if not math.isclose(self.bin_lower_m, distance_bin.lower_m, abs_tol=1e-12) or not math.isclose(
            self.bin_upper_m, distance_bin.upper_m, abs_tol=1e-12
        ):
            raise ValueError("stored bin bounds do not match canonical bounds")
        if not distance_bin.contains(self.target_signed_distance_m):
            raise ValueError("sampled target is outside its bin")
        if not distance_bin.contains(self.exact_signed_distance_m):
            raise ValueError("exact signed distance is outside its bin or exclusion gap")
        if self.ground_truth_inside != (self.exact_signed_distance_m < 0):
            raise ValueError("ground-truth sign does not match exact signed distance")
        if self.repeat_index >= repeats_for_bin(self.bin_id):
            raise ValueError("repeat index is invalid for bin")
        return self


class RenderRecord(StrictRecord):
    run_id: NonEmptyStr
    render_id: NonEmptyStr
    pin_id: NonEmptyStr
    preset_id: NonEmptyStr
    marker_shape: NonEmptyStr
    marker_color: NonEmptyStr
    marker_size_px: PositiveInt
    marker_opacity: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    prompt_variant: NonEmptyStr
    image_path: NonEmptyStr
    image_sha256: Sha256

    @field_validator("image_path")
    @classmethod
    def relative_image_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("image_path must be relative to the dataset root")
        return value


class DeviationRecord(StrictRecord):
    run_id: NonEmptyStr
    deviation_id: NonEmptyStr
    deviation_type: NonEmptyStr
    source_id: NonEmptyStr | None
    origin_box_id: NonEmptyStr | None
    destination_box_id: NonEmptyStr | None
    details: dict[str, object]

    @model_validator(mode="after")
    def validate_spill(self) -> Self:
        if self.deviation_type == "spill" and (self.origin_box_id is None or self.destination_box_id is None):
            raise ValueError("spill deviations require origin and destination boxes")
        return self


class ManifestBundle(StrictRecord):
    run: RunRecord
    boxes: tuple[BoxRecord, ...]
    sources: tuple[SourceRecord, ...]
    pins: tuple[PinRecord, ...]
    renders: tuple[RenderRecord, ...]
    deviations: tuple[DeviationRecord, ...]


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        schema_version INTEGER, run_id VARCHAR PRIMARY KEY, created_at TIMESTAMPTZ,
        config_hash VARCHAR, git_commit VARCHAR, code_version VARCHAR, overture_release VARCHAR,
        area_log_cuts DOUBLE[], coverage_threshold DOUBLE, screen_per_box_cap INTEGER,
        spill_revisit_fraction DOUBLE, delta_m JSON, census_urban_vintage VARCHAR,
        ecoregion_artifact VARCHAR, seeds MAP(VARCHAR, BIGINT),
        ecoregion_merge_mapping MAP(VARCHAR, VARCHAR)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS boxes (
        run_id VARCHAR, box_id VARCHAR, ecoregion_code VARCHAR, ecoregion_name VARCHAR,
        state VARCHAR, center_rural BOOLEAN, full_pool_count INTEGER, content_pool_count INTEGER,
        rural_content_pool_count INTEGER, eligible_count INTEGER, screened_count INTEGER,
        scene_rural_loss INTEGER, box_containment_loss INTEGER, tile_containment_loss INTEGER,
        spacing_loss INTEGER, main_quota INTEGER, realized_main INTEGER, spill_in INTEGER,
        spill_out INTEGER, zero_eligible_reason VARCHAR, box_selection_probability DOUBLE,
        imagery_dates DATE[], imagery_seasons VARCHAR[],
        footprint_provenance_mix MAP(VARCHAR, DOUBLE), PRIMARY KEY (run_id, box_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sources (
        run_id VARCHAR, source_id VARCHAR PRIMARY KEY, box_id VARCHAR, quota_box_id VARCHAR,
        ecoregion_code VARCHAR, building_stratum VARCHAR, assignment VARCHAR, state VARCHAR,
        source_footprint_provenance VARCHAR, footprint_area_m2 DOUBLE,
        source_anchor_lat DOUBLE, source_anchor_lon DOUBLE, nearest_neighbor_distance_m DOUBLE,
        neighbors_within_80m INTEGER, scene_id VARCHAR, naip_item_id VARCHAR,
        naip_source_crs VARCHAR, naip_item_transform DOUBLE[], imagery_date DATE,
        imagery_season VARCHAR, bin_availability MAP(VARCHAR, BOOLEAN),
        unavailability_reasons MAP(VARCHAR, VARCHAR), innermost_signed_distance_m DOUBLE,
        outermost_signed_distance_m DOUBLE, complete_source BOOLEAN, complete_inside BOOLEAN,
        scene_rural BOOLEAN, rejection_provenance VARCHAR[], scene_window INTEGER[],
        scene_transform DOUBLE[]
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pins (
        run_id VARCHAR, pin_id VARCHAR PRIMARY KEY, source_id VARCHAR, bin_id INTEGER,
        bin_lower_m DOUBLE, bin_upper_m DOUBLE, repeat_index INTEGER,
        target_signed_distance_m DOUBLE, exact_signed_distance_m DOUBLE,
        ground_truth_inside BOOLEAN, tip_row INTEGER, tip_col INTEGER, tip_x DOUBLE, tip_y DOUBLE,
        tip_lat DOUBLE, tip_lon DOUBLE, owning_polygon_id VARCHAR,
        nearest_polygon_id VARCHAR, nearest_polygon_provenance VARCHAR,
        nearest_edge_bearing_radians DOUBLE, radial_displacement_m DOUBLE,
        source_to_pin_bearing_radians DOUBLE, crop_window INTEGER[], crop_transform DOUBLE[]
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS renders (
        run_id VARCHAR, render_id VARCHAR PRIMARY KEY, pin_id VARCHAR, preset_id VARCHAR,
        marker_shape VARCHAR, marker_color VARCHAR, marker_size_px INTEGER,
        marker_opacity DOUBLE, prompt_variant VARCHAR,
        image_path VARCHAR, image_sha256 VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS deviations (
        run_id VARCHAR, deviation_id VARCHAR PRIMARY KEY, deviation_type VARCHAR,
        source_id VARCHAR, origin_box_id VARCHAR, destination_box_id VARCHAR, details JSON
    )
    """,
)


def _validate_foreign_keys(bundle: ManifestBundle) -> None:
    run_id = bundle.run.run_id
    collections = (bundle.boxes, bundle.sources, bundle.pins, bundle.renders, bundle.deviations)
    if any(record.run_id != run_id for records in collections for record in records):
        raise ValueError("every manifest record must reference the bundle run")
    box_ids = {record.box_id for record in bundle.boxes}
    for source in bundle.sources:
        if source.box_id not in box_ids or source.quota_box_id not in box_ids:
            raise ValueError(f"source {source.source_id} references an unknown box")
    source_ids = {record.source_id for record in bundle.sources}
    for pin in bundle.pins:
        if pin.source_id not in source_ids:
            raise ValueError(f"pin {pin.pin_id} references an unknown source")
    pin_ids = {record.pin_id for record in bundle.pins}
    for render in bundle.renders:
        if render.pin_id not in pin_ids:
            raise ValueError(f"render {render.render_id} references an unknown pin")


def _values(record: BaseModel) -> tuple:
    result = []
    for field_name, value in record.model_dump(mode="python").items():
        scalar_types = (str, int, float, bool, type(None))
        if isinstance(value, dict) and any(not isinstance(item, scalar_types) for item in value.values()):
            result.append(json.dumps(value, sort_keys=True))
        elif field_name == "delta_m" and isinstance(value, tuple):
            result.append(json.dumps(value))
        else:
            result.append(value)
    return tuple(result)


def _insert_many(connection, table: str, records: Iterable[BaseModel]) -> None:
    rows = list(records)
    if not rows:
        return
    columns = tuple(type(rows[0]).model_fields)
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        [_values(record) for record in rows],
    )


_CREATE_COMPATIBILITY_VIEW = """
CREATE OR REPLACE VIEW samples AS
SELECT
    2 AS manifest_schema_version,
    v.render_id AS sample_id,
    v.image_path,
    r.config_hash,
    r.git_commit,
    s.source_id AS building_id,
    s.box_id AS bbox_name,
    s.state,
    s.ecoregion_code,
    b.box_selection_probability AS bbox_selection_probability,
    b.eligible_count AS eligible_buildings_in_bbox,
    b.realized_main AS sampled_buildings_in_bbox,
    1.0::DOUBLE AS sampling_weight,
    r.overture_release,
    s.source_footprint_provenance AS footprint_source,
    s.footprint_area_m2,
    s.source_anchor_lat AS target_anchor_lat,
    s.source_anchor_lon AS target_anchor_lon,
    'fixed_level' AS displacement_role,
    p.target_signed_distance_m AS target_signed_edge_distance_m,
    p.exact_signed_distance_m AS actual_signed_edge_distance_m,
    p.radial_displacement_m,
    p.source_to_pin_bearing_radians AS bearing_radians,
    p.tip_lat AS pin_lat,
    p.tip_lon AS pin_lon,
    p.ground_truth_inside,
    TRUE AS include_in_distance_curve,
    s.naip_item_id,
    year(s.imagery_date)::INTEGER AS naip_year,
    lower(s.state) AS naip_state,
    0.6::DOUBLE AS naip_native_gsd_x_m,
    0.6::DOUBLE AS naip_native_gsd_y_m,
    s.naip_source_crs,
    0.6::DOUBLE AS rendered_gsd_m,
    512 AS image_px,
    v.marker_shape,
    v.marker_color,
    v.marker_size_px,
    v.marker_opacity,
    r.census_urban_vintage AS census_urban_artifact,
    0.0::DOUBLE AS census_buffer_m,
    p.pin_id,
    p.bin_id,
    p.repeat_index,
    p.nearest_polygon_id,
    p.nearest_polygon_provenance,
    p.nearest_edge_bearing_radians,
    v.prompt_variant,
    v.image_sha256
FROM renders v
JOIN pins p USING (run_id, pin_id)
JOIN sources s USING (run_id, source_id)
JOIN boxes b USING (run_id, box_id)
JOIN runs r USING (run_id)
"""


def write_manifest(path: str | Path, bundle: ManifestBundle) -> None:
    bundle = ManifestBundle.model_validate(bundle)
    _validate_foreign_keys(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    connection = duckdb.connect(str(temporary))
    try:
        connection.execute("BEGIN")
        for statement in _SCHEMA:
            connection.execute(statement)
        _insert_many(connection, "runs", (bundle.run,))
        _insert_many(connection, "boxes", bundle.boxes)
        _insert_many(connection, "sources", bundle.sources)
        _insert_many(connection, "pins", bundle.pins)
        _insert_many(connection, "renders", bundle.renders)
        _insert_many(connection, "deviations", bundle.deviations)
        connection.execute(_CREATE_COMPATIBILITY_VIEW)
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    os.replace(temporary, path)


def append_renders(path: str | Path, records: Iterable[RenderRecord | dict]) -> None:
    validated = tuple(RenderRecord.model_validate(record) for record in records)
    if not validated:
        return
    with duckdb.connect(str(path)) as connection:
        run_ids = {row[0] for row in connection.execute("SELECT run_id FROM runs").fetchall()}
        pin_ids = {row[0] for row in connection.execute("SELECT pin_id FROM pins").fetchall()}
        if any(record.run_id not in run_ids or record.pin_id not in pin_ids for record in validated):
            raise ValueError("render references an unknown run or pin")
        connection.execute("BEGIN")
        try:
            _insert_many(connection, "renders", validated)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def _read_manifest(path: str | Path) -> tuple[tuple[str, ...], dict[str, tuple]]:
    with duckdb.connect(str(path), read_only=True) as connection:
        columns = tuple(row[1] for row in connection.execute("PRAGMA table_info('samples')").fetchall())
        sample_id_index = columns.index("sample_id")
        rows = connection.execute("SELECT * FROM samples ORDER BY sample_id").fetchall()
    return columns, {row[sample_id_index]: row for row in rows}


def diff_manifests(first: str | Path, second: str | Path) -> list[str]:
    first_columns, first_rows = _read_manifest(first)
    second_columns, second_rows = _read_manifest(second)
    if first_columns != second_columns:
        return ["schema changed"]
    differences = []
    for sample_id in sorted(first_rows.keys() | second_rows.keys()):
        if sample_id not in second_rows:
            differences.append(f"removed: {sample_id}")
        elif sample_id not in first_rows:
            differences.append(f"added: {sample_id}")
        elif first_rows[sample_id] != second_rows[sample_id]:
            differences.append(f"changed: {sample_id}")
    return differences
