import hashlib
import json
from collections.abc import Iterable, Mapping
from math import tau
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

import duckdb
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(gt=0, le=1)]
UnitInterval = Annotated[float, Field(ge=0, le=1)]
PositiveInt = Annotated[int, Field(gt=0)]
Latitude = Annotated[float, Field(ge=-90, le=90)]
Longitude = Annotated[float, Field(ge=-180, le=180)]
Bearing = Annotated[float, Field(ge=0, lt=tau)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
DisplacementRole = Literal["fixed_level", "deep_inside_control"]

_CREATE_SAMPLES = """
CREATE TABLE IF NOT EXISTS samples (
    manifest_schema_version INTEGER NOT NULL CHECK (manifest_schema_version = 1),
    sample_id VARCHAR PRIMARY KEY,
    image_path VARCHAR NOT NULL,
    config_hash VARCHAR NOT NULL,
    git_commit VARCHAR NOT NULL,
    building_id VARCHAR NOT NULL,
    bbox_name VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    ecoregion_code VARCHAR NOT NULL,
    bbox_selection_probability DOUBLE NOT NULL,
    eligible_buildings_in_bbox INTEGER NOT NULL,
    sampled_buildings_in_bbox INTEGER NOT NULL,
    sampling_weight DOUBLE NOT NULL,
    overture_release VARCHAR NOT NULL,
    footprint_source VARCHAR NOT NULL,
    footprint_area_m2 DOUBLE NOT NULL,
    target_anchor_lat DOUBLE NOT NULL,
    target_anchor_lon DOUBLE NOT NULL,
    displacement_role VARCHAR NOT NULL,
    target_signed_edge_distance_m DOUBLE,
    actual_signed_edge_distance_m DOUBLE NOT NULL,
    radial_displacement_m DOUBLE NOT NULL,
    bearing_radians DOUBLE,
    pin_lat DOUBLE NOT NULL,
    pin_lon DOUBLE NOT NULL,
    ground_truth_inside BOOLEAN NOT NULL,
    include_in_distance_curve BOOLEAN NOT NULL,
    naip_item_id VARCHAR NOT NULL,
    naip_year INTEGER NOT NULL,
    naip_state VARCHAR NOT NULL,
    naip_native_gsd_x_m DOUBLE NOT NULL,
    naip_native_gsd_y_m DOUBLE NOT NULL,
    naip_source_crs VARCHAR NOT NULL,
    rendered_gsd_m DOUBLE NOT NULL,
    image_px INTEGER NOT NULL,
    marker_shape VARCHAR NOT NULL,
    marker_color VARCHAR NOT NULL,
    marker_size_px INTEGER NOT NULL,
    marker_opacity DOUBLE NOT NULL,
    census_urban_artifact VARCHAR NOT NULL,
    census_buffer_m DOUBLE NOT NULL
)
"""


class ManifestSchemaError(RuntimeError):
    pass


def sample_id_for(
    building_id: str,
    displacement_role: DisplacementRole,
    target_signed_edge_distance_m: float | None,
    marker_shape: str,
    marker_color: str,
) -> str:
    identity = [building_id, displacement_role, target_signed_edge_distance_m, marker_shape, marker_color]
    encoded = json.dumps(identity, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    manifest_schema_version: Literal[1]
    sample_id: Sha256
    image_path: NonEmptyStr
    config_hash: Sha256
    git_commit: GitCommit

    building_id: NonEmptyStr
    bbox_name: NonEmptyStr
    state: NonEmptyStr
    ecoregion_code: NonEmptyStr
    bbox_selection_probability: Probability
    eligible_buildings_in_bbox: PositiveInt
    sampled_buildings_in_bbox: PositiveInt
    sampling_weight: PositiveFloat

    overture_release: NonEmptyStr
    footprint_source: NonEmptyStr
    footprint_area_m2: PositiveFloat
    target_anchor_lat: Latitude
    target_anchor_lon: Longitude

    displacement_role: DisplacementRole
    target_signed_edge_distance_m: float | None
    actual_signed_edge_distance_m: float
    radial_displacement_m: NonNegativeFloat
    bearing_radians: Bearing | None
    pin_lat: Latitude
    pin_lon: Longitude
    ground_truth_inside: bool
    include_in_distance_curve: bool

    naip_item_id: NonEmptyStr
    naip_year: PositiveInt
    naip_state: NonEmptyStr
    naip_native_gsd_x_m: PositiveFloat
    naip_native_gsd_y_m: PositiveFloat
    naip_source_crs: NonEmptyStr

    rendered_gsd_m: PositiveFloat
    image_px: PositiveInt
    marker_shape: NonEmptyStr
    marker_color: NonEmptyStr
    marker_size_px: PositiveInt
    marker_opacity: UnitInterval
    census_urban_artifact: NonEmptyStr
    census_buffer_m: NonNegativeFloat

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError("image_path must be relative to the dataset root")
        return value

    @model_validator(mode="after")
    def validate_displacement_role(self) -> Self:
        if self.displacement_role == "fixed_level":
            valid = (
                self.target_signed_edge_distance_m is not None
                and self.bearing_radians is not None
                and self.include_in_distance_curve
            )
        else:
            valid = (
                self.target_signed_edge_distance_m is None
                and self.bearing_radians is None
                and not self.include_in_distance_curve
            )
        if not valid:
            raise ValueError("displacement fields do not match displacement_role")
        return self

    @model_validator(mode="after")
    def validate_ground_truth(self) -> Self:
        if self.ground_truth_inside != (self.actual_signed_edge_distance_m <= 0):
            raise ValueError("ground truth does not match actual signed edge distance")
        if self.displacement_role == "deep_inside_control" and self.actual_signed_edge_distance_m >= 0:
            raise ValueError("deep-inside control must be inside the footprint")
        if self.sampled_buildings_in_bbox > self.eligible_buildings_in_bbox:
            raise ValueError("sampled building count exceeds eligible building count")
        return self

    @model_validator(mode="after")
    def validate_sample_id(self) -> Self:
        expected = sample_id_for(
            building_id=self.building_id,
            displacement_role=self.displacement_role,
            target_signed_edge_distance_m=self.target_signed_edge_distance_m,
            marker_shape=self.marker_shape,
            marker_color=self.marker_color,
        )
        if self.sample_id != expected:
            raise ValueError("sample_id does not match semantic sample identity")
        return self


def write_manifest(path: str | Path, rows: Iterable[ManifestRow | Mapping[str, object]]) -> None:
    validated = [ManifestRow.model_validate(row) for row in rows]
    columns = tuple(ManifestRow.model_fields)
    placeholders = ", ".join("?" for _ in columns)
    values = [tuple(getattr(row, column) for column in columns) for row in validated]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        connection.execute("BEGIN")
        connection.execute(_CREATE_SAMPLES)
        stored_columns = tuple(row[1] for row in connection.execute("PRAGMA table_info('samples')").fetchall())
        if stored_columns != columns:
            raise ManifestSchemaError("samples table does not match manifest schema v1")
        if values:
            connection.executemany(
                f"INSERT INTO samples ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
