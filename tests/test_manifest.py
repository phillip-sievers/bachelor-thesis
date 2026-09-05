import duckdb
import pytest
from pydantic import ValidationError


def fixed_level_row(**overrides):
    row = {
        "manifest_schema_version": 1,
        "sample_id": "sample-1",
        "image_path": "images/sample-1.png",
        "config_hash": "a" * 64,
        "git_commit": "b" * 40,
        "building_id": "building-1",
        "bbox_name": "eco_9_2_box_01",
        "state": "KS",
        "ecoregion_code": "9.2",
        "bbox_selection_probability": 0.01,
        "eligible_buildings_in_bbox": 100,
        "sampled_buildings_in_bbox": 20,
        "sampling_weight": 500.0,
        "overture_release": "2026-08-19.0",
        "footprint_source": "overture",
        "footprint_area_m2": 150.0,
        "target_anchor_lat": 38.5,
        "target_anchor_lon": -98.0,
        "displacement_role": "fixed_level",
        "target_signed_edge_distance_m": 1.0,
        "actual_signed_edge_distance_m": 1.02,
        "radial_displacement_m": 8.0,
        "bearing_radians": 1.5,
        "pin_lat": 38.50001,
        "pin_lon": -97.99999,
        "ground_truth_inside": False,
        "include_in_distance_curve": True,
        "naip_item_id": "m_3809801_ne_14_060_20220830",
        "naip_year": 2022,
        "naip_state": "ks",
        "naip_native_gsd_x_m": 0.6,
        "naip_native_gsd_y_m": 0.6,
        "naip_source_crs": "EPSG:26914",
        "rendered_gsd_m": 0.6,
        "image_px": 512,
        "marker_shape": "google_pin",
        "marker_color": "red",
        "marker_size_px": 15,
        "marker_opacity": 1.0,
        "census_urban_artifact": "tl_2025_us_uac20.zip",
        "census_buffer_m": 100.0,
    }
    row.update(overrides)
    if "sample_id" not in overrides:
        from thesis.data.manifest import sample_id_for

        row["sample_id"] = sample_id_for(
            building_id=row["building_id"],
            displacement_role=row["displacement_role"],
            target_signed_edge_distance_m=row["target_signed_edge_distance_m"],
            marker_shape=row["marker_shape"],
            marker_color=row["marker_color"],
        )
    return row


def test_manifest_row_accepts_the_locked_fixed_level_schema():
    from thesis.data.manifest import ManifestRow

    row = ManifestRow.model_validate(fixed_level_row())

    assert row.manifest_schema_version == 1
    assert row.target_signed_edge_distance_m == pytest.approx(1.0)


def test_manifest_row_accepts_the_deep_inside_control():
    from thesis.data.manifest import ManifestRow

    row = ManifestRow.model_validate(
        fixed_level_row(
            displacement_role="deep_inside_control",
            target_signed_edge_distance_m=None,
            actual_signed_edge_distance_m=-8.2,
            bearing_radians=None,
            ground_truth_inside=True,
            include_in_distance_curve=False,
        )
    )

    assert row.target_signed_edge_distance_m is None
    assert row.bearing_radians is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"displacement_role": "unknown"},
        {"target_signed_edge_distance_m": None},
        {"bearing_radians": None},
        {"include_in_distance_curve": False},
        {
            "displacement_role": "deep_inside_control",
            "target_signed_edge_distance_m": -5.0,
            "actual_signed_edge_distance_m": -8.2,
            "bearing_radians": None,
            "ground_truth_inside": True,
            "include_in_distance_curve": False,
        },
        {
            "displacement_role": "deep_inside_control",
            "target_signed_edge_distance_m": None,
            "actual_signed_edge_distance_m": -8.2,
            "bearing_radians": 1.5,
            "ground_truth_inside": True,
            "include_in_distance_curve": False,
        },
        {
            "displacement_role": "deep_inside_control",
            "target_signed_edge_distance_m": None,
            "actual_signed_edge_distance_m": -8.2,
            "bearing_radians": None,
            "ground_truth_inside": True,
            "include_in_distance_curve": True,
        },
    ],
)
def test_manifest_row_rejects_displacement_role_mismatches(overrides):
    from thesis.data.manifest import ManifestRow

    with pytest.raises(ValidationError):
        ManifestRow.model_validate(fixed_level_row(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"manifest_schema_version": 2},
        {"unexpected_future_field": "silent schema drift"},
    ],
)
def test_manifest_row_rejects_schema_drift(overrides):
    from thesis.data.manifest import ManifestRow

    with pytest.raises(ValidationError):
        ManifestRow.model_validate(fixed_level_row(**overrides))


@pytest.mark.parametrize("image_path", ["/tmp/sample.png", "../sample.png", ""])
def test_manifest_row_requires_a_relative_dataset_image_path(image_path):
    from thesis.data.manifest import ManifestRow

    with pytest.raises(ValidationError):
        ManifestRow.model_validate(fixed_level_row(image_path=image_path))


@pytest.mark.parametrize(
    "overrides",
    [
        {"actual_signed_edge_distance_m": 1.0, "ground_truth_inside": True},
        {"actual_signed_edge_distance_m": -1.0, "ground_truth_inside": False},
        {
            "displacement_role": "deep_inside_control",
            "target_signed_edge_distance_m": None,
            "actual_signed_edge_distance_m": 1.0,
            "bearing_radians": None,
            "ground_truth_inside": False,
            "include_in_distance_curve": False,
        },
    ],
)
def test_manifest_row_rejects_inconsistent_ground_truth(overrides):
    from thesis.data.manifest import ManifestRow

    with pytest.raises(ValidationError):
        ManifestRow.model_validate(fixed_level_row(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"bbox_selection_probability": 0.0},
        {"eligible_buildings_in_bbox": 0},
        {"sampled_buildings_in_bbox": 0},
        {"sampled_buildings_in_bbox": 101},
        {"sampling_weight": 0.0},
        {"footprint_area_m2": 0.0},
        {"target_anchor_lat": 91.0},
        {"target_anchor_lon": 181.0},
        {"radial_displacement_m": -1.0},
        {"bearing_radians": 6.283185307179586},
        {"pin_lat": -91.0},
        {"pin_lon": -181.0},
        {"naip_year": 0},
        {"naip_native_gsd_x_m": 0.0},
        {"naip_native_gsd_y_m": 0.0},
        {"rendered_gsd_m": 0.0},
        {"image_px": 0},
        {"marker_size_px": 0},
        {"marker_opacity": 1.1},
        {"census_buffer_m": -1.0},
        {"image_px": "512"},
    ],
)
def test_manifest_row_rejects_out_of_domain_values(overrides):
    from thesis.data.manifest import ManifestRow

    with pytest.raises(ValidationError):
        ManifestRow.model_validate(fixed_level_row(**overrides))


def test_sample_id_is_stable_for_the_semantic_sample_identity():
    from thesis.data.manifest import sample_id_for

    sample_id = sample_id_for(
        building_id="building-1",
        displacement_role="fixed_level",
        target_signed_edge_distance_m=1.0,
        marker_shape="google_pin",
        marker_color="red",
    )

    assert sample_id == "e2fdaec7c8a7c35f26c9b51d56959601c3cf2868559f6b9e3e2601297863bef2"


def test_manifest_row_rejects_a_sample_id_from_different_semantics():
    from thesis.data.manifest import ManifestRow

    with pytest.raises(ValidationError):
        ManifestRow.model_validate(fixed_level_row(sample_id="wrong"))


def test_write_manifest_creates_the_typed_samples_table(tmp_path):
    from thesis.data.manifest import write_manifest

    path = tmp_path / "manifest.duckdb"
    row = fixed_level_row()

    write_manifest(path, [row])

    with duckdb.connect(str(path), read_only=True) as connection:
        stored = connection.execute(
            "SELECT sample_id, image_path, target_signed_edge_distance_m FROM samples"
        ).fetchone()
    assert stored == (row["sample_id"], "images/sample-1.png", 1.0)


def test_write_manifest_rejects_duplicate_sample_ids(tmp_path):
    from thesis.data.manifest import write_manifest

    row = fixed_level_row()

    with pytest.raises(duckdb.ConstraintException):
        write_manifest(tmp_path / "manifest.duckdb", [row, row])


def test_write_manifest_rejects_an_existing_incompatible_schema(tmp_path):
    from thesis.data.manifest import ManifestSchemaError, write_manifest

    path = tmp_path / "manifest.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE samples (sample_id VARCHAR PRIMARY KEY)")

    with pytest.raises(ManifestSchemaError):
        write_manifest(path, [])


def test_diff_manifests_reports_changed_sample_ids(tmp_path):
    from thesis.data.manifest import diff_manifests, write_manifest

    first = tmp_path / "first.duckdb"
    second = tmp_path / "second.duckdb"
    row = fixed_level_row()
    write_manifest(first, [row])
    write_manifest(second, [{**row, "naip_year": 2023}])

    assert diff_manifests(first, first) == []
    assert diff_manifests(first, second) == [f"changed: {row['sample_id']}"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_hash": ""},
        {"config_hash": "a" * 63},
        {"git_commit": ""},
        {"git_commit": "g" * 40},
        {"building_id": ""},
    ],
)
def test_manifest_row_requires_complete_provenance_identifiers(overrides):
    from thesis.data.manifest import ManifestRow

    with pytest.raises(ValidationError):
        ManifestRow.model_validate(fixed_level_row(**overrides))
