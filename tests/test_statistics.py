import math

import numpy as np
import pandas as pd
import pytest

from thesis.data.statistics import (
    GateFailure,
    area_tertile_cuts,
    assign_area_stratum,
    availability_smd_report,
    coverage_report,
    enforce_gate,
    weighted_quantile,
    weighted_smd,
    weighted_variance,
)

BIN_IDS = (-5, -3, -1, 1, 3, 5, 10, 20, 40, 80)


def test_weighted_quantile_interpolates_the_midpoint_weighted_cdf():
    values = np.array([0.0, 10.0, 20.0, 30.0])
    weights = np.array([1.0, 1.0, 8.0, 1.0])

    assert weighted_quantile(values, weights, (0.5,)).item() == pytest.approx(18.8888888889)


def test_area_tertiles_give_each_box_total_weight_one():
    frame = pd.DataFrame(
        {
            "box_id": ["sparse", "sparse", "dense", "dense", "dense", "dense", "dense", "dense"],
            "footprint_area_m2": [100, 121, 144, 169, 196, 225, 256, 289],
        }
    )

    cuts = area_tertile_cuts(frame)
    counts = frame.groupby("box_id")["box_id"].transform("size")
    expected = weighted_quantile(
        np.log(frame["footprint_area_m2"].to_numpy()),
        (1 / counts).to_numpy(),
        (1 / 3, 2 / 3),
    )

    assert cuts == pytest.approx(tuple(expected))


def test_area_stratum_cut_boundaries_are_stable():
    cuts = (math.log(150), math.log(250))

    assert assign_area_stratum(149.9, cuts) == "small"
    assert assign_area_stratum(150, cuts) == "medium"
    assert assign_area_stratum(250, cuts) == "large"


def test_weighted_variance_and_smd_use_weights_in_denominator():
    selected = pd.DataFrame({"x": [0.0, 2.0], "w": [3.0, 1.0]})
    reference = pd.DataFrame({"x": [0.0, 4.0], "w": [1.0, 1.0]})

    assert weighted_variance(selected["x"], selected["w"]) == pytest.approx(0.75)
    assert weighted_variance(reference["x"], reference["w"]) == pytest.approx(4.0)
    assert weighted_smd(selected, reference, "x", "w") == pytest.approx(-0.9733285268)


def _sources(*, unavailable: dict[int, int] | None = None) -> pd.DataFrame:
    unavailable = unavailable or {}
    rows = []
    for index in range(2_000):
        availability = {str(bin_id): index >= unavailable.get(bin_id, 0) for bin_id in BIN_IDS}
        rows.append(
            {
                "source_id": f"source-{index}",
                "box_id": f"box-{index % 120:03}",
                "assignment": "main",
                "bin_availability": availability,
            }
        )
    return pd.DataFrame(rows)


def _quotas(frame: pd.DataFrame) -> dict[str, int]:
    return frame.groupby("box_id").size().astype(int).to_dict()


def test_coverage_report_includes_global_and_per_box_rows():
    frame = _sources(unavailable={-5: 200})

    report = coverage_report(frame, BIN_IDS)

    global_minus_five = report[(report["scope"] == "global") & (report["bin_id"] == -5)].iloc[0]
    assert global_minus_five["available_sources"] == 1_800
    assert global_minus_five["coverage"] == pytest.approx(0.9)
    assert len(report[report["scope"] == "box"]) == 120 * len(BIN_IDS)


def test_coverage_gate_accepts_exactly_fifty_percent_and_rejects_less():
    passing = _sources(unavailable={-5: 1_000})
    enforce_gate(
        coverage_report(passing, BIN_IDS),
        passing,
        _box_gate_frame(),
        expected_main_total=2_000,
        coverage_threshold=0.5,
    )

    failing = _sources(unavailable={-5: 1_001})
    with pytest.raises(GateFailure, match=r"bin -5 coverage"):
        enforce_gate(
            coverage_report(failing, BIN_IDS),
            failing,
            _box_gate_frame(),
            expected_main_total=2_000,
            coverage_threshold=0.5,
        )


def _box_gate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"box_id": [f"box-{index:03}" for index in range(120)], "eligible_count": 1, "zero_eligible_reason": None}
    )


def test_gate_requires_main_total_but_allows_deficit_spill_between_boxes():
    frame = _sources()
    enforce_gate(
        coverage_report(frame, BIN_IDS),
        frame,
        _box_gate_frame(),
        expected_main_total=2_000,
        coverage_threshold=0.5,
    )
    with pytest.raises(GateFailure, match="main total"):
        enforce_gate(
            coverage_report(frame.iloc[:-1], BIN_IDS),
            frame.iloc[:-1],
            _box_gate_frame(),
            expected_main_total=2_000,
            coverage_threshold=0.5,
        )


def test_gate_requires_a_reason_for_every_zero_eligible_box():
    frame = _sources()
    boxes = _box_gate_frame()
    boxes.loc[0, "eligible_count"] = 0
    with pytest.raises(GateFailure, match="lack a recorded reason"):
        enforce_gate(coverage_report(frame, BIN_IDS), frame, boxes, expected_main_total=2_000, coverage_threshold=0.5)
    boxes.loc[0, "zero_eligible_reason"] = "tile_containment"
    enforce_gate(coverage_report(frame, BIN_IDS), frame, boxes, expected_main_total=2_000, coverage_threshold=0.5)


def test_availability_audit_reports_smds_without_gating():
    eligible = pd.DataFrame(
        {
            "source_id": ["a", "b", "c", "d"],
            "box_id": ["sparse", "sparse", "dense", "dense"],
            "footprint_area_m2": [100.0, 400.0, 100.0, 400.0],
            "nearest_neighbor_distance_m": [10.0, 20.0, 30.0, 40.0],
            "neighbors_within_80m": [0, 1, 2, 3],
            "building_stratum": ["small", "large", "small", "large"],
            "bin_availability": [
                {"-5": True},
                {"-5": False},
                {"-5": True},
                {"-5": False},
            ],
        }
    )

    report = availability_smd_report(eligible, (-5,))

    assert set(report["covariate"]) == {
        "log_footprint_area",
        "log_nearest_neighbor_distance",
        "neighbors_within_80m",
        "stratum_small",
        "stratum_medium",
        "stratum_large",
        "footprint_area_m2_within_small",
        "footprint_area_m2_within_large",
    }
    assert (report["bin_id"] == -5).all()
    unavailable_stratum = report.loc[report["covariate"] == "footprint_area_m2_within_large", "smd"]
    assert unavailable_stratum.isna().all()
    assert np.isfinite(report.loc[report["covariate"] != "footprint_area_m2_within_large", "smd"]).all()
