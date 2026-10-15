"""Weighted construction audits and gate calculations (D18, D22)."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class GateFailure(RuntimeError):
    pass


def _numeric_arrays(values: Sequence[float], weights: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if value_array.ndim != 1 or weight_array.ndim != 1 or len(value_array) != len(weight_array):
        raise ValueError("values and weights must be one-dimensional and equally sized")
    if len(value_array) == 0:
        raise ValueError("values and weights must not be empty")
    if not np.isfinite(value_array).all() or not np.isfinite(weight_array).all():
        raise ValueError("values and weights must be finite")
    if np.any(weight_array < 0) or not np.any(weight_array > 0):
        raise ValueError("weights must be non-negative with a positive sum")
    return value_array, weight_array


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    probabilities: Sequence[float],
) -> np.ndarray:
    value_array, weight_array = _numeric_arrays(values, weights)
    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 1 or np.any((probability_array < 0) | (probability_array > 1)):
        raise ValueError("probabilities must be one-dimensional values in [0, 1]")

    order = np.argsort(value_array, kind="stable")
    sorted_values = value_array[order]
    sorted_weights = weight_array[order]
    positive = sorted_weights > 0
    sorted_values = sorted_values[positive]
    sorted_weights = sorted_weights[positive]
    cumulative_midpoints = (np.cumsum(sorted_weights) - sorted_weights / 2) / sorted_weights.sum()
    return np.interp(probability_array, cumulative_midpoints, sorted_values)


def area_tertile_cuts(
    frame: pd.DataFrame,
    *,
    box_column: str = "box_id",
    area_column: str = "footprint_area_m2",
) -> tuple[float, float]:
    if (frame[area_column] <= 0).any():
        raise ValueError("footprint areas must be positive")
    counts = frame.groupby(box_column)[box_column].transform("size").astype(float)
    cuts = weighted_quantile(
        np.log(frame[area_column].to_numpy(dtype=float)),
        (1.0 / counts).to_numpy(),
        (1 / 3, 2 / 3),
    )
    return float(cuts[0]), float(cuts[1])


def assign_area_stratum(area_m2: float, log_area_cuts: tuple[float, float]) -> str:
    if area_m2 <= 0:
        raise ValueError("footprint area must be positive")
    log_area = math.log(area_m2)
    if log_area < log_area_cuts[0]:
        return "small"
    if log_area < log_area_cuts[1]:
        return "medium"
    return "large"


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    value_array, weight_array = _numeric_arrays(values, weights)
    return float(np.average(value_array, weights=weight_array))


def weighted_variance(values: Sequence[float], weights: Sequence[float]) -> float:
    value_array, weight_array = _numeric_arrays(values, weights)
    mean = float(np.average(value_array, weights=weight_array))
    return float(np.average((value_array - mean) ** 2, weights=weight_array))


def weighted_smd(
    selected: pd.DataFrame,
    reference: pd.DataFrame,
    value_column: str,
    weight_column: str,
) -> float:
    selected_mean = weighted_mean(selected[value_column], selected[weight_column])
    reference_mean = weighted_mean(reference[value_column], reference[weight_column])
    selected_variance = weighted_variance(selected[value_column], selected[weight_column])
    reference_variance = weighted_variance(reference[value_column], reference[weight_column])
    denominator = math.sqrt((selected_variance + reference_variance) / 2)
    difference = selected_mean - reference_mean
    if denominator == 0:
        return 0.0 if difference == 0 else math.copysign(math.inf, difference)
    return difference / denominator


def _availability(value: object, bin_id: int) -> bool:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError("bin_availability must be a mapping or JSON object")
    return bool(value.get(str(bin_id), value.get(bin_id, False)))


def coverage_report(sources: pd.DataFrame, bin_ids: Iterable[int]) -> pd.DataFrame:
    main = sources.loc[sources["assignment"] == "main"].copy()
    if main.empty:
        raise ValueError("coverage requires main sources")

    rows: list[dict[str, object]] = []
    for bin_id in bin_ids:
        available = main["bin_availability"].map(lambda value, current_bin=bin_id: _availability(value, current_bin))
        rows.append(
            {
                "scope": "global",
                "box_id": None,
                "bin_id": int(bin_id),
                "available_sources": int(available.sum()),
                "total_sources": int(len(main)),
                "coverage": float(available.mean()),
            }
        )
        for box_id, box_rows in main.groupby("box_id", sort=True):
            box_available = box_rows["bin_availability"].map(
                lambda value, current_bin=bin_id: _availability(value, current_bin)
            )
            rows.append(
                {
                    "scope": "box",
                    "box_id": str(box_id),
                    "bin_id": int(bin_id),
                    "available_sources": int(box_available.sum()),
                    "total_sources": int(len(box_rows)),
                    "coverage": float(box_available.mean()),
                }
            )
    return pd.DataFrame(rows)


def _audit_frame(eligible: pd.DataFrame) -> pd.DataFrame:
    result = eligible.copy()
    if "audit_weight" not in result:
        counts = result.groupby("box_id")["box_id"].transform("size").astype(float)
        result["audit_weight"] = 1.0 / counts
    result["log_footprint_area"] = np.log(result["footprint_area_m2"].to_numpy(dtype=float))
    nearest = result["nearest_neighbor_distance_m"].to_numpy(dtype=float)
    result["log_nearest_neighbor_distance"] = np.log1p(nearest)
    for stratum in ("small", "medium", "large"):
        result[f"stratum_{stratum}"] = (result["building_stratum"] == stratum).astype(float)
    return result


def availability_smd_report(eligible: pd.DataFrame, bin_ids: Iterable[int]) -> pd.DataFrame:
    reference = _audit_frame(eligible)
    covariates = (
        "log_footprint_area",
        "log_nearest_neighbor_distance",
        "neighbors_within_80m",
        "stratum_small",
        "stratum_medium",
        "stratum_large",
    )
    rows: list[dict[str, object]] = []
    for bin_id in bin_ids:
        current_bin = int(bin_id)
        available = reference.loc[
            reference["bin_availability"].map(
                lambda value, selected_bin=current_bin: _availability(value, selected_bin)
            )
        ]
        if available.empty:
            for covariate in covariates:
                rows.append({"bin_id": int(bin_id), "covariate": covariate, "smd": math.nan})
            continue
        for covariate in covariates:
            rows.append(
                {
                    "bin_id": int(bin_id),
                    "covariate": covariate,
                    "smd": weighted_smd(available, reference, covariate, "audit_weight"),
                }
            )
        for stratum in ("small", "medium", "large"):
            stratum_reference = reference.loc[reference["building_stratum"] == stratum]
            stratum_available = available.loc[available["building_stratum"] == stratum]
            if stratum_reference.empty:
                continue
            rows.append(
                {
                    "bin_id": int(bin_id),
                    "covariate": f"footprint_area_m2_within_{stratum}",
                    "smd": math.nan
                    if stratum_available.empty
                    else weighted_smd(
                        stratum_available,
                        stratum_reference,
                        "footprint_area_m2",
                        "audit_weight",
                    ),
                }
            )
    return pd.DataFrame(rows)


def selection_smd_report(selected: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    reference = _audit_frame(eligible)
    selected_ids = set(selected["source_id"].astype(str))
    selected_frame = reference.loc[reference["source_id"].astype(str).isin(selected_ids)]
    if selected_frame.empty:
        raise ValueError("selection audit requires selected sources")
    covariates = (
        "log_footprint_area",
        "log_nearest_neighbor_distance",
        "neighbors_within_80m",
        "stratum_small",
        "stratum_medium",
        "stratum_large",
    )
    rows = [
        {
            "covariate": covariate,
            "smd": weighted_smd(selected_frame, reference, covariate, "audit_weight"),
        }
        for covariate in covariates
    ]
    for stratum in ("small", "medium", "large"):
        stratum_reference = reference.loc[reference["building_stratum"] == stratum]
        stratum_selected = selected_frame.loc[selected_frame["building_stratum"] == stratum]
        if stratum_reference.empty or stratum_selected.empty:
            continue
        rows.append(
            {
                "covariate": f"footprint_area_m2_within_{stratum}",
                "smd": weighted_smd(
                    stratum_selected,
                    stratum_reference,
                    "footprint_area_m2",
                    "audit_weight",
                ),
            }
        )
    return pd.DataFrame(rows)


def enforce_gate(
    coverage: pd.DataFrame,
    selected_sources: pd.DataFrame,
    boxes: pd.DataFrame,
    *,
    expected_main_total: int,
    coverage_threshold: float,
) -> None:
    global_rows = coverage.loc[coverage["scope"] == "global"]
    failures = global_rows.loc[global_rows["coverage"] < coverage_threshold]
    if not failures.empty:
        details = ", ".join(
            f"bin {int(row.bin_id)} coverage {row.coverage:.3f}" for row in failures.itertuples(index=False)
        )
        raise GateFailure(f"{details} below {coverage_threshold:.3f}")

    main = selected_sources.loc[selected_sources["assignment"] == "main"]
    if len(main) != expected_main_total:
        raise GateFailure(f"main total {len(main)} does not equal {expected_main_total}")

    required = {"box_id", "eligible_count", "zero_eligible_reason"}
    missing = required - set(boxes.columns)
    if missing:
        raise ValueError(f"box gate columns missing: {sorted(missing)}")
    unexplained = boxes.loc[(boxes["eligible_count"] == 0) & boxes["zero_eligible_reason"].isna(), "box_id"]
    if not unexplained.empty:
        raise GateFailure(f"zero-eligible boxes lack a recorded reason: {sorted(unexplained.astype(str))}")
