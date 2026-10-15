from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from thesis.config import load_config

COUNTS = ("tp", "pos", "tn", "neg", "pred_pos")


def predictions(items: pd.DataFrame, invalid_policy: str) -> pd.DataFrame:
    items = items.copy()
    valid = items["parsed_answer"].isin(["yes", "no"])
    if invalid_policy == "exclude":
        items = items.loc[valid].copy()
        valid = valid.loc[items.index]
    elif invalid_policy != "wrong":
        raise ValueError(f"unknown invalid_policy: {invalid_policy}")
    # An invalid answer becomes the incorrect answer.
    items["predicted_inside"] = np.where(valid, items["parsed_answer"] == "yes", ~items["inside"].astype(bool))
    return items


def distance_sign_predictions(items: pd.DataFrame, invalid_policy: str) -> pd.DataFrame:
    items = items.copy()
    distance = pd.to_numeric(items["parsed_answer"], errors="coerce")
    # A distance of 0 counts as inside.
    items["parsed_answer"] = np.where(distance.isna(), None, np.where(distance <= 0, "yes", "no"))
    return predictions(items, invalid_policy)


def item_weights(items: pd.DataFrame, estimand: str) -> pd.Series:
    if estimand == "pooled":
        return pd.Series(1.0, index=items.index)
    per_box = 1.0 / items.groupby(["model", "box_id"])["pin_id"].transform("size")
    if estimand == "per_box_mean":
        return per_box
    if estimand == "design_weighted":
        stratum_weights = {str(k): float(v) for k, v in load_config("scoring")["stratum_weights"].items()}
        return per_box * items["ecoregion"].astype(str).map(stratum_weights)
    raise ValueError(f"unknown estimand: {estimand}")


def box_counts(items: pd.DataFrame, weights: pd.Series, by: list[str]) -> pd.DataFrame:
    truth = items["inside"].astype(bool)
    predicted = items["predicted_inside"].astype(bool)
    frame = items[["box_id", *by]].assign(
        tp=weights * (truth & predicted),
        pos=weights * truth,
        tn=weights * (~truth & ~predicted),
        neg=weights * ~truth,
        pred_pos=weights * predicted,
    )
    summed = frame.groupby(["box_id", *by])[list(COUNTS)].sum()
    wide = summed.unstack(by) if by else summed
    if by:
        wide = wide.reorder_levels([*range(1, len(by) + 1), 0], axis=1)
    return wide.fillna(0.0).sort_index(axis=1)


def metrics(counts: pd.DataFrame) -> pd.DataFrame:
    def get(name):
        return counts.xs(name, axis=1, level=-1) if counts.columns.nlevels > 1 else counts[name]

    # A bin has one class only. Thus balanced accuracy per bin is NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = get("tp") / get("pos")
        tnr = get("tn") / get("neg")
        return pd.concat(
            {
                "balanced_accuracy": 0.5 * (tpr + tnr),
                "accuracy": (get("tp") + get("tn")) / (get("pos") + get("neg")),
                "precision": get("tp") / get("pred_pos"),
            },
            axis=1,
        )


def bootstrap_draws(box_ecoregion: pd.Series, replicates: int, seed: int) -> np.ndarray:
    # Rao-Wu: draw m_h - 1 boxes per ecoregion and multiply by m_h / (m_h - 1).
    # This removes the variance bias of the naive bootstrap. All models use the same draws.
    rng = np.random.default_rng(seed)
    draws = np.zeros((replicates, len(box_ecoregion)))
    for ecoregion in sorted(box_ecoregion.unique()):
        columns = np.flatnonzero(box_ecoregion.to_numpy() == ecoregion)
        m = len(columns)
        if m < 2:
            raise ValueError(f"ecoregion {ecoregion} has {m} box; the bootstrap needs at least 2")
        counts = rng.multinomial(m - 1, np.full(m, 1 / m), size=replicates)
        draws[:, columns] = counts * m / (m - 1)
    return draws


def score(
    items: pd.DataFrame,
    by: list[str],
    *,
    estimand: str | None = None,
    invalid_policy: str | None = None,
    replicates: int | None = None,
    seed: int | None = None,
    sign_from_distance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config("scoring")
    estimand = estimand or config["estimand"]
    invalid_policy = invalid_policy or config["invalid_policy"]
    replicates = replicates or int(config["bootstrap_replicates"])
    seed = seed if seed is not None else int(config["bootstrap_seed"])

    scored = (distance_sign_predictions if sign_from_distance else predictions)(items, invalid_policy)
    counts = box_counts(scored, item_weights(scored, estimand), by)
    point = metrics(counts.sum().to_frame().T).iloc[0].unstack(0)

    box_ecoregion = items.groupby("box_id")["ecoregion"].first().loc[counts.index]
    draws = bootstrap_draws(box_ecoregion, replicates, seed)
    replicate_counts = pd.DataFrame(draws @ counts.to_numpy(), columns=counts.columns)
    return point, metrics(replicate_counts)


def reliability_boundary(replicates: pd.DataFrame, model: str, tau: float, bins=(3, 5, 10, 20, 40, 80)) -> dict:
    lcb = {b: float(np.nanquantile(replicates[("accuracy", model, b)], 0.025)) for b in bins}
    boundary = "not reached"
    for b in sorted(bins, reverse=True):
        if lcb[b] < tau:
            break
        boundary = b
    return {"lcb": lcb, "b_star": boundary}


def percentile_interval(replicates: pd.Series | np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(replicates, dtype=float)
    values = values[~np.isnan(values)]
    tail = (1 - level) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))


def bootstrap_p_value(
    difference_replicates: pd.Series | np.ndarray, null: float = 0.0, alternative: str = "two-sided"
) -> float:
    # One-sided "greater": H0 theta <= null. The +1 keeps p >= 1/(B+1).
    values = np.asarray(difference_replicates, dtype=float)
    values = values[~np.isnan(values)]
    below = (1 + np.sum(values <= null)) / (len(values) + 1)
    if alternative == "greater":
        return float(below)
    if alternative != "two-sided":
        raise ValueError(f"unknown alternative: {alternative}")
    above = (1 + np.sum(values >= null)) / (len(values) + 1)
    return float(min(1.0, 2 * min(below, above)))


def paired_difference(
    point: pd.DataFrame, replicates: pd.DataFrame, metric: str, a: str, b: str, level: float = 0.95
) -> dict[str, float]:
    difference = replicates[(metric, a)] - replicates[(metric, b)]
    low, high = percentile_interval(difference, level)
    return {
        "difference": float(point.loc[a, metric] - point.loc[b, metric]),
        "ci_low": low,
        "ci_high": high,
        "p_value": bootstrap_p_value(difference),
    }


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    reject, still_rejecting = {}, True
    for rank, (name, p) in enumerate(ordered):
        still_rejecting = still_rejecting and p <= alpha / (len(ordered) - rank)
        reject[name] = still_rejecting
    return reject


DistanceError = Callable[[np.ndarray, np.ndarray], np.ndarray]


def absolute_error(predicted_m: np.ndarray, true_m: np.ndarray) -> np.ndarray:
    return np.abs(predicted_m - true_m)


def log_error(predicted_m: np.ndarray, true_m: np.ndarray) -> np.ndarray:
    return np.abs(np.log((np.abs(predicted_m) + 1) / (np.abs(true_m) + 1)))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def bootstrap_median_error(
    items: pd.DataFrame,
    *,
    by: list[str] | None = None,
    error: DistanceError = log_error,
    estimand: str | None = None,
    invalid_policy: str | None = None,
    replicates: int | None = None,
    seed: int | None = None,
    chunk: int = 500,
) -> tuple[pd.Series, pd.DataFrame]:
    config = load_config("scoring")
    estimand = estimand or config["estimand"]
    invalid_policy = invalid_policy or config["invalid_policy"]
    replicates = replicates or int(config["bootstrap_replicates"])
    seed = seed if seed is not None else int(config["bootstrap_seed"])

    scored = distance_errors(items, error)
    if invalid_policy == "exclude":
        scored = scored.loc[scored["error"].notna()]
    elif invalid_policy == "wrong":
        # An invalid answer gets an infinite error.
        scored = scored.assign(error=scored["error"].fillna(np.inf))
    else:
        raise ValueError(f"unknown invalid_policy: {invalid_policy}")
    scored = scored.assign(weight=item_weights(scored, estimand))

    box_ecoregion = items.groupby("box_id")["ecoregion"].first().sort_index()
    draws = bootstrap_draws(box_ecoregion, replicates, seed)
    box_column = {box_id: i for i, box_id in enumerate(box_ecoregion.index)}

    point, reps = {}, {}
    # Weights and draws come from all items, so a per-bin median uses the same ones as the overall median.
    for key, group in scored.groupby(["model", *by] if by else "model"):
        group = group.sort_values("error", kind="stable")
        values = group["error"].to_numpy()
        weights = group["weight"].to_numpy()
        columns = group["box_id"].map(box_column).to_numpy()
        point[key] = weighted_median(values, weights)
        medians = np.empty(replicates)
        for start in range(0, replicates, chunk):
            cumulative = np.cumsum(draws[start : start + chunk][:, columns] * weights, axis=1)
            index = (cumulative >= cumulative[:, -1:] / 2).argmax(axis=1)
            medians[start : start + chunk] = values[index]
        reps[key] = medians
    return pd.Series(point), pd.DataFrame(reps)


def distance_errors(items: pd.DataFrame, error: DistanceError = log_error) -> pd.DataFrame:
    predicted = pd.to_numeric(items["parsed_answer"], errors="coerce").to_numpy(dtype=float)
    return items.assign(error=error(predicted, items["exact_signed_distance_m"].to_numpy(dtype=float)))
