from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.config import load_config


def draw_subsample(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    config = load_config("subsamples")
    draw = config["draws"][name]
    if draw["n"] is None:
        raise ValueError(f"subsample size for {name} is not set in configs/subsamples.yaml")
    per_box = int(config["sources_per_box"])
    rng = np.random.default_rng(int(draw["seed"]))

    box_ecoregion = rows.groupby("box_id")["ecoregion"].first().sort_index()
    boxes_per_ecoregion = box_ecoregion.value_counts().sort_index()
    target_boxes = int(np.ceil(int(draw["n"]) / per_box))
    allocation = (boxes_per_ecoregion / boxes_per_ecoregion.sum() * target_boxes).round().astype(int)
    allocation = allocation.clip(lower=int(config["min_boxes_per_ecoregion"]), upper=boxes_per_ecoregion)

    chosen_boxes = [
        box_id
        for ecoregion, k in allocation.items()
        for box_id in rng.choice(box_ecoregion.index[box_ecoregion == ecoregion], size=k, replace=False)
    ]
    picked = []
    for box_id in sorted(chosen_boxes):
        in_box = rows.loc[rows["box_id"] == box_id].sort_index()
        picked.append(in_box.iloc[rng.choice(len(in_box), size=min(per_box, len(in_box)), replace=False)])
    return pd.concat(picked)


def one_pin_per_source(pins: pd.DataFrame, sources: pd.DataFrame, name: str) -> pd.DataFrame:
    rng = np.random.default_rng(int(load_config("subsamples")["draws"][name]["seed"]))
    candidates = pins.loc[pins["source_id"].isin(sources["source_id"])].sort_values("pin_id")
    return candidates.groupby("source_id").sample(1, random_state=rng)


def draw_failure_modes(errors: pd.DataFrame) -> pd.DataFrame:
    # Per model: shuffle, then round-robin over bins, so each bin gets an equal share (capped by its errors).
    rng = np.random.default_rng(int(load_config("subsamples")["draws"]["failure_mode"]["seed"]))
    n = int(load_config("subsamples")["draws"]["failure_mode"]["n"])
    shuffled = errors.sort_values(["model", "pin_id"]).sample(frac=1, random_state=rng)
    rank = shuffled.groupby(["model", "bin_id"]).cumcount()
    return (
        shuffled.assign(rank=rank)
        .sort_values(["model", "rank", "bin_id"])
        .groupby("model")
        .head(n)
        .drop(columns="rank")
    )
