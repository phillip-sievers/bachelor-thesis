from collections import Counter

import pandas as pd
from shapely.geometry import box

from thesis.data.sampling import (
    assign_ablation,
    fill_deficits_by_ecoregion,
    main_box_quotas,
    reduce_primary_pins,
    select_ablation_boxes,
    select_joint_sources,
    usable_source_pool,
)


def test_main_quota_is_80_times_17_plus_40_times_16():
    quotas = main_box_quotas([f"box-{index:03}" for index in range(120)], seed=19)

    assert Counter(quotas.values()) == {17: 80, 16: 40}
    assert sum(quotas.values()) == 2_000
    assert quotas == main_box_quotas(list(reversed(sorted(quotas))), seed=19)


def _spacing_candidates():
    return pd.DataFrame(
        {
            "source_id": ["a", "b", "c", "d"],
            "box_id": ["box-a", "box-a", "box-b", "box-b"],
            "ecoregion": ["eco-1", "eco-1", "eco-2", "eco-2"],
            "building_stratum": ["small", "large", "small", "large"],
            "scene_geometry": [box(0, 0, 10, 10), box(5, 0, 15, 10), box(30, 0, 40, 10), box(50, 0, 60, 10)],
        }
    )


def test_candidate_order_depends_on_seed_not_input_order():
    candidates = _spacing_candidates()
    capacities = {"box-a": 2, "box-b": 1}

    forward = select_joint_sources(candidates, seed=41, capacity_by_box=capacities)
    reverse = select_joint_sources(candidates.iloc[::-1], seed=41, capacity_by_box=capacities)

    assert forward.selected["source_id"].tolist() == reverse.selected["source_id"].tolist()
    assert forward.spacing_rejections["source_id"].tolist() == reverse.spacing_rejections["source_id"].tolist()
    assert set(forward.spacing_rejections["reason"]) <= {"scene_overlap"}


def test_joint_selection_rejects_overlapping_scenes():
    result = select_joint_sources(_spacing_candidates(), seed=41, capacity_by_box={"box-a": 2, "box-b": 2})
    selected_geometries = result.selected["scene_geometry"].tolist()

    assert all(
        not first.intersects(second)
        for index, first in enumerate(selected_geometries)
        for second in selected_geometries[index + 1 :]
    )
    assert len(result.spacing_rejections) == 1


def test_deficits_spill_within_ecoregion_before_an_adjacent_region():
    candidates = pd.DataFrame(
        {
            "source_id": ["origin", "same-1", "same-2", "adjacent"],
            "box_id": ["sparse", "receiver-a", "receiver-b", "receiver-c"],
            "ecoregion": ["eco-1", "eco-1", "eco-1", "eco-2"],
            "scene_geometry": [box(0, 0, 1, 1), box(10, 0, 11, 1), box(20, 0, 21, 1), box(30, 0, 31, 1)],
        }
    )
    initial = select_joint_sources(candidates.iloc[:1], seed=1, capacity_by_box={"sparse": 3})

    result = fill_deficits_by_ecoregion(
        initial,
        candidates,
        ecoregion_by_box={"sparse": "eco-1"},
        adjacent_ecoregions={"eco-1": ("eco-2",)},
        seed=2,
    )

    assert len(result.selected) == 3
    assert {spill["scope"] for spill in result.spills} == {"same_ecoregion"}
    assert {spill["destination_box_id"] for spill in result.spills} == {"receiver-a", "receiver-b"}
    assert set(result.selected.loc[result.selected["box_id"] != "sparse", "quota_box_id"]) == {"sparse"}


def _ablation_pool():
    rows = []
    for box_index in range(12):
        ecoregion = f"eco-{box_index % 2}"
        for stratum in ("small", "medium", "large"):
            rows.append(
                {
                    "source_id": f"source-{box_index}-{stratum}",
                    "box_id": f"box-{box_index}",
                    "ecoregion": ecoregion,
                    "building_stratum": stratum,
                    "assignment": "unassigned",
                }
            )
    return pd.DataFrame(rows)


def test_ablation_has_at_most_one_source_per_box_and_balances_cells():
    assigned = assign_ablation(_ablation_pool(), seed=43, target=12)
    ablation = assigned.loc[assigned["assignment"] == "ablation"]
    counts = ablation.groupby(["ecoregion", "building_stratum"]).size()

    assert len(ablation) == 12
    assert ablation["box_id"].is_unique
    assert counts.max() - counts.min() <= 1
    assert set(assigned["assignment"]) == {"main", "ablation"}


def test_ablation_can_be_locked_to_preselected_extra_capacity_boxes():
    pool = _ablation_pool()
    required_boxes = {"box-1", "box-4", "box-7"}

    assigned = assign_ablation(pool, seed=43, target=3, required_box_ids=required_boxes)

    assert set(assigned.loc[assigned["assignment"] == "ablation", "box_id"]) == required_boxes


def test_ablation_box_reservations_are_seeded_ecoregion_balanced_and_feasible():
    rows = []
    quotas = {}
    for box_index in range(8):
        box_id = f"box-{box_index}"
        quotas[box_id] = 2
        candidate_count = 3 if box_index != 7 else 2
        for source_index in range(candidate_count):
            rows.append(
                {
                    "source_id": f"source-{box_index}-{source_index}",
                    "box_id": box_id,
                    "ecoregion": f"eco-{box_index % 2}",
                }
            )
    candidates = pd.DataFrame(rows)

    selected = select_ablation_boxes(candidates, quotas, seed=47, target=6)
    reversed_selected = select_ablation_boxes(candidates.iloc[::-1], quotas, seed=47, target=6)

    assert selected == reversed_selected
    assert len(selected) == 6
    assert "box-7" not in selected
    ecoregions = candidates.drop_duplicates("box_id").set_index("box_id")["ecoregion"]
    counts = Counter(ecoregions.loc[list(selected)])
    assert counts == {"eco-0": 3, "eco-1": 3}


def test_reserved_ablation_is_one_joint_spacing_pass_and_leaves_main_quota_per_box():
    pool = _ablation_pool()
    pool["scene_geometry"] = [box(index * 10, 0, index * 10 + 1, 1) for index in range(len(pool))]
    quotas = {f"box-{index}": 2 for index in range(12)}
    reserved = select_ablation_boxes(pool, quotas, seed=47, target=6)
    combined = {box_id: quota + int(box_id in reserved) for box_id, quota in quotas.items()}

    spacing = select_joint_sources(pool, seed=41, capacity_by_box=combined)
    assigned = assign_ablation(spacing.selected, seed=43, target=6, required_box_ids=reserved)
    main = assigned.loc[assigned["assignment"] == "main"]

    assert len(spacing.selected) == sum(combined.values())
    assert assigned.loc[assigned["assignment"] == "ablation", "box_id"].is_unique
    assert main.groupby("quota_box_id").size().to_dict() == quotas


def test_only_zero_bin_sources_are_replacement_candidates():
    screened = pd.DataFrame({"source_id": ["complete", "partial", "zero"], "available_bin_count": [10, 1, 0]})

    usable, replacement_candidates = usable_source_pool(screened)

    assert usable["source_id"].tolist() == ["complete", "partial"]
    assert replacement_candidates[["source_id", "reason"]].to_dict("records") == [
        {"source_id": "zero", "reason": "zero_usable_bins"}
    ]


def test_delta_reduction_keeps_one_near_pin_for_primary_and_all_survivors_for_curve():
    pins = pd.DataFrame(
        {
            "pin_id": ["n1", "n2", "n3", "p1", "p2", "p3", "far"],
            "source_id": ["source"] * 7,
            "bin_id": [-1, -1, -1, 1, 1, 1, 10],
            "exact_signed_distance_m": [-1.4, -0.8, -0.4, 0.4, 0.8, 1.4, 9.0],
        }
    )

    reduced = reduce_primary_pins(pins, delta_m=0.7, seed=53)
    survivors = reduced.loc[reduced["survives_delta"]]

    assert survivors.groupby(["source_id", "bin_id"])["primary"].sum().to_dict() == {
        ("source", -1): 1,
        ("source", 1): 1,
        ("source", 10): 1,
    }
    assert survivors["include_in_curve"].all()
    assert not reduced.loc[~reduced["survives_delta"], "primary"].any()
    shuffled = reduce_primary_pins(pins.sample(frac=1, random_state=2), delta_m=0.7, seed=53)
    pd.testing.assert_frame_equal(reduced, shuffled)


def test_delta_reduction_handles_a_near_bin_with_no_survivor():
    pins = pd.DataFrame(
        {
            "pin_id": ["p1", "p2", "p3"],
            "source_id": ["source"] * 3,
            "bin_id": [1, 1, 1],
            "exact_signed_distance_m": [0.31, 0.5, 0.69],
        }
    )

    reduced = reduce_primary_pins(pins, delta_m=0.7, seed=53)

    assert not reduced["survives_delta"].any()
    assert not reduced["primary"].any()
    assert not reduced["include_in_curve"].any()
