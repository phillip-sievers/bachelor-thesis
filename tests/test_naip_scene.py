from datetime import date

import pytest
from rasterio.transform import from_origin
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box

from thesis.data.naip import (
    NoValidItem,
    TileOutOfBounds,
    aligned_scene_window,
    crop_window_for_slot,
    imagery_season,
    pick_containing_native_item,
    window_cache_key,
)


def _item(item_id, year, gsd, geometry):
    return {"id": item_id, "year": year, "gsd": gsd, "geometry": geometry}


def test_containing_item_checks_all_overlapping_quarter_quads():
    items = [
        _item("new-a", 2024, 0.6, box(0, 0, 5, 10)),
        _item("new-b", 2024, 0.6, box(4, 0, 10, 10)),
    ]

    selected = pick_containing_native_item(items, box(4.5, 2, 8, 8), 0.6, 0.01)

    assert selected["id"] == "new-b"


def test_containing_item_uses_latest_native_vintage_and_id_tiebreak():
    scene = box(2, 2, 8, 8)
    items = [
        _item("old", 2022, 0.6, box(0, 0, 10, 10)),
        _item("new-b", 2024, 0.6, box(0, 0, 10, 10)),
        _item("new-a", 2024, 0.6, box(0, 0, 10, 10)),
    ]

    assert pick_containing_native_item(items, scene, 0.6, 0.01)["id"] == "new-a"


def test_latest_vintage_is_computed_only_from_items_overlapping_the_scene():
    scene = box(2, 2, 8, 8)
    items = [
        _item("local", 2022, 0.6, box(0, 0, 10, 10)),
        _item("elsewhere", 2024, 0.6, box(20, 20, 30, 30)),
    ]

    assert pick_containing_native_item(items, scene, 0.6, 0.01)["id"] == "local"


def test_non_native_and_mosaic_only_scenes_reject():
    with pytest.raises(NoValidItem):
        pick_containing_native_item([_item("30cm", 2024, 0.3, box(0, 0, 10, 10))], box(2, 2, 8, 8), 0.6, 0.01)

    with pytest.raises(TileOutOfBounds):
        pick_containing_native_item(
            [
                _item("left", 2024, 0.6, box(0, 0, 5, 10)),
                _item("right", 2024, 0.6, box(5, 0, 10, 10)),
            ],
            box(4, 2, 6, 8),
            0.6,
            0.01,
        )


def test_aligned_scene_window_rounds_outward_to_native_grid():
    transform = from_origin(100, 200, 0.6, 0.6)

    window = aligned_scene_window(transform, width=1_000, height=1_000, bounds=(101.0, 196.9, 103.1, 199.0))
    left, bottom, right, top = window_bounds(window, transform)

    assert (window.col_off, window.row_off, window.width, window.height) == (1, 1, 5, 5)
    assert left <= 101.0 and bottom <= 196.9 and right >= 103.1 and top >= 199.0


def test_aligned_scene_window_rejects_dataset_edge():
    transform = from_origin(100, 200, 0.6, 0.6)

    with pytest.raises(TileOutOfBounds):
        aligned_scene_window(transform, width=10, height=10, bounds=(99.9, 195, 103, 199))


def test_slot_is_se_central_crop_pixel():
    window = crop_window_for_slot(600, 700, (1_400, 1_400), 512, 256, 256)

    assert (window.row_off, window.col_off, window.height, window.width) == (344, 444, 512, 512)


def test_scene_cache_key_includes_both_window_dimensions():
    from rasterio.windows import Window

    assert window_cache_key("item", Window(1, 2, 700, 800)) != window_cache_key("item", Window(1, 2, 800, 700))


def test_crop_window_rejects_scene_edge_slots():
    with pytest.raises(TileOutOfBounds):
        crop_window_for_slot(255, 700, (1_400, 1_400), 512, 256, 256)
    with pytest.raises(TileOutOfBounds):
        crop_window_for_slot(600, 1_145, (1_400, 1_400), 512, 256, 256)


def test_tip_directional_margins_at_native_gsd():
    assert (256 + 0.5) * 0.6 == pytest.approx(153.9)
    assert (512 - 256 - 0.5) * 0.6 == pytest.approx(153.3)


@pytest.mark.parametrize(
    "acquired,expected",
    [
        (date(2024, 1, 15), "winter"),
        (date(2024, 4, 15), "spring"),
        (date(2024, 7, 15), "summer"),
        (date(2024, 10, 15), "autumn"),
    ],
)
def test_imagery_season_is_derived_from_acquisition_date(acquired, expected):
    assert imagery_season(acquired) == expected
