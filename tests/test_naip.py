from pathlib import Path

import numpy as np
import pytest
import yaml
from rasterio.transform import from_origin, xy
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

from thesis.data.naip import (
    NoValidItem,
    ResolutionMismatch,
    TileOutOfBounds,
    assert_native_gsd,
    cache_key,
    fetch_tile,
    load_cached,
    pick_item,
    store_atomic,
    window_for,
)

QUAD = dict(
    transform=from_origin(600_000.0, 4_500_000.0, 0.6, 0.6),
    width=12330,
    height=9600,
)
SIZE = 512


def test_window_centres_on_point():
    col, row = 1000, 800
    x = 600_000.0 + 0.6 * col + 0.3
    y = 4_500_000.0 - 0.6 * row - 0.3

    win, px, py = window_for(**QUAD, x=x, y=y, size=SIZE)

    assert (win.col_off, win.row_off) == (col - SIZE // 2, row - SIZE // 2)
    assert (win.width, win.height) == (SIZE, SIZE)
    assert (px, py) == (SIZE // 2, SIZE // 2)


# def test_px_py_from_window()


@pytest.mark.parametrize("col,row", [(100, 5000), (12300, 5000), (5000, 100), (5000, 9500)])
def test_window_off_quad_edge_rejects(col, row):
    x = 600_000.0 + 0.6 * col + 0.3
    y = 4_500_000.0 - 0.6 * row - 0.3
    with pytest.raises(TileOutOfBounds):
        window_for(**QUAD, x=x, y=y, size=SIZE)


def test_ground_extent_is_307m():
    x = 600_000.0 + 0.6 * 1000 + 0.3
    y = 4_500_000.0 - 0.6 * 800 - 0.3
    win, _, _ = window_for(**QUAD, x=x, y=y, size=SIZE)

    left, bottom, right, top = window_bounds(win, QUAD["transform"])
    assert right - left == pytest.approx(307.2, abs=1e-6)
    assert top - bottom == pytest.approx(307.2, abs=1e-6)


def test_px_py_roundtrips_to_requested_point():
    x = 600_000.0 + 0.6 * 1000 + 0.3
    y = 4_500_000.0 - 0.6 * 800 - 0.3
    win, px, py = window_for(**QUAD, x=x, y=y, size=SIZE)

    wt = window_transform(win, QUAD["transform"])
    bx, by = xy(wt, py, px)
    assert (bx, by) == pytest.approx((x, y), abs=0.6)


@pytest.mark.parametrize("res", [(1.0, 1.0), (0.3, 0.3), (0.6, 1.0)])
def test_wrong_native_gsd_raises(res):
    with pytest.raises(ResolutionMismatch):
        assert_native_gsd(res, gsd=0.6, tol=0.01)


# this should never happen, safety
def test_near_06_is_accepted():
    assert_native_gsd((0.5999999, 0.6000001), gsd=0.6, tol=0.01)


# from notebook
ITEMS = [
    {"id": "m_4007317_nw_18_060_20200612", "gsd": 0.6, "year": 2020},
    {"id": "m_4007317_nw_18_060_20220719", "gsd": 0.6, "year": 2022},
    {"id": "m_4007317_nw_18_030_20220719", "gsd": 0.3, "year": 2022},
]


def test_pick_item_newest_wins():
    assert pick_item(ITEMS, gsd=0.6, tolerance=0.01)["id"].endswith("20220719")
    assert pick_item(ITEMS, gsd=0.6, tolerance=0.01)["gsd"] == 0.6


def test_pick_item_tiebreak_is_lowest_id():
    # also just for safety
    same_year = [
        {"id": "m_bbb", "gsd": 0.6, "year": 2022},
        {"id": "m_aaa", "gsd": 0.6, "year": 2022},
    ]
    assert pick_item(same_year, gsd=0.6, tolerance=0.01)["id"] == "m_aaa"


def test_pick_item_rejects_non_06m():
    with pytest.raises(NoValidItem):
        pick_item([ITEMS[2]], gsd=0.6, tolerance=0.01)["id"]


def test_cache_key_stable():
    a = cache_key("m_4007317_nw_18_060_20220719", 744, 544, 512)
    b = cache_key("m_4007317_nw_18_060_20220719", 744, 544, 512)
    assert a == b and len(a) == 16


@pytest.mark.parametrize(
    "args",
    [
        ("m_other", 744, 544, 512),
        ("m_4007317_nw_18_060_20220719", 745, 544, 512),
        ("m_4007317_nw_18_060_20220719", 744, 545, 512),
        ("m_4007317_nw_18_060_20220719", 744, 544, 256),
    ],
)
def test_cache_key_changes(args):
    base = cache_key("m_4007317_nw_18_060_20220719", 744, 544, 512)
    assert cache_key(*args) != base


def test_store_then_load(tmp_path):
    arr = np.zeros((3, SIZE, SIZE), dtype=np.uint8)
    meta = {"item_id": "m_x", "col_off": 744, "row_off": 544}
    store_atomic(tmp_path, "abc123", arr, meta)

    tile = load_cached(tmp_path, "abc123")
    assert tile.array.shape == (3, SIZE, SIZE)  # type: ignore
    assert tile.metadata["item_id"] == "m_x"  # type: ignore


def test_partial_write_is_not_a_cache_hit(tmp_path):
    key = cache_key("m_4007317_nw_18_060_20220719", 744, 544, SIZE)
    (tmp_path / f"{key}.npy").write_bytes(b"truncated")  # no .json
    assert load_cached(tmp_path, key) is None


FARMHOUSE = (44.075796, -105.718358)


@pytest.mark.network
def test_fetch_tile_against_planetary_computer(tmp_path):
    config = yaml.safe_load(Path("configs/imagery.yaml").read_text())
    config["cache_dir"] = str(tmp_path)
    lat, lon = FARMHOUSE

    tile = fetch_tile(lat, lon, gsd=config["gsd_m"], size=config["image_px"], config=config)

    assert tile.array.shape == (3, config["image_px"], config["image_px"])
    assert tile.array.dtype == np.uint8
    assert tile.metadata["native_gsd"] == pytest.approx([0.6, 0.6])
    assert tile.metadata["extent_m"] == pytest.approx(307.2)
    assert "?" not in tile.metadata["href"]  # token stripped

    again = fetch_tile(lat, lon, gsd=config["gsd_m"], size=config["image_px"], config=config)
    assert again.metadata["fetched_at"] == tile.metadata["fetched_at"]
    assert len(list(tmp_path.glob("*.npy"))) == 1
