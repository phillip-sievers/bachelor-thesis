"""Native NAIP discovery and integer scene/crop windows (D04--D06, D17)."""

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import planetary_computer as pc
import rasterio
from numpy.typing import NDArray
from pystac_client import Client
from rasterio.io import DatasetReader
from rasterio.transform import Affine, rowcol
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping as geometry_mapping
from shapely.geometry import shape as shapely_shape


class TileOutOfBounds(Exception):
    pass


class ResolutionMismatch(Exception):
    pass


class NoValidItem(Exception):
    pass


class Tile(NamedTuple):
    array: NDArray[np.uint8]
    metadata: dict[str, Any]


def cache_key(
    item_id: str,
    col_off: int | float,
    row_off: int | float,
    size: int,
) -> str:
    raw = f"{item_id}|{col_off}|{row_off}|{size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def window_cache_key(item_id: str, window: Window) -> str:
    raw = f"{item_id}|{int(window.col_off)}|{int(window.row_off)}|{int(window.width)}|{int(window.height)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.npy", cache_dir / f"{key}.json"


def pick_item(
    items: Sequence[Mapping[str, Any]],
    gsd: float,
    tolerance: float,
) -> Mapping[str, Any]:
    ok = [i for i in items if abs(i["gsd"] - gsd) <= tolerance]
    if not ok:
        raise NoValidItem(f"No items with GSD {gsd}")
    return min(ok, key=lambda i: (-i["year"], i["id"]))


def pick_containing_native_item(
    items: Sequence[Mapping[str, Any]],
    scene_geometry,
    gsd: float,
    tolerance: float,
) -> Mapping[str, Any]:
    return containing_native_items(items, scene_geometry, gsd, tolerance)[0]


def containing_native_items(
    items: Sequence[Mapping[str, Any]],
    scene_geometry,
    gsd: float,
    tolerance: float,
) -> tuple[Mapping[str, Any], ...]:
    native = []
    for item in items:
        geometry = item["geometry"]
        if isinstance(geometry, Mapping):
            geometry = shapely_shape(geometry)
        if abs(float(item["gsd"]) - gsd) <= tolerance and geometry.intersects(scene_geometry):
            native.append(item)
    if not native:
        raise NoValidItem(f"No items with native GSD {gsd}")
    latest_year = max(int(item["year"]) for item in native)
    latest = sorted((item for item in native if int(item["year"]) == latest_year), key=lambda item: str(item["id"]))
    containing = []
    for item in latest:
        geometry = item["geometry"]
        if isinstance(geometry, Mapping):
            geometry = shapely_shape(geometry)
        if geometry.covers(scene_geometry):
            containing.append(item)
    if not containing:
        raise TileOutOfBounds(
            f"no single native-GSD NAIP item from latest vintage {latest_year} contains the enlarged scene"
        )
    return tuple(containing)


def aligned_scene_window(
    transform: Affine,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> Window:
    fractional = from_bounds(*bounds, transform=transform)
    col_off = math.floor(fractional.col_off)
    row_off = math.floor(fractional.row_off)
    col_end = math.ceil(fractional.col_off + fractional.width)
    row_end = math.ceil(fractional.row_off + fractional.height)
    if col_off < 0 or row_off < 0 or col_end > width or row_end > height:
        raise TileOutOfBounds(
            f"aligned scene window ({col_off}, {row_off}, {col_end}, {row_end}) exceeds ({width}, {height})"
        )
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def crop_window_for_slot(
    row: int,
    col: int,
    scene_shape: tuple[int, int],
    size: int,
    tip_row: int,
    tip_col: int,
) -> Window:
    if not 0 <= tip_row < size or not 0 <= tip_col < size:
        raise ValueError("tip must lie inside the crop")
    row_off = int(row) - tip_row
    col_off = int(col) - tip_col
    if row_off < 0 or col_off < 0 or row_off + size > scene_shape[0] or col_off + size > scene_shape[1]:
        raise TileOutOfBounds(f"slot ({row}, {col}) cannot place tip ({tip_row}, {tip_col}) in a {size}px crop")
    return Window(col_off, row_off, size, size)


def imagery_season(acquired: date) -> str:
    if acquired.month in (12, 1, 2):
        return "winter"
    if acquired.month in (3, 4, 5):
        return "spring"
    if acquired.month in (6, 7, 8):
        return "summer"
    return "autumn"


def window_for(
    transform: Affine,
    width: int,
    height: int,
    x: float,
    y: float,
    size: int,
) -> tuple[Window, int, int]:
    row, col = rowcol(transform, x, y)
    row, col = int(row), int(col)
    col_off, row_off = col - size // 2, row - size // 2
    if col_off < 0 or row_off < 0 or col_off + size > width or row_off + size > height:
        raise TileOutOfBounds(
            f"window at ({col_off}, {row_off}) with size {size} is out of bounds for image of size ({width}, {height})"
        )
    return Window(col_off, row_off, size, size), col - col_off, row - row_off


def transform_coords(dst_crs, lon: float, lat: float) -> tuple[float, float]:
    (x,), (y,) = warp_transform("EPSG:4326", dst_crs, [lon], [lat])
    return x, y


def assert_native_gsd(
    res: tuple[float, float],
    gsd: float,
    tol: float,
) -> None:
    if any(abs(r - gsd) > tol for r in res):
        raise ResolutionMismatch(f"native GSD {res} != GSD {gsd} with tolerance {tol}")


def store_atomic(
    cache_dir: Path,
    key: str,
    array: NDArray[np.uint8],
    metadata: Mapping[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy, js = cache_paths(cache_dir, key)

    tmp_npy = cache_dir / f"{key}.tmp.npy"
    np.save(tmp_npy, array)
    os.replace(tmp_npy, npy)

    tmp_js = cache_dir / f"{key}.tmp.json"
    tmp_js.write_text(json.dumps(metadata))
    os.replace(tmp_js, js)


def load_cached(cache_dir: Path, key: str) -> Tile | None:
    npy, js = cache_paths(cache_dir, key)
    if not npy.exists() or not js.exists():
        return None
    array = np.load(npy)
    metadata = json.loads(js.read_text())
    return Tile(array, metadata)


def search_items(
    lon: float,
    lat: float,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    catalog = Client.open(config["stac_endpoint"], modifier=pc.sign_inplace)
    search = catalog.search(
        collections=[config["collection"]],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{config['year_min']}-01-01/..",
    )
    return [_item_record(item, config) for item in search.items()]


def _item_record(item, config: Mapping[str, Any]) -> dict[str, Any]:
    acquired = item.properties.get("datetime") or item.properties.get("start_datetime")
    return {
        "id": item.id,
        "gsd": float(item.properties["gsd"]),
        "year": int(item.properties["naip:year"]),
        "state": item.properties["naip:state"],
        "href": item.assets[config["asset"]].href,
        "geometry": shapely_shape(item.geometry),
        "acquired": acquired,
        "item": item,
    }


def search_items_for_geometry(scene_geometry_wgs84, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    catalog = Client.open(config["stac_endpoint"], modifier=pc.sign_inplace)
    search = catalog.search(
        collections=[config["collection"]],
        intersects=geometry_mapping(scene_geometry_wgs84),
        datetime=f"{config['year_min']}-01-01/..",
    )
    return [_item_record(item, config) for item in search.items()]


def _open_item(record: Mapping[str, Any]) -> DatasetReader:
    signed = pc.sign(record["item"])
    return rasterio.open(signed.assets["image"].href)


def item_grid_metadata(record: Mapping[str, Any], gsd: float, tolerance: float) -> dict[str, Any]:
    with _open_item(record) as src:
        assert_native_gsd(src.res, gsd, tolerance)
        acquired = record.get("acquired")
        acquired_date = (
            datetime.fromisoformat(str(acquired).replace("Z", "+00:00")).date()
            if acquired
            else date(int(record["year"]), 7, 1)
        )
        return {
            "item_id": str(record["id"]),
            "src_crs": src.crs.to_string(),
            "transform": tuple(float(value) for value in src.transform[:6]),
            "width": int(src.width),
            "height": int(src.height),
            "native_gsd": tuple(float(value) for value in src.res),
            "acquired_date": acquired_date.isoformat(),
            "season": imagery_season(acquired_date),
            "href": str(record["href"]).split("?")[0],
        }


def read_item_window(record: Mapping[str, Any], window: Window, gsd: float, tolerance: float) -> Tile:
    with _open_item(record) as src:
        assert_native_gsd(src.res, gsd, tolerance)
        array = read_window(src, window)
        transform = src.window_transform(window)
        metadata = item_grid_metadata_from_open(record, src, window, transform)
    return Tile(array, metadata)


def item_grid_metadata_from_open(
    record: Mapping[str, Any],
    src: DatasetReader,
    window: Window,
    transform: Affine,
) -> dict[str, Any]:
    acquired = record.get("acquired")
    acquired_date = (
        datetime.fromisoformat(str(acquired).replace("Z", "+00:00")).date()
        if acquired
        else date(int(record["year"]), 7, 1)
    )
    return {
        "item_id": str(record["id"]),
        "year": int(record["year"]),
        "state": str(record["state"]),
        "acquired_date": acquired_date.isoformat(),
        "season": imagery_season(acquired_date),
        "native_gsd": [float(value) for value in src.res],
        "src_crs": src.crs.to_string(),
        "transform": [float(value) for value in transform[:6]],
        "col_off": int(window.col_off),
        "row_off": int(window.row_off),
        "width": int(window.width),
        "height": int(window.height),
        "href": str(record["href"]).split("?")[0],
    }


def read_window(src: DatasetReader, window: Window) -> NDArray[np.uint8]:
    arr = src.read([1, 2, 3], window=window)  # drop NIR
    if arr.shape != (3, window.height, window.width):
        raise ValueError(f"expected shape (3, {window.height}, {window.width}), got {arr.shape}")
    return arr


def build_meta(
    item: Mapping[str, Any],
    src: DatasetReader,
    window: Window,
    px: int,
    py: int,
    lon: float,
    lat: float,
    x: float,
    y: float,
    size: int,
    gsd: float,
) -> dict[str, Any]:
    return {
        "item_id": item["id"],
        "year": item["year"],
        "state": item["state"],
        "native_gsd": [float(r) for r in src.res],
        "src_crs": src.crs.to_string(),
        "col_off": int(window.col_off),
        "row_off": int(window.row_off),
        "size": int(size),
        "px": int(px),
        "py": int(py),
        "req_lon": lon,
        "req_lat": lat,
        "x": x,
        "y": y,
        "extent_m": size * gsd,
        "href": item["href"].split("?")[0],
        "fetched_at": datetime.now(UTC).isoformat(),
    }


# v1 old
# def fetch_tile(
#     lat: float,
#     lon: float,
#     gsd: float,
#     size: int,
#     config: Mapping[str, Any],
# ) -> Tile:
#     # time.sleep(0.75)

#     cache_dir = Path(config["cache_dir"])
#     items = search_items(lon, lat, config)
#     item = pick_item(items, gsd, tolerance=config["gsd_tolerance"])
#     with _open_item(item) as src:
#         x, y = transform_coords(src.crs, lon, lat)
#         assert_native_gsd(src.res, gsd, tol=config["gsd_tolerance"])
#         window, px, py = window_for(src.transform, src.width, src.height, x, y, size)
#         key = cache_key(item["id"], window.col_off, window.row_off, size)
#         if (cache_hit := load_cached(cache_dir, key)) is not None:
#             return cache_hit
#         arr = read_window(src, window)
#         meta = build_meta(item, src, window, px, py, lon, lat, x, y, size, gsd)
#     store_atomic(cache_dir, key, arr, meta)
#     return Tile(arr, meta)
