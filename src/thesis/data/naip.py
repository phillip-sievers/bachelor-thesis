import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
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
from rasterio.windows import Window


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
    return [
        {
            "id": item.id,
            "gsd": float(item.properties["gsd"]),
            "year": int(item.properties["naip:year"]),
            "state": item.properties["naip:state"],
            "href": item.assets[config["asset"]].href,
            "item": item,
        }
        for item in search.items()
    ]


def _open_item(record: Mapping[str, Any]) -> DatasetReader:
    signed = pc.sign(record["item"])
    return rasterio.open(signed.assets["image"].href)


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


def fetch_tile(
    lat: float,
    lon: float,
    gsd: float,
    size: int,
    config: Mapping[str, Any],
) -> Tile:
    # time.sleep(0.75)

    cache_dir = Path(config["cache_dir"])
    items = search_items(lon, lat, config)
    item = pick_item(items, gsd, tolerance=config["gsd_tolerance"])
    with _open_item(item) as src:
        x, y = transform_coords(src.crs, lon, lat)
        assert_native_gsd(src.res, gsd, tol=config["gsd_tolerance"])
        window, px, py = window_for(src.transform, src.width, src.height, x, y, size)
        key = cache_key(item["id"], window.col_off, window.row_off, size)
        if (cache_hit := load_cached(cache_dir, key)) is not None:
            return cache_hit
        arr = read_window(src, window)
        meta = build_meta(item, src, window, px, py, lon, lat, x, y, size, gsd)
    store_atomic(cache_dir, key, arr, meta)
    return Tile(arr, meta)
