"""
Calculate the signed distance of a point to a polygon. And utils around it
"""

from functools import lru_cache

from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform


def utm_epsg(longitude: float, latitude: float) -> int:
    zone = int((longitude + 180) // 6) + 1
    return (26900 if latitude >= 0 else 32700) + zone


@lru_cache(maxsize=16)
def _transformer(epsg: int) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def to_utm(geom, epsg: int):
    tf = _transformer(epsg)
    return shapely_transform(tf.transform, geom)


def is_inside(point: Point, polygon: Polygon) -> bool:
    return polygon.covers(point)


def signed_distance_m(point: Point, polygon: Polygon) -> float:
    """negative means inside, positive outside. distance to nearest edge of shape"""
    d = polygon.boundary.distance(point)
    return -d if is_inside(point, polygon) else d
