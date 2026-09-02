"""
Calculate the signed distance of a point to a polygon. And utils around it
"""

from shapely.geometry import Point


def utm_epsg(longitude: float, latitude: float) -> int:
    return 2


def to_utm(geom, epsg: int) -> Point:
    return Point(0, 0)


def is_inside(point, polygon) -> bool:
    return False


def signed_distance_m(point, polygon) -> float:
    return 0.0
