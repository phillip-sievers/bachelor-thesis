from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from math import cos, hypot, sin, tau

from shapely import maximum_inscribed_circle
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from thesis.geometry.signed_distance import signed_distance_m


@dataclass(frozen=True)
class DisplacedPin:
    target_signed_distance_m: float | None
    actual_signed_distance_m: float | None
    bearing_radians: float | None
    point: Point | None


def point_at_signed_distance(
    footprint: Polygon,
    anchor: Point,
    target_m: float,
    bearing_radians: float,
    tolerance_m: float,
) -> Point | None:
    level = footprint.buffer(target_m, quad_segs=32)
    minx, miny, maxx, maxy = footprint.bounds
    reach = max(hypot(anchor.x - x, anchor.y - y) for x in (minx, maxx) for y in (miny, maxy))
    ray_length = reach + abs(target_m) + 1
    ray = LineString(
        [
            anchor,
            Point(
                anchor.x + ray_length * cos(bearing_radians),
                anchor.y + ray_length * sin(bearing_radians),
            ),
        ]
    )
    crossing = ray.intersection(level.boundary)
    if crossing.is_empty:
        return None
    pin = nearest_points(anchor, crossing)[1]
    actual_m = signed_distance_m(pin, footprint)
    if abs(actual_m - target_m) > tolerance_m:
        raise RuntimeError(f"signed-distance tolerance: target={target_m}, actual={actual_m}")
    return pin


def generate_displacements(
    footprint: Polygon,
    anchor: Point,
    target_distances_m: Iterable[float],
    seed: int,
    building_id: str,
    tolerance_m: float = 0.1,
) -> tuple[DisplacedPin, ...]:
    digest = sha256(f"{seed}:{building_id}".encode()).digest()
    bearing = tau * int.from_bytes(digest[:8], "big") / 2**64
    results = []
    for target_m in target_distances_m:
        pin = point_at_signed_distance(footprint, anchor, target_m, bearing, tolerance_m)
        actual_m = signed_distance_m(pin, footprint) if pin is not None else None
        results.append(DisplacedPin(float(target_m), actual_m, bearing, pin))

    circle = maximum_inscribed_circle(footprint, tolerance=tolerance_m)
    center = Point(circle.coords[0])
    results.append(DisplacedPin(None, signed_distance_m(center, footprint), None, center))
    return tuple(results)
