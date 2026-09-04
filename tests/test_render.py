from itertools import product

import numpy as np
import pytest

from thesis.render.marker import draw_marker
from thesis.render.presets import load_presets


def test_marker_leaves_pixels_outside_footprint_untouched():
    tile = np.random.randint(0, 255, size=(3, 64, 64), dtype=np.uint8)
    out = draw_marker(
        tile,
        px=32,
        py=32,
        shape="dot",
        color=(230, 20, 20),
        size_px=10,
        opacity=1.0,
    )
    x0, y0, x1, y1 = 32 - 10, 32 - 10, 32 + 10, 32 + 10
    mask = np.ones((64, 64), dtype=bool)
    mask[y0:y1, x0:x1] = False

    np.testing.assert_array_equal(np.transpose(out, (1, 2, 0))[mask], np.transpose(tile, (1, 2, 0))[mask])


def test_colors_matching_input():
    tile = np.zeros((3, 64, 64), dtype=np.uint8)
    color = (230, 20, 20)
    out = draw_marker(tile, px=32, py=32, shape="dot", color=color, size_px=10, opacity=1.0)
    center_rgb = np.transpose(out, (1, 2, 0))[32, 32, :3]
    assert tuple(center_rgb) == color


@pytest.mark.parametrize("size_px", [15, 20])
def test_crosshair_stroke_stays_inside_its_sprite(size_px):
    tile = np.zeros((3, 64, 64), dtype=np.uint8)

    out = draw_marker(
        tile,
        px=32,
        py=32,
        shape="crosshair",
        color=(230, 20, 20),
        size_px=size_px,
        opacity=1.0,
    )

    start = 32 - size_px // 2
    sprite = out[:, start : start + size_px, start : start + size_px]
    assert sprite.any()
    assert not sprite[:, 0, :].any()
    assert not sprite[:, -1, :].any()
    assert not sprite[:, :, 0].any()
    assert not sprite[:, :, -1].any()


def test_google_pin_uses_bottom_center_as_hotspot():
    tile = np.zeros((3, 64, 64), dtype=np.uint8)

    out = draw_marker(
        tile,
        px=32,
        py=32,
        shape="google_pin",
        color=(230, 20, 20),
        size_px=20,
        opacity=1.0,
    )

    changed = np.any(out != tile, axis=0)
    changed_y, _ = np.nonzero(changed)
    assert changed_y.max() == 32
    assert changed[32, 32]


def test_presets_loading():
    versions = list(
        product(
            ["dot", "crosshair", "google_pin"],
            [(230, 20, 20), (0, 130, 255), (30, 255, 130)],
        )
    )
    presets = load_presets(
        {
            "shapes": ["dot", "crosshair", "google_pin"],
            "colors": {
                "red": [230, 20, 20],
                "neon_blue": [0, 130, 255],
                "neon_green": [30, 255, 130],
            },
            "size_px": 20,
            "outline_px": 2,
            "opacity": 1.0,
        }
    )
    actual = [(p["shape"], p["color"]) for p in presets.values()]
    assert actual == versions
