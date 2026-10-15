"""D06 empirical marker-hotspot tests."""

import numpy as np
import pytest

from thesis.render.marker import draw_marker, marker_hotspot, render_at_tip
from thesis.render.presets import load_presets


def _locate_tip(rendered, shape):
    changed = np.any(rendered != 0, axis=0)
    rows, cols = np.nonzero(changed)
    if shape == "google_pin":
        bottom = rows.max()
        bottom_cols = cols[rows == bottom]
        return int(bottom), int(bottom_cols.max())
    weights = rendered.astype(float).sum(axis=0)
    row = int(np.floor((np.average(np.arange(512), weights=weights.sum(axis=1))) + 0.5))
    col = int(np.floor((np.average(np.arange(512), weights=weights.sum(axis=0))) + 0.5))
    return row, col


@pytest.mark.parametrize("shape", ["dot", "crosshair", "google_pin"])
def test_every_marker_shape_has_empirical_tip_on_se_central_pixel(shape):
    blank = np.zeros((3, 512, 512), dtype=np.uint8)

    rendered = draw_marker(
        blank,
        px=256,
        py=256,
        shape=shape,
        color=(230, 20, 20),
        size_px=20,
        opacity=1.0,
    )

    assert _locate_tip(rendered, shape) == (256, 256)
    assert np.any(rendered[:, 256, 256] != 0)


def test_every_configured_marker_variant_uses_the_same_hotspot():
    presets = load_presets(
        {
            "shapes": ["dot", "crosshair", "google_pin"],
            "colors": {"red": [230, 20, 20], "blue": [0, 130, 255], "green": [30, 255, 130]},
            "size_px": 20,
            "outline_px": 2,
            "opacity": 1.0,
        }
    )

    for preset in presets.values():
        rendered = render_at_tip(np.zeros((3, 512, 512), dtype=np.uint8), preset)
        assert _locate_tip(rendered, preset["shape"]) == (256, 256)


def test_marker_hotspot_is_explicit_for_each_shape():
    assert marker_hotspot("dot", 20) == (10, 10)
    assert marker_hotspot("crosshair", 20) == (10, 10)
    assert marker_hotspot("google_pin", 20) == (10, 19)


def test_marker_free_render_preserves_crop_and_authoritative_tip():
    crop = np.random.default_rng(4).integers(0, 256, (3, 512, 512), dtype=np.uint8)

    rendered = render_at_tip(crop, None)

    np.testing.assert_array_equal(rendered, crop)


@pytest.mark.parametrize("shape", [(3, 511, 512), (3, 512, 511), (4, 512, 512)])
def test_render_at_tip_requires_a_native_512_rgb_crop(shape):
    with pytest.raises(ValueError, match="512"):
        render_at_tip(np.zeros(shape, dtype=np.uint8), None)
