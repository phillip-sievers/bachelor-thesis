import io
from importlib.resources import files

import cairo
import cairosvg
import numpy as np


def surface_to_rgba(surface: cairo.ImageSurface) -> np.ndarray:
    w, h = surface.get_width(), surface.get_height()
    buf = np.ndarray(shape=(h, w, 4), dtype=np.uint8, buffer=surface.get_data())
    bgra = buf.copy()
    rgba = bgra[..., [2, 1, 0, 3]]
    return rgba


def composite_patch(
    tile_chw: np.ndarray,
    patch_rgba: np.ndarray,
    cx: int,
    cy: int,
    hotspot: tuple[int, int] | None = None,
) -> np.ndarray:
    out = tile_chw.copy()
    tile_hwc = np.transpose(out, (1, 2, 0))
    patch_h, patch_w = patch_rgba.shape[:2]
    hotspot_x, hotspot_y = hotspot or (patch_w // 2, patch_h // 2)
    x0, y0 = cx - hotspot_x, cy - hotspot_y

    region = tile_hwc[y0 : y0 + patch_h, x0 : x0 + patch_w]
    alpha = patch_rgba[:, :, 3:4] / 255.0
    region[:] = (patch_rgba[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)

    return np.transpose(tile_hwc, (2, 0, 1))


def draw_dot(context: cairo.Context, x, y, px_size, color, opacity) -> cairo.Context:
    r, g, b = [c / 255 for c in color]
    context.set_source_rgba(r, g, b, opacity)
    context.arc(x, y, px_size / 2, 0, 2 * np.pi)
    context.fill()
    return context


def draw_crosshair(context: cairo.Context, x, y, px_size, color, opacity):
    r, g, b = [c / 255 for c in color]
    line_width = px_size / 10
    radius = px_size / 2 - line_width / 2 - 1
    context.set_source_rgba(r, g, b, opacity)
    context.set_line_width(line_width)
    context.arc(x, y, radius, 0, 2 * np.pi)
    context.move_to(x - radius, y)
    context.line_to(x + radius, y)
    context.move_to(x, y - radius)
    context.line_to(x, y + radius)
    context.stroke()
    return context


def draw_google_pin(px_size, color) -> cairo.ImageSurface:
    def _load_pin_svg_text() -> str:
        svg_path = files("thesis.render") / "assets" / "Google_Maps_pin.svg"
        return svg_path.read_text()

    def _recolor_pin_svg(svg_text: str, color: tuple[int, int, int]) -> str:
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        svg_text = svg_text.replace("fill:#ff4646", f"fill:{hex_color}")
        svg_text = svg_text.replace("stroke:#d73534", f"stroke:{hex_color}")
        return svg_text

    svg_text = _recolor_pin_svg(_load_pin_svg_text(), color)
    png_bytes = cairosvg.svg2png(bytestring=svg_text.encode(), output_width=px_size, output_height=px_size)
    assert png_bytes, "Failed to convert SVG to PNG"
    return cairo.ImageSurface.create_from_png(io.BytesIO(png_bytes))


def draw_marker(array, px, py, shape, color, size_px, opacity):
    if shape == "google_pin":
        surface = draw_google_pin(size_px, color)
    else:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size_px, size_px)
        ctx = cairo.Context(surface)
        if shape == "dot":
            draw_dot(ctx, size_px / 2, size_px / 2, size_px, color, opacity)
        elif shape == "crosshair":
            draw_crosshair(ctx, size_px / 2, size_px / 2, size_px, color, opacity)
        else:
            raise ValueError(f"Unknown shape: {shape}")
    patch_rgba = surface_to_rgba(surface)
    hotspot = (patch_rgba.shape[1] // 2, patch_rgba.shape[0] - 1) if shape == "google_pin" else None
    return composite_patch(array, patch_rgba, px, py, hotspot=hotspot)
