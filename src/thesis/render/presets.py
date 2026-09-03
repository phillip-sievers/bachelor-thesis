from itertools import product


def load_presets(config: dict) -> dict[str, dict]:
    presets = {}
    for shape, (color_name, rgb) in product(config["shapes"], config["colors"].items()):
        presets[f"{shape}_{color_name}"] = {
            "shape": shape,
            "color": tuple(rgb),
            "size_px": config["size_px"],
            "opacity": config["opacity"],
        }
    return presets
