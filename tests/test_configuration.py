from pathlib import Path

import yaml


def test_imports():
    pass


def test_run_config_has_all_references():
    run = yaml.safe_load(Path("configs/run.yaml").read_text())
    for file in Path("configs").glob("*.yaml"):
        filename = file.name[:-5]  # without yaml
        if filename == "run":
            continue
        assert filename in run, f"{filename} not found in run.yaml"
