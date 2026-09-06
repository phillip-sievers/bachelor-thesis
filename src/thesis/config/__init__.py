from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent.parent.parent
YAML_CONFIGS = REPO_ROOT / "configs"
KEYS = ("bins", "imagery", "overture", "models", "render", "sampling", "prompts")


def load_run_config() -> dict:
    path = Path(REPO_ROOT / "configs" / "run.yaml")
    return yaml.safe_load(path.read_text())


def load_config(key: str) -> dict:
    run_config = load_run_config()
    if key not in KEYS:
        raise ValueError(f"Invalid config key: {key}. Must be one of {KEYS}")

    path = Path(REPO_ROOT / run_config[key])
    return yaml.safe_load(path.read_text())
