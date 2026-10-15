from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent.parent.parent
YAML_CONFIGS = REPO_ROOT / "configs"
KEYS = ("bins", "imagery", "overture", "models", "render", "sampling", "prompts")
# These configs are not in run.yaml. Thus the frozen configuration hash does not change.
UNHASHED_KEYS = ("subsamples", "inference", "scoring")


def load_run_config() -> dict:
    path = Path(REPO_ROOT / "configs" / "run.yaml")
    return yaml.safe_load(path.read_text())


def load_config(key: str) -> dict:
    if key in UNHASHED_KEYS:
        return yaml.safe_load((YAML_CONFIGS / f"{key}.yaml").read_text())
    run_config = load_run_config()
    if key not in KEYS:
        raise ValueError(f"Invalid config key: {key}. Must be one of {KEYS}")

    path = Path(REPO_ROOT / run_config[key])
    return yaml.safe_load(path.read_text())
