from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


def load_models_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_arm_config(arm: str, config: dict | None = None) -> dict:
    if config is None:
        config = load_models_config()

    if arm not in config["arms"]:
        available = ", ".join(sorted(config["arms"]))
        raise ValueError(f"Unknown arm '{arm}'. Available arms: {available}")

    arm_config = dict(config["arms"][arm])
    backend = arm_config.get("backend")

    if backend is not None:
        defaults = config.get("backend_defaults", {}).get(backend, {})
        retry = config.get("retry", {})
        merged = dict(defaults)
        merged.update(retry)
        merged.update(arm_config)
        arm_config = merged

    arm_config["arm"] = arm
    return arm_config
