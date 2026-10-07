import json
import os
from pathlib import Path

DEFAULT_PATH = "config/config.json"


def _resolve(value):
    """Strings of the form "env:NAME" are replaced by the environment variable NAME."""
    if isinstance(value, str) and value.startswith("env:"):
        return os.environ.get(value[4:], "")
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def load_config(path=None):
    path = Path(path or os.environ.get("AVGEN_CONFIG", DEFAULT_PATH))
    if not path.exists():
        raise FileNotFoundError(
            f"Config {path} not found. Copy config/config.example.json to {DEFAULT_PATH}."
        )
    with open(path, encoding="utf-8") as f:
        return _resolve(json.load(f))
