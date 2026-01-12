"""Experiment registry loader."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_experiments(registry_path: Path) -> Dict[str, Any]:
    """Load experiments from a registry YAML file."""
    return yaml.safe_load(registry_path.read_text())["experiments"]
