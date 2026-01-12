"""Data caching utilities for analysis scripts."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from responserank.llm.analysis.plotting_utils import get_loss_type_colors
from responserank.llm.analysis.results_loader import load_results


def compute_run_styling(
    experiment_labels: List[str],
    style_overrides: Optional[Dict[str, Dict[str, str]]],
) -> Dict[str, Dict[str, str]]:
    """Compute run styling for experiment labels.

    Args:
        experiment_labels: List of experiment labels
        style_overrides: Optional dict of style overrides per experiment
                        Format: {"pattern": {"color": "red", "marker": "o", "linestyle": "-"}, ...}
                        Patterns support regex matching

    Returns:
        Dict with computed styling: {"run_colors": {}, "run_markers": {}, "run_linestyles": {}}
    """
    default_colors = get_loss_type_colors(experiment_labels)
    default_markers = ["o", "s", "D", "^", "p", "*", "X", "v", "<", ">"]
    default_linestyle = "-"
    style_overrides = {} if style_overrides is None else style_overrides

    run_colors = {}
    run_markers = {}
    run_linestyles = {}

    for i, rid in enumerate(experiment_labels):
        run_colors[rid] = default_colors[rid]
        run_markers[rid] = default_markers[i % len(default_markers)]
        run_linestyles[rid] = default_linestyle

        for pattern, overrides in style_overrides.items():
            if re.match(pattern, rid):
                if "color" in overrides:
                    run_colors[rid] = overrides["color"]
                if "marker" in overrides:
                    run_markers[rid] = overrides["marker"]
                if "linestyle" in overrides:
                    run_linestyles[rid] = overrides["linestyle"]
                break  # Use first matching pattern

    return {
        "run_colors": run_colors,
        "run_markers": run_markers,
        "run_linestyles": run_linestyles,
    }


def compute_display_name_aliases(
    experiment_labels: List[str],
    display_name_overrides: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Compute legend label overrides for collected experiments."""

    if not display_name_overrides:
        return {}

    alias_map: Dict[str, str] = {}

    for label in experiment_labels:
        for pattern, override in display_name_overrides.items():
            if re.match(pattern, label):
                alias_map[label] = override
                break

    return alias_map


def save_cached_data(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    project_name: str,
    egid_patterns: Set[str],
    experiments: Dict[str, Any],
    style_overrides: Optional[Dict[str, Dict[str, str]]],
    display_name_overrides: Optional[Dict[str, str]],
    seed_filter: Optional[int],
    command_line: Optional[str],
) -> None:
    """
    Save DataFrame and metadata to cache files.

    Args:
        df: DataFrame from load_results()
        output_dir: Output directory (will contain data.pkl and meta.json)
        project_name: wandb project name
        egid_patterns: Set of regex patterns that were collected
        experiments: Full experiments dict from registry (for generate command)
        style_overrides: Optional style specifications
        display_name_overrides: Optional legend label overrides
        seed_filter: Optional seed filter
        command_line: Optional full command line for reference
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "data.pkl"
    meta_path = output_dir / "meta.json"

    df.to_pickle(pkl_path)

    all_egids = sorted(df["egid"].unique())
    computed_styling = compute_run_styling(all_egids, style_overrides)
    computed_display_names = compute_display_name_aliases(
        all_egids, display_name_overrides
    )

    metadata = {
        "experiments": experiments,
        "experiment_specs": {
            "egid_patterns": sorted(egid_patterns),
        },
        "styling": style_overrides or {},
        "computed_styling": computed_styling,
        "display_name_overrides": display_name_overrides or {},
        "computed_display_names": computed_display_names,
        "data_summary": {
            "unique_egids": sorted(df["egid"].unique().tolist()),
            "unique_display_names": sorted(df["display_name"].unique().tolist()),
            "seeds": sorted(df["seed"].unique().tolist()),
            "fractions": sorted(df["fraction"].unique().tolist()),
            "columns": df.columns.tolist(),
            "metrics_available": [
                col
                for col in df.columns
                if col.startswith("eval") and not col.endswith("_mean")
            ],
        },
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Data saved to: {pkl_path}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Collected {len(df)} runs")


def load_cached_data(data_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Load DataFrame and metadata from cache files.

    Args:
        data_dir: Directory containing data.pkl and meta.json

    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    data_dir = Path(data_dir)
    pkl_path = data_dir / "data.pkl"
    meta_path = data_dir / "meta.json"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Data file not found: {pkl_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_pickle(pkl_path)

    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"Loaded {len(df)} runs from cache")
    print(f"Unique experiments: {', '.join(metadata['data_summary']['unique_egids'])}")

    return df, metadata


def collect_and_save_data(
    project_name: str,
    output_dir: Path,
    egid_patterns: Set[str],
    experiments: Dict[str, Any],
    style_overrides: Optional[Dict[str, Dict[str, str]]],
    display_name_overrides: Optional[Dict[str, str]],
    seed_filter: Optional[int],
    command_line: Optional[str],
) -> None:
    """
    Collect data from wandb and save to cache files.

    This is the main function called by the collect command.
    """
    print(f"Collecting data from wandb project: {project_name}")
    if seed_filter is not None:
        print(f"Filtering to seed: {seed_filter}")

    print("Experiment patterns:")
    for pattern in sorted(egid_patterns):
        print(f"  {pattern}")

    df = load_results(
        project=project_name,
        seed_filter=seed_filter,
        loss_type_exclusions=None,
        accuracy_field_name="eval_accuracy",
        egid_patterns=egid_patterns,
    )

    if df.empty:
        raise ValueError("No data collected. Check experiment specifications.")

    save_cached_data(
        df=df,
        output_dir=output_dir,
        project_name=project_name,
        egid_patterns=egid_patterns,
        experiments=experiments,
        style_overrides=style_overrides,
        display_name_overrides=display_name_overrides,
        seed_filter=seed_filter,
        command_line=command_line,
    )
