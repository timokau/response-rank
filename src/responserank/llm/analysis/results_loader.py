"""Data loading utilities for analysis scripts."""

import os
import pickle
import re
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import wandb


def _get_cache_path(run_id: str) -> str:
    """Get cache file path for a wandb run."""
    cache_dir = ".wandb_run_cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"history_{run_id}.pkl")


def _load_cached_history(run_id: str) -> Optional[pd.DataFrame]:
    """Load cached run history if it exists."""
    cache_path = _get_cache_path(run_id)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except (OSError, pickle.PickleError):
            print(f"Warning: Could not load cache for run {run_id}")
            os.remove(cache_path)
    return None


def _save_cached_history(run_id: str, history: pd.DataFrame) -> None:
    """Save run history to cache."""
    cache_path = _get_cache_path(run_id)
    with open(cache_path, "wb") as f:
        pickle.dump(history, f)


def matches_any_pattern(egid: str, patterns: Set[str]) -> bool:
    """Check if the given egid matches any of the patterns.

    Args:
        egid: The experiment group ID to match
        patterns: Set of regex patterns to match against

    Returns:
        True if egid matches at least one pattern, False otherwise
    """
    for pattern in patterns:
        regex = re.compile(pattern)
        if regex.match(egid):
            return True
    return False


def extract_base_metadata(
    run,
    seed_filter: Optional[int],
    loss_type_exclusions: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    """Extract base metadata from run, applying filters.

    Returns None if run should be filtered out.
    """
    config = run.config
    egid = config.get("egid")

    hydra_cfg = config["hydra_cfg"]
    seed = hydra_cfg["seed"]
    exp_config = hydra_cfg["experiment"]
    fraction = exp_config["fraction"]
    baseline = exp_config["baseline"]

    loss_type = "baseline" if baseline else "responserank"

    if seed_filter is not None and seed != seed_filter:
        return None

    if loss_type_exclusions and loss_type in loss_type_exclusions:
        return None

    if seed is None or fraction is None:
        return None

    return {
        "run_id": run.id,
        "run_name": run.name,
        "egid": egid,
        "experiment_name": exp_config["name"],
        "seed": seed,
        "fraction": fraction,
    }


def get_epoch_summaries(
    history: pd.DataFrame,
    epoch_col: str = "train/epoch",
    columns: Optional[List[str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Extract final logged values for each full epoch from history dataframe.

    Only includes rows that have an explicit epoch value; rows without epochs are dropped.

    Args:
        history: DataFrame from run.history()
        epoch_col: Name of the epoch column (default: "train/epoch")
        columns: Optional list of columns to include. If provided, history is first
                reduced to only these columns plus epoch_col for efficiency.

    Returns:
        Dict mapping epoch number (int) to dict of column values
        Example: {1: {'eval/accuracy': 0.65, 'eval/rewardbench_accuracy': 0.70, ...},
                  2: {'eval/accuracy': 0.67, 'eval/rewardbench_accuracy': 0.72, ...}}
    """
    if epoch_col not in history.columns:
        return {}

    if columns is not None:
        cols_to_keep = list(set(columns + [epoch_col]))
        cols_to_keep = [col for col in cols_to_keep if col in history.columns]
        history = history[cols_to_keep]

    df = history.copy()

    mask = df[epoch_col].notna()
    df_filtered = df[mask].copy()

    # Convert to integer epochs
    df_filtered["epoch_int"] = df_filtered[epoch_col].astype(int)

    # Group by epoch and get the last non-null value for each column
    # This gives us the final state of each metric at the end of each epoch
    epoch_summaries = {
        epoch: row.drop(["epoch_int", epoch_col], errors="ignore").dropna().to_dict()
        for epoch, row in df_filtered.groupby("epoch_int").last().iterrows()
    }

    return epoch_summaries


def extract_all_epoch_metrics(run) -> List[Dict[str, Any]]:
    """Extract eval metrics for all epochs from run history.

    Args:
        run: wandb run object

    Returns:
        List of dicts with epoch metrics. Each dict contains:
        - epoch: int (epoch number)
        - eval_accuracy: float (evaluation accuracy at this epoch)
        - eval/rewardbench_v1/overall/accuracy_score: float or None (rewardbench accuracy at this epoch)
        - eval/rewardbench_v2/overall/accuracy_score: float or None (rewardbench v2 overall score at this epoch)
    """
    cached_history = _load_cached_history(run.id)
    if cached_history is not None:
        history = cached_history
    else:
        history = run.history(samples=None)
        _save_cached_history(run.id, history)

    epoch_summaries = get_epoch_summaries(history, epoch_col="train/epoch")

    epoch_list = []
    for epoch_num in sorted(epoch_summaries.keys()):
        epoch_data = {
            "epoch": epoch_num,
            "eval_accuracy": epoch_summaries[epoch_num].get("eval/accuracy"),
            "eval/rewardbench_v1/overall/accuracy_score": epoch_summaries[
                epoch_num
            ].get("eval/rewardbench/overall/accuracy_score")
            or epoch_summaries[epoch_num].get("eval/rewardbench_accuracy"),
            "eval/rewardbench_v2/overall/accuracy_score": epoch_summaries[
                epoch_num
            ].get("eval/rewardbench_v2/overall/accuracy_score"),
        }

        # Extract all RewardBench v2 section scores
        for key in epoch_summaries[epoch_num]:
            if key.startswith("eval/rewardbench_v2/section/"):
                epoch_data[key] = epoch_summaries[epoch_num][key]

        if epoch_data["eval_accuracy"] is not None:
            epoch_list.append(epoch_data)

    return epoch_list


def fetch_runs(
    project: str,
    filters: Optional[Dict[str, Any]],
    egid_patterns: Optional[List[str]] = None,
) -> List[Any]:
    """Fetch runs from wandb project.

    Args:
        project: wandb project name
        filters: Base filters for wandb API
        egid_patterns: List of regex patterns to match against config.egid

    Returns:
        List of wandb runs matching the filters
    """
    api = wandb.Api()

    combined_filters = filters.copy() if filters else {}

    if egid_patterns is not None:
        # Add word boundary prefix to patterns for better server-side filtering (in place of anchoring, which wandb does not support)
        bounded_patterns = [
            f"\\b{pattern}" if not pattern.startswith("\\b") else pattern
            for pattern in egid_patterns
        ]
        egid_conditions = [
            {"config.egid": {"$regex": pattern}} for pattern in bounded_patterns
        ]

        combined_filters["$or"] = combined_filters.get("$or", []) + egid_conditions

    return api.runs(project, filters=combined_filters, per_page=1000)


def process_runs(
    runs: List[Any],
    egid_patterns: Set[str],
    seed_filter: Optional[int],
    loss_type_exclusions: Optional[List[str]],
    accuracy_field_name: str,
) -> pd.DataFrame:
    """Process wandb runs into a DataFrame.

    Args:
        runs: List of wandb runs
        egid_patterns: Set of regex patterns to match egids against
        seed_filter: If provided, only include runs with this seed
        loss_type_exclusions: List of loss types to exclude
        accuracy_field_name: Name of the accuracy field in output DataFrame

    Returns:
        DataFrame with columns: run_id, run_name, egid, experiment_name, seed, fraction,
        run_state, display_name, <accuracy_field_name>, eval/rewardbench_v1/overall/accuracy_score,
        eval/rewardbench_v2/overall/accuracy_score, train_examples, test_examples,
        non_singleton_fraction, avg_fragment_size, eval/rewardbench_v2/section/*
    """

    all_run_data = []

    for run in runs:
        if run.config["hydra_cfg"].get("dry_run", False):
            continue

        run_state = run.state

        base_data = extract_base_metadata(run, seed_filter, loss_type_exclusions)
        if base_data is None:
            continue

        base_data["run_state"] = run_state

        if not matches_any_pattern(base_data["egid"], egid_patterns):
            print(
                f"Info: Skipping run {run.name} with egid {base_data['egid']} - "
                f"fetched by wandb (no anchor support) but doesn't match from start"
            )
            continue

        summary = run.summary
        run_entry = base_data.copy()
        update_dict = {
            "display_name": f"{base_data['egid']} (final)",
            accuracy_field_name: summary.get("eval/accuracy"),
            "eval/rewardbench_v1/overall/accuracy_score": summary.get(
                "eval/rewardbench/overall/accuracy_score"
            )
            or summary.get("eval/rewardbench_accuracy"),
            "eval/rewardbench_v2/overall/accuracy_score": summary.get(
                "eval/rewardbench_v2/overall/accuracy_score"
            ),
            "train_examples": summary.get("train_examples"),
            "test_examples": summary.get("test_examples"),
            "non_singleton_fraction": (
                summary.get("partitioner/train/non_singleton_fraction")
                or summary.get("stratification/non_singleton_fraction")
                or summary.get("non_singleton_fraction")
            ),
            "avg_fragment_size": (run.summary.get("packing/avg_fragment_size")),
        }

        for key in summary.keys():
            if key.startswith("eval/rewardbench_v2/section/") and key.endswith(
                "_score"
            ):
                update_dict[key] = summary[key]

        run_entry.update(update_dict)
        all_run_data.append(run_entry)

        if "eval/accuracy" in summary:
            print(
                f"Processed run {run.name}: egid={base_data['egid']}, seed={base_data['seed']}, "
                f"summary acc={summary['eval/accuracy']:.4f}"
            )
        else:
            print(
                f"Processed run {run.name}: egid={base_data['egid']}, seed={base_data['seed']}, "
                f"state={run_state} (no eval metrics yet)"
            )

    return pd.DataFrame(all_run_data)


def load_results(
    project: str,
    seed_filter: Optional[int],
    loss_type_exclusions: Optional[List[str]],
    accuracy_field_name: str,
    egid_patterns: Set[str],
) -> pd.DataFrame:
    """Complete data loading pipeline for analysis.

    Args:
        project: wandb project name
        seed_filter: If provided, only include runs with this seed
        loss_type_exclusions: List of loss types to exclude
        accuracy_field_name: Name of the accuracy field in output DataFrame
        egid_patterns: Set of regex patterns to match egids against

    Returns:
        DataFrame ready for analysis
    """
    runs = fetch_runs(project, filters=None, egid_patterns=sorted(egid_patterns))

    df = process_runs(
        runs,
        egid_patterns,
        seed_filter,
        loss_type_exclusions,
        accuracy_field_name,
    )

    return df
