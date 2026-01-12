"""Collect and aggregate trial results."""

import json
import re
from pathlib import Path

# We include only the metrics used in the analysis script to avoid excessively large files.
INCLUDED_SUMMARY_METRICS = {"choice_accuracy", "pearson_distance_correlation"}
INCLUDED_DATASET_METRICS = {
    "test_abs_logit_diff_rt_kendall_tau",
    "test_confusion_matrix",
}


def parse_result_filename(filename: str) -> tuple[str, float]:
    """Parse learner and fraction from result filename."""
    match = re.match(r"(.+)_frac_(\d+)_results\.json$", filename)
    if not match:
        raise ValueError(f"Unexpected result filename format: {filename}")
    learner = match.group(1)
    fraction = int(match.group(2)) / 100
    return learner, fraction


def extract_dataset_name(experiment_dir: Path) -> str:
    """Extract dataset name from experiment directory path."""
    match = re.search(r"dataset=(\w+)", experiment_dir.name)
    if not match:
        raise ValueError(f"Cannot extract dataset name from: {experiment_dir}")
    return match.group(1)


def aggregate_experiment(experiment_dir: Path, output_dir: Path):
    """Aggregate trial results from an experiment directory."""
    # Sort by trial id
    trial_dirs = sorted(
        experiment_dir.glob("trial_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if len(trial_dirs) < 1:
        raise ValueError(f"No trial directories found in {experiment_dir}")

    print(f"Found {len(trial_dirs)} trial directories")

    with open(experiment_dir / "metadata.json") as f:
        metadata = json.load(f)
    aggregated = {"metadata": metadata, "dataset_metrics": {}}
    expected_runs_per_trial = None
    sorted_learners = None
    largest_fraction = None

    for trial_dir in trial_dirs:
        result_files = sorted(trial_dir.glob("*_frac_*_results.json"))

        parsed_results = [(f, *parse_result_filename(f.name)) for f in result_files]
        trial_runs = {(learner, fraction) for _, learner, fraction in parsed_results}

        if expected_runs_per_trial is None:
            expected_runs_per_trial = trial_runs
            sorted_learners = sorted({lf[0] for lf in expected_runs_per_trial})
            largest_fraction = max(lf[1] for lf in expected_runs_per_trial)
        elif trial_runs != expected_runs_per_trial:
            # All trials should have the same learners with the same fractions.
            raise ValueError(
                f"Inconsistent learner/fraction pairs in {trial_dir.name}: "
                f"expected {expected_runs_per_trial}, got {trial_runs}"
            )

        for result_file, learner, fraction in parsed_results:
            with open(result_file) as f:
                results = json.load(f)

            if learner not in aggregated:
                aggregated[learner] = {"sizes": {}}
            frac_key = str(fraction)
            if frac_key not in aggregated[learner]["sizes"]:
                aggregated[learner]["sizes"][frac_key] = {"summary_metrics": {}}

            for key, value in results["summary_metrics"].items():
                if isinstance(value, dict) and value.get("dataset_metric", False):
                    # Keep dataset metrics only once (for the full dataset / largest fraction)
                    if (
                        fraction == largest_fraction
                        and learner == sorted_learners[0]
                        and key in INCLUDED_DATASET_METRICS
                    ):
                        if key not in aggregated["dataset_metrics"]:
                            aggregated["dataset_metrics"][key] = []
                        aggregated["dataset_metrics"][key].append(value)
                    continue

                if key not in INCLUDED_SUMMARY_METRICS:
                    continue

                if key not in aggregated[learner]["sizes"][frac_key]["summary_metrics"]:
                    aggregated[learner]["sizes"][frac_key]["summary_metrics"][key] = []
                aggregated[learner]["sizes"][frac_key]["summary_metrics"][key].append(
                    value
                )

    dataset_name = extract_dataset_name(experiment_dir)
    output_path = output_dir / f"{dataset_name}.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregated, f)

    print(f"  -> {output_path}")


def collect_all(experiment_paths_file: Path, output_dir: Path):
    """Aggregate experiments listed in experiment_paths.json."""
    with open(experiment_paths_file) as f:
        experiment_paths = json.load(f)

    if not experiment_paths:
        raise ValueError(f"No experiments in {experiment_paths_file}")

    print(f"Found {len(experiment_paths)} experiments")
    for dataset_name, exp_path in experiment_paths.items():
        exp_dir = Path(exp_path)
        print(f"\nAggregating {dataset_name}: {exp_dir.name}")
        aggregate_experiment(exp_dir, output_dir)
