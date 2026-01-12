"""
Experiment runner for ResponseRank experiments.

This module provides the main entry point for running experiments. It handles:

- Experiment configuration through Hydra
- Dataset generation and preparation
- Parallel execution of multiple learners and trials
- Result aggregation and initial visualization
- Experiment monitoring and progress tracking

Usage:
    python -m responserank.synthetic.experiment_runner dataset=deterministic_all
"""

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from responserank.synthetic.execution_backends import ExecutionBackend
from responserank.synthetic.visualization import plot_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aggregate_dataset_metrics(trial_dir, learner_keys, dataset_metrics):
    """Aggregates dataset-specific metrics from the first learner's results."""
    first_learner = learner_keys[0]
    first_learner_file = trial_dir / f"{first_learner}_results.json"
    with open(first_learner_file, "r") as f:
        first_results = json.load(f)
        # Collect dataset-specific metrics
        for key, value in first_results["summary_metrics"].items():
            if isinstance(value, dict) and value.get("dataset_metric", False):
                if key not in dataset_metrics["summary_metrics"]:
                    dataset_metrics["summary_metrics"][key] = []
                dataset_metrics["summary_metrics"][key].append(value)


def aggregate_learner_results(trial_dir, learner_key, aggregated_results):
    """Aggregates summary and epoch metrics for a given learner."""
    learner_file = trial_dir / f"{learner_key}_results.json"
    try:
        with open(learner_file, "r") as f:
            results = json.load(f)

        # Aggregate summary metrics
        for key, value in results["summary_metrics"].items():
            if key not in aggregated_results[learner_key]["summary_metrics"]:
                aggregated_results[learner_key]["summary_metrics"][key] = []
            aggregated_results[learner_key]["summary_metrics"][key].append(value)

        # Aggregate epoch metrics
        aggregated_results[learner_key]["epoch_metrics"].append(
            results["epoch_metrics"]
        )

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading results for {learner_key}: {e}")


def create_aborted_marker(output_dir):
    (output_dir / "ABORTED").touch()
    logger.info(f"Created ABORTED marker in {output_dir}")


def generate_dataset(cfg: DictConfig) -> str:
    random_state = np.random.RandomState(cfg.random_seed)
    dataset_seed = random_state.randint(0, 2**32)
    # Create a new random seed to avoid reuse of the global seed
    cfg.random_seed = random_state.randint(0, 2**32)

    dataset_cmd = [
        sys.executable,
        "-m",
        "responserank.synthetic.generate_dataset",
        f"random_seed={dataset_seed}",
    ]

    # Automatically pass all dataset configuration parameters
    dataset_cmd.extend(get_overrides_for_key("dataset", "dataset"))

    logger.info(f"Generating dataset with command: {' '.join(dataset_cmd)}")
    try:
        result = subprocess.run(dataset_cmd, capture_output=True, text=True, check=True)
        df_file = result.stdout.strip().splitlines()[-1]
        logger.info(f"Dataset generated and saved to: {df_file}")
        return df_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating dataset: {e.stderr}")
        raise


def get_overrides_for_key(key, new_key):
    result = []
    overrides = HydraConfig.get().overrides["task"]
    valid_prefixes = ["", "+", "++", "~"]
    for override in overrides:
        for prefix in valid_prefixes:
            if override.startswith(f"{prefix}{key}.") or override.startswith(
                f"{prefix}{key}="
            ):
                if new_key is None:
                    result.append(override)
                else:
                    result.append(override.replace(key, new_key, 1))
    return result


def run_learner(
    train_df_file: Path,
    test_df_file: Path,
    overrides: List[str],
    output_dir: Path,
    learner_name: str,
) -> None:
    def hydra_escape(s):
        # Escape a string for hydra, assuming it is double quoted
        # https://hydra.cc/docs/advanced/override_grammar/basic/#quoted-values
        # - Double any sequence of backslashes preceding a double or single quote
        s = re.sub(r'(\\+)"', r'\g<1>\g<1>"', s)
        # - Replace double quotes (") with escaped double quotes (\")
        s = s.replace('"', r"\"")
        return s

    pre_run = time.time()
    cmd = [
        sys.executable,
        "-m",
        "responserank.synthetic.run_learner",
        # Shell excape is not needed, since we use subprocess.run with a list.
        # Inner escape is needed for hydra.
        f'train_df_file="{hydra_escape(str(train_df_file))}"',
        f'test_df_file="{hydra_escape(str(test_df_file))}"',
    ]

    cmd.extend(overrides)

    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
        result_dir = Path(result.stdout.strip().splitlines()[0])
        result_file = result_dir / "results.json"
        output_file = output_dir / f"{learner_name}_results.json"
        shutil.copy(result_file, output_file)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running learner: {e.stderr}")
        raise
    post_run = time.time()
    logger.info(
        f"Completed learner {learner_name} in {post_run - pre_run:.2f} seconds. Results saved to {output_file}"
    )


def prepare_trial(cfg, trial, output_dir, learner_keys, learner_overrides):
    logger.info(f"Preparing trial {trial + 1}/{cfg.num_trials}")
    trial_dir = output_dir / f"trial_{trial + 1}"
    trial_dir.mkdir()

    df_file = generate_dataset(cfg)
    df = pd.read_csv(df_file)
    train_df = df.iloc[: cfg.dataset.num_train_samples]
    test_df = df.iloc[cfg.dataset.num_train_samples :]

    dataset_name = Path(df_file).stem
    dataset_dir = trial_dir / dataset_name
    dataset_dir.mkdir()
    train_df_file = dataset_dir / "train.csv"
    test_df_file = dataset_dir / "test.csv"
    train_df.to_csv(train_df_file, index=False)
    test_df.to_csv(test_df_file, index=False)

    jobs = []
    # Create a mapping between result names and metadata
    result_metadata = {}

    # Create jobs for all dataset sizes and learners
    for size_fraction in cfg.dataset_sizes:
        for learner in learner_keys:
            # Create a descriptive result name that includes fraction for better UX
            result_name = f"{learner}_frac_{int(size_fraction * 100)}"

            # Extend learner overrides with dataset_fraction parameter
            size_overrides = learner_overrides[learner].copy()
            size_overrides.append(f"dataset_fraction={size_fraction}")

            # Store metadata about this result (flat structure)
            result_metadata[result_name] = {
                "learner": learner,
                "fraction": size_fraction,
            }

            jobs.append(
                (
                    train_df_file,
                    test_df_file,
                    size_overrides,
                    trial_dir,
                    result_name,
                )
            )

    return jobs, trial_dir, result_metadata


def _run_trials(
    cfg: DictConfig,
    output_dir: Path,
    backend: ExecutionBackend,
    learner_keys: List[str],
    learner_overrides: Dict[str, List[str]],
):
    completed_trials = set()

    # Create a nested structure for results with learners and sizes
    aggregated_results = {"dataset_metrics": {}}
    for learner_key in learner_keys:
        aggregated_results[learner_key] = {
            "sizes": {}  # Will be populated with dataset fractions as keys
        }

    trial_dirs = {}  # trial index -> trial dir
    trial_job_map = defaultdict(list)  # trial_index -> list of (result_name, job_id)
    all_result_metadata = {}  # Map to store all result metadata across trials
    trials_completed = 0

    for trial in range(cfg.num_trials):
        jobs, trial_dir, result_metadata = prepare_trial(
            cfg, trial, output_dir, learner_keys, learner_overrides
        )

        # Store the result metadata for this trial
        all_result_metadata[trial] = result_metadata

        # Submit all jobs for this trial
        for job in jobs:
            result_name = job[4]  # Extract result name from job tuple
            job_id = backend.submit_job(*job)
            trial_job_map[trial].append((result_name, job_id))

        trial_dirs[trial] = trial_dir

        # Check for completed trials and visualize results occasionally
        newly_completed = _check_and_aggregate_completed_trials(
            cfg,
            completed_trials,
            trial_dirs,
            trial_job_map,
            all_result_metadata,
            backend,
            aggregated_results,
            output_dir,
            learner_keys,
        )
        if len(newly_completed) > 0:
            tc_prev = trials_completed
            trials_completed += len(newly_completed)
            if trials_completed // 10 > tc_prev // 10:
                print(f"Plotting results of first {trials_completed} trials")
                plot_results(aggregated_results, output_dir, learner_order=None)

    # Now that all trials are submitted, monitor and visualize until they are completed
    while len(completed_trials) < cfg.num_trials:
        time.sleep(5)
        newly_completed = _check_and_aggregate_completed_trials(
            cfg,
            completed_trials,
            trial_dirs,
            trial_job_map,
            all_result_metadata,
            backend,
            aggregated_results,
            output_dir,
            learner_keys,
        )
        if len(newly_completed) > 0:
            tc_prev = trials_completed
            trials_completed += len(newly_completed)
            if (
                trials_completed // 10 > tc_prev // 10
                or len(completed_trials) >= cfg.num_trials
            ):
                print(f"Plotting results of first {trials_completed} trials")
                plot_results(aggregated_results, output_dir, learner_order=None)

    return aggregated_results


def _check_and_aggregate_completed_trials(
    cfg: DictConfig,
    completed_trials: set,
    trial_dirs: Dict[int, Path],
    trial_job_map: Dict[int, List[Tuple[str, str]]],
    all_result_metadata: Dict[int, Dict[str, Dict[str, any]]],
    backend: ExecutionBackend,
    aggregated_results: Dict[str, Dict],
    output_dir: Path,
    learner_keys: List[str],
) -> set:
    newly_completed = set()
    for trial_id in trial_dirs.keys():
        if trial_id in completed_trials:
            continue

        trial_complete = True
        for result_name, job_id in trial_job_map[trial_id]:
            if not backend.get_job_status(job_id):
                trial_complete = False
                break

        if trial_complete:
            completed_trials.add(trial_id)
            newly_completed.add(trial_id)
            logger.info(f"Trial {trial_id} complete. Aggregating results.")

            trial_dir = trial_dirs[trial_id]

            # Get result metadata for this trial
            result_metadata = all_result_metadata[trial_id]

            # Process all result files using the metadata
            for result_name, job_id in trial_job_map[trial_id]:
                result_file = trial_dir / f"{result_name}_results.json"

                # Extract learner and fraction from metadata
                learner = result_metadata[result_name]["learner"]
                fraction = result_metadata[result_name]["fraction"]

                # Read the results file
                with open(result_file, "r") as f:
                    results = json.load(f)

                # Ensure the learner exists in aggregated_results
                if learner not in aggregated_results:
                    aggregated_results[learner] = {"sizes": {}}

                # Ensure the size fraction exists for this learner
                if fraction not in aggregated_results[learner]["sizes"]:
                    aggregated_results[learner]["sizes"][fraction] = {
                        "summary_metrics": {},
                        "epoch_metrics": [],
                    }

                # Process dataset metrics for the largest dataset size
                for key, value in results["summary_metrics"].items():
                    if isinstance(value, dict) and value.get("dataset_metric", False):
                        largest_size = max(cfg.dataset_sizes, key=float)
                        if float(fraction) == float(largest_size):
                            if key not in aggregated_results["dataset_metrics"]:
                                aggregated_results["dataset_metrics"][key] = []

                            # Only add if this is the first learner for this trial to avoid duplication
                            if learner == learner_keys[0]:
                                aggregated_results["dataset_metrics"][key].append(value)

                # Aggregate summary metrics for this learner and size
                for key, value in results["summary_metrics"].items():
                    # Dataset metrics are handled above
                    if isinstance(value, dict) and value.get("dataset_metric", False):
                        continue

                    if (
                        key
                        not in aggregated_results[learner]["sizes"][fraction][
                            "summary_metrics"
                        ]
                    ):
                        aggregated_results[learner]["sizes"][fraction][
                            "summary_metrics"
                        ][key] = []

                    aggregated_results[learner]["sizes"][fraction]["summary_metrics"][
                        key
                    ].append(value)

                # Aggregate epoch metrics
                aggregated_results[learner]["sizes"][fraction]["epoch_metrics"].append(
                    results["epoch_metrics"]
                )

    return newly_completed


@hydra.main(
    config_path="../../../conf/synthetic", config_name="config", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Register the abort marker creation, but don't create it yet
    def aborted_marker_fn():
        return create_aborted_marker(output_dir)

    atexit.register(aborted_marker_fn)

    backend = instantiate(cfg.execution_backend, run_learner_func=run_learner)
    try:
        logger.info(f"Starting experiments with dataset: {cfg.dataset}")

        # None means removed using ~ operator on command line
        learner_keys = [key for key in cfg.learners.keys() if key is not None]

        learner_overrides = {}
        for key in learner_keys:
            overrides = [f"learner={key}"]
            overrides.extend(get_overrides_for_key(f"learners.{key}", "learner"))
            learner_overrides[key] = overrides

        aggregated_results = _run_trials(
            cfg, output_dir, backend, learner_keys, learner_overrides
        )

        logger.info("All experiments completed.")

        # Save metadata for easier experiment identification
        metadata = {
            "dataset": cfg.dataset.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_trials": cfg.num_trials,
            "learners": learner_keys,
            "dataset_sizes": list(cfg.dataset_sizes),
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        # Generate plots
        plot_results(aggregated_results, output_dir, learner_order=None)
        logger.info(f"Generated plots in {output_dir}")

        # Save experiment config
        config_dir = output_dir / "config"
        os.makedirs(config_dir, exist_ok=True)
        for key, value in cfg.items():
            file_path = config_dir / f"{key}.yaml"
            yaml_str = OmegaConf.to_yaml({key: value})
            with open(file_path, "w") as f:
                f.write(yaml_str)

        logger.info(f"Experiment config saved to {config_dir}")
        logger.info(
            "All requested experiments completed, results aggregated and plotted."
        )

        # If we reach here, experiment completed successfully, so unregister the abort marker
        atexit.unregister(aborted_marker_fn)
    except KeyboardInterrupt:
        logger.info("Experiment aborted by user. Cleaning up processes...")
        backend.cleanup()
        # The atexit handler will create the ABORTED marker


if __name__ == "__main__":
    main()
