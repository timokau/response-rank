#!/usr/bin/env python3
"""
CLI for analysis and visualization of results.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

from responserank.synthetic.analysis.collect import collect_all
from responserank.synthetic.analysis.plot_combined_dataset_metrics import (
    run_analysis as run_kendall_analysis,
)
from responserank.synthetic.analysis.plot_combined_metrics import (
    run_analysis as run_combined_metrics_analysis,
)
from responserank.synthetic.analysis.plot_datasize import (
    run_analysis as run_datasize_analysis,
)
from responserank.synthetic.visualization import plot_confusion_matrix

LEARNERS = [
    "bt",
    "rt_regression",
    "rt_regression_perm",
    "rr",
    "rr_pooled",
    "rr_perm",
]

WITH_VARIABILITY_FILES = [
    "deterministic_all.json",
    "stochastic.json",
    "drift_diffusion.json",
]

NO_VARIABILITY_FILES = [
    "deterministic_all_no_variability.json",
    "stochastic_no_variability.json",
    "drift_diffusion_no_variability.json",
]

DATASETS = {
    "with_variability": [
        {"name": "Deterministic", "config_name": "deterministic_all"},
        {"name": "Stochastic", "config_name": "stochastic"},
        {"name": "Drift Diffusion", "config_name": "drift_diffusion"},
    ],
    "no_variability": [
        {
            "name": "Deterministic No Variability",
            "config_name": "deterministic_all_no_variability",
        },
        {
            "name": "Stochastic No Variability",
            "config_name": "stochastic_no_variability",
        },
        {
            "name": "Drift Diffusion No Variability",
            "config_name": "drift_diffusion_no_variability",
        },
    ],
}


def load_experiment_paths(paths_file):
    """Load experiment paths from JSON file."""
    if os.path.exists(paths_file):
        with open(paths_file, "r") as f:
            return json.load(f)
    return {}


def save_experiment_paths(paths, paths_file):
    """Save experiment paths to JSON file."""
    os.makedirs(paths_file.parent, exist_ok=True)
    with open(paths_file, "w") as f:
        json.dump(paths, f, indent=2)


def find_latest_experiments(experiment_dir):
    """Find the latest experiment directory for each dataset by parsing metadata.json files."""
    latest_by_dataset = {}

    dataset_metadata_to_config = {
        "Deterministic": "deterministic_all",
        "Stochastic": "stochastic",
        "Drift Diffusion": "drift_diffusion",
        "Deterministic No Variability": "deterministic_all_no_variability",
        "Stochastic No Variability": "stochastic_no_variability",
        "Drift Diffusion No Variability": "drift_diffusion_no_variability",
    }

    # Walk through all experiment directories
    for root, _, files in os.walk(experiment_dir):
        if "metadata.json" in files:
            metadata_path = Path(root) / "metadata.json"

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            dataset_name = metadata.get("dataset")
            if not dataset_name or dataset_name not in dataset_metadata_to_config:
                continue

            config_dataset = dataset_metadata_to_config[dataset_name]
            timestamp_str = metadata["timestamp"]
            timestamp_dt = datetime.datetime.strptime(
                timestamp_str, "%Y-%m-%d %H:%M:%S"
            )

            if (
                config_dataset not in latest_by_dataset
                or timestamp_dt.timestamp()
                > latest_by_dataset[config_dataset]["timestamp_dt"].timestamp()
            ):
                latest_by_dataset[config_dataset] = {
                    "path": str(Path(root)),
                    "timestamp_dt": timestamp_dt,
                }

    paths = {}
    for dataset, info in latest_by_dataset.items():
        paths[dataset] = info["path"]

    return paths


def find_command(output_dir, experiment_dir):
    """Find latest experiment directories and update paths file"""
    paths_file = output_dir / "experiment_paths.json"
    print("Finding latest experiment directories...")
    paths = find_latest_experiments(experiment_dir)
    save_experiment_paths(paths, paths_file)
    print(f"Updated experiment paths in {paths_file}")

    print("Found latest experiments:")
    for dataset, path in paths.items():
        path_obj = Path(path)
        print(f"- {dataset}: {path_obj.name}")


def generate_confusion_matrices(results_dir: Path, output_dir: Path):
    """Generate confusion matrix plots from collected results."""
    configs = {
        "deterministic_all": "deterministic",
        "drift_diffusion": "ddm",
        "stochastic": "stochastic",
        "deterministic_all_no_variability": "deterministic_no_variability",
        "drift_diffusion_no_variability": "ddm_no_variability",
        "stochastic_no_variability": "stochastic_no_variability",
    }

    for config_name, output_name in configs.items():
        json_file = results_dir / f"{config_name}.json"
        with open(json_file) as f:
            data = json.load(f)

        confusion_data = data["dataset_metrics"]["test_confusion_matrix"]

        # Sum confusion matrices across all trials
        joint_matrix = np.array(confusion_data[0]["value"])
        for item in confusion_data[1:]:
            joint_matrix += np.array(item["value"])

        plot_confusion_matrix(
            joint_matrix,
            output_name,
            output_dir,
            escape_latex=True,
            annot_fontsize=14,
            tick_fontsize=12,
            label_fontsize=14,
        )


def run_combined_metrics(result_dirs, learners, output_dir, output_suffix):
    """Run the combined metrics analysis."""
    metrics = ["pearson_distance_correlation", "choice_accuracy"]

    # Call the analysis function directly
    run_combined_metrics_analysis(
        result_dirs=result_dirs,
        learners=learners,
        output_dir=output_dir,
        metrics=metrics,
        output_suffix=output_suffix,
    )


def boxplots_command(output_dir, results_dir, variability):
    """Generate combined metric boxplots"""
    if variability in ["with", "both"]:
        print("Generating boxplots with variability...")
        files = [results_dir / f for f in WITH_VARIABILITY_FILES]
        run_combined_metrics(files, LEARNERS, output_dir, "with_variability")

    if variability in ["without", "both"]:
        print("Generating boxplots without variability...")
        files = [results_dir / f for f in NO_VARIABILITY_FILES]
        run_combined_metrics(files, LEARNERS, output_dir, "no_variability")


def run_datasize_plots(result_dirs, learners, output_dir, output_filename):
    """Run the dataset size vs metrics analysis."""
    metrics = ["pearson_distance_correlation", "choice_accuracy"]

    # Call the analysis function directly
    run_datasize_analysis(
        result_dirs=result_dirs,
        learners=learners,
        output_dir=output_dir,
        metrics=metrics,
        output_filename=output_filename,
    )


def datasize_command(output_dir, results_dir, variability):
    """Generate dataset size vs metrics plots"""
    if variability in ["with", "both"]:
        print("Generating dataset size plots with variability...")
        files = [results_dir / f for f in WITH_VARIABILITY_FILES]
        run_datasize_plots(
            files,
            LEARNERS,
            output_dir,
            "metrics_vs_datasize_combined_with_variability",
        )

    if variability in ["without", "both"]:
        print("Generating dataset size plots without variability...")
        files = [results_dir / f for f in NO_VARIABILITY_FILES]
        run_datasize_plots(
            files,
            LEARNERS,
            output_dir,
            "metrics_vs_datasize_combined_no_variability",
        )


def run_kendall_tau(result_dirs, output_dir, output_suffix):
    """Run the Kendall tau analysis."""
    metrics = ["test_abs_logit_diff_rt_kendall_tau"]

    # Call the analysis function directly
    run_kendall_analysis(
        result_dirs=result_dirs,
        metrics=metrics,
        output_dir=output_dir,
        output_suffix=output_suffix,
    )


def kendall_command(output_dir, results_dir, variability):
    """Generate Kendall tau histograms"""
    if variability in ["with", "both"]:
        print("Generating Kendall tau histograms with variability...")
        files = [results_dir / f for f in WITH_VARIABILITY_FILES]
        run_kendall_tau(files, output_dir, "with_variability")

    if variability in ["without", "both"]:
        print("Generating Kendall tau histograms without variability...")
        files = [results_dir / f for f in NO_VARIABILITY_FILES]
        run_kendall_tau(files, output_dir, "no_variability")


def plots_command(output_dir, results_dir):
    """Run all plot commands"""
    boxplots_command(output_dir, results_dir, variability="both")
    datasize_command(output_dir, results_dir, variability="both")
    kendall_command(output_dir, results_dir, variability="both")
    generate_confusion_matrices(results_dir, output_dir)
    print("All analyses completed!")


def main():
    parser = argparse.ArgumentParser(description="ResponseRank Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Analysis command to run")

    find_parser = subparsers.add_parser(
        "find", help="Find latest experiment directories"
    )
    find_parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Directory containing experiment results",
    )
    find_parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write experiment_paths.json",
    )

    collect_parser = subparsers.add_parser(
        "collect", help="Collect results using experiment_paths.json"
    )
    collect_parser.add_argument(
        "cache_dir",
        type=Path,
        help="Directory for aggregated result files",
    )
    collect_parser.add_argument(
        "--paths-file",
        type=Path,
        required=True,
        help="Path to experiment_paths.json",
    )

    generate_parser = subparsers.add_parser("generate", help="Generate all plots")
    generate_parser.add_argument(
        "cache_dir",
        type=Path,
        help="Directory containing aggregated result files",
    )
    generate_parser.add_argument(
        "figures_dir",
        type=Path,
        help="Directory for output figures",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "find":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        find_command(args.output_dir, args.experiment_dir)
    elif args.command == "collect":
        collect_all(args.paths_file, args.cache_dir)
    elif args.command == "generate":
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        plots_command(args.figures_dir, args.cache_dir)


if __name__ == "__main__":
    main()
