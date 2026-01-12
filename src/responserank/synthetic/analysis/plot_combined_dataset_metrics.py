"""
Plot combined dataset metrics like Kendall tau across different datasets.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from responserank.llm.analysis.plotting_utils import save_figure_reproducible

# Enable LaTeX rendering
rc("text", usetex=True)
rc("font", family="serif")


def load_results(result_path):
    """Load results from a result file."""
    with open(result_path) as f:
        data = json.load(f)
    metadata = data.pop("metadata")
    dataset_name = metadata.get("dataset")
    return data, dataset_name


def extract_metric_values(results, metric_name):
    """Extract values and x-axis range for a specified dataset metric from results.

    Returns:
        Tuple of (values, x_range) where x_range is (min, max) or None.
    """
    values = []
    x_range = None

    metrics = results["dataset_metrics"].get(metric_name, [])

    for metric in metrics:
        value_data = metric["value"]
        x_values = value_data["x"]
        values.extend(x_values)

        if "range" in value_data:
            range_data = (value_data["range"][0], value_data["range"][1])
            if x_range is None:
                x_range = range_data
            elif x_range != range_data:
                raise ValueError(
                    f"Inconsistent ranges for {metric_name}: {x_range} vs {range_data}"
                )

    return values, x_range


def plot_histograms(
    data_dict,
    metric_name,
    output_dir,
    output_suffix,
    x_range: Optional[Tuple[float, float]],
):
    """Plot histograms of metric values for each dataset.

    Args:
        data_dict: Mapping of dataset names to metric values.
        metric_name: Name of the metric being visualized.
        output_dir: Directory where the plot should be written.
        output_suffix: Filename suffix for the saved plot.
        x_range: Optional fixed x-axis range. Pass ``None`` to compute from data.
    """
    datasets = list(data_dict.keys())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5), sharey=True)

    if n_datasets == 1:
        axes = [axes]

    dataset_display_names = {
        "Deterministic": "Deterministic",
        "Deterministic No Variability": "Deterministic",
        "Drift Diffusion": "Drift-Diffusion",
        "Drift Diffusion No Variability": "Drift-Diffusion",
        "Stochastic": "Stochastic",
        "Stochastic No Variability": "Stochastic",
    }

    metric_display_names = {
        "test_abs_logit_diff_rt_kendall_tau": "Kendall $\\tau$ Correlation",
    }

    if x_range is None:
        min_val = min(min(values) for values in data_dict.values() if values)
        max_val = max(max(values) for values in data_dict.values() if values)
        x_range = (min_val, max_val)

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        values = data_dict[dataset]

        ax.hist(values, bins=25, range=x_range, alpha=0.7, color="#1D9A6C")

        # Add mean indicator (line and text)
        mean_val = np.mean(values)
        ax.axvline(mean_val, color="black", linestyle="dashed", linewidth=2)
        ax.text(
            0.05,
            0.90,
            f"Mean: {mean_val:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        display_name = dataset_display_names.get(dataset, dataset.capitalize())
        ax.set_title(r"\textbf{" + display_name + r"}", fontsize=14)

        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel(r"\textbf{Frequency}", fontsize=14)

        # No X label, will use shared label
        ax.set_xlabel("")

        # Shared ylim, use max
        ax.set_ylim(0, len(values))
        ax.set_xlim(x_range)

        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Shared X label
    display_name = metric_display_names.get(metric_name, metric_name)
    fig.text(
        0.5,
        0.02,
        r"\textbf{" + display_name + r"}",
        ha="center",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_path = output_dir / f"{metric_name}_combined_histograms{suffix}.pdf"
    save_figure_reproducible(fig, str(output_path))

    print(f"Saved {metric_name} histograms to {output_path}")
    plt.close()


def run_analysis(result_dirs, metrics, output_dir, output_suffix):
    """Run the combined dataset metrics analysis.

    Args:
        result_dirs: List of directories containing experiment results
        metrics: List of metrics to analyze
        output_dir: Directory to save output files
        output_suffix: Suffix for output filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for metric_name in metrics:
        data_dict = {}
        x_range = None
        for result_dir in result_dirs:
            results, dataset_name = load_results(result_dir)
            values, range_from_data = extract_metric_values(results, metric_name)
            data_dict[dataset_name] = values
            if range_from_data is not None:
                if x_range is None:
                    x_range = range_from_data
                elif x_range != range_from_data:
                    raise ValueError(
                        f"Inconsistent ranges for {metric_name} across datasets: "
                        f"{x_range} vs {range_from_data}"
                    )
            print(f"Extracted {len(values)} {metric_name} values from {dataset_name}")

        plot_histograms(
            data_dict,
            metric_name,
            output_dir,
            output_suffix,
            x_range=x_range,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot dataset metric histograms across datasets"
    )
    parser.add_argument(
        "--result_dirs",
        nargs="+",
        required=True,
        help="Paths to aggregated result files (e.g., results/synthetic/deterministic_all.json)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["test_abs_logit_diff_rt_kendall_tau"],
        help="Dataset metrics to visualize (default: test_abs_logit_diff_rt_kendall_tau)",
    )
    parser.add_argument(
        "--output",
        default="./",
        help="Output directory for histograms",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output filenames",
    )

    args = parser.parse_args()

    run_analysis(
        result_dirs=args.result_dirs,
        metrics=args.metrics,
        output_dir=args.output,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
