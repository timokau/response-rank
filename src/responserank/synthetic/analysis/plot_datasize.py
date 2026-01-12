#!/usr/bin/env python3
"""
Plot dataset size vs. accuracy

This script generates a plot showing how choice accuracy changes with
training dataset size across different models and datasets.
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import rc

from responserank.llm.analysis.plotting_utils import save_figure_reproducible

# Enable LaTeX rendering
rc("text", usetex=True)
rc("font", family="serif")


def load_results(result_path):
    """Load aggregated results and metadata from a result file."""
    with open(result_path) as f:
        data = json.load(f)
    metadata = data.pop("metadata")
    return data, metadata


def ci95(x):
    return 1.96 * x.std() / np.sqrt(len(x))


def extract_accuracy_by_size(results, dataset_name, metric_name="choice_accuracy"):
    """
    Extract accuracy values grouped by learner and dataset size.

    Args:
        results: Results dictionary with nested structure by learner and size
        dataset_name: Name of the dataset for labeling
        metric_name: Name of the metric to extract

    Returns:
        DataFrame with columns: Dataset, Learner, Size, Value
    """
    data = []

    for learner_name, learner_data in results.items():
        if learner_name == "dataset_metrics":
            continue

        for size_fraction, size_data in learner_data.get("sizes", {}).items():
            if metric_name in size_data.get("summary_metrics", {}):
                for acc_summary in size_data["summary_metrics"][metric_name]:
                    data.append(
                        {
                            "Dataset": dataset_name,
                            "Learner": learner_name,
                            "Size": float(size_fraction) * 100,
                            "Value": acc_summary["value"],
                        }
                    )

    return pd.DataFrame(data)


def plot_combined_metrics_vs_size(
    datasets_metrics_data,
    output_dir,
    output_filename,
    learner_order,
    use_interpolation: bool,
    dataset_order: List[str],
):
    """Create a combined plot of metrics vs. dataset size for all datasets.

    Args:
        datasets_metrics_data: Dictionary of dictionaries with keys as metric names
                              and values as DataFrames containing metric data
        output_dir: Directory to save output files
        output_filename: Base filename for output files (without extension)
        learner_order: List specifying the order of learners
        use_interpolation: Whether to use interpolation to find the exact crossing point
                          If False, uses the first data point that exceeds the threshold
        dataset_order: List specifying the order of datasets to display
    """
    # Get unique datasets and learners from the first metric's data
    metrics = list(datasets_metrics_data.keys())
    first_metric = metrics[0]
    first_dataset = list(datasets_metrics_data[first_metric].keys())[0]

    datasets = dataset_order
    datasets_metrics_data[first_metric][first_dataset]
    learners = learner_order

    # Set up figure with subplots (rows for metrics, columns for datasets)
    fig, axes = plt.subplots(
        len(metrics),
        len(datasets),
        figsize=(len(datasets) * 5, len(metrics) * 4.5),
        sharex="col",
        sharey="row",
    )

    # Make axes indexable as [row, col] even with single row or column
    if len(metrics) == 1 and len(datasets) > 1:
        axes = axes.reshape(1, -1)
    elif len(datasets) == 1 and len(metrics) > 1:
        axes = axes.reshape(-1, 1)
    elif len(metrics) == 1 and len(datasets) == 1:
        axes = np.array([[axes]])

    # Learner display names for cleaner labels
    learner_display_names = {
        "bt": "BT",
        "rt_regression": "RtRegression",
        "rt_regression_perm": "RtRegression-Perm",
        "rr": "ResponseRank",
        "rr_pooled": "ResponseRank-Pool",
        "rr_perm": "ResponseRank-Perm",
        "rr_bidir": "ResponseRank-Bidir",
    }

    learner_colors = {
        "rr": "#1D9A6C",  # teal for ResponseRank
        "rr_perm": "#1D9A6C",
        "rr_pooled": "#1D9A6C",  # same teal for ResponseRank-Pool
        "rr_bidir": "#1D9A6C",  # same teal for ResponseRank-Bidir
        "rt_regression": "#D4A017",  # gold for RtRegression
        "rt_regression_perm": "#D4A017",
        "bt": "#E74C3C",  # red for BT
    }

    # Define line styles by variant
    learner_styles = {
        # Base methods (solid lines)
        "rt_regression": {"linestyle": "-", "marker": "o"},
        "rr": {"linestyle": "-", "marker": "D"},
        # Ablations (dash-dot line)
        "rr_pooled": {"linestyle": "dashdot", "marker": "s"},
        "rr_bidir": {"linestyle": "dashdot", "marker": "v"},
        # Baseline
        "bt": {"linestyle": "--", "marker": "^"},
        # Permutation controls (dotted lines)
        "rt_regression_perm": {"linestyle": ":", "marker": "o"},
        "rr_perm": {"linestyle": ":", "marker": "D"},
    }

    # Metric display names for titles
    metric_display_names = {
        "choice_accuracy": "Accuracy",
        "pearson_distance_correlation": "PDC",
    }

    data_efficiency_thresholds = {}

    # Iterate through metrics (rows)
    for row, metric_name in enumerate(metrics):
        # Iterate through datasets (columns)
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            dataset_data = datasets_metrics_data[metric_name][dataset]

            # Calculate aggregated statistics by learner and size
            grouped_data = (
                dataset_data.groupby(["Learner", "Size"])["Value"]
                .agg(["mean", "std", ci95])
                .reset_index()
            )

            # Get performance at 100% data for BT and ResponseRank
            bt_data = grouped_data[(grouped_data["Learner"] == "bt")]
            bt_full_size_data = (
                bt_data[bt_data["Size"] == 100.0] if not bt_data.empty else None
            )

            responserank_data = grouped_data[(grouped_data["Learner"] == "rr")]
            responserank_full_size_data = (
                responserank_data[responserank_data["Size"] == 100.0]
                if not responserank_data.empty
                else None
            )

            if bt_full_size_data is not None and not bt_full_size_data.empty:
                bt_full_size_value = bt_full_size_data["mean"].values[0]
                data_efficiency_thresholds[(dataset, metric_name)] = bt_full_size_value

                if (
                    responserank_full_size_data is not None
                    and not responserank_full_size_data.empty
                ):
                    responserank_full_size_value = responserank_full_size_data[
                        "mean"
                    ].values[0]
                    data_efficiency_thresholds[
                        (dataset, metric_name, "improvement")
                    ] = (responserank_full_size_value, bt_full_size_value)

            # Plot each learner as a line
            for j, learner in enumerate(learners):
                learner_data = grouped_data[grouped_data["Learner"] == learner]

                # Sort by size
                learner_data = learner_data.sort_values(by="Size")

                display_name = learner_display_names.get(learner, learner)
                style = learner_styles.get(learner, {"linestyle": "-", "marker": "o"})
                ax.plot(
                    learner_data["Size"],
                    learner_data["mean"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    label=display_name,
                    color=learner_colors[learner],
                    linewidth=2,
                    markersize=8,
                )

                # Add 95% confidence interval bands
                ax.fill_between(
                    learner_data["Size"],
                    learner_data["mean"] - learner_data["ci95"],
                    learner_data["mean"] + learner_data["ci95"],
                    alpha=0.3,
                    color=learner_colors[learner],
                )

            # Format subplot
            if row == 0:  # Only add title on the first row
                ax.set_title(f"\\textbf{{{dataset.capitalize()}}}", fontsize=14)

            # Only add x-label on the bottom row
            if row == len(metrics) - 1:
                ax.set_xlabel(r"\textbf{Training Set Size (\%)}", fontsize=12)
            else:
                ax.set_xlabel("")

            # Only add y-label on the first column
            if col == 0:
                metric_display = metric_display_names.get(metric_name, metric_name)
                ax.set_ylabel(f"\\textbf{{{metric_display}}}", fontsize=12)

            # Format y-axis based on the metric
            if metric_name == "choice_accuracy":
                # Format as percentage
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}")
                )
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            elif metric_name == "pearson_distance_correlation":
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

            # Only show horizontal gridlines
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)

            # Add horizontal line and annotation for data efficiency if threshold is available
            if (dataset, metric_name) in data_efficiency_thresholds:
                threshold_value = data_efficiency_thresholds[(dataset, metric_name)]

                # Draw the horizontal dashed line at BT's 100% performance
                ax.axhline(
                    y=threshold_value,
                    color="#E74C3C",  # bt red
                    linestyle="--",
                    alpha=0.7,
                    zorder=0,
                )

                # Find where ResponseRank crosses this threshold (if it does)
                responserank_data = grouped_data[grouped_data["Learner"] == "rr"]
                if not responserank_data.empty:
                    responserank_data = responserank_data.sort_values(by="Size")
                    responserank_sizes = responserank_data["Size"].values
                    responserank_means = responserank_data["mean"].values

                    # Find the point where ResponseRank line crosses the threshold
                    efficiency_point = None
                    crossing_idx = None
                    for i in range(len(responserank_sizes)):
                        if responserank_means[i] >= threshold_value:
                            crossing_idx = i
                            break

                    if crossing_idx is not None:
                        if use_interpolation and crossing_idx > 0:
                            # Get the two points that bracket the threshold crossing
                            x1, y1 = (
                                responserank_sizes[crossing_idx - 1],
                                responserank_means[crossing_idx - 1],
                            )
                            x2, y2 = (
                                responserank_sizes[crossing_idx],
                                responserank_means[crossing_idx],
                            )

                            # Linear interpolation to find exact crossing point
                            slope = (y2 - y1) / (x2 - x1)
                            x_interpolated = x1 + (threshold_value - y1) / slope

                            efficiency_point = (x_interpolated, threshold_value)
                        else:
                            # Non-interpolated approach: use the first data point that exceeds threshold
                            efficiency_point = (
                                responserank_sizes[crossing_idx],
                                responserank_means[crossing_idx],
                            )

                    # If we found a crossing point, annotate it
                    if efficiency_point:
                        # vertical line
                        # ax.axvline(
                        #     x=efficiency_point[0],
                        #     color="#E74C3C",  # bt red
                        #     linestyle="--",
                        #     alpha=0.5,
                        #     zorder=0,
                        # )

                        # Add annotation text - with arrow pointing from top-right to bottom-left
                        percentage_text = (
                            f"\\textbf{{{round(efficiency_point[0], 1):.1f}\\%}}"
                            if efficiency_point[0] != int(efficiency_point[0])
                            else f"{int(efficiency_point[0])}\\%"
                        )
                        ax.annotate(
                            percentage_text,
                            xy=(efficiency_point[0], threshold_value),
                            textcoords="offset points",
                            xytext=(-6, 45),
                            arrowprops=dict(
                                arrowstyle="->",
                                connectionstyle="arc3,rad=-.2",
                                color="black",
                            ),
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                fc="white",
                                ec="gray",
                                alpha=0.8,
                            ),
                            fontsize=12,  # Plot title is 14, x-axis label 12
                            horizontalalignment="right",
                            verticalalignment="top",
                        )

            # Add annotation for performance improvement at 100% data size
            if (dataset, metric_name, "improvement") in data_efficiency_thresholds:
                responserank_value, bt_value = data_efficiency_thresholds[
                    (dataset, metric_name, "improvement")
                ]

                absolute_diff = responserank_value - bt_value

                # Format the improvement text based on the metric
                if metric_name == "pearson_distance_correlation":
                    # For PDC, use absolute difference with 2 decimal places
                    if absolute_diff >= 0:
                        improvement_text = f"\\textbf{{+{absolute_diff:.2f}}}"
                    else:
                        improvement_text = f"\\textbf{{{absolute_diff:.2f}}}"
                else:
                    # For accuracy, show as percentage points (xx% format)
                    abs_diff_percent = absolute_diff * 100
                    if abs_diff_percent >= 0:
                        improvement_text = f"\\textbf{{+{abs_diff_percent:.1f}\\%}}"
                    else:
                        improvement_text = f"\\textbf{{{abs_diff_percent:.1f}\\%}}"

                # First draw a direct arrow between the points
                ax.annotate(
                    "",  # No text on the arrow itself
                    xy=(100.0, responserank_value),
                    xytext=(100.0, bt_value),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="black",
                        lw=1.5,
                    ),
                    xycoords="data",
                    textcoords="data",
                )

                # Then add the text annotation
                midpoint = (responserank_value + bt_value) / 2
                ax.annotate(
                    improvement_text,
                    xy=(100.0, midpoint),
                    xytext=(98.5, midpoint),
                    xycoords="data",
                    textcoords="data",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc="white",
                        ec="gray",
                        alpha=0.8,
                    ),
                    fontsize=12,
                    horizontalalignment="right",
                    verticalalignment="center",
                )

    # Add a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        fancybox=True,
        shadow=True,
        ncol=len(learners),
        handlelength=5,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend

    # Save plots
    output_path_pdf = output_dir / f"{output_filename}.pdf"
    save_figure_reproducible(fig, str(output_path_pdf))
    print(f"Saved combined metrics vs. dataset size plots to {output_path_pdf}")


def run_analysis(result_dirs, learners, output_dir, metrics, output_filename):
    """Run the dataset size vs metrics analysis.

    Args:
        result_dirs: List of directories containing experiment results
        learners: List of learner names to include in the analysis
        output_dir: Directory to save output files
        metrics: List of metrics to analyze
        output_filename: Base filename for output files (without extension)
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Dictionary to store all metrics data by dataset
    all_metrics_data = {}

    # Initialize for all requested metrics
    for metric_name in metrics:
        all_metrics_data[metric_name] = {}

    dataset_order = []

    for result_dir in result_dirs:
        results, metadata = load_results(result_dir)
        dataset_name = metadata["dataset"]

        if dataset_name not in dataset_order:
            dataset_order.append(dataset_name)

        # Get dataset sizes from metadata
        dataset_sizes = metadata.get("dataset_sizes", [1.0])
        print(f"Dataset sizes in {dataset_name}: {dataset_sizes}")

        # Extract data for each metric
        for metric_name in metrics:
            # Extract metric data by dataset size
            data = extract_accuracy_by_size(results, dataset_name, metric_name)

            # Filter by learners if specified
            if learners:
                data = data[data["Learner"].isin(learners)]

            # Store in the metrics data dictionary
            all_metrics_data[metric_name][dataset_name] = data

    # Create combined metrics plot
    plot_combined_metrics_vs_size(
        all_metrics_data,
        output_dir,
        output_filename,
        learners,
        use_interpolation=True,
        dataset_order=dataset_order,
    )

    print(f"Generated combined metrics vs. dataset size plots: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Plot metrics vs. dataset size")
    parser.add_argument(
        "--result_dirs",
        nargs="+",
        required=True,
        help="Paths to aggregated result files (e.g., results/synthetic/deterministic_all.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Directory to save the plots",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["pearson_distance_correlation", "choice_accuracy"],
        help="Metrics to plot (default: choice_accuracy and pearson_distance_correlation)",
    )
    parser.add_argument(
        "--learners",
        nargs="+",
        default=None,
        help="Learners to include in the visualization (default: all available)",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        required=True,
        help="Base filename for output files (without extension)",
    )

    args = parser.parse_args()

    run_analysis(
        result_dirs=args.result_dirs,
        learners=args.learners,
        output_dir=args.output,
        metrics=args.metrics,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
