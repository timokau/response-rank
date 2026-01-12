#!/usr/bin/env python3
"""
Plot combined metrics across multiple datasets.

This script generates boxplots for choice_accuracy and distance_pearson metrics
across multiple datasets, showing the performance of different learners.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from scipy import stats

from responserank.llm.analysis.plotting_utils import save_figure_reproducible

# Enable LaTeX rendering
rc("text", usetex=True)
rc("font", family="serif")


def _ci95_half_width(values):
    """Calculate 95% CI half-width."""
    values = np.array(values)
    assert len(values) > 1
    sem = stats.sem(values)
    return sem * stats.t.ppf(0.975, len(values) - 1)


def load_results(result_path):
    """Load aggregated results from a result file.

    Args:
        result_path: Path to the aggregated result file (e.g., deterministic_all.json)

    Returns:
        Tuple of (results_dict, dataset_name)
    """
    with open(result_path) as f:
        data = json.load(f)
    metadata = data.pop("metadata")
    dataset_name = metadata.get("dataset")
    return data, dataset_name


def extract_metric_data(
    results,
    dataset_name,
    metric_name,
    learners: Optional[List[str]],
):
    """Extract metric data for specified learners from results.

    Args:
        results: Dictionary of aggregated results
        dataset_name: Name of the dataset
        metric_name: Name of the metric to extract
        learners: List of learner names to include (None for all)

    Returns:
        DataFrame with columns: Dataset, Learner, Size, Trial, Value
    """
    data_rows = []

    # If no learners specified, use all available
    if learners is None:
        learners = [k for k in results.keys() if k != "dataset_metrics"]

    for learner in learners:
        if learner not in results:
            print(f"Warning: Learner '{learner}' not found in dataset '{dataset_name}'")
            continue

        sizes = results[learner]["sizes"].keys()
        sizes = sorted(sizes, key=lambda x: float(x))

        for size in sizes:
            if "summary_metrics" not in results[learner]["sizes"][size]:
                print(
                    f"Warning: No summary metrics for '{learner}' in '{dataset_name}' at size {size}"
                )
                continue

            if metric_name not in results[learner]["sizes"][size]["summary_metrics"]:
                print(
                    f"Warning: Metric '{metric_name}' not found for '{learner}' in '{dataset_name}' at size {size}"
                )
                continue

            metric_values = results[learner]["sizes"][size]["summary_metrics"][
                metric_name
            ]

            for trial, value in enumerate(metric_values):
                # Handle both scalar metrics and metric dictionaries
                if isinstance(value, dict) and "type" in value and "value" in value:
                    if value["type"] == "scalar":
                        metric_value = value["value"]
                    else:
                        # For now, skip non-scalar metrics
                        continue
                else:
                    metric_value = value

                size_percentage = int(float(size) * 100)

                data_rows.append(
                    {
                        "Dataset": dataset_name,
                        "Learner": learner,
                        "Size": size_percentage,
                        "Trial": trial + 1,
                        "Value": metric_value,
                    }
                )

    return pd.DataFrame(data_rows)


def plot_combined_metrics(metric_data_dict, output_dir, output_suffix, learner_order):
    """Create a combined boxplot visualization for multiple metrics across datasets.

    Args:
        metric_data_dict: Dictionary mapping metric names to DataFrames with columns: Dataset, Learner, Trial, Value
        output_dir: Directory to save the plot
        output_suffix: Suffix for output filenames
        learner_order: List specifying the order of learners
    """
    # Get unique datasets and learners (assuming consistent across metrics)
    metrics = list(metric_data_dict.keys())
    first_metric = metrics[0]

    # Get unique datasets and learners
    datasets = sorted(metric_data_dict[first_metric]["Dataset"].unique())
    learners = learner_order

    # Learner display names for cleaner labels
    learner_display_names = {
        "bt": "BT",
        "rt_regression": "RtRegression",
        "rt_regression_perm": "RtRegression-Perm",
        "rr": "ResponseRank",
        "rr_pooled": "ResponseRank-Pool",
        "rr_perm": "ResponseRank-Perm",
    }

    # Set up the figure: 2 rows (one for each metric), 3 columns (one for each dataset)
    fig, axes = plt.subplots(
        len(metrics),
        len(datasets),
        figsize=(len(datasets) * 5, len(metrics) * 4.5),
        sharex="col",
        sharey="row",
    )

    # Create color mapping by method family for consistent colors with plot_datasize.py
    learner_colors = {
        # Main method families (same color for base/permutation)
        "rr": "#1D9A6C",  # teal for ResponseRank
        "rr_perm": "#1D9A6C",  # same teal
        "rr_pooled": "#1D9A6C",  # same teal for ResponseRank-Pool
        "rt_regression": "#D4A017",  # gold for RtRegression
        "rt_regression_perm": "#D4A017",  # same gold
        # Baseline method
        "bt": "#E74C3C",  # red for BT
    }

    # Metric display names for y-axis labels
    metric_display_names = {
        "pearson_distance_correlation": "PDC",
        "choice_accuracy": "Accuracy",
    }

    # Create boxplots for each metric and dataset
    for row, metric_name in enumerate(metrics):
        metric_data = metric_data_dict[metric_name]

        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            dataset_data = metric_data[metric_data["Dataset"] == dataset]
            # Sort for deterministic PDF output (fix matplotlib drawing order)
            dataset_data = dataset_data.sort_values(
                ["Learner", "Size", "Value"]
            ).reset_index(drop=True)

            # Create the boxplot
            sns.boxplot(
                x="Learner",
                y="Value",
                hue="Learner",
                data=dataset_data,
                ax=ax,
                palette=learner_colors,
                notch=True,
                legend=False,
                order=learners,
            )

            ax.yaxis.grid(True, linestyle="--", alpha=0.7)

            # Hide x-tick marks for all subplots but keep labels
            ax.tick_params(axis="x", which="both", length=0)

            # Format y-axis based on the metric
            if metric_name == "choice_accuracy":
                # Convert y values to percentages
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda y, _: "{:.0f}\\%".format(y * 100))
                )
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            elif metric_name == "pearson_distance_correlation":
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

            # Customize the plot
            if row == 0:  # Only set title on the first row
                ax.set_title(r"\textbf{" + dataset.capitalize() + r"}", fontsize=14)

            # Only set y-label on the first column
            if col == 0:
                ylabel = (
                    r"\textbf{"
                    + metric_display_names.get(metric_name, metric_name)
                    + r"}"
                )

                ax.set_ylabel(ylabel, fontsize=14, labelpad=15)
                ax.yaxis.set_label_coords(-0.12, 0.5)
            else:
                ax.set_ylabel("")

            # Handle x-labels
            if row == len(metrics) - 1:
                # Remove default x-labels to avoid warnings
                ax.set_xlabel("")
                ax.set_xticklabels([])

                # Get the current tick positions
                tick_positions = ax.get_xticks()

                # Explicitly set ticks and labels for bottom row
                if len(tick_positions) == len(learners):
                    ax.set_xticks(tick_positions)
                    display_labels = [
                        learner_display_names.get(learner, learner)
                        for learner in learners
                    ]
                    ax.set_xticklabels(display_labels, rotation=20, ha="center")

                    # Hide x-tick marks but keep labels
                    ax.tick_params(axis="x", which="both", length=0)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

    # No legend needed as the x-tick labels already show this information

    # Adjust layout
    plt.tight_layout()

    combined_filename = f"combined_metrics_boxplot{output_suffix}.pdf"
    output_path = output_dir / combined_filename
    save_figure_reproducible(fig, str(output_path))

    print(f"Saved combined metrics visualization to {output_path}")
    plt.close()

    # Get trial counts from the first metric (should be the same for all metrics)
    first_metric = list(metric_data_dict.keys())[0]
    first_metric_data = metric_data_dict[first_metric]

    # Count trials for each dataset/learner/size combination
    sizes = sorted(first_metric_data["Size"].unique())
    trial_counts = []
    for dataset in datasets:
        for learner in learners:
            for size in sizes:
                learner_data = first_metric_data[
                    (first_metric_data["Dataset"] == dataset)
                    & (first_metric_data["Learner"] == learner)
                    & (first_metric_data["Size"] == size)
                ]
                if len(learner_data) > 0:
                    trial_counts.append(len(learner_data))

    # Get unique trial counts and format them for vars.tex
    unique_counts = sorted(set(trial_counts), reverse=True)
    trials_text = "/".join(str(count) for count in unique_counts)

    # Write vars.tex with trial count information (numbers only)
    vars_path = output_dir / f"vars{output_suffix}.tex"
    with open(vars_path, "w") as f:
        f.write(f"\\newcommand{{\\numTrials}}{{{trials_text}}}\n")

    # Warn about inconsistent trial counts
    if len(unique_counts) > 1:
        print(f"Warning: Inconsistent trial counts detected: {unique_counts}")
        print(
            "         This may indicate missing data or inconsistent experimental setups."
        )

    # Now process each metric for CSV and LaTeX tables

    # Save numerical results as CSV summary and LaTeX table
    for metric_name, metric_data in metric_data_dict.items():
        # Create summary table with mean, std, and CI for each dataset/learner/size combination
        summary_data = (
            metric_data.groupby(["Dataset", "Learner", "Size"])["Value"]
            .agg(["mean", "std", _ci95_half_width])
            .reset_index()
        )
        summary_data.columns = ["Dataset", "Learner", "Size", "mean", "std", "ci"]

        # Create and save a LaTeX table
        latex_path = output_dir / f"{metric_name}_table{output_suffix}.tex"

        # Format the data for LaTeX table
        latex_table = []

        datasets = sorted(summary_data["Dataset"].unique())
        all_learners = learners
        all_sizes = sorted(summary_data["Size"].unique())

        # Table header (right-aligned numeric columns)
        header = (
            "\\begin{tabular}{l" + "".join(["r" for _ in range(len(all_sizes))]) + "}"
        )
        latex_table.append(header)
        latex_table.append("\\toprule")

        # Column headers (size percentages)
        size_header = (
            "Dataset / Learner & "
            + " & ".join([f"{size}\\%" for size in all_sizes])
            + " \\\\"
        )
        latex_table.append(size_header)
        latex_table.append("\\midrule")

        # Find highest (assumed best) result for each dataset/size combination
        best_values = {}
        for dataset in datasets:
            for size in all_sizes:
                dataset_size_data = summary_data[
                    (summary_data["Dataset"] == dataset)
                    & (summary_data["Size"] == size)
                ]

                best_idx = (
                    dataset_size_data["mean"].idxmax()
                    if len(dataset_size_data) > 0
                    else None
                )

                if best_idx is not None:
                    best_values[(dataset, size)] = dataset_size_data.loc[
                        best_idx, "Learner"
                    ]

        for dataset in datasets:
            # Add dataset name as a multirow header with bold text
            latex_table.append(
                f"\\multicolumn{{{len(all_sizes) + 1}}}{{l}}{{\\textbf{{{dataset.capitalize()}}}}} \\\\"
            )

            # Format rows for each learner in this dataset
            for learner in all_learners:
                learner_data = summary_data[
                    (summary_data["Dataset"] == dataset)
                    & (summary_data["Learner"] == learner)
                ]

                learner_row = [learner_display_names.get(learner, learner)]

                for size in all_sizes:
                    size_data = learner_data[learner_data["Size"] == size]

                    mean_val = size_data["mean"].values[0]
                    std_val = size_data["std"].values[0]
                    ci_val = size_data["ci"].values[0]

                    # Format based on metric type
                    if metric_name == "choice_accuracy":
                        # Format as percentage
                        val = f"{mean_val * 100:.1f}"
                        std = f"{std_val * 100:.1f}"
                        ci = f"{ci_val * 100:.1f}"
                        suffix = "\\%"
                    else:
                        # Format as decimal
                        val = f"{mean_val:.2f}"
                        std = f"{std_val:.2f}"
                        ci = f"{ci_val:.2f}"
                        suffix = ""

                    if (dataset, size) in best_values and best_values[
                        (dataset, size)
                    ] == learner:
                        val = f"\\mathbf{{{val}}}"

                    cell = f"${val} \\pm {std}{suffix}$ [$\\pm$ {ci}]"

                    learner_row.append(cell)

                latex_table.append(" & ".join(learner_row) + " \\\\")

            # Separate datasets (except for last)
            if dataset != datasets[-1]:
                latex_table.append("\\midrule")

        # Table footer
        latex_table.append("\\bottomrule")
        latex_table.append("\\end{tabular}")

        # Write LaTeX table to file
        with open(latex_path, "w") as f:
            f.write("\n".join(latex_table))

        print(f"Saved {metric_name} summary results to {latex_path}")


def generate_statistical_table(results_data, output_dir, output_suffix):
    """Generate a LaTeX table summarizing the statistical test results.

    Args:
        results_data: List of dictionaries containing test results
        output_dir: Directory to save the LaTeX table
        output_suffix: Suffix for the output filename.
    """
    # File to save the table
    table_path = output_dir / f"statistical_table{output_suffix}.tex"

    table_lines = [
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Dataset} & \\textbf{Comparison} & \\textbf{p-value} & \\textbf{Effect} & \\textbf{Mean Diff. (Effect Size)} \\\\",
        "\\midrule",
    ]

    # Group results by metric
    metrics = sorted(set(item["metric"] for item in results_data))

    for i, metric in enumerate(metrics):
        metric_results = [item for item in results_data if item["metric"] == metric]

        # Group by comparison type
        comparisons = sorted(set(item["hypothesis"] for item in metric_results))

        for j, comparison in enumerate(comparisons):
            comparison_results = [
                item for item in metric_results if item["hypothesis"] == comparison
            ]

            # Sort by dataset
            comparison_results.sort(key=lambda x: x["dataset"])

            for result in comparison_results:
                # Format p-value
                if result.get("p_value_formatted") in ["N/A", "\\text{N/A}"]:
                    p_value_str = "$\\text{N/A}$"  # Properly wrapped in math mode
                else:
                    p_value_str = "$" + result.get("p_value_formatted", "N/A") + "$"

                # Format effect direction
                direction = result.get("direction_symbol", "")

                # Format mean difference and effect size
                mean_diff = result.get("mean_diff", float("nan"))
                effect_size = result.get("effect_size", float("nan"))

                if np.isnan(mean_diff) or np.isnan(effect_size):
                    diff_str = "$\\text{N/A}$"
                else:
                    # Format as percentage for accuracy, raw value for other metrics
                    if metric == "choice_accuracy":
                        # Convert to percentage
                        diff_str = f"${mean_diff * 100:.1f}\\%$ (${effect_size:.2f}$)"
                    else:
                        diff_str = f"${mean_diff:.4f}$ (${effect_size:.2f}$)"

                # Format metric display name
                metric_display = {
                    "choice_accuracy": "Accuracy",
                    "pearson_distance_correlation": "PDC",
                }.get(metric, metric.upper())

                # Add row to table
                row = f"{metric_display} & {result['dataset'].capitalize()} & {result['hypothesis']} & {p_value_str} & {direction} & {diff_str} \\\\"
                table_lines.append(row)

            # Add midrule between comparison types (except after the last one)
            if j < len(comparisons) - 1:
                table_lines.append("\\midrule")

        # Add midrule between metrics (except after the last one)
        if i < len(metrics) - 1:
            table_lines.append("\\midrule")

    # Finish the table
    table_lines.extend(["\\bottomrule", "\\end{tabular}"])

    # Write table to file
    with open(table_path, "w") as f:
        f.write("\n".join(table_lines))

    print(f"Statistical comparison table saved to {table_path}")


def perform_statistical_tests(metric_data_dict, output_dir, learner_order):
    """Perform statistical tests to compare different learners.

    Args:
        metric_data_dict: Dictionary mapping metric names to DataFrames with columns: Dataset, Learner, Trial, Value
        output_dir: Directory to save the statistical test results
        learner_order: List specifying the order of learners

    Returns:
        List of dictionaries containing test results
    """
    # Pairs to compare
    comparisons = [
        ("bt", "rr", "ResponseRank vs BT"),
        ("rr_perm", "rr", "ResponseRank vs ResponseRank-Perm"),
        ("rt_regression", "rr", "ResponseRank vs RtRegression"),
        ("bt", "rt_regression", "RtRegression vs BT"),
        ("rt_regression_perm", "rt_regression", "RtRegression vs RtRegression-Perm"),
        ("rr_pooled", "rr", "ResponseRank vs ResponseRank-Pool"),
    ]

    # Collect all test results for table generation
    all_test_results = []

    for metric_name, metric_data in metric_data_dict.items():
        # Per-dataset analysis
        for dataset in sorted(metric_data["Dataset"].unique()):
            dataset_data = metric_data[metric_data["Dataset"] == dataset]

            for model1, model2, hypothesis in comparisons:
                # Get data for both models
                model1_data = dataset_data[dataset_data["Learner"] == model1][
                    "Value"
                ].values
                model2_data = dataset_data[dataset_data["Learner"] == model2][
                    "Value"
                ].values

                # Initialize result dictionary for this test
                test_result = {
                    "metric": metric_name,
                    "dataset": dataset,
                    "model1": model1,
                    "model2": model2,
                    "hypothesis": hypothesis,
                }

                # Match trials if uneven (take smaller set)
                model_1_num_trials = len(model1_data)
                model_2_num_trials = len(model2_data)
                if model_1_num_trials != model_2_num_trials:
                    print(
                        f"Warning: Model {model1} vs {model2} data mismatched: {model_1_num_trials} vs {model_2_num_trials} trials."
                    )
                min_trials = min(len(model1_data), len(model2_data))
                model1_data = model1_data[:min_trials]
                model2_data = model2_data[:min_trials]

                # Perform statistical tests

                # Calculate basic statistics for both models
                n = len(model1_data)
                mean_diff = np.mean(model2_data - model1_data)
                test_result["n"] = n
                test_result["mean_diff"] = mean_diff

                # Always use Wilcoxon signed-rank test without checking for normality
                statistic, p_value = stats.wilcoxon(model1_data, model2_data)
                test_name = "Wilcoxon signed-rank test"

                test_result["test_name"] = test_name
                test_result["p_value"] = p_value

                # Calculate effect size (Cohen's d for paired samples)
                mean_1 = np.mean(model1_data)
                mean_2 = np.mean(model2_data)
                all_data = np.concatenate([model1_data, model2_data])
                pooled_std = np.std(all_data, ddof=1)

                effect_size = (mean_2 - mean_1) / pooled_std
                test_result["effect_size"] = effect_size

                # Determine direction (is model2 better than model1?)
                mean1 = np.mean(model1_data)
                mean2 = np.mean(model2_data)
                test_result["mean1"] = mean1
                test_result["mean2"] = mean2

                lower_is_better = metric_name != "choice_accuracy"

                if (lower_is_better and mean2 < mean1) or (
                    not lower_is_better and mean2 > mean1
                ):
                    direction = "improvement"
                    direction_symbol = "↑"  # Improvement
                else:
                    direction = "decline"
                    direction_symbol = "↓"  # Decline

                test_result["direction"] = direction
                test_result["direction_symbol"] = direction_symbol

                # Format result for output
                significance = "significant" if p_value < 0.05 else "not significant"
                test_result["significance"] = significance

                # Format p-value for display
                if np.isnan(p_value):
                    p_val_formatted = "\\text{N/A}"
                elif p_value < 0.0001:
                    p_val_formatted = "< 0.0001"
                else:
                    p_val_formatted = f"{p_value:.4f}"

                test_result["p_value_formatted"] = p_val_formatted

                all_test_results.append(test_result)

    return all_test_results


def run_analysis(result_dirs, learners, output_dir, metrics, output_suffix):
    """Run the combined metrics analysis with the given parameters.

    Args:
        result_dirs: List of directories containing experiment results
        learners: List of learner names to include in the analysis
        output_dir: Directory to save output files
        metrics: List of metrics to analyze
        output_suffix: Suffix for output filenames
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load results from all directories
    all_results = []
    for result_dir in result_dirs:
        try:
            results, dataset_name = load_results(result_dir)
            all_results.append((results, dataset_name))
            print(f"Loaded results from {result_dir} (dataset: {dataset_name})")
        except Exception as e:
            print(f"Error loading results from {result_dir}: {e}")

    # Collect all metric data in a dictionary
    metric_data_dict = {}
    for metric_name in metrics:
        combined_data = pd.DataFrame()
        for results, dataset_name in all_results:
            metric_data = extract_metric_data(
                results, dataset_name, metric_name, learners
            )
            combined_data = pd.concat([combined_data, metric_data])
        metric_data_dict[metric_name] = combined_data

    # Create and save the combined visualization
    output_suffix_formatted = "" if len(output_suffix) == 0 else f"_{output_suffix}"
    plot_combined_metrics(
        metric_data_dict, output_dir, output_suffix_formatted, learners
    )

    # Perform statistical tests and save results
    generate_statistical_table(
        perform_statistical_tests(metric_data_dict, output_dir, learners),
        output_dir,
        output_suffix_formatted,
    )


def main():
    """Main function to parse arguments and create visualizations."""
    parser = argparse.ArgumentParser(
        description="Create combined metric visualizations across datasets"
    )
    parser.add_argument(
        "--result_dirs",
        nargs="+",
        required=True,
        help="Paths to aggregated result files (e.g., results/synthetic/deterministic_all.json)",
    )
    parser.add_argument(
        "--learners",
        nargs="+",
        default=None,
        help="Learners to include in the visualization (default: all available)",
    )
    parser.add_argument(
        "--output",
        default="./combined",
        help="Output directory for visualizations (default: ./combined)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["pearson_distance_correlation", "choice_accuracy"],
        help="Metrics to visualize (default: pearson_distance_correlation and choice_accuracy)",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output filenames",
    )

    args = parser.parse_args()

    run_analysis(
        result_dirs=args.result_dirs,
        learners=args.learners,
        output_dir=args.output,
        metrics=args.metrics,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
