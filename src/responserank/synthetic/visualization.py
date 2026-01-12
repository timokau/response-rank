import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from responserank.llm.analysis.plotting_utils import save_figure_reproducible


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    title,
    results_basedir,
    escape_latex: bool,
    annot_fontsize: int,
    tick_fontsize: int,
    label_fontsize: int,
):
    """Plot a confusion matrix using seaborn's heatmap."""
    plt.figure(figsize=(6, 6))

    # Calculate percentages for annotations
    total = confusion_matrix.sum()
    percentages = confusion_matrix / total * 100

    # Create annotations with both count and percentage
    pct_symbol = r"\%" if escape_latex else "%"
    annotations = np.array(
        [
            [
                f"{count}\n({percentage:.1f}{pct_symbol})"
                for count, percentage in zip(row, row_percentages)
            ]
            for row, row_percentages in zip(confusion_matrix, percentages)
        ]
    )

    sns.heatmap(
        confusion_matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=["Option 1", "Option 2"],
        yticklabels=["Should 1", "Should 2"],
        square=True,
        annot_kws={"size": annot_fontsize},
    )

    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xlabel("Chosen Option", fontsize=label_fontsize)
    plt.ylabel("True Preference", fontsize=label_fontsize)
    fig = plt.gcf()
    save_figure_reproducible(
        fig, f"{results_basedir}/{title.lower()}_confusion_matrix.pdf"
    )
    plt.close()


def plot_results(results, results_basedir, learner_order: Optional[List[str]]):
    os.makedirs(results_basedir, exist_ok=True)
    model_keys = [k for k in results.keys() if k != "dataset_metrics"]
    model_types = learner_order if learner_order is not None else model_keys

    # Plot dataset metrics
    dataset_metrics = {}

    for metric_name, metric_data in results["dataset_metrics"].items():
        scalar_plot_data = []
        histogram_data = {}

        if metric_data[0].get("type") == "confusion_matrix":
            joint_confusion_matrix = np.array(metric_data[0]["value"])
            for data in metric_data[1:]:
                joint_confusion_matrix += np.array(data["value"])
            plot_confusion_matrix(
                joint_confusion_matrix,
                metric_name,
                results_basedir,
                escape_latex=False,
                annot_fontsize=10,
                tick_fontsize=10,
                label_fontsize=10,
            )
        elif metric_data[0].get("type") == "histogram":
            histogram_data["dataset"] = [m["value"] for m in metric_data]
            plot_histogram(
                histogram_data,
                metric_name,
                f"{metric_name}",
                results_basedir,
            )
        else:
            for value in metric_data:
                # Use empty string as column name to avoid x-axis label while keeping tick label "dataset"
                scalar_plot_data.append({"": "dataset", metric_name: value["value"]})
            if scalar_plot_data:
                plot_boxplot(
                    scalar_plot_data,
                    "",
                    metric_name,
                    f"{metric_name}",
                    results_basedir,
                    ["dataset"],
                )
        dataset_metrics[metric_name] = True

    # Plot learner metrics
    all_summary_metrics = set()
    for model_type in model_types:
        sizes = results[model_type]["sizes"].keys()
        largest_size = sorted(sizes, key=lambda x: float(x))[-1]
        model_metrics = results[model_type]["sizes"][largest_size]["summary_metrics"]
        all_summary_metrics.update(model_metrics.keys())

    # Remove dataset metrics from learner plots
    all_summary_metrics = all_summary_metrics - set(dataset_metrics.keys())

    for metric_name in all_summary_metrics:
        scalar_plot_data = []
        histogram_data = {}

        for model_type in model_types:
            # Get metrics for the largest dataset size
            sizes = results[model_type]["sizes"].keys()
            largest_size = sorted(sizes, key=lambda x: float(x))[-1]
            metric_dicts = results[model_type]["sizes"][largest_size][
                "summary_metrics"
            ].get(metric_name, [])

            for metric_dict in metric_dicts:
                if metric_dict.get("type") == "scalar":
                    scalar_plot_data.append(
                        {"Model": model_type, metric_name: metric_dict["value"]}
                    )
                elif metric_dict.get("type") == "histogram":
                    if model_type not in histogram_data:
                        histogram_data[model_type] = []
                    histogram_data[model_type].append(metric_dict["value"])

        if len(scalar_plot_data) > 0:
            plot_boxplot(
                scalar_plot_data,
                "Model",
                metric_name,
                f"{metric_name}",
                results_basedir,
                model_types,
            )
            # plot_violinplot(
            #     scalar_plot_data,
            #     "Model",
            #     metric_name,
            #     f"{metric_name.capitalize()} Comparison (Violin Plot)",
            #     results_basedir,
            #     model_types,
            # )

        if histogram_data:
            plot_histogram(
                histogram_data,
                metric_name,
                f"{metric_name}",
                results_basedir,
            )

    # Plot all scalar metrics over epochs
    plot_scalar_metrics(results, model_types, results_basedir)


def plot_boxplot(data, x, y, title, results_basedir, order):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x, y=y, data=pd.DataFrame(data), order=order, notch=True)
    # plt.title(title)
    plt.xticks(rotation=45)
    fig = plt.gcf()
    save_figure_reproducible(
        fig, f"{results_basedir}/{y.lower()}_comparison_boxplot.pdf"
    )
    plt.close()


def plot_violinplot(data, x, y, title, results_basedir, order):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=x, y=y, data=pd.DataFrame(data), order=order)
    # plt.title(title)
    plt.xticks(rotation=45)
    fig = plt.gcf()
    save_figure_reproducible(
        fig, f"{results_basedir}/{y.lower()}_comparison_violinplot.pdf"
    )
    plt.close()


def plot_histogram(
    data, metric_name, title, results_basedir, subplot_width=6, subplot_padding=0.5
):
    n_models = len(data)
    fig_width = n_models * subplot_width + (n_models - 1) * subplot_padding
    plt.figure(figsize=(fig_width, 6))

    # First combine all data for each model
    combined_data = {}
    for model_type, hist_list in data.items():
        if not hist_list:
            continue

        # Get the histogram parameters from first item
        combined_data[model_type] = {
            k: v for k, v in hist_list[0].items() if k not in ["dataset_metric", "type"]
        }

        # Combine x values from all trials
        combined_x = []
        for hist_item in hist_list:
            combined_x.extend(hist_item["x"])
        combined_data[model_type]["x"] = combined_x

    # Now calculate frequencies and plot
    max_freq = 0
    for idx, (model_type, hist_params) in enumerate(combined_data.items(), 1):
        plt.subplot(1, n_models, idx)

        hist_kwargs = {k: v for k, v in hist_params.items() if k not in ["x"]}
        freq, _ = np.histogram(hist_params["x"], **hist_kwargs)
        max_freq = max(max_freq, max(freq))

        plt.hist(**hist_params, alpha=0.5, label=model_type)

        if n_models > 1:
            # plt.title(f"{model_type}")
            pass
        if idx == 1:  # Only add y-label to the leftmost subplot
            plt.ylabel("Frequency")

        # Use metric name as xlabel for dataset metrics, model type otherwise
        xlabel = metric_name if model_type == "dataset" else model_type
        plt.xlabel(xlabel)

    # Apply consistent y-axis limits
    for idx in range(1, n_models + 1):
        plt.subplot(1, n_models, idx)
        plt.ylim(0, max_freq * 1.05)  # Add 5% padding to the top

    # plt.suptitle(title)
    fig = plt.gcf()
    save_figure_reproducible(
        fig, f"{results_basedir}/{metric_name.lower()}_histograms.pdf"
    )
    plt.close()


def plot_scalar_metrics(results, model_types, results_basedir):
    first_model = model_types[0]
    sizes = results[first_model]["sizes"].keys()
    largest_size = sorted(sizes, key=lambda x: float(x))[-1]

    # Get all scalar metrics across all model types
    all_scalar_metrics = set()
    for model_type in model_types:
        if not results[model_type]["sizes"][largest_size]["epoch_metrics"]:
            continue

        for trial in results[model_type]["sizes"][largest_size]["epoch_metrics"]:
            for epoch in trial:
                all_scalar_metrics.update(
                    metric_name
                    for metric_name, metric_dict in epoch.items()
                    if metric_dict.get("type") == "scalar"
                )

    for metric_name in all_scalar_metrics:
        # Determine which model types have this metric
        model_types_with_metric = [
            model_type
            for model_type in model_types
            if results[model_type]["sizes"][largest_size]["epoch_metrics"]
            and any(
                metric_name in epoch
                for trial in results[model_type]["sizes"][largest_size]["epoch_metrics"]
                for epoch in trial
            )
        ]

        # Skip this metric if no model type has it
        if not model_types_with_metric:
            continue

        plt.figure(figsize=(12, 6))
        for model_type in model_types_with_metric:
            # Ensure all trials have the same number of epochs
            max_epochs = max(
                len(trial)
                for trial in results[model_type]["sizes"][largest_size]["epoch_metrics"]
            )
            padded_metrics = []
            for trial in results[model_type]["sizes"][largest_size]["epoch_metrics"]:
                padded_trial = [
                    m.get(metric_name, {}).get("value", float("nan")) for m in trial
                ] + [float("nan")] * (max_epochs - len(trial))
                padded_metrics.append(padded_trial)

            avg_metric = np.nanmean(padded_metrics, axis=0)
            std_metric = np.nanstd(padded_metrics, axis=0)

            # Plot mean line
            plt.plot(range(len(avg_metric)), avg_metric, label=model_type)

            # Add shaded area for standard deviation
            plt.fill_between(
                range(len(avg_metric)),
                avg_metric - std_metric,
                avg_metric + std_metric,
                alpha=0.3,
            )

        # plt.title(f"{metric_name} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        fig = plt.gcf()
        save_figure_reproducible(
            fig, f"{results_basedir}/{metric_name}_over_epochs.pdf"
        )
        plt.close()
