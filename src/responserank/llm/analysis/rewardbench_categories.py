"""RewardBench v2 category plotting functionality."""

import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

from responserank.llm.analysis.joint_perf_plot import (
    calculate_ci95_half_width,
    calculate_data_bounds,
    plot_metric_on_axis,
)
from responserank.llm.analysis.plotting_utils import (
    save_figure_reproducible,
    setup_plot_style,
)

# Enable LaTeX rendering
rc("text", usetex=True)
rc("font", family="serif")


def prepare_rewardbench_data(
    df: pd.DataFrame, metadata: dict, include_experiments: Optional[List[List[str]]]
):
    """Extract and prepare data for RewardBench category plotting.

    Args:
        df: DataFrame loaded from cache
        metadata: Metadata dict from cache
        include_experiments: Optional list of experiment display name arrays to include.
            If None, includes all experiments. Experiments are specified as an array of strings,
            indicating how they will be placed in the legend grid.

    Returns:
        dict with styling derived from ``metadata["computed_styling"]``.
    """
    if include_experiments is None:
        experiment_labels = sorted(df["egid"].unique())
        include_experiments = [experiment_labels]
    else:
        # Filter experiments based on include_experiments parameter
        computed_display_names = metadata["computed_display_names"]
        alias_to_raw = {v: k for k, v in computed_display_names.items()}

        # Transpose include_experiments to convert from row-major to column-major
        # for matplotlib's columnwise legend filling
        # Rows must have equal length (pad with None at callsite)
        transposed = [list(col) for col in zip(*include_experiments)]

        # Flatten column-major order
        filter_names = []
        display_labels_map = {}
        for col in transposed:
            for exp_display_name in col:
                if exp_display_name is None:
                    filter_names.append(None)
                else:
                    raw_name = alias_to_raw.get(exp_display_name, exp_display_name)
                    filter_names.append(raw_name)
                    display_labels_map[raw_name] = exp_display_name

        # Filter out None entries when querying data
        real_filter_names = [name for name in filter_names if name is not None]
        df = df[df["egid"].isin(real_filter_names)]

        # Keep experiment_labels in the transposed order (including None)
        experiment_labels = []
        for name in filter_names:
            if name is None:
                experiment_labels.append(None)
            elif name in df["egid"].unique():
                experiment_labels.append(name)
            else:
                # Maintain grid structure for missing experiments
                experiment_labels.append(None)

    fractions = sorted(df["fraction"].unique())

    display_name_aliases = metadata.get("computed_display_names", {})

    computed_styling = metadata["computed_styling"]
    run_colors_lookup = computed_styling["run_colors"]
    run_markers_lookup = computed_styling["run_markers"]
    run_linestyles_lookup = computed_styling["run_linestyles"]

    run_colors = {}
    run_markers = {}
    run_linestyles = {}

    for label in experiment_labels:
        if label is None:
            # None entries for legend spacing - use invisible style
            run_colors[label] = "none"
            run_markers[label] = ""
            run_linestyles[label] = ""
            continue

        if (
            label not in run_colors_lookup
            or label not in run_markers_lookup
            or label not in run_linestyles_lookup
        ):
            raise ValueError(f"Missing computed styling for display name '{label}'")

        run_colors[label] = run_colors_lookup[label]
        run_markers[label] = run_markers_lookup[label]
        run_linestyles[label] = run_linestyles_lookup[label]

    majority_counts = {}
    run_has_deviations = {}

    for label in experiment_labels:
        if label is None:
            majority_counts[label] = 0
            run_has_deviations[label] = False
            continue

        label_data = df[df["egid"] == label]
        if label_data.empty:
            continue

        if "run_id" not in label_data.columns:
            raise ValueError(
                "RewardBench plotting requires 'run_id' column to compute sample counts"
            )

        per_fraction_counts = label_data.groupby("fraction")["run_id"].nunique()

        if per_fraction_counts.empty:
            raise ValueError(f"No fraction data found for display name '{label}'")

        majority_count = int(per_fraction_counts.mode().iloc[0])
        majority_counts[label] = majority_count
        run_has_deviations[label] = (per_fraction_counts != majority_count).any()

    rb_v2_section_cols = [
        col
        for col in df.columns
        if col.startswith("eval/rewardbench_v2/section/") and col.endswith("_score")
    ]

    rb_v2_overall_col = "eval/rewardbench_v2/overall/accuracy_score"

    if rb_v2_overall_col not in df.columns and not rb_v2_section_cols:
        return None

    all_v2_cols = []
    all_v2_cols.extend(sorted(rb_v2_section_cols))

    cols_with_data = []
    for col in all_v2_cols:
        if col in df.columns and df[col].notna().any():
            cols_with_data.append(col)

    if not cols_with_data:
        return None

    return {
        "experiment_labels": experiment_labels,
        "fractions": fractions,
        "run_colors": run_colors,
        "run_markers": run_markers,
        "run_linestyles": run_linestyles,
        "majority_counts": majority_counts,
        "run_has_deviations": run_has_deviations,
        "cols_with_data": cols_with_data,
        "df": df,
        "display_name_aliases": display_name_aliases,
    }


def build_rewardbench_plot(
    data_bundle: Dict[str, Any],
    plot_width: float,
    show_points: bool,
    show_std: str,
    *,
    use_ci95: bool,
    use_median: bool,
    height_per_row: float,
    marker_size: float,
    line_width: float,
    dash_scale: float,
    title_fontsize: float,
    xlabel_fontsize: float = 14.0,
    ylabel_fontsize: float = 14.0,
    xtick_fontsize: float = 12.0,
    ytick_fontsize: float = 12.0,
    shared_ylabel: str = r"\bf{}Score",
    supylabel_kwargs: Optional[Dict[str, Any]] = None,
    shared_xlabel: str = r"\bf{}Training Set Size (\%)",
    supxlabel_kwargs: Optional[Dict[str, Any]] = None,
    include_experiments: Optional[List[List[str]]],
) -> Tuple[plt.Figure, np.ndarray, List[Any], List[str], int]:
    """Construct RewardBench category plots and return legend payload.

    Args:
        data_bundle: Prepared data from ``prepare_rewardbench_data``.
        plot_width: Width of the figure in inches.
        show_points: Whether to draw jittered sample points.
        show_std: Spread visualization mode.
        use_ci95: Whether to use 95% confidence intervals for spread.
        use_median: Whether to aggregate with medians (disables spread visuals).
        height_per_row: Base figure height in inches for a full-range (0-1) row.
            Actual row height scales with the maximum value observed in that row.
        marker_size: Marker size (points) for line plots and jitter points.
        line_width: Width of plotted lines in points.
        dash_scale: Scale factor applied to dash patterns for dashed lines.
        title_fontsize: Font size for subplot titles.
        xlabel_fontsize: Font size for x-axis labels.
        ylabel_fontsize: Font size for y-axis labels.
        xtick_fontsize: Font size for x-axis tick labels.
        ytick_fontsize: Font size for y-axis tick labels.
        shared_ylabel: Text to use for the figure-level y-axis label.
        supylabel_kwargs: Additional keyword arguments forwarded to ``fig.supylabel``.
        shared_xlabel: Text to use for the figure-level x-axis label.
        supxlabel_kwargs: Additional keyword arguments forwarded to ``fig.supxlabel``.
        include_experiments: Optional list of experiment display name arrays to include.
            If None, includes all experiments. Experiments are specified as an array of strings,
            indicating how they will be placed in the legend grid.

    Returns:
        Tuple containing the figure, axes grid, legend handles, labels, and legend grid columns.
    """
    assert show_std in {"shade", "errorbar", "none"}
    setup_plot_style()

    if use_median:
        show_std = "none"

    experiment_labels = data_bundle["experiment_labels"]
    fractions = data_bundle["fractions"]
    run_colors = data_bundle["run_colors"]
    run_markers = data_bundle["run_markers"]
    run_linestyles = data_bundle["run_linestyles"]
    majority_counts = data_bundle["majority_counts"]
    run_has_deviations = data_bundle["run_has_deviations"]
    cols_with_data = data_bundle["cols_with_data"]
    df = data_bundle["df"]
    display_name_aliases = data_bundle.get("display_name_aliases", {})

    n_plots = len(cols_with_data)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    metric_entries: List[Dict[str, Any]] = []

    for metric_col in cols_with_data:
        metric_name = metric_col.replace("eval/rewardbench_v2/section/", "").replace(
            "_score", ""
        )
        metric_name = " ".join(word.capitalize() for word in metric_name.split("_"))

        df_metric = df[df[metric_col].notna()]

        if use_median:
            agg_stats = (
                df_metric.groupby(["egid", "experiment_name", "fraction"])[metric_col]
                .agg(mean="median", count="count")
                .reset_index()
            )
            agg_stats["std"] = 0.0  # Add dummy std column
        elif use_ci95:
            agg_stats = (
                df_metric.groupby(["egid", "experiment_name", "fraction"])[metric_col]
                .agg(mean="mean", std=calculate_ci95_half_width, count="count")
                .reset_index()
            )
        else:
            agg_stats = (
                df_metric.groupby(["egid", "experiment_name", "fraction"])[metric_col]
                .agg(mean="mean", std="std", count="count")
                .reset_index()
            )

        if not agg_stats.empty:
            _, metric_upper_bound = calculate_data_bounds(
                agg_stats, df_metric, show_std, 0.0, 1.0, metric_col, show_points
            )
        else:
            metric_upper_bound = 1.0

        metric_entries.append(
            {
                "metric_col": metric_col,
                "metric_name": metric_name,
                "agg_stats": agg_stats,
                "df_metric": df_metric,
                "upper_bound": metric_upper_bound,
            }
        )

    # Sort by headroom to optimize space usage (plots with similar headroom are in the same row, allows trimming)
    metric_entries.sort(key=lambda entry: entry["upper_bound"], reverse=True)

    for idx, entry in enumerate(metric_entries):
        entry["row_idx"] = idx // n_cols
        entry["col_idx"] = idx % n_cols

    row_upper_bounds: List[float] = [0.0 for _ in range(n_rows)]
    for entry in metric_entries:
        row_idx = entry["row_idx"]
        row_upper_bounds[row_idx] = max(row_upper_bounds[row_idx], entry["upper_bound"])

    min_height_ratio = 0.05
    height_ratios = [max(bound, min_height_ratio) for bound in row_upper_bounds]
    total_height_units = sum(height_ratios) if height_ratios else 1.0
    fig_height = height_per_row * total_height_units

    fig = plt.figure(figsize=(plot_width, fig_height))
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios)
    axes = gs.subplots()

    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif n_rows == 1 and axes.ndim == 1:
        axes = axes.reshape(1, -1)
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    # Random generator for consistent jitter
    rng = random.Random(12345)

    for entry in metric_entries:
        row_idx = entry["row_idx"]
        col_idx = entry["col_idx"]
        ax = axes[row_idx, col_idx]

        row_upper_bound = max(row_upper_bounds[row_idx], min_height_ratio)
        row_upper_bound = min(1.0, row_upper_bound * 1.02)
        add_xlabel = row_idx == n_rows - 1

        plot_metric_on_axis(
            ax,
            entry["agg_stats"],
            entry["df_metric"],
            experiment_labels,
            fractions,
            entry["metric_col"],
            entry["metric_name"],
            "Score",
            show_points,
            show_std,
            run_colors,
            run_markers,
            run_linestyles,
            majority_counts,
            run_has_deviations,
            add_xlabel=add_xlabel,
            rng=rng,
            marker_size=marker_size,
            line_width=line_width,
            dash_scale=dash_scale,
            title_position="bottom",
            title_fontsize=title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_fontsize=xtick_fontsize,
            ytick_fontsize=ytick_fontsize,
            display_name_aliases=display_name_aliases,
            y_bounds=None,
        )

        ax.set_ylim(0.0, row_upper_bound)
        tick_percent_values = [20, 40, 60, 80, 100]
        yticks = []
        labels = []
        for val in tick_percent_values:
            frac_val = val / 100.0
            if frac_val <= row_upper_bound + 1e-8:
                yticks.append(frac_val)
                labels.append(f"{val}")
        ax.set_yticks(yticks)
        ax.set_yticklabels(labels, fontsize=ytick_fontsize)

        if col_idx > 0:
            ax.tick_params(labelleft=False)

    axes_flat = list(np.ravel(axes))
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    if supylabel_kwargs is None:
        supylabel_kwargs = {}
    else:
        supylabel_kwargs = dict(supylabel_kwargs)

    if supxlabel_kwargs is None:
        supxlabel_kwargs = {}
    else:
        supxlabel_kwargs = dict(supxlabel_kwargs)

    for ax in axes_flat:
        if ax.get_visible():
            ax.set_ylabel("")
            ax.set_xlabel("")

    fig.supylabel(shared_ylabel, fontsize=ylabel_fontsize, **supylabel_kwargs)
    fig.supxlabel(shared_xlabel, fontsize=xlabel_fontsize, **supxlabel_kwargs)

    if axes_flat and n_plots > 0:
        handles, labels = axes_flat[0].get_legend_handles_labels()
    else:
        handles, labels = [], []

    # Insert invisible handles for None entries to maintain legend layout
    label_to_handle = dict(zip(labels, handles))

    # Rebuild in the order of experiment_labels (which includes None)
    final_handles = []
    final_labels = []
    for exp_label in experiment_labels:
        if exp_label is None:
            invisible_patch = mpatches.Patch(color="none", label="")
            final_handles.append(invisible_patch)
            final_labels.append("")
        else:
            display_label = display_name_aliases.get(exp_label, exp_label)
            if display_label in label_to_handle:
                final_handles.append(label_to_handle[display_label])
                final_labels.append(display_label)

    handles = final_handles
    labels = final_labels

    # Calculate legend grid layout
    if include_experiments is None:
        legend_grid_cols = len(handles)
    else:
        legend_grid_cols = (
            max(len(row) for row in include_experiments)
            if include_experiments
            else len(handles)
        )

    return fig, axes, handles, labels, legend_grid_cols


def create_rewardbench_plot(
    data_bundle: Dict[str, Any],
    plot_width: float,
    show_points: bool,
    show_std: str,
    *,
    use_ci95: bool,
    use_median: bool,
) -> plt.Figure:
    """Create RewardBench category plot figure with in-plot legend."""
    fig, axes, handles, labels, legend_grid_cols = build_rewardbench_plot(
        data_bundle,
        plot_width,
        show_points,
        show_std,
        use_ci95=use_ci95,
        use_median=use_median,
        height_per_row=6.0,
        marker_size=10.0,
        line_width=3.0,
        dash_scale=1.0,
        title_fontsize=14.0,
        supylabel_kwargs={"x": 0.04},
        supxlabel_kwargs={"y": 0.04},
        include_experiments=None,
    )

    # Add shared legend to the first subplot
    if handles:
        axes[0, 0].legend(
            handles,
            labels,
            title="Run Type",
            loc="lower right",
            fontsize=10,
            title_fontsize=12,
            frameon=True,
            framealpha=0.9,
        )

    fig.tight_layout()
    return fig


def generate_rewardbench_categories(
    df: pd.DataFrame,
    metadata: dict,
    plot_width: float,
    show_points: bool,
    show_std: str,
    *,
    use_ci95: bool,
    use_median: bool,
    output_path: str,
):
    """Generate grid plot for RewardBench v2 category scores from cached data.

    Args:
        df: DataFrame loaded from cache
        metadata: Metadata dict from cache
        plot_width: Width of the plot in inches
        show_points: Whether to show individual data points
        show_std: One of {"shade", "errorbar", "none"} to control std visualization
        use_ci95: If True, use 95% confidence interval of mean instead of std
        use_median: If True, use median instead of mean (disables uncertainty ranges)
        output_path: Path where to save the plot PDF (default: average_performance_across_seeds_rb.pdf)
    """
    data_bundle = prepare_rewardbench_data(df, metadata)

    if data_bundle is None:
        print("No RewardBench v2 data found, skipping category plot")
        return

    if not data_bundle["cols_with_data"]:
        print("No RewardBench v2 data with non-null values found")
        return

    fig = create_rewardbench_plot(
        data_bundle,
        plot_width,
        show_points,
        show_std,
        use_ci95=use_ci95,
        use_median=use_median,
    )

    save_figure_reproducible(fig, output_path)
    plt.close(fig)

    print(f"RewardBench v2 category plot saved to {output_path}")
