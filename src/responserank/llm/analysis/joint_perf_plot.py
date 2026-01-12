import random
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from matplotlib.lines import Line2D
from scipy import stats

from responserank.llm.analysis.plotting_utils import (
    save_figure_reproducible,
    setup_plot_style,
    warn_on_low_seed_counts,
)

# Enable LaTeX rendering
rc("text", usetex=True)
rc("font", family="serif")


def _resolve_linestyle(style: str, dash_scale: float):
    """Scale dashed line patterns while leaving solid lines untouched."""
    if style in {"-", "solid"}:
        return "solid"

    if style in {"--", "dashed"}:
        pattern = (6.0, 6.0)
    elif style in {"-.", "dashdot"}:
        pattern = (6.0, 3.0, 1.0, 3.0)
    elif style in {":", "dotted"}:
        pattern = (1.0, 3.0)
    else:
        return style

    scaled = tuple(max(segment * dash_scale, 0.1) for segment in pattern)
    return (0, scaled)


def calculate_ci95_half_width(values):
    """Calculate half-width of 95% confidence interval for the mean.

    Args:
        values: Array-like of data points

    Returns:
        Half-width of 95% CI for the mean

    Note:
        This assumes the values are normally distributed and uses the t-distribution
        for small sample sizes. When comparing multiple points at different fractions,
        this approach might underestimate statistical power since it doesn't account
        for multiple comparisons.
    """
    if len(values) <= 1:
        return 0.0

    standard_error_of_mean = stats.sem(values)
    ci_half_width = standard_error_of_mean * stats.t.ppf(0.975, len(values) - 1)

    return ci_half_width


def calculate_data_bounds(
    agg_data: pd.DataFrame,
    raw_data: Optional[pd.DataFrame],
    show_std: str,
    min_bound: float,
    max_bound: float,
    metric_col: Optional[str],
    show_points: bool,
) -> Tuple[float, float]:
    """Calculate tight data bounds from aggregated statistics.

    Args:
        agg_data: DataFrame with 'mean' and 'std' columns
        raw_data: Optional raw data for additional bound calculation
        show_std: Whether std is being shown ("shade", "errorbar", "none")
        min_bound: Minimum allowed bound (typically 0.0)
        max_bound: Maximum allowed bound (typically 1.0)
        metric_col: Column name in raw_data to use for bounds
        show_points: Whether individual points are shown (controls raw data expansion)

    Returns:
        Tuple of (y_min, y_max) bounds that tightly fit the data envelope
    """
    if agg_data.empty:
        return min_bound, max_bound

    if show_std in {"shade", "errorbar"}:
        lower_envelope = agg_data["mean"] - agg_data["std"]
        upper_envelope = agg_data["mean"] + agg_data["std"]
        y_min = max(min_bound, lower_envelope.min())
        y_max = upper_envelope.max()
    else:
        y_min = max(min_bound, agg_data["mean"].min())
        y_max = agg_data["mean"].max()

    if (
        show_points
        and raw_data is not None
        and not raw_data.empty
        and metric_col is not None
        and metric_col in raw_data.columns
    ):
        raw_min = raw_data[metric_col].min()
        raw_max = raw_data[metric_col].max()
        if pd.api.types.is_numeric_dtype(raw_data[metric_col]):
            y_min = max(min_bound, min(y_min, raw_min))
            y_max = max(y_max, raw_max)

    y_max = min(max_bound, y_max)

    return float(y_min), float(y_max)


def compute_sample_counts(agg_stats: pd.DataFrame, experiment_labels: List[str]):
    """Compute majority sample counts and deviation indicators.

    Args:
        agg_stats: Aggregated statistics DataFrame
        experiment_labels: List of experiment labels

    Returns:
        Tuple of (majority_counts, run_has_deviations)
    """
    majority_counts = {}
    for run_label in experiment_labels:
        run_counts = agg_stats[agg_stats["egid"] == run_label]["count"]
        majority_counts[run_label] = run_counts.mode().iloc[0]

    def has_sample_count_deviations(run_label: str) -> bool:
        run_data = agg_stats[agg_stats["egid"] == run_label]
        majority_count = majority_counts[run_label]
        return (run_data["count"] != majority_count).any()

    run_has_deviations = {
        run_label: has_sample_count_deviations(run_label)
        for run_label in experiment_labels
    }

    return majority_counts, run_has_deviations


def plot_metric_on_axis(
    ax,
    agg_data: pd.DataFrame,
    raw_data: pd.DataFrame,
    experiment_labels: List[str],
    fractions: List[float],
    metric_col: str,
    title: str,
    ylabel: str,
    show_points: bool,
    show_std: str,
    run_colors: Dict[str, str],
    run_markers: Dict[str, str],
    run_linestyles: Dict[str, str],
    majority_counts: Dict[str, int],
    run_has_deviations: Dict[str, bool],
    add_xlabel: bool,
    rng,
    *,
    marker_size: float,
    line_width: float,
    dash_scale: float,
    title_position: str,
    title_fontsize: float,
    xlabel_fontsize: float = 14.0,
    ylabel_fontsize: float = 14.0,
    xtick_fontsize: float = 12.0,
    ytick_fontsize: float = 12.0,
    display_name_aliases: Optional[Dict[str, str]] = None,
    y_bounds: Optional[tuple],
):
    """Plot metric data on a given axis.

    Args:
        ax: Matplotlib axis to plot on
        agg_data: Aggregated statistics DataFrame
        raw_data: Raw data DataFrame for jittered points
        experiment_labels: List of experiment labels
        fractions: List of data fractions
        metric_col: Column name for the metric to plot
        title: Plot title
        ylabel: Y-axis label
        show_points: Whether to show jittered data points
        show_std: One of {"shade", "errorbar", "none"}
        run_colors: Dict mapping run labels to colors
        run_markers: Dict mapping run labels to markers
        run_linestyles: Dict mapping run labels to linestyles
        majority_counts: Dict mapping run labels to majority sample counts
        run_has_deviations: Dict mapping run labels to deviation flags
        add_xlabel: Whether to add the xlabel (Training Set Size).
        rng: Random number generator for jitter
        marker_size: Marker size for line plots (points)
        line_width: Width of plotted lines in points
        dash_scale: Scale factor applied to dash patterns for dashed lines
        title_position: Where to place the subplot title. One of {"top", "bottom", "none"}.
        title_fontsize: Font size to use for the subplot title text.
        xlabel_fontsize: Font size for the x-axis label.
        ylabel_fontsize: Font size for the y-axis label.
        xtick_fontsize: Font size for x-axis tick labels.
        ytick_fontsize: Font size for y-axis tick labels.
        display_name_aliases: Optional mapping from raw display names to legend labels.
    """
    for run_label in experiment_labels:
        run_df = agg_data[agg_data["egid"] == run_label]

        if not run_df.empty:
            # Sort by fraction to ensure correct line connection order
            run_df = run_df.sort_values("fraction")
            alias_label = (
                display_name_aliases.get(run_label)
                if display_name_aliases is not None
                else None
            )

            if alias_label is None:
                sample_count_str = f"n={majority_counts[run_label]}"
                if run_has_deviations[run_label]:
                    sample_count_str += "*"
                label = f"{run_label} ({sample_count_str})"
            else:
                label = alias_label

            xs = run_df["fraction"].values
            means = run_df["mean"].values
            stds = run_df["std"].values

            if show_std == "shade":
                ax.fill_between(
                    xs,
                    means - stds,
                    means + stds,
                    color=run_colors[run_label],
                    alpha=0.15,
                    linewidth=0,
                    zorder=2,
                )
            elif show_std == "errorbar":
                ax.errorbar(
                    xs,
                    means,
                    yerr=stds,
                    fmt="none",
                    ecolor=run_colors[run_label],
                    alpha=0.5,
                    capsize=5,
                    zorder=2,
                )

            linestyle = _resolve_linestyle(run_linestyles[run_label], dash_scale)

            (line,) = ax.plot(
                xs,
                means,
                marker=run_markers[run_label],
                color=run_colors[run_label],
                linestyle=linestyle,
                linewidth=line_width,
                markersize=marker_size,
                label=label,
                zorder=3,
            )
            if isinstance(linestyle, tuple):
                line.set_linestyle(linestyle)

            # Optionally, add the raw data points with jitter for transparency
            if show_points:
                for frac in fractions:
                    raw_points = raw_data[
                        (raw_data["egid"] == run_label) & (raw_data["fraction"] == frac)
                    ]

                    if not raw_points.empty:
                        # Add some jitter to x-values to prevent overlap
                        jittered_x = [
                            frac + rng.uniform(-0.01, 0.01)
                            for _ in range(len(raw_points))
                        ]

                        ax.scatter(
                            jittered_x,
                            raw_points[metric_col],
                            marker=run_markers[run_label],
                            color=run_colors[run_label],
                            alpha=0.3,
                            s=marker_size**2,
                            edgecolors="white",
                            linewidths=0.5,
                            zorder=1,
                        )

    if title_position == "top":
        ax.set_title(f"\\bf{{}}{title}", fontsize=title_fontsize)
    elif title_position == "bottom":
        ax.set_title("")
        ax.text(
            0.5,
            0.02,
            f"\\bf{{}}{title}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=title_fontsize,
        )
    elif title_position == "none":
        ax.set_title("")
    else:
        raise ValueError(
            f"Unsupported title_position '{title_position}'. Expected 'top', 'bottom', or 'none'."
        )

    if add_xlabel:
        ax.set_xlabel(r"\bf{}Training Set Size (\%)", fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    percentage_ticks = [f * 100 for f in fractions]
    ax.set_xticks(fractions)
    ax.set_xticklabels(
        [f"{int(p)}%" if p == int(p) else f"{p:.1f}%" for p in percentage_ticks],
        fontsize=xtick_fontsize,
    )

    spine_color = ax.spines["bottom"].get_edgecolor()

    ax.tick_params(
        axis="x",
        which="major",
        direction="inout",
        length=4,
        width=1.0,
        bottom=True,
        top=False,
        # Only show xtick labels in bottommost row
        labelbottom=add_xlabel,
        labelsize=xtick_fontsize,
        colors=spine_color,
        labelcolor="black",
    )

    y_spine_color = ax.spines["left"].get_edgecolor()
    ax.tick_params(
        axis="y",
        which="major",
        direction="inout",
        length=4,
        width=1.0,
        left=True,
        right=False,
        colors=y_spine_color,
        labelcolor="black",
        labelsize=ytick_fontsize,
    )

    # Only horizontal grid
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    if y_bounds is not None:
        y_lower, y_upper = y_bounds
        ax.set_ylim(y_lower, y_upper)
    elif not agg_data.empty:
        y_min, y_max = calculate_data_bounds(
            agg_data, raw_data, show_std, 0.0, 1.0, metric_col, show_points
        )
        ax.set_ylim(y_min, y_max)
        y_lower, y_upper = y_min, y_max

    if not agg_data.empty or y_bounds is not None:
        tick_percent_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        yticks = []
        ylabels = []
        for val in tick_percent_values:
            frac_val = val / 100.0
            if y_lower <= frac_val <= y_upper - 1e-8:
                yticks.append(frac_val)
                ylabels.append(f"{val}")

        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=ytick_fontsize)


def build_reproduction_command(
    df: pd.DataFrame,
    project_name: str,
    seed_filter: Optional[int],
    plot_width: float,
    show_points: bool,
    show_std: str,
    style_overrides: Optional[Dict[str, Dict[str, str]]],
    baseline_egid_regex: Optional[str],
    primary_egid_regex: Optional[str],
) -> str:
    """Build the command to reproduce these plots.

    Args:
        df: DataFrame with egid and epoch columns
        project_name: wandb project name
        seed_filter: Optional seed filter
        plot_width: Plot width
        show_points: Whether points are shown
        show_std: Standard deviation display mode
        style_overrides: Style specifications per pattern (not egid)
        baseline_egid_regex: Original baseline pattern
        primary_egid_regex: Original primary pattern

    Returns:
        Complete reproduction command string
    """

    # Get unique (egid, epoch) combinations
    unique_specs = (
        df[["egid", "epoch"]].drop_duplicates().sort_values(["egid", "epoch"])
    )

    def find_styling_for_egid(egid: str) -> Optional[Dict[str, str]]:
        if style_overrides is not None:
            for pattern, styles in style_overrides.items():
                if re.match(pattern, egid):
                    return styles
        return {}

    def format_egid_spec(egid: str, epoch: int) -> str:
        if epoch == -1:
            base_spec = egid
        else:
            base_spec = f"{egid}:{epoch}"

        styles = find_styling_for_egid(egid)
        style_parts = []
        if "color" in styles:
            style_parts.append(styles["color"])
        if "marker" in styles:
            style_parts.append(styles["marker"])
        if "linestyle" in styles:
            style_parts.append(styles["linestyle"])

        if style_parts:
            base_spec += f"[{'|'.join(style_parts)}]"

        return base_spec

    # Identify baseline and primary egids from the actual data
    baseline_egid = None
    primary_egid = None
    comparison_egids = []

    for _, row in unique_specs.iterrows():
        egid, epoch = row["egid"], row["epoch"]
        formatted_spec = format_egid_spec(egid, epoch)

        if baseline_egid_regex and re.match(baseline_egid_regex, egid):
            baseline_egid = formatted_spec
        elif primary_egid_regex and re.match(primary_egid_regex, egid):
            primary_egid = formatted_spec
        else:
            comparison_egids.append(formatted_spec)

    cmd_parts = ["python -m llm_rt.analysis.main joint-perf"]
    cmd_parts.append(f"--project-name {project_name}")

    if seed_filter is not None:
        cmd_parts.append(f"--seed-filter {seed_filter}")

    if plot_width != 24:  # Only include if non-default
        cmd_parts.append(f"--plot-width {plot_width}")

    if baseline_egid:
        cmd_parts.append(f"--baseline-egid {baseline_egid}")

    if primary_egid:
        cmd_parts.append(f"--primary-egid {primary_egid}")

    if comparison_egids:
        cmd_parts.append(f"--comparison-egids {','.join(comparison_egids)}")

    if not show_points:
        cmd_parts.append("--no-points")

    if show_std == "errorbar":
        cmd_parts.append("--errorbar")

    return " ".join(cmd_parts)


def generate_joint_perf_plot(
    df: pd.DataFrame,
    metadata: Dict,
    plot_width: float,
    plot_height: float,
    show_points: bool,
    show_std: str,
    *,
    use_ci95: bool,
    use_median: bool,
    include_metrics: Sequence[str],
    output_path: str,
    title_fontsize: float,
    marker_size: float,
    line_width: float,
    dash_scale: float,
    supxlabel_fontsize: float,
    supylabel_fontsize: float,
    tick_fontsize: float,
    legend_mode: Literal["axis", "figure_top"],
    legend_kwargs: Optional[Dict[str, Any]],
    include_experiments: Optional[List[List[str]]],
    baseline_experiment: Optional[str],
    primary_experiment: Optional[str],
    min_seed_warning: Optional[int] = None,
    custom_styles: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Generate joint performance plot from cached data.

    Args:
        df: DataFrame loaded from cache
        metadata: Metadata dict from cache
        plot_width: Width of the plot in inches
        plot_height: Height of the plot in inches
        show_points: Whether to show jittered data points
        show_std: One of {"shade", "errorbar", "none"} to control std visualization.
        use_ci95: If True, use 95% confidence interval of mean instead of std
        use_median: If True, use median instead of mean (disables uncertainty ranges)
        include_metrics: Ordered collection selecting which panels to render.
            Supported values: "multipref", "rb_v1", "rb_v2".
        output_path: Path where to save the plot PDF (default: average_performance_across_seeds.pdf)
        title_fontsize: Font size for subplot titles
        marker_size: Size of markers in the plot
        line_width: Width of lines in the plot
        dash_scale: Scale factor for dashed line patterns
        supxlabel_fontsize: Font size for super x-axis label
        supylabel_fontsize: Font size for super y-axis label
        legend_mode: Placement strategy for the legend. Use "axis" to attach the legend
            to the rightmost axis or "figure_top" to span the full figure width.
        legend_kwargs: Optional overrides forwarded to the legend placement call.
        include_experiments: Optional list of experiment display name arrays to include.
            If None, includes all experiments. Experiments are specified as an array of strings,
            indicating how they will be placed in the legend grid.
        baseline_experiment: Display name of baseline experiment (e.g. "BT (baseline)").
            If provided, this experiment will be used as the baseline for comparisons.
        primary_experiment: Display name of primary experiment (e.g. "ResponseRank-Agree (ours)").
            If provided, this experiment will get delta arrows showing improvement over baseline.
        min_seed_warning: Emit warnings when an experiment/fraction combination uses fewer
            than this number of unique seeds. ``None`` disables the warning.
        custom_styles: Optional dict mapping canonical display names to style dicts with "label" key.
            Applied after alias resolution to override legend labels.
    """
    assert show_std in {"shade", "errorbar", "none"}
    rng = random.Random(12345)  # For consistent jitter

    if include_experiments is None:
        all_experiments = sorted(df["egid"].unique())
        include_experiments = [all_experiments]

    # Filter experiments and construct the legend
    computed_display_names = metadata["computed_display_names"]
    alias_to_raw = {alias: raw for raw, alias in computed_display_names.items()}
    filter_names = []
    display_labels_map = {}

    for legend_row in include_experiments:
        for exp_display_name in legend_row:
            raw_name = alias_to_raw.get(exp_display_name, exp_display_name)
            filter_names.append(raw_name)
            display_labels_map[raw_name] = exp_display_name

    # The legend is ordered in a grid as specified in include_experiments, with
    # padding added as needed if not all rows have the same length.
    max_cols = max(len(row) for row in include_experiments)
    legend_grid_cols = max_cols

    df = df[df["egid"].isin(filter_names)]

    if use_median:
        print("The median option is not compatible with std/ci95")
        show_std = "none"

    # Set the plot style
    setup_plot_style()

    matched_experiments = sorted(df["egid"].unique())

    def format_exps(egid_list, prefix):
        return "\n".join([prefix + f"- {egid}" for egid in egid_list])

    print("Using cached data:")
    print(f"  Experiments:\n{format_exps(matched_experiments, '    ')}")
    print(f"  Total runs: {len(df)}")

    if df.empty:
        print("No valid data found in cache.")
    else:
        experiment_labels = [name for name in filter_names if name in df["egid"].values]
        fractions = sorted(df["fraction"].unique())
        display_name_aliases = metadata.get("computed_display_names", {})

        if min_seed_warning is not None:
            warn_on_low_seed_counts(
                df,
                experiments=experiment_labels,
                groupby_cols=["fraction"],
                threshold=min_seed_warning,
                display_name_aliases=display_name_aliases,
            )

        if not df.empty:
            if use_median:
                # Use median instead of mean, no std/CI calculation
                agg_stats = (
                    df.groupby(["egid", "experiment_name", "fraction"])[
                        ["eval_accuracy", "train_examples", "test_examples"]
                    ]
                    .agg(
                        {
                            "eval_accuracy": ["median", "count"],
                            "train_examples": "median",
                            "test_examples": "median",
                        }
                    )
                    .reset_index()
                )
                agg_stats.columns = [
                    "egid",
                    "experiment_name",
                    "fraction",
                    "mean",  # Actually median when use_median=True
                    "count",
                    "train_examples_mean",
                    "test_examples_mean",
                ]
                # Add dummy std column filled with zeros (required by plotting code)
                agg_stats["std"] = 0.0
                # Reorder columns to match expected format
                agg_stats = agg_stats[
                    [
                        "display_name",
                        "experiment_name",
                        "fraction",
                        "mean",
                        "std",
                        "count",
                        "train_examples_mean",
                        "test_examples_mean",
                    ]
                ]
            elif use_ci95:
                agg_stats = (
                    df.groupby(["egid", "experiment_name", "fraction"])[
                        ["eval_accuracy", "train_examples", "test_examples"]
                    ]
                    .agg(
                        {
                            "eval_accuracy": [
                                "mean",
                                calculate_ci95_half_width,
                                "count",
                            ],
                            "train_examples": "mean",
                            "test_examples": "mean",
                        }
                    )
                    .reset_index()
                )
                agg_stats.columns = [
                    "egid",
                    "experiment_name",
                    "fraction",
                    "mean",
                    "std",  # Actually CI95 half-width when use_ci95=True
                    "count",
                    "train_examples_mean",
                    "test_examples_mean",
                ]
            else:
                agg_stats = (
                    df.groupby(["egid", "experiment_name", "fraction"])[
                        ["eval_accuracy", "train_examples", "test_examples"]
                    ]
                    .agg(
                        {
                            "eval_accuracy": ["mean", "std", "count"],
                            "train_examples": "mean",
                            "test_examples": "mean",
                        }
                    )
                    .reset_index()
                )
                agg_stats.columns = [
                    "egid",
                    "experiment_name",
                    "fraction",
                    "mean",
                    "std",
                    "count",
                    "train_examples_mean",
                    "test_examples_mean",
                ]

            df_rb = df[df["eval/rewardbench_v1/overall/accuracy_score"].notna()]
            if use_median:
                agg_stats_rb = (
                    df_rb.groupby(["egid", "experiment_name", "fraction"])[
                        "eval/rewardbench_v1/overall/accuracy_score"
                    ]
                    .agg(mean="median", count="count")
                    .reset_index()
                )
                agg_stats_rb["std"] = 0.0  # Add dummy std column
            elif use_ci95:
                agg_stats_rb = (
                    df_rb.groupby(["egid", "experiment_name", "fraction"])[
                        "eval/rewardbench_v1/overall/accuracy_score"
                    ]
                    .agg(mean="mean", std=calculate_ci95_half_width, count="count")
                    .reset_index()
                )
            else:
                agg_stats_rb = (
                    df_rb.groupby(["egid", "experiment_name", "fraction"])[
                        "eval/rewardbench_v1/overall/accuracy_score"
                    ]
                    .agg(mean="mean", std="std", count="count")
                    .reset_index()
                )

            # Rewardbench v2 (only if available)
            df_rb_v2 = df[df["eval/rewardbench_v2/overall/accuracy_score"].notna()]
            agg_stats_rb_v2 = None
            if len(df_rb_v2) > 0:
                if use_median:
                    agg_stats_rb_v2 = (
                        df_rb_v2.groupby(["egid", "experiment_name", "fraction"])[
                            "eval/rewardbench_v2/overall/accuracy_score"
                        ]
                        .agg(mean="median", count="count")
                        .reset_index()
                    )
                    agg_stats_rb_v2["std"] = 0.0  # Add dummy std column
                elif use_ci95:
                    agg_stats_rb_v2 = (
                        df_rb_v2.groupby(["egid", "experiment_name", "fraction"])[
                            "eval/rewardbench_v2/overall/accuracy_score"
                        ]
                        .agg(mean="mean", std=calculate_ci95_half_width, count="count")
                        .reset_index()
                    )
                else:
                    agg_stats_rb_v2 = (
                        df_rb_v2.groupby(["egid", "experiment_name", "fraction"])[
                            "eval/rewardbench_v2/overall/accuracy_score"
                        ]
                        .agg(mean="mean", std="std", count="count")
                        .reset_index()
                    )

            majority_counts, run_has_deviations = compute_sample_counts(
                agg_stats, experiment_labels
            )

            computed_styling = metadata["computed_styling"]
            run_colors = computed_styling["run_colors"]
            run_markers = computed_styling["run_markers"]
            run_linestyles = computed_styling["run_linestyles"]

            ordered_metrics: List[str] = []
            panels_info: List[Dict[str, Dict[str, str]]] = []
            for metric_key in include_metrics:
                if metric_key == "multipref":
                    panels_info.append(
                        {
                            "key": "multipref",
                            "agg": agg_stats,
                            "raw": df,
                            "title": "MultiPref Test Accuracy",
                            "metric_col": "eval_accuracy",
                        }
                    )
                    ordered_metrics.append(metric_key)
                elif metric_key == "rb_v1":
                    panels_info.append(
                        {
                            "key": "rb_v1",
                            "agg": agg_stats_rb,
                            "raw": df_rb,
                            "title": "RewardBench 1 Score",
                            "metric_col": "eval/rewardbench_v1/overall/accuracy_score",
                        }
                    )
                    ordered_metrics.append(metric_key)
                elif metric_key == "rb_v2":
                    panels_info.append(
                        {
                            "key": "rb_v2",
                            "agg": agg_stats_rb_v2,
                            "raw": df_rb_v2,
                            "title": "RewardBench 2 Score",
                            "metric_col": "eval/rewardbench_v2/overall/accuracy_score",
                        }
                    )
                    ordered_metrics.append(metric_key)
                else:
                    raise ValueError(f"Unsupported metric '{metric_key}'.")

            # Calculate natural y-bounds for each panel
            panel_y_ranges = []
            for panel in panels_info:
                agg_data = panel["agg"]
                if agg_data is not None and not agg_data.empty:
                    y_min, y_max = calculate_data_bounds(
                        agg_data,
                        panel["raw"],
                        "shade",
                        0.0,
                        float("inf"),
                        panel["metric_col"],
                        show_points,
                    )
                    delta = y_max - y_min
                    panel_y_ranges.append(
                        {"y_min": y_min, "y_max": y_max, "delta": delta}
                    )
                else:
                    panel_y_ranges.append({"y_min": 0.0, "y_max": 1.0, "delta": 1.0})

            # Apply unified scaling: ensure one centimeter on y-axis means same score difference
            unified_y_bounds = []
            if panel_y_ranges:
                max_delta = max(pr["delta"] for pr in panel_y_ranges if pr["delta"] > 0)
                for pr in panel_y_ranges:
                    if pr["delta"] < max_delta and pr["delta"] > 0:
                        # Keep upper bound, adjust lower bound to match largest delta
                        new_y_min = pr["y_max"] - max_delta
                        unified_y_bounds.append((new_y_min, pr["y_max"]))
                    else:
                        # Panel with largest range stays tight
                        unified_y_bounds.append((pr["y_min"], pr["y_max"]))

            n_cols = len(panels_info)
            fig, axes_row = plt.subplots(
                1,
                n_cols,
                figsize=(plot_width, plot_height),
                constrained_layout=True,
            )
            # Set outer padding to 0, control xlabel and legend spacing individually
            fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)
            if n_cols == 1:
                axes_row = [axes_row]
            else:
                axes_row = list(axes_row)

            axes_map = {}
            for i, (axis, panel) in enumerate(zip(axes_row, panels_info)):
                bounds = (
                    unified_y_bounds[i]
                    if unified_y_bounds and i < len(unified_y_bounds)
                    else None
                )
                axes_map[panel["key"]] = axis
                plot_metric_on_axis(
                    axis,
                    panel["agg"],
                    panel["raw"],
                    experiment_labels,
                    fractions,
                    panel["metric_col"],
                    panel["title"],
                    "",
                    show_points,
                    show_std,
                    run_colors,
                    run_markers,
                    run_linestyles,
                    majority_counts,
                    run_has_deviations,
                    add_xlabel=True,
                    rng=rng,
                    marker_size=marker_size,
                    line_width=line_width,
                    dash_scale=dash_scale,
                    title_position="top",
                    title_fontsize=title_fontsize,
                    display_name_aliases=display_name_aliases,
                    y_bounds=bounds,
                    xtick_fontsize=tick_fontsize,
                    ytick_fontsize=tick_fontsize,
                )

            axes_in_row = axes_row

            ax1 = axes_map.get("multipref")
            ax2 = axes_map.get("rb_v1")
            ax3 = axes_map.get("rb_v2")

            if baseline_experiment is not None:
                # Convert aliases to raw display names for baseline/primary lookup
                computed_display_names = metadata["computed_display_names"]
                alias_to_raw = {
                    alias: raw for raw, alias in computed_display_names.items()
                }

                baseline_raw = alias_to_raw.get(
                    baseline_experiment, baseline_experiment
                )
                primary_raw = (
                    alias_to_raw.get(primary_experiment, primary_experiment)
                    if primary_experiment
                    else None
                )

                if ax1 is not None or ax2 is not None or ax3 is not None:
                    add_baseline_parity_lines(
                        ax1,
                        ax2,
                        ax3,
                        baseline_raw,
                        experiment_labels,
                        agg_stats,
                        agg_stats_rb,
                        agg_stats_rb_v2,
                        run_colors,
                        fractions,
                        primary_raw,
                    )

                    if primary_raw is not None:
                        add_delta_arrows(
                            ax1,
                            ax2,
                            ax3,
                            baseline_raw,
                            primary_raw,
                            experiment_labels,
                            agg_stats,
                            agg_stats_rb,
                            agg_stats_rb_v2,
                            run_colors,
                            fractions,
                        )

            # Place legend on the rightmost available axis
            legend_axis_key = None
            for candidate in ("rb_v2", "rb_v1", "multipref"):
                if candidate in axes_map:
                    legend_axis_key = candidate
                    break
            if legend_axis_key is None:
                legend_axis = axes_row[-1]
            else:
                legend_axis = axes_map[legend_axis_key]
            legend_handles, legend_labels = legend_axis.get_legend_handles_labels()

            legend_labels = [
                display_labels_map.get(label, label) for label in legend_labels
            ]

            if custom_styles is not None:
                legend_labels = [
                    (
                        custom_styles[label]["label"]
                        if label in custom_styles and "label" in custom_styles[label]
                        else label
                    )
                    for label in legend_labels
                ]

            num_rows = len(include_experiments)

            grid_entries = []
            entry_idx = 0

            for row_idx, exp_row in enumerate(include_experiments):
                for col_idx, _ in enumerate(exp_row):
                    if entry_idx < len(legend_handles):
                        grid_entries.append((row_idx, col_idx, entry_idx))
                        entry_idx += 1

            total_entries = len(legend_handles)
            entries_needed = num_rows * legend_grid_cols
            if total_entries < entries_needed:
                empty_entries_needed = entries_needed - total_entries
                for i in range(empty_entries_needed):
                    dummy_handle = Line2D([0], [0], alpha=0)
                    legend_handles.append(dummy_handle)
                    legend_labels.append("")
                    remaining_row = (total_entries + i) // legend_grid_cols
                    remaining_col = (total_entries + i) % legend_grid_cols
                    if remaining_row < num_rows:
                        grid_entries.append(
                            (remaining_row, remaining_col, total_entries + i)
                        )

            reordered_handles = [None] * len(legend_handles)
            reordered_labels = [None] * len(legend_labels)

            for row_idx, col_idx, orig_idx in grid_entries:
                mpl_idx = col_idx * num_rows + row_idx
                if mpl_idx < len(reordered_handles):
                    reordered_handles[mpl_idx] = legend_handles[orig_idx]
                    reordered_labels[mpl_idx] = legend_labels[orig_idx]

            legend_handles[:] = [h for h in reordered_handles if h is not None]
            legend_labels[:] = [
                label for label in reordered_labels if label is not None
            ]

            if legend_mode == "axis":
                axis_legend_kwargs: Dict[str, Any] = {
                    "loc": "lower right",
                    "fontsize": 12,
                    "frameon": True,
                    "framealpha": 0.9,
                }
                if legend_kwargs is not None:
                    axis_legend_kwargs.update(legend_kwargs)
                legend_axis.legend(**axis_legend_kwargs)
            elif legend_mode == "figure_top":
                if not legend_handles:
                    raise ValueError(
                        "No legend entries available for figure-level legend."
                    )

                visible_axes = [ax for ax in axes_in_row if ax.get_visible()]
                if visible_axes:
                    left = min(ax.get_position().x0 for ax in visible_axes)
                    right = max(ax.get_position().x1 for ax in visible_axes)
                else:
                    left, right = 0.0, 1.0
                legend_width = right - left

                figure_legend_kwargs: Dict[str, Any] = {
                    "loc": "lower left",
                    "bbox_to_anchor": (left, 0.92, legend_width, 0.1),
                    "bbox_transform": fig.transFigure,
                    "mode": "expand",
                    "ncol": legend_grid_cols,
                    "fontsize": 12,
                    "frameon": False,
                    "borderaxespad": 0.0,
                }
                if legend_kwargs is not None:
                    figure_legend_kwargs.update(legend_kwargs)

                fig.legend(legend_handles, legend_labels, **figure_legend_kwargs)
            else:
                raise ValueError(
                    f"Unsupported legend_mode '{legend_mode}'. Expected 'axis' or 'figure_top'."
                )

            for axis in axes_in_row:
                axis.set_xlabel("")

            fig.supxlabel(
                r"\bf{}Training Set Size (\%)", fontsize=supxlabel_fontsize, y=0.03
            )

            # Add text annotations for sample counts (only for deviations from majority)
            for run_label in experiment_labels:
                for i, frac in enumerate(fractions):
                    # Annotations for multipref plot
                    stats = agg_stats[
                        (agg_stats["egid"] == run_label)
                        & (agg_stats["fraction"] == frac)
                    ]

                    if ax1 is not None and not stats.empty:
                        count = stats["count"].values[0]
                        if count != majority_counts[run_label]:
                            ax1.annotate(
                                f"n={count}",
                                xy=(frac, stats["mean"].values[0]),
                                xytext=(0, 10),  # 10 points vertical offset
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                alpha=0.7,
                            )

                    # Annotations for RewardBench v1 plot
                    stats_rb = agg_stats_rb[
                        (agg_stats_rb["egid"] == run_label)
                        & (agg_stats_rb["fraction"] == frac)
                    ]

                    if ax2 is not None and not stats_rb.empty:
                        count_rb = stats_rb["count"].values[0]
                        if count_rb != majority_counts[run_label]:
                            ax2.annotate(
                                f"n={count_rb}",
                                xy=(frac, stats_rb["mean"].values[0]),
                                xytext=(0, 10),  # 10 points vertical offset
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                alpha=0.7,
                            )

                    # Annotations for RewardBench v2 plot (if available)
                    if agg_stats_rb_v2 is not None:
                        stats_rb_v2 = agg_stats_rb_v2[
                            (agg_stats_rb_v2["egid"] == run_label)
                            & (agg_stats_rb_v2["fraction"] == frac)
                        ]

                        if ax3 is not None and not stats_rb_v2.empty:
                            count_rb_v2 = stats_rb_v2["count"].values[0]
                            if count_rb_v2 != majority_counts[run_label]:
                                ax3.annotate(
                                    f"n={count_rb_v2}",
                                    xy=(frac, stats_rb_v2["mean"].values[0]),
                                    xytext=(0, 10),
                                    textcoords="offset points",
                                    ha="center",
                                    va="bottom",
                                    fontsize=8,
                                    alpha=0.7,
                                )

            # Manually reduce gap between subplots (constrained_layout doesn't go tight enough)
            if n_cols == 2:
                pos0 = axes_row[0].get_position()
                pos1 = axes_row[1].get_position()

                gap = pos1.x0 - pos0.x1
                reduction = gap * 0.1

                # Shift right plot left and expand both plots to fill recovered space
                new_pos1 = [
                    pos1.x0 - reduction,
                    pos1.y0,
                    pos1.width + reduction / 2,
                    pos1.height,
                ]
                new_pos0 = [pos0.x0, pos0.y0, pos0.width + reduction / 2, pos0.height]

                axes_row[0].set_position(new_pos0)
                axes_row[1].set_position(new_pos1)

            # Save the figure
            save_figure_reproducible(plt.gcf(), output_path)

            # Display the plot
            plt.show()

            print(f"Plot has been saved to {output_path}")


def add_baseline_parity_lines(
    ax_multipref,
    ax_rb_v1,
    ax_rb_v2,
    baseline_experiment: str,
    experiment_labels: List[str],
    agg_stats: pd.DataFrame,
    agg_stats_rb: pd.DataFrame,
    agg_stats_rb_v2: Optional[pd.DataFrame],
    run_colors: Dict[str, str],
    fractions: List[float],
    primary_experiment: Optional[str],
):
    """Add horizontal baseline parity lines to all panels.

    Args:
        ax_multipref: Multipref axis (or None if not included)
        ax_rb_v1: RewardBench v1 axis (or None if not included)
        ax_rb_v2: RewardBench v2 axis (or None if not included)
        baseline_experiment: Display name of baseline experiment
        experiment_labels: List of all experiment labels
        agg_stats: Aggregated multipref statistics
        agg_stats_rb: Aggregated RewardBench v1 statistics
        agg_stats_rb_v2: Aggregated RewardBench v2 statistics (optional)
        run_colors: Dict mapping experiment labels to colors
        fractions: List of data fractions
        primary_experiment: Primary experiment for crossing point annotation (or None)
    """
    if (
        baseline_experiment not in experiment_labels
        or baseline_experiment not in run_colors
    ):
        print(
            f"Warning: Baseline experiment '{baseline_experiment}' not found or not styled"
        )
        return

    baseline_color = run_colors[baseline_experiment]
    max_fraction = max(fractions)

    # Get baseline performance at max fraction for each metric
    multipref_baseline = agg_stats[
        (agg_stats["egid"] == baseline_experiment)
        & (agg_stats["fraction"] == max_fraction)
    ]

    rb_baseline = agg_stats_rb[
        (agg_stats_rb["egid"] == baseline_experiment)
        & (agg_stats_rb["fraction"] == max_fraction)
    ]

    # Add parity line to multipref panel if present
    if ax_multipref is not None and not multipref_baseline.empty:
        baseline_value = multipref_baseline["mean"].values[0]
        ax_multipref.axhline(
            y=baseline_value,
            color=baseline_color,
            linestyle="--",
            alpha=0.6,
            zorder=0,
            label=f"{baseline_experiment} @ max fraction",
        )

        if primary_experiment is not None and primary_experiment in experiment_labels:
            _add_crossing_annotation(
                ax_multipref,
                agg_stats,
                baseline_experiment,
                primary_experiment,
                baseline_value,
                fractions,
            )

    # Add parity line to RewardBench v1 panel if present
    if ax_rb_v1 is not None and not rb_baseline.empty:
        baseline_value = rb_baseline["mean"].values[0]
        ax_rb_v1.axhline(
            y=baseline_value, color=baseline_color, linestyle="--", alpha=0.6, zorder=0
        )

        if primary_experiment is not None and primary_experiment in experiment_labels:
            _add_crossing_annotation(
                ax_rb_v1,
                agg_stats_rb,
                baseline_experiment,
                primary_experiment,
                baseline_value,
                fractions,
            )

    # Add parity line to RewardBench v2 panel if present
    if ax_rb_v2 is not None and agg_stats_rb_v2 is not None:
        rb_v2_baseline = agg_stats_rb_v2[
            (agg_stats_rb_v2["egid"] == baseline_experiment)
            & (agg_stats_rb_v2["fraction"] == max_fraction)
        ]

        if not rb_v2_baseline.empty:
            baseline_value = rb_v2_baseline["mean"].values[0]
            ax_rb_v2.axhline(
                y=baseline_value,
                color=baseline_color,
                linestyle="--",
                alpha=0.6,
                zorder=0,
            )

            if (
                primary_experiment is not None
                and primary_experiment in experiment_labels
            ):
                _add_crossing_annotation(
                    ax_rb_v2,
                    agg_stats_rb_v2,
                    baseline_experiment,
                    primary_experiment,
                    baseline_value,
                    fractions,
                )


def _add_crossing_annotation(
    ax,
    agg_data: pd.DataFrame,
    baseline_exp: str,
    primary_exp: str,
    threshold_value: float,
    fractions: List[float],
):
    """Add annotation showing where primary crosses baseline threshold.

    Args:
        ax: Matplotlib axis
        agg_data: Aggregated statistics for this metric
        baseline_exp: Baseline experiment name
        primary_exp: Primary experiment name
        threshold_value: Y-value of baseline at max fraction
        fractions: List of available fractions
    """
    primary_data = agg_data[agg_data["egid"] == primary_exp]
    if primary_data.empty:
        return

    primary_data = primary_data.sort_values(by="fraction")
    primary_fractions = primary_data["fraction"].values
    primary_means = primary_data["mean"].values

    crossing_idx = None
    for i in range(len(primary_fractions)):
        if primary_means[i] >= threshold_value:
            crossing_idx = i
            break

    if crossing_idx is None or crossing_idx == 0:
        return

    x1, y1 = primary_fractions[crossing_idx - 1], primary_means[crossing_idx - 1]
    x2, y2 = primary_fractions[crossing_idx], primary_means[crossing_idx]
    slope = (y2 - y1) / (x2 - x1)
    x_interpolated = x1 + (threshold_value - y1) / slope
    efficiency_point = (x_interpolated, threshold_value)

    percentage = efficiency_point[0] * 100
    percentage_text = (
        f"{percentage:.1f}\\%"
        if percentage != int(percentage)
        else f"{int(percentage)}\\%"
    )

    ax.annotate(
        percentage_text,
        xy=(efficiency_point[0], threshold_value),
        textcoords="offset points",
        xytext=(-6, 30),
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
        fontsize=8,
        horizontalalignment="right",
        verticalalignment="top",
    )


def find_max_difference_fraction(
    baseline_data: pd.DataFrame, primary_data: pd.DataFrame, fractions: List[float]
) -> float:
    """Find fraction with largest absolute difference between primary and baseline.

    Args:
        baseline_data: DataFrame with baseline experiment data
        primary_data: DataFrame with primary experiment data
        fractions: List of available fractions

    Returns:
        Fraction where absolute difference is largest.
    """
    max_diff = 0
    max_diff_frac = None

    for fraction in fractions:
        baseline_row = baseline_data[baseline_data["fraction"] == fraction]
        primary_row = primary_data[primary_data["fraction"] == fraction]

        if not baseline_row.empty and not primary_row.empty:
            baseline_value = baseline_row["mean"].values[0]
            primary_value = primary_row["mean"].values[0]
            diff = abs(primary_value - baseline_value)

            if diff > max_diff:
                max_diff = diff
                max_diff_frac = fraction

    assert max_diff_frac is not None
    return max_diff_frac


def add_arrow_to_panel(
    ax,
    baseline_data: pd.DataFrame,
    primary_data: pd.DataFrame,
    fractions: List[float],
    primary_color: str,
):
    """Add delta arrow to a single panel showing primary vs baseline at rightmost fraction.

    Args:
        ax: Matplotlib axis to draw on
        baseline_data: DataFrame with baseline experiment data for this metric
        primary_data: DataFrame with primary experiment data for this metric
        fractions: List of available fractions
        primary_color: Color to use for arrow and text
    """
    if baseline_data.empty or primary_data.empty:
        return

    target_fraction = max(fractions)

    baseline_at_target = baseline_data[baseline_data["fraction"] == target_fraction]
    primary_at_target = primary_data[primary_data["fraction"] == target_fraction]

    if baseline_at_target.empty or primary_at_target.empty:
        return

    baseline_value = baseline_at_target["mean"].values[0]
    primary_value = primary_at_target["mean"].values[0]
    delta = primary_value - baseline_value

    delta_percent = delta * 100
    delta_text = (
        f"+{delta_percent:.1f}%" if delta_percent >= 0 else f"{delta_percent:.1f}%"
    )

    # Draw arrow from baseline to primary
    ax.annotate(
        "",  # No text on the arrow itself
        xy=(target_fraction, primary_value),
        xytext=(target_fraction, baseline_value),
        arrowprops=dict(
            arrowstyle="-|>",
            color=primary_color,
            lw=1.5,
        ),
        xycoords="data",
        textcoords="data",
    )

    # Add text annotation slightly below midpoint
    midpoint = (primary_value + baseline_value) / 2
    ax.annotate(
        delta_text,
        xy=(target_fraction, midpoint),
        xytext=(5, -4),  # 5 points right, 4 down
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec=primary_color,
            alpha=0.8,
            linewidth=1,
        ),
    )


def add_delta_arrows(
    ax_multipref,
    ax_rb_v1,
    ax_rb_v2,
    baseline_experiment: str,
    primary_experiment: str,
    experiment_labels: List[str],
    agg_stats: pd.DataFrame,
    agg_stats_rb: pd.DataFrame,
    agg_stats_rb_v2: Optional[pd.DataFrame],
    run_colors: Dict[str, str],
    fractions: List[float],
):
    """Add delta arrows showing largest primary method baseline delta.

    Args:
        ax_multipref: Multipref axis (or None if not included)
        ax_rb_v1: RewardBench v1 axis (or None if not included)
        ax_rb_v2: RewardBench v2 axis (or None if not included)
        baseline_experiment: Display name of baseline experiment
        primary_experiment: Display name of primary experiment
        experiment_labels: List of all experiment labels
        agg_stats: Aggregated multipref statistics
        agg_stats_rb: Aggregated RewardBench v1 statistics
        agg_stats_rb_v2: Aggregated RewardBench v2 statistics (optional)
        run_colors: Dict mapping experiment labels to colors
        fractions: List of data fractions
    """
    if (
        baseline_experiment not in experiment_labels
        or primary_experiment not in experiment_labels
    ):
        return

    multipref_baseline_all = agg_stats[agg_stats["egid"] == baseline_experiment]
    multipref_primary_all = agg_stats[agg_stats["egid"] == primary_experiment]

    rb_baseline_all = agg_stats_rb[agg_stats_rb["egid"] == baseline_experiment]
    rb_primary_all = agg_stats_rb[agg_stats_rb["egid"] == primary_experiment]

    rb_v2_baseline_all = None
    rb_v2_primary_all = None
    if agg_stats_rb_v2 is not None:
        rb_v2_baseline_all = agg_stats_rb_v2[
            agg_stats_rb_v2["egid"] == baseline_experiment
        ]
        rb_v2_primary_all = agg_stats_rb_v2[
            agg_stats_rb_v2["egid"] == primary_experiment
        ]

    if ax_multipref is not None:
        add_arrow_to_panel(
            ax_multipref,
            multipref_baseline_all,
            multipref_primary_all,
            fractions,
            "black",
        )

    if ax_rb_v1 is not None:
        add_arrow_to_panel(
            ax_rb_v1, rb_baseline_all, rb_primary_all, fractions, "black"
        )

    if (
        ax_rb_v2 is not None
        and rb_v2_baseline_all is not None
        and rb_v2_primary_all is not None
    ):
        add_arrow_to_panel(
            ax_rb_v2, rb_v2_baseline_all, rb_v2_primary_all, fractions, "black"
        )
