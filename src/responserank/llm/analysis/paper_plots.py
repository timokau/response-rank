"""Generate paper plots and tables from cached experimental data."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from responserank.llm.analysis import significance_tests
from responserank.llm.analysis.joint_perf_plot import (
    calculate_ci95_half_width,
    generate_joint_perf_plot,
)
from responserank.llm.analysis.multipref_table import generate_multipref_tables
from responserank.llm.analysis.plotting_utils import (
    save_figure_reproducible,
    setup_plot_style,
    warn_on_low_seed_counts,
)
from responserank.llm.analysis.rewardbench_categories import (
    build_rewardbench_plot,
    prepare_rewardbench_data,
)

ABLATION_HEIGHTS = 4.0
PAPER_PLOTS_MIN_SEEDS = 30


def config_names_to_display_names(
    config_names: List[str], experiments: Dict[str, Any]
) -> List[str]:
    return [experiments[name]["display_name"] for name in config_names]


def build_custom_styles_from_configs(
    styles_by_config: Dict[str, Dict[str, Any]], experiments: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    return {
        experiments[config_name]["display_name"]: style
        for config_name, style in styles_by_config.items()
    }


def create_grouped_bar_chart(
    df: pd.DataFrame,
    metadata: dict,
    hyperparameter_experiments: List[List[str]],
    output_path: str,
    title: str,
    x_labels: List[str],
    custom_styles: Dict[str, Dict[str, Any]],
    reference_lines: Dict[str, Dict[str, Any]],
    min_seed_warning: Optional[int],
    metric_ylims: Dict[str, tuple],
    figure_height: float,
    title_fontsize: float,
    tick_fontsize: float,
    label_fontsize: float,
) -> None:
    """Create grouped bar chart for hyperparameter sensitivity analysis.

    Args:
        df: DataFrame loaded from cache
        metadata: Metadata dict from cache
        hyperparameter_experiments: List of experiment display name arrays for each hyperparameter value
        output_path: Path to save the plot
        title: Title for subplot titles (e.g., "Learning Rate")
        x_labels: Labels for x-axis (e.g., ["1e-5", "2e-5", "baseline"])
        custom_styles: Dict mapping experiment names to style dicts with "label" key (required) and optional "bold" and "hatch" keys
        reference_lines: Dict mapping experiment names to horizontal line styles (use {} if none)
        min_seed_warning: If provided, warn when an experiment uses fewer unique seeds.
        metric_ylims: Dict mapping metric keys to (min, max) y-axis limit tuples
        figure_height: Height of the figure in inches
    """
    rc("text", usetex=True)
    rc("font", family="serif")
    setup_plot_style()

    df_filtered = df[df["fraction"] == 1.0].copy()

    computed_display_names = metadata["computed_display_names"]
    alias_to_raw = {alias: raw for raw, alias in computed_display_names.items()}

    filter_names = []
    display_labels_map = {}

    for experiment_group in hyperparameter_experiments:
        for exp_display_name in experiment_group:
            raw_name = alias_to_raw.get(exp_display_name, exp_display_name)
            filter_names.append(raw_name)
            display_labels_map[raw_name] = exp_display_name

    df_filtered = df_filtered[df_filtered["egid"].isin(filter_names)]

    if min_seed_warning is not None:
        warn_on_low_seed_counts(
            df_filtered,
            experiments=sorted(set(filter_names)),
            groupby_cols=[],
            threshold=min_seed_warning,
            display_name_aliases=computed_display_names,
        )

    computed_styling = metadata["computed_styling"]
    run_colors_lookup = computed_styling["run_colors"]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, figure_height))

    metric_config = {
        "multipref": {
            "column": "eval_accuracy",
            "title": "MultiPref Test Accuracy",
            "ylim": metric_ylims["multipref"],
            "tick_resolution": 0.1,
        },
        "rb_v2": {
            "column": "eval/rewardbench_v2/overall/accuracy_score",
            "title": "RewardBench 2 Score",
            "ylim": metric_ylims["rb_v2"],
            "tick_resolution": 0.1,
        },
    }

    metrics = ["multipref", "rb_v2"]
    n_groups = len(hyperparameter_experiments)
    x_pos = np.arange(n_groups)
    max_bars_per_group = max(len(group) for group in hyperparameter_experiments)
    bar_width = 0.8 / max_bars_per_group

    for i, (ax, metric_key) in enumerate(zip(axes, metrics)):
        metric_column = metric_config[metric_key]["column"]
        metric_title = metric_config[metric_key]["title"]

        for group_idx, experiment_group in enumerate(hyperparameter_experiments):
            for bar_idx, exp_name in enumerate(experiment_group):
                raw_name = alias_to_raw.get(exp_name, exp_name)
                exp_data = df_filtered[df_filtered["egid"] == raw_name]

                values = exp_data[metric_column].values
                # I want to be able to add pending runs to the plots, use 0 as a placeholder for those
                mean_val = np.mean(values) if len(values) > 0 else 0.0
                ci_val = calculate_ci95_half_width(values) if len(values) > 1 else 0.0
                seed_count = int(exp_data["seed"].dropna().nunique())

                bar_offset = (bar_idx - max_bars_per_group / 2 + 0.5) * bar_width
                bar_pos = x_pos[group_idx] + bar_offset

                color = run_colors_lookup.get(raw_name, "#CCCCCC")

                if exp_name not in custom_styles:
                    raise ValueError(
                        f"Missing custom_styles entry for experiment: {exp_name}"
                    )

                style = custom_styles[exp_name]
                if "label" not in style:
                    raise ValueError(
                        f"Missing 'label' key in custom_styles for experiment: {exp_name}"
                    )

                hyperparam_label = style["label"]
                hatch_pattern = style.get("hatch")

                if style.get("bold", False):
                    hyperparam_label = f"\\textbf{{{hyperparam_label}}}"

                ax.bar(
                    bar_pos,
                    mean_val,
                    bar_width,
                    yerr=ci_val,
                    capsize=3,
                    color=color,
                    hatch=hatch_pattern,
                )

                # Add custom x-axis label for each bar in each group
                ax.text(
                    bar_pos,
                    -0.025,
                    hyperparam_label,
                    ha="center",
                    va="top",
                    fontsize=7,
                    rotation=45,
                    transform=ax.get_xaxis_transform(),
                )

                # Add seed count annotation if below threshold
                if min_seed_warning is not None and seed_count < min_seed_warning:
                    text_y = mean_val + ci_val + 0.02
                    ax.text(
                        bar_pos,
                        text_y,
                        f"n={seed_count}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="#C0392B",
                        weight="bold",
                    )

        for exp_name, line_style in reference_lines.items():
            raw_name = alias_to_raw.get(exp_name, exp_name)
            exp_data = df[df["fraction"] == 1.0]
            exp_data = exp_data[exp_data["egid"] == raw_name]
            values = exp_data[metric_column].values
            mean_val = np.mean(values)

            ax.axhline(
                y=mean_val,
                color=line_style.get("color", "gray"),
                linestyle=line_style.get("linestyle", "--"),
                alpha=line_style.get("alpha", 0.7),
                linewidth=line_style.get("linewidth", 1.5),
                label=line_style.get("label", exp_name),
            )

        ax.set_title(f"\\bf{{}}{metric_title}", fontsize=title_fontsize)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=tick_fontsize)
        ax.tick_params(axis="x", pad=25)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.15)

        if i == 0:
            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                right=False,
                labelsize=tick_fontsize,
                color=(0, 0, 0, 0.3),
            )
        else:
            ax.yaxis.tick_right()
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                right=True,
                labelsize=tick_fontsize,
                color=(0, 0, 0, 0.3),
            )

        if "ylim" in metric_config[metric_key]:
            ax.set_ylim(metric_config[metric_key]["ylim"])

        if "tick_resolution" in metric_config[metric_key]:
            tick_res = metric_config[metric_key]["tick_resolution"]
            ax.yaxis.set_major_locator(MultipleLocator(tick_res))
            ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout(rect=[-0.04, 0.05, 1, 1])

    save_figure_reproducible(fig, output_path)
    plt.close(fig)

    print(f"Bar chart saved to {output_path}")


def generate_rewardbench_categories_paper(
    df: pd.DataFrame,
    metadata: dict,
    plot_width: float,
    show_points: bool,
    show_std: str,
    *,
    use_ci95: bool,
    use_median: bool,
    output_path: str,
    include_experiments: Optional[List[List[str]]],
    min_seed_warning: Optional[int],
    custom_styles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Generate RewardBench category plots with legend spanning the plot width."""
    data_bundle = prepare_rewardbench_data(df, metadata, include_experiments)

    if min_seed_warning is not None:
        warn_on_low_seed_counts(
            data_bundle["df"],
            experiments=data_bundle["experiment_labels"],
            groupby_cols=["fraction"],
            threshold=min_seed_warning,
            display_name_aliases=data_bundle["display_name_aliases"],
        )

    fig, axes, handles, labels, legend_grid_cols = build_rewardbench_plot(
        data_bundle,
        plot_width,
        show_points,
        show_std,
        use_ci95=use_ci95,
        use_median=use_median,
        height_per_row=3.0,
        marker_size=4.5,
        line_width=2.0,
        dash_scale=0.2,
        title_fontsize=10.0,
        xlabel_fontsize=10.0,
        ylabel_fontsize=10.0,
        xtick_fontsize=8.0,
        ytick_fontsize=8.0,
        supylabel_kwargs={"x": 0.04},
        supxlabel_kwargs={"y": 0.03},
        include_experiments=include_experiments,
    )

    if handles:
        if custom_styles is not None:
            labels = [
                (
                    custom_styles[label]["label"]
                    if label in custom_styles and "label" in custom_styles[label]
                    else label
                )
                for label in labels
            ]

        # Place legend on top, spanning both figures but not the left y-axis label.
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])  # Leave 10% headroom for legend
        visible_axes = [ax for ax in axes.flat if ax.get_visible()]
        if visible_axes:
            left = min(ax.get_position().x0 for ax in visible_axes)
            right = max(ax.get_position().x1 for ax in visible_axes)
            legend_width = right - left
            fig.legend(
                handles,
                labels,
                loc="lower left",
                bbox_to_anchor=(left, 0.9, legend_width, 0.1),
                bbox_transform=fig.transFigure,
                mode="expand",
                ncol=legend_grid_cols,
                fontsize=10,
                frameon=False,
                borderaxespad=0.0,
            )

    save_figure_reproducible(fig, output_path)
    plt.close(fig)

    print(f"RewardBench v2 category plot saved to {output_path}")


def generate_size_ablation_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
    tex_macros_path: Path,
):
    """Generate size ablation bar plot."""
    config_experiments = [
        [
            "bt",
            "rr_agree_spread2",
            "rr_agree_spread3",
            "rr_agree_spread4",
            "rr_agree",
        ],
    ]

    styles_by_config = {
        "bt": {"label": "BT", "hatch": "///"},
        "rr_agree_spread2": {"label": "Size 2"},
        "rr_agree_spread3": {"label": "Size 3"},
        "rr_agree_spread4": {"label": "Size 4"},
        "rr_agree": {
            "label": "Full Size",
            "hatch": "\\\\\\",
            "bold": True,
        },
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/responserank_size_ablation_bar.pdf",
        title="Ranking Size Constraints",
        x_labels=["Ranking Size"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )

    write_fragment_size_macros(df, metadata, tex_macros_path)
    write_singleton_fraction_macros(df, metadata, tex_macros_path)


def write_fragment_size_macros(
    df: pd.DataFrame,
    metadata: dict,
    tex_macros_path: Path,
) -> None:
    """Write LaTeX macros for size ablation avg_fragment_size values."""
    computed_display_names = metadata["computed_display_names"]
    alias_to_raw = {alias: raw for raw, alias in computed_display_names.items()}

    experiments = {
        "SizeTwo": "RR-Agree (spread2)",
        "SizeThree": "RR-Agree (spread3)",
        "SizeFour": "RR-Agree (spread4)",
        "Full": "RR-Agree (ours)",
    }

    macros = []
    for short_name, exp_name in experiments.items():
        raw_name = alias_to_raw.get(exp_name, exp_name)
        exp_data = df[(df["egid"] == raw_name) & (df["fraction"] == 1.0)]

        if len(exp_data) > 0 and "avg_fragment_size" in exp_data.columns:
            avg_frag = exp_data["avg_fragment_size"].mean()
            macros.append(
                f"\\newcommand{{\\sizeAblation{short_name}FragSize}}{{{avg_frag:.2f}}}"
            )

    if macros:
        existing = tex_macros_path.read_text() if tex_macros_path.exists() else ""
        new_content = existing + "\n" + "\n".join(macros) + "\n"
        tex_macros_path.write_text(new_content)


def write_meanpref_fragment_size_macros(
    df: pd.DataFrame,
    metadata: dict,
    tex_macros_path: Path,
) -> None:
    """Write LaTeX macros for MeanPref size ablation avg_fragment_size values."""
    computed_display_names = metadata["computed_display_names"]
    alias_to_raw = {alias: raw for raw, alias in computed_display_names.items()}

    experiments = {
        "SizeTwo": "RR-Consensus-2",
        "SizeFour": "RR-Consensus-4",
        "SizeSix": "RR-Consensus-6",
        "Full": "RR-Consensus",
    }

    macros = []
    for short_name, exp_name in experiments.items():
        raw_name = alias_to_raw.get(exp_name, exp_name)
        exp_data = df[(df["egid"] == raw_name) & (df["fraction"] == 1.0)]

        avg_frag = exp_data["avg_fragment_size"].mean()
        macros.append(
            f"\\newcommand{{\\meanPref{short_name}FragSize}}{{{avg_frag:.2f}}}"
        )

    if macros:
        existing = tex_macros_path.read_text() if tex_macros_path.exists() else ""
        new_content = existing + "\n" + "\n".join(macros) + "\n"
        tex_macros_path.write_text(new_content)


def write_singleton_fraction_macros(
    df: pd.DataFrame,
    metadata: dict,
    tex_macros_path: Path,
) -> None:
    """Write LaTeX macros for singleton fraction values."""
    computed_display_names = metadata["computed_display_names"]
    alias_to_raw = {alias: raw for raw, alias in computed_display_names.items()}

    experiments = {
        "stated": "RR-Stated",
        "rt": "RR-RT (annostrat spread2)",
    }

    macros = []
    for short_name, exp_name in experiments.items():
        raw_name = alias_to_raw.get(exp_name, exp_name)
        exp_data = df[(df["egid"] == raw_name) & (df["fraction"] == 1.0)]

        # Convert from non-singleton fraction to singleton percentage
        non_singleton_frac = exp_data["non_singleton_fraction"].mean()
        singleton_frac = 1.0 - non_singleton_frac
        singleton_pct = singleton_frac * 100
        macros.append(
            f"\\newcommand{{\\{short_name}SingletonFraction}}{{{singleton_pct:.1f}}}"
        )

    if macros:
        existing = tex_macros_path.read_text() if tex_macros_path.exists() else ""
        new_content = existing + "\n" + "\n".join(macros) + "\n"
        tex_macros_path.write_text(new_content)


def generate_lr_sensitivity_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate learning rate sensitivity bar chart."""
    config_experiments = [
        ["bt_lr1em5", "bt", "bt_lr2em5"],
        ["rr_agree_lr1em5", "rr_agree", "rr_agree_lr2em5"],
    ]

    styles_by_config = {
        "bt_lr1em5": {"label": "1e-05"},
        "bt": {"label": "5e-06"},
        "bt_lr2em5": {"label": "2e-05"},
        "rr_agree_lr1em5": {"label": "1e-05"},
        "rr_agree": {"label": "5e-06", "bold": True},
        "rr_agree_lr2em5": {"label": "2e-05"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/hyperparam_lr_sensitivity_bar.pdf",
        title="Learning Rate",
        x_labels=["BT variants", "RR variants"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_acc_sensitivity_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate accumulation steps sensitivity bar chart."""
    config_experiments = [
        ["bt_acc2", "bt", "bt_acc8"],
        ["rr_agree_gradacc2", "rr_agree", "rr_agree_gradacc8"],
    ]

    styles_by_config = {
        "bt_acc2": {"label": "2"},
        "bt": {"label": "4", "bold": True},
        "bt_acc8": {"label": "8"},
        "rr_agree_gradacc2": {"label": "2"},
        "rr_agree": {"label": "4", "bold": True},
        "rr_agree_gradacc8": {"label": "8"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/hyperparam_acc_sensitivity_bar.pdf",
        title="Gradient Accumulation Steps",
        x_labels=["BT variants", "RR variants"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_clip_sensitivity_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate gradient clipping sensitivity bar chart."""
    config_experiments = [
        ["bt_clip0p5", "bt", "bt_clip2p0"],
        ["rr_agree_gradclip0p5", "rr_agree", "rr_agree_gradclip2p0"],
    ]

    styles_by_config = {
        "bt_clip0p5": {"label": "0.5"},
        "bt": {"label": "1.0", "bold": True},
        "bt_clip2p0": {"label": "2.0"},
        "rr_agree_gradclip0p5": {"label": "0.5"},
        "rr_agree": {"label": "1.0", "bold": True},
        "rr_agree_gradclip2p0": {"label": "2.0"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/hyperparam_clip_sensitivity_bar.pdf",
        title="Gradient Clipping",
        x_labels=["BT variants", "RR variants"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_epochs_sensitivity_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate training epochs sensitivity bar chart."""
    config_experiments = [
        ["bt_ep2", "bt", "bt_ep4"],
        ["rr_agree_ep2", "rr_agree", "rr_agree_ep4"],
    ]

    styles_by_config = {
        "bt_ep2": {"label": "2"},
        "bt": {"label": "3", "bold": True},
        "bt_ep4": {"label": "4"},
        "rr_agree_ep2": {"label": "2"},
        "rr_agree": {"label": "3", "bold": True},
        "rr_agree_ep4": {"label": "4"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/hyperparam_epochs_sensitivity_bar.pdf",
        title="Training Epochs",
        x_labels=["BT variants", "RR variants"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_warmup_sensitivity_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate warmup ratio sensitivity bar chart."""
    config_experiments = [
        ["bt_warmup0p01", "bt", "bt_warmup0p1"],
        [
            "rr_agree_warmup0p01",
            "rr_agree",
            "rr_agree_warmup0p1",
        ],
    ]

    styles_by_config = {
        "bt_warmup0p01": {"label": "0.01"},
        "bt": {"label": "0.05", "bold": True},
        "bt_warmup0p1": {"label": "0.1"},
        "rr_agree_warmup0p01": {"label": "0.01"},
        "rr_agree": {"label": "0.05", "bold": True},
        "rr_agree_warmup0p1": {"label": "0.1"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/hyperparam_warmup_sensitivity_bar.pdf",
        title="Warmup Ratio",
        x_labels=["BT variants", "RR variants"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_wd_sensitivity_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate weight decay sensitivity bar chart."""
    config_experiments = [
        ["bt_wd0p001", "bt"],
        ["rr_agree_weightdecay0p001", "rr_agree"],
    ]

    styles_by_config = {
        "bt_wd0p001": {"label": "0.001"},
        "bt": {"label": "0.1", "bold": True},
        "rr_agree_weightdecay0p001": {"label": "0.001"},
        "rr_agree": {"label": "0.1", "bold": True},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/hyperparam_wd_sensitivity_bar.pdf",
        title="Weight Decay",
        x_labels=["BT variants", "RR variants"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_noise_robustness_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate noise robustness sensitivity bar chart using partial shuffle."""
    config_experiments = [
        [
            "rr_agree_spread2",
            "rr_agree_spread2_partialshuffle25",
            "rr_agree_spread2_partialshuffle50",
            "rr_agree_spread2_partialshuffle75",
            "rr_agree_spread2_partialshuffle100",
        ],
        [
            "rr_agree",
            "rr_agree_partialshuffle25",
            "rr_agree_partialshuffle50",
            "rr_agree_partialshuffle75",
            "rr_agree_partialshuffle100",
        ],
    ]

    styles_by_config = {
        "rr_agree": {"label": "0\\%"},
        "rr_agree_partialshuffle25": {"label": "25\\%"},
        "rr_agree_partialshuffle50": {"label": "50\\%"},
        "rr_agree_partialshuffle75": {"label": "75\\%"},
        "rr_agree_partialshuffle100": {"label": "100\\%"},
        "rr_agree_spread2": {"label": "0\\%"},
        "rr_agree_spread2_partialshuffle25": {"label": "25\\%"},
        "rr_agree_spread2_partialshuffle50": {"label": "50\\%"},
        "rr_agree_spread2_partialshuffle75": {"label": "75\\%"},
        "rr_agree_spread2_partialshuffle100": {"label": "100\\%"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/noise_robustness_sensitivity_bar.pdf",
        title="Partial Shuffle Percentage",
        x_labels=["Size 2", "Full Size"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={
            "BT (baseline)": {
                "color": "#E74C3C",
                "linestyle": "--",
                "alpha": 0.7,
                "linewidth": 1.5,
                "label": "BT baseline",
            }
        },
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_filter_robustness_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate filter robustness sensitivity bar chart."""

    config_experiments = [
        [
            "rr_agree",
            "rr_agree_filter25",
            "rr_agree_filter50",
            "rr_agree_filter75",
            "rr_agree_filter95",
            "bt",
        ],
    ]

    styles_by_config = {
        "bt": {"label": "BT"},
        "rr_agree": {"label": "0", "bold": True},
        "rr_agree_filter25": {"label": "25%"},
        "rr_agree_filter50": {"label": "50%"},
        "rr_agree_filter75": {"label": "75%"},
        "rr_agree_filter95": {"label": "95%"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/filter_robustness_sensitivity_bar.pdf",
        title="Filter Fraction",
        x_labels=["Filter Strength"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_response_time_length_strat_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate response time with length stratification bar chart."""
    config_experiments = [
        [
            "rr_rt_spread2_global",
            "rr_rt_annostrat_spread2",
            "rr_rt_annolength2_spread2",
            "rr_rt_annolength4_spread2",
            "rr_rt_annolength8_spread2",
            "rr_rt_annolength16_spread2",
            "rr_rt_annolength32_spread2",
        ],
    ]

    styles_by_config = {
        "rr_rt_spread2_global": {"label": "Global"},
        "rr_rt_annostrat_spread2": {"label": "No buckets"},
        "rr_rt_annolength2_spread2": {"label": "2 buckets"},
        "rr_rt_annolength4_spread2": {"label": "4 buckets"},
        "rr_rt_annolength8_spread2": {"label": "8 buckets"},
        "rr_rt_annolength16_spread2": {"label": "16 buckets"},
        "rr_rt_annolength32_spread2": {"label": "32 buckets"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/response_time_length_strat_bar.pdf",
        title="Response Time Length Stratification",
        x_labels=[""],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={
            "BT (baseline)": {
                "color": "#E74C3C",
                "linestyle": "--",
                "alpha": 0.7,
                "linewidth": 1.5,
                "label": "BT baseline",
            },
            "RR-RandRank": {
                "color": "#95A5A6",
                "linestyle": "--",
                "alpha": 0.7,
                "linewidth": 1.5,
                "label": "RandRank",
            },
        },
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_response_time_length_strat_bar_plot_simplified(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate simplified response time bar chart with only key comparisons."""
    config_experiments = [
        [
            "bt",
            "bt_ep2",
            "rr_randrank_randsize2",
            "rr_rt_annostrat_spread2",
            "rr_stated",
            "rr_agree",
        ],
    ]

    styles_by_config = {
        "rr_rt_annostrat_spread2": {"label": "RR-RT"},
        "bt": {"label": "BT", "hatch": "///"},
        "bt_ep2": {"label": "BT@2", "hatch": "..."},
        "rr_randrank_randsize2": {"label": "RR-Random"},
        "rr_stated": {"label": "RR-Stated", "hatch": "..."},
        "rr_agree": {"label": "RR-Agree", "hatch": "\\\\\\"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/response_time_length_strat_bar_simplified.pdf",
        title="",
        x_labels=[""],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_slight_weight_experiment_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate slight weight experiment bar chart."""
    config_experiments = [
        [
            "rr_agree_slight25",
            "rr_agree",
            "rr_agree_slight75",
            "bt",
        ],
    ]

    styles_by_config = {
        "rr_agree_slight25": {"label": "S25"},
        "rr_agree": {"label": "S50", "bold": True},
        "rr_agree_slight75": {"label": "S75"},
        "bt": {"label": "BT", "hatch": "///"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/slight_weight_experiment_bar.pdf",
        title="Slight Weight Experiment",
        x_labels=["Slight Weight"],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_stated_strength_ablation_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate stated strength ablation bar chart."""
    config_experiments = [
        [
            "rr_stated",
            "rr_stated_global",
            "rr_stated_partialshuffle100",
            "bt",
            "rr_agree",
        ],
    ]

    styles_by_config = {
        "rr_stated": {"label": "Stated"},
        "rr_stated_global": {"label": "Stated-Global"},
        "rr_stated_partialshuffle100": {"label": "Stated-Shuf"},
        "bt": {"label": "BT", "hatch": "///"},
        "rr_agree": {
            "label": "RR",
            "hatch": "\\\\\\",
            "bold": True,
        },
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/stated_strength_ablation_bar.pdf",
        title="Stated Strength Ablation",
        x_labels=[""],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_randrank_size_ablation_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate random ranking size ablation bar chart."""
    config_experiments = [
        [
            "bt",
            "rr_randrank_randsize2",
            "rr_randrank_randsize4",
        ],
    ]

    styles_by_config = {
        "rr_randrank_randsize2": {"label": "Size 2"},
        "rr_randrank_randsize4": {"label": "Size 4"},
        "bt": {"label": "BT", "hatch": "///"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/randrank_size_ablation_bar.pdf",
        title="Random Ranking Size Ablation",
        x_labels=[""],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={},
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_meanpref_size_comparison_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate RR-Agree vs RR-MeanPref size ablation bar chart."""
    config_experiments = [
        [
            "bt",
            "rr_agree",
            "rr_consensus_spread2",
            "rr_consensus_spread4",
            "rr_consensus_spread6",
            "rr_consensus",
        ],
    ]

    styles_by_config = {
        "bt": {"label": "BT"},
        "rr_agree": {"label": "RR-Agree", "bold": True, "hatch": "\\\\\\"},
        "rr_consensus": {"label": "RR-MeanPref (full)", "hatch": "///"},
        "rr_consensus_spread2": {"label": "RR-MeanPref-2", "hatch": "///"},
        "rr_consensus_spread4": {"label": "RR-MeanPref-4", "hatch": "///"},
        "rr_consensus_spread6": {"label": "RR-MeanPref-6", "hatch": "///"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/meanpref_size_comparison_bar.pdf",
        title="RR-Agree vs RR-MeanPref",
        x_labels=[""],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={
            "BT (baseline)": {
                "color": "#E74C3C",
                "linestyle": "--",
                "alpha": 0.7,
                "linewidth": 1.5,
                "label": "BT baseline",
            }
        },
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_bt_soft_labels_comparison_bar_plot(
    df: pd.DataFrame,
    metadata: dict,
    experiments: Dict[str, Any],
    figures_dir: str,
    min_seed_warning: Optional[int],
):
    """Generate BT soft labels vs RR comparison bar chart."""
    config_experiments = [
        [
            "bt",
            "bt_agree",
            "bt_meanpref",
            "rr_agree",
        ],
    ]

    styles_by_config = {
        "bt": {"label": "BT"},
        "bt_agree": {"label": "BT-Agree"},
        "bt_meanpref": {"label": "BT-MeanPref"},
        "rr_agree": {"label": "RR-Agree", "bold": True, "hatch": "\\\\\\"},
    }

    create_grouped_bar_chart(
        df=df,
        metadata=metadata,
        hyperparameter_experiments=[
            config_names_to_display_names(group, experiments)
            for group in config_experiments
        ],
        output_path=f"{figures_dir}/bt_soft_labels_comparison_bar.pdf",
        title="Soft Labels Comparison",
        x_labels=[""],
        custom_styles=build_custom_styles_from_configs(styles_by_config, experiments),
        reference_lines={
            "BT (baseline)": {
                "color": "#E74C3C",
                "linestyle": "--",
                "alpha": 0.7,
                "linewidth": 1.5,
                "label": "BT baseline",
            }
        },
        min_seed_warning=min_seed_warning,
        metric_ylims={"multipref": (0.6, 0.8), "rb_v2": (0.4, 0.6)},
        figure_height=3.0,
        title_fontsize=8.0,
        tick_fontsize=8.0,
        label_fontsize=8.0,
    )


def generate_paper_plots(
    df,
    metadata,
    experiments: Dict[str, Any],
    figures_dir,
    min_seed_warning: int = PAPER_PLOTS_MIN_SEEDS,
):
    """Generate all paper plots and tables from cached data.

    Args:
        df: DataFrame loaded from cache
        metadata: Metadata dict from cache
        experiments: Dict mapping config names to experiment metadata (display_name, color, etc.)
        figures_dir: Directory where to save paper figures and tables
        min_seed_warning: Threshold for seed-count warnings in generated figures.
    """
    tex_macros_path = Path(figures_dir) / "stats.tex"

    # Filter to only finished runs for plotting
    df = df[df["run_state"] == "finished"].copy()

    generate_joint_perf_plot(
        df=df,
        metadata=metadata,
        plot_width=6.07,  # Tuned to get 5.49in output with constrained_layout + manual spacing
        plot_height=ABLATION_HEIGHTS,
        show_points=False,
        show_std="shade",
        use_ci95=True,
        use_median=False,
        include_metrics=("multipref", "rb_v2"),
        output_path=f"{figures_dir}/multipref_datasize.pdf",
        title_fontsize=8.0,
        tick_fontsize=8.0,
        marker_size=4.5,
        line_width=2.0,
        dash_scale=0.2,
        supxlabel_fontsize=8.0,
        supylabel_fontsize=8.0,
        legend_mode="figure_top",
        legend_kwargs={
            "fontsize": 8.0,
        },
        include_experiments=[
            config_names_to_display_names(["bt", "rr_stated", "rr_agree"], experiments),
        ],
        baseline_experiment=experiments["bt"]["display_name"],
        primary_experiment=experiments["rr_agree"]["display_name"],
        min_seed_warning=min_seed_warning,
        custom_styles={},
    )

    generate_rewardbench_categories_paper(
        df=df,
        metadata=metadata,
        plot_width=5.5,
        show_points=False,
        show_std="shade",
        use_ci95=True,
        use_median=False,
        output_path=f"{figures_dir}/multipref_datasize_rb.pdf",
        include_experiments=[
            config_names_to_display_names(
                ["bt", "bt_ep2", "rr_randrank_randsize2"], experiments
            ),
            config_names_to_display_names(
                [
                    "rr_rt_annostrat_spread2",
                    "rr_stated",
                    "rr_agree",
                ],
                experiments,
            ),
        ],
        min_seed_warning=min_seed_warning,
        custom_styles=build_custom_styles_from_configs(
            {
                "rr_rt_annostrat_spread2": {"label": "RR-RT"},
                "bt_ep2": {"label": "BT@2"},
                "rr_randrank_randsize2": {"label": "RR-Random"},
            },
            experiments,
        ),
    )

    generate_multipref_tables(
        df=df,
        metadata=metadata,
        output_dir=Path(figures_dir),
        experiments=[
            experiments["bt"]["display_name"],
            experiments["bt_ep2"]["display_name"],
            experiments["rr_randrank_randsize2"]["display_name"],
            experiments["rr_rt_annostrat_spread2"]["display_name"],
            experiments["rr_stated"]["display_name"],
            experiments["rr_agree"]["display_name"],
        ],
        labels={
            experiments["bt"]["display_name"]: "BT",
            experiments["bt_ep2"]["display_name"]: "BT@2",
            experiments["rr_randrank_randsize2"]["display_name"]: "RR-Random",
            experiments["rr_rt_annostrat_spread2"]["display_name"]: "RR-RT",
            experiments["rr_stated"]["display_name"]: "RR-Stated",
            experiments["rr_agree"]["display_name"]: "RR-Agree",
        },
    )

    generate_meanpref_size_comparison_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )

    generate_bt_soft_labels_comparison_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )

    significance_tests.run_batching_validation(df, metadata, tex_macros_path)
    significance_tests.run_stated_validation(df, metadata, tex_macros_path)
    significance_tests.run_agree_validation(df, metadata, tex_macros_path)
    significance_tests.run_stated_global_validation(df, metadata, tex_macros_path)
    significance_tests.run_meanpref_size_validation(df, metadata, tex_macros_path)
    significance_tests.run_bt_agree_validation(df, metadata, tex_macros_path)
    significance_tests.run_bt_meanpref_validation(df, metadata, tex_macros_path)

    generate_size_ablation_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
        tex_macros_path=tex_macros_path,
    )

    write_meanpref_fragment_size_macros(df, metadata, tex_macros_path)

    generate_lr_sensitivity_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_acc_sensitivity_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_clip_sensitivity_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_epochs_sensitivity_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_warmup_sensitivity_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_wd_sensitivity_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_noise_robustness_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_filter_robustness_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_response_time_length_strat_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    significance_tests.write_length_bucketing_singleton_macros(
        df, metadata, tex_macros_path
    )
    generate_response_time_length_strat_bar_plot_simplified(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_slight_weight_experiment_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_stated_strength_ablation_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
    generate_randrank_size_ablation_bar_plot(
        df,
        metadata,
        experiments,
        figures_dir,
        min_seed_warning=min_seed_warning,
    )
