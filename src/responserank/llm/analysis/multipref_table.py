"""Utilities to build the MultiPref LaTeX results tables."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

EXPECTED_SEEDS = 30
FRACTION_RANGES = [
    ((0.1, 0.5), "_part1"),
    ((0.6, 1.0), "_part2"),
]


@dataclass
class MetricConfig:
    """Configuration for a metric."""

    column: str
    output_filename: str
    is_percentage: bool


METRICS = {
    "test_accuracy": MetricConfig(
        column="eval_accuracy",
        output_filename="multipref_test_accuracy_table.tex",
        is_percentage=True,
    ),
    "rewardbench_v1": MetricConfig(
        column="eval/rewardbench_v1/overall/accuracy_score",
        output_filename="multipref_rewardbench_v1_table.tex",
        is_percentage=True,
    ),
    "rewardbench_v2": MetricConfig(
        column="eval/rewardbench_v2/overall/accuracy_score",
        output_filename="multipref_rewardbench_v2_table.tex",
        is_percentage=True,
    ),
}


def _ci95_half_width(values: np.ndarray) -> float:
    """Calculate 95% CI half-width."""
    values = np.array(values)
    assert len(values) > 1
    sem = stats.sem(values)
    return sem * stats.t.ppf(0.975, len(values) - 1)


def _aggregate_experiment_data(
    df: pd.DataFrame,
    metadata: dict,
    experiments: List[str],
    labels: Dict[str, str],
) -> pd.DataFrame:
    """Aggregate data for specified experiments.

    Returns DataFrame with columns: display_name, fraction, metric_mean, metric_std, metric_ci
    for each metric in METRICS.
    """
    computed_display_names = metadata["computed_display_names"]
    alias_to_raw = {alias: raw for raw, alias in computed_display_names.items()}

    raw_egids = []
    for exp_name in experiments:
        raw_egids.append(alias_to_raw[exp_name])

    filtered_df = df[df["egid"].isin(raw_egids)].copy()

    if filtered_df.empty:
        raise ValueError(f"No data found for experiments: {experiments}")

    agg_dict = {}
    for metric_name, config in METRICS.items():
        agg_dict[config.column] = ["mean", "std", _ci95_half_width, "count"]

    agg_stats = filtered_df.groupby(["egid", "fraction"]).agg(agg_dict).reset_index()

    new_columns = ["egid", "fraction"]
    for metric_name, config in METRICS.items():
        new_columns.extend(
            [
                f"{metric_name}_mean",
                f"{metric_name}_std",
                f"{metric_name}_ci",
                f"{metric_name}_count",
            ]
        )
    agg_stats.columns = new_columns

    egid_to_display = computed_display_names
    agg_stats["display_name"] = agg_stats["egid"].map(egid_to_display)

    first_metric = list(METRICS.keys())[0]
    count_col = f"{first_metric}_count"
    for _, row in agg_stats.iterrows():
        count = row[count_col]
        if count < EXPECTED_SEEDS:
            label = labels.get(row["display_name"], row["display_name"])
            fraction_pct = int(row["fraction"] * 100)
            print(
                f"WARNING: {label} at {fraction_pct}% has only {int(count)} seeds "
                f"(expected {EXPECTED_SEEDS})"
            )

    return agg_stats


def _format_cell(
    mean: float,
    std: float,
    ci: float,
    is_percentage: bool,
    is_best: bool,
) -> str:
    """Format a table cell with mean, std, and CI.

    Format: $mean \\pm std [\\pm CI]$ with optional bold for best.
    """
    if is_percentage:
        mean_val = mean * 100
        std_val = std * 100
        ci_val = ci * 100
        suffix = "\\%"
    else:
        mean_val = mean
        std_val = std
        ci_val = ci
        suffix = ""

    if is_best:
        mean_str = f"\\mathbf{{{mean_val:.1f}}}"
    else:
        mean_str = f"{mean_val:.1f}"

    return f"${mean_str} \\pm {std_val:.1f}{suffix}$ [$\\pm$ {ci_val:.1f}]"


def _generate_table_for_metric(
    agg_stats: pd.DataFrame,
    experiments: List[str],
    labels: Dict[str, str],
    metric_name: str,
    config: MetricConfig,
    fraction_range: Tuple[float, float],
) -> str:
    """Generate LaTeX table for a single metric."""
    all_fractions = sorted(agg_stats["fraction"].unique())
    fractions = [
        f for f in all_fractions if fraction_range[0] <= f <= fraction_range[1]
    ]

    header_cols = "l" + "r" * len(fractions)
    lines = [
        f"\\begin{{tabular}}{{@{{}}{header_cols}@{{}}}}",
        "\\toprule",
    ]

    fraction_headers = " & ".join(f"{int(f * 100)}\\%" for f in fractions)
    lines.append(f"Learner & {fraction_headers} \\\\")
    lines.append("\\midrule")

    best_per_fraction: Dict[float, Tuple[str, float]] = {}
    for fraction in fractions:
        fraction_data = agg_stats[agg_stats["fraction"] == fraction]
        if not fraction_data.empty:
            # Highest is best
            best_idx = fraction_data[f"{metric_name}_mean"].idxmax()
            best_learner = fraction_data.loc[best_idx, "display_name"]
            best_per_fraction[fraction] = best_learner

    for exp_name in experiments:
        exp_data = agg_stats[agg_stats["display_name"] == exp_name]
        if exp_data.empty:
            continue

        label = labels.get(exp_name, exp_name)

        cells = [label]
        for fraction in fractions:
            fraction_row = exp_data[exp_data["fraction"] == fraction]
            if fraction_row.empty:
                cells.append("--")
                continue

            row = fraction_row.iloc[0]
            is_best = best_per_fraction.get(fraction) == exp_name

            cell = _format_cell(
                mean=row[f"{metric_name}_mean"],
                std=row[f"{metric_name}_std"],
                ci=row[f"{metric_name}_ci"],
                is_percentage=config.is_percentage,
                is_best=is_best,
            )
            cells.append(cell)

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def generate_multipref_tables(
    df: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
    experiments: List[str],
    labels: Dict[str, str],
) -> None:
    """Create MultiPref LaTeX tables from cached DataFrame.

    Generates two tables per metric (split by fraction range) with:
    - Rows: Learners (experiments)
    - Columns: Training fractions
    - Cell format: $mean \\pm std$ [± CI] with best values bolded

    Args:
        df: DataFrame loaded from cache
        metadata: Metadata dict from cache
        output_dir: Directory for output LaTeX tables
        experiments: List of experiment display names to include
        labels: Mapping from experiment display names to table labels
    """
    agg_stats = _aggregate_experiment_data(df, metadata, experiments, labels)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, config in METRICS.items():
        for fraction_range, suffix in FRACTION_RANGES:
            latex_content = _generate_table_for_metric(
                agg_stats=agg_stats,
                experiments=experiments,
                labels=labels,
                metric_name=metric_name,
                config=config,
                fraction_range=fraction_range,
            )

            base_name = config.output_filename.replace(".tex", "")
            output_path = output_dir / f"{base_name}{suffix}.tex"
            output_path.write_text(latex_content + "\n")
            print(f"Wrote {metric_name} table ({suffix}) to {output_path}")

    print(f"Experiments included: {', '.join(experiments)}")
