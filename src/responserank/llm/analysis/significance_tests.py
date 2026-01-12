"""Statistical significance tests for paper results.

This module provides statistical tests to validate experimental findings
and generate LaTeX macros for the paper.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

STANDARD_METRICS = {
    "MultiPref": "eval_accuracy",
    "RewardBench v2": "eval/rewardbench_v2/overall/accuracy_score",
}


def compute_paired_comparison(
    df: pd.DataFrame,
    metadata: dict,
    exp_a: str,
    exp_b: str,
    metrics: dict[str, str],
    fraction: float,
) -> dict:
    """Compute paired statistical comparison between two experiments.

    Performs paired t-tests and Levene's tests to compare two experimental
    conditions across multiple metrics.

    Args:
        df: DataFrame with experimental results
        metadata: Metadata dict containing computed_display_names mapping
        exp_a: Display name of first experiment
        exp_b: Display name of second experiment
        metrics: Dict mapping metric names to column names
        fraction: Data fraction to filter on

    Returns:
        Dictionary mapping metric names to dicts with keys:
            - mean_a: Mean score for experiment A
            - mean_b: Mean score for experiment B
            - mean_diff: Mean difference (B - A)
            - p_mean: P-value from paired t-test
            - cohens_d: Cohen's d effect size
            - std_a: Standard deviation for experiment A
            - std_b: Standard deviation for experiment B
            - std_ratio: Ratio of standard deviations (B/A)
            - p_var: P-value from Levene's test for variance equality

    Raises:
        ValueError: If required experiments not found or data validation fails
    """
    computed_display_names = metadata["computed_display_names"]
    alias_to_raw = {alias: raw for raw, alias in computed_display_names.items()}

    exp_a_raw = alias_to_raw.get(exp_a, exp_a)
    exp_b_raw = alias_to_raw.get(exp_b, exp_b)

    data_a = df[(df["egid"] == exp_a_raw) & (df["fraction"] == fraction)]
    data_b = df[(df["egid"] == exp_b_raw) & (df["fraction"] == fraction)]

    if len(data_a) == 0:
        raise ValueError(f"No data found for {exp_a} experiment (raw: {exp_a_raw})")
    if len(data_b) == 0:
        raise ValueError(f"No data found for {exp_b} experiment (raw: {exp_b_raw})")

    data_a = data_a.sort_values("seed").reset_index(drop=True)
    data_b = data_b.sort_values("seed").reset_index(drop=True)

    if len(data_a) != len(data_b):
        raise ValueError(
            f"Mismatched sample sizes: {exp_a} has {len(data_a)} runs, {exp_b} has {len(data_b)} runs"
        )

    if not (data_a["seed"].values == data_b["seed"].values).all():
        raise ValueError(f"Seeds don't match between {exp_a} and {exp_b} experiments")

    results = {}

    for metric_name, metric_col in metrics.items():
        scores_a = data_a[metric_col].values
        scores_b = data_b[metric_col].values

        differences = scores_b - scores_a

        _, p_mean = stats.ttest_rel(scores_b, scores_a)

        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        cohens_d = mean_diff / std_diff

        std_a = np.std(scores_a, ddof=1)
        std_b = np.std(scores_b, ddof=1)
        std_ratio = std_b / std_a

        _, p_var = stats.levene(scores_a, scores_b, center="median")

        results[metric_name] = {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": mean_diff,
            "p_mean": p_mean,
            "cohens_d": cohens_d,
            "std_a": std_a,
            "std_b": std_b,
            "std_ratio": std_ratio,
            "p_var": p_var,
        }

    return results


def format_comparison_macros(
    results: dict, macro_prefix: str, metric_short_names: dict[str, str]
) -> str:
    """Format statistical results as LaTeX macro definitions.

    Args:
        results: Dictionary with statistical test results for each metric
        macro_prefix: Prefix for macro names (e.g., "batching")
        metric_short_names: Dict mapping full metric names to short names for macros

    Returns:
        LaTeX macro definitions as a string
    """
    macros = []

    for metric_name, short_name in metric_short_names.items():
        res = results[metric_name]

        # Convert to percentage points (assumes metrics in [0, 1] range like accuracy)
        mean_diff_pp = res["mean_diff"] * 100
        mean_diff_str = f"{mean_diff_pp:+.1f}".replace("+", r"+")

        p_mean_str = "<0.001" if res["p_mean"] < 0.001 else f"= {res['p_mean']:.3f}"
        p_var_str = "<0.01" if res["p_var"] < 0.01 else f"= {res['p_var']:.2f}"

        macros.append(
            f"\\newcommand{{\\{macro_prefix}{short_name}MeanP}}{{{p_mean_str}}}"
        )
        macros.append(
            f"\\newcommand{{\\{macro_prefix}{short_name}MeanD}}{{{res['cohens_d']:.2f}}}"
        )
        macros.append(
            f"\\newcommand{{\\{macro_prefix}{short_name}MeanDiff}}{{{mean_diff_str}}}"
        )
        macros.append(
            f"\\newcommand{{\\{macro_prefix}{short_name}VarP}}{{{p_var_str}}}"
        )

    return "\n".join(macros) + "\n"


def write_macros(latex_content: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_content)


def append_macros(latex_content: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write("\n" + latex_content)


def run_batching_validation(
    df: pd.DataFrame, metadata: dict, tex_macros_path: Path
) -> dict:
    """Run batching validation stats and write LaTeX macros.

    Compares BT baseline with RR-Agree-BT (BT loss with RR batching)
    to validate that the custom batching infrastructure does not affect baseline
    performance.

    Args:
        df: DataFrame with experimental results
        metadata: Metadata dict containing computed_display_names mapping
        tex_macros_path: Path to write LaTeX macros

    Returns:
        Dictionary with statistical test results
    """
    results = compute_paired_comparison(
        df,
        metadata,
        "BT (baseline)",
        "RR-Agree-BT",
        STANDARD_METRICS,
        1.0,
    )

    # Short names for LaTeX macro identifiers (remove spaces and special chars)
    metric_short_names = {
        "MultiPref": "Multipref",
        "RewardBench v2": "RewardBench",
    }

    latex_content = format_comparison_macros(results, "batching", metric_short_names)
    write_macros(latex_content, tex_macros_path)

    return results


def run_stated_validation(
    df: pd.DataFrame, metadata: dict, tex_macros_path: Path
) -> dict:
    results = compute_paired_comparison(
        df,
        metadata,
        "BT (baseline)",
        "RR-Stated",
        {"RewardBench v2": "eval/rewardbench_v2/overall/accuracy_score"},
        1.0,
    )

    metric_short_names = {
        "RewardBench v2": "RewardBench",
    }

    latex_content = format_comparison_macros(results, "stated", metric_short_names)
    append_macros(latex_content, tex_macros_path)

    return results


def run_agree_validation(
    df: pd.DataFrame, metadata: dict, tex_macros_path: Path
) -> dict:
    results = compute_paired_comparison(
        df,
        metadata,
        "BT (baseline)",
        "RR-Agree (ours)",
        {"RewardBench v2": "eval/rewardbench_v2/overall/accuracy_score"},
        1.0,
    )

    metric_short_names = {
        "RewardBench v2": "RewardBench",
    }

    latex_content = format_comparison_macros(results, "agree", metric_short_names)
    append_macros(latex_content, tex_macros_path)

    return results


def run_stated_global_validation(
    df: pd.DataFrame, metadata: dict, tex_macros_path: Path
) -> dict:
    results = compute_paired_comparison(
        df,
        metadata,
        "RR-Stated",
        "RR-Stated (global)",
        STANDARD_METRICS,
        1.0,
    )

    metric_short_names = {
        "MultiPref": "Multipref",
        "RewardBench v2": "RewardBench",
    }

    latex_content = format_comparison_macros(
        results, "statedGlobal", metric_short_names
    )
    append_macros(latex_content, tex_macros_path)

    return results


def run_meanpref_size_validation(
    df: pd.DataFrame, metadata: dict, tex_macros_path: Path
) -> dict:
    """Compare RR-MeanPref-6 vs RR-MeanPref (full) for size ablation analysis."""
    results = compute_paired_comparison(
        df,
        metadata,
        "RR-Consensus",
        "RR-Consensus-6",
        STANDARD_METRICS,
        1.0,
    )

    metric_short_names = {
        "MultiPref": "Multipref",
        "RewardBench v2": "RewardBench",
    }

    latex_content = format_comparison_macros(
        results, "meanPrefSixVsFull", metric_short_names
    )
    append_macros(latex_content, tex_macros_path)

    return results


def write_length_bucketing_singleton_macros(
    df: pd.DataFrame, metadata: dict, tex_macros_path: Path
) -> dict:
    computed_display_names = metadata["computed_display_names"]
    alias_to_raw = {alias: raw for raw, alias in computed_display_names.items()}

    experiments = {
        "Global": "RR-RT (spread2 global)",
        "NoBuckets": "RR-RT (annostrat spread2)",
        "BucketsTwo": "RR-RT (2 buckets spread2)",
        "BucketsFour": "RR-RT (4 buckets spread2)",
        "BucketsEight": "RR-RT (8 buckets spread2)",
        "BucketsSixteen": "RR-RT (16 buckets spread2)",
        "BucketsThirtyTwo": "RR-RT (32 buckets spread2)",
    }

    results = {}
    macros = []

    for key, display_name in experiments.items():
        exp_raw = alias_to_raw.get(display_name, display_name)
        data = df[(df["egid"] == exp_raw) & (df["fraction"] == 1.0)]

        non_singleton_mean = data["non_singleton_fraction"].mean()
        singleton_mean = 1.0 - non_singleton_mean

        results[display_name] = singleton_mean

        fmt = ".2f" if singleton_mean >= 0.01 else ".1g"

        macro_name = f"\\newcommand{{\\singletonFrac{key}}}{{{singleton_mean:{fmt}}}}"
        macros.append(macro_name)

    latex_content = "\n".join(macros) + "\n"
    append_macros(latex_content, tex_macros_path)

    return results


def run_bt_agree_validation(
    df: pd.DataFrame, metadata: dict, tex_macros_path: Path
) -> dict:
    """Compare BT (baseline) vs BT-Agree on both metrics."""
    results = compute_paired_comparison(
        df,
        metadata,
        "BT (baseline)",
        "BT-Agree",
        STANDARD_METRICS,
        1.0,
    )

    metric_short_names = {
        "MultiPref": "Multipref",
        "RewardBench v2": "RewardBench",
    }

    latex_content = format_comparison_macros(results, "btAgree", metric_short_names)
    append_macros(latex_content, tex_macros_path)

    return results


def run_bt_meanpref_validation(
    df: pd.DataFrame, metadata: dict, tex_macros_path: Path
) -> dict:
    """Compare BT (baseline) vs BT-MeanPref on both metrics."""
    results = compute_paired_comparison(
        df,
        metadata,
        "BT (baseline)",
        "BT-MeanPref",
        STANDARD_METRICS,
        1.0,
    )

    metric_short_names = {
        "MultiPref": "Multipref",
        "RewardBench v2": "RewardBench",
    }

    latex_content = format_comparison_macros(results, "btMeanPref", metric_short_names)
    append_macros(latex_content, tex_macros_path)

    return results
