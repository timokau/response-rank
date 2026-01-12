"""Dataset analysis module for MultiPref.

Analyzes raw dataset properties (correlations, distributions) and outputs
LaTeX macros and figures for the paper.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from scipy import stats

from responserank.llm.data.datasets.multipref import (
    filter_valid_annotations,
)
from responserank.llm.data.stratification import (
    _calculate_example_length,
    compute_within_stratum_rt_agreement_correlations,
)


def _get_preference_ordinal(pref: str, slight_preference_value: float) -> float:
    """Convert preference string to ordinal value."""
    if pref == "A-is-clearly-better":
        return 1.0
    elif pref == "A-is-slightly-better":
        return slight_preference_value
    elif pref == "Tie":
        return 0.0
    elif pref == "B-is-slightly-better":
        return -slight_preference_value
    elif pref == "B-is-clearly-better":
        return -1.0
    raise ValueError(f"Unknown preference: {pref}")


def _compute_correlation(annotations: List[Dict], field1: str, field2: str) -> Dict:
    """Compute Spearman correlations between two fields globally and within annotators.

    Args:
        annotations: List of annotation records
        field1: First field name (e.g., 'response_time')
        field2: Second field name (e.g., 'consensus_score')

    Returns:
        Dictionary with global_corr, global_p, and within_annotator_median
    """
    values1 = [a[field1] for a in annotations]
    values2 = [a[field2] for a in annotations]

    global_corr, global_p = stats.spearmanr(values1, values2)

    by_annotator = defaultdict(list)
    for ann in annotations:
        by_annotator[ann["annotator_id"]].append(ann)

    within_annotator_corrs = []
    for _, ann_list in by_annotator.items():
        if len(ann_list) < 3:
            continue

        ann_values1 = [a[field1] for a in ann_list]
        ann_values2 = [a[field2] for a in ann_list]

        corr, _ = stats.spearmanr(ann_values1, ann_values2)
        # May be NaN if the annotator has no variance in one of the fields.
        if not np.isnan(corr):
            within_annotator_corrs.append(corr)

    return {
        "global_corr": global_corr,
        "global_p": global_p,
        "within_annotator_median": np.median(within_annotator_corrs),
    }


def load_multipref_annotations() -> List[Dict]:
    """Load and extract annotation-level data from MultiPref.

    Returns:
        List of annotation records with keys:
            - annotator_id: str
            - response_time: float (seconds)
            - text_length: int (characters)
            - consensus_score: float (-1 to +1, chosen-relative)
    """
    ds = load_dataset("allenai/multipref", "default")["train"]

    annotations = []

    for item in ds:
        all_annotations = (
            item["normal_worker_annotations"] + item["expert_worker_annotations"]
        )

        valid_annotations = filter_valid_annotations(all_annotations)

        slight_preference_value = 0.5
        mean_preference_a_relative = sum(
            _get_preference_ordinal(a["overall_pref"], slight_preference_value)
            for a in all_annotations
        ) / len(all_annotations)

        conversation_a = [
            {"role": "user", "content": item["text"]},
            {"role": "assistant", "content": item["completion_a"]},
        ]
        conversation_b = [
            {"role": "user", "content": item["text"]},
            {"role": "assistant", "content": item["completion_b"]},
        ]

        example = {"chosen": conversation_a, "rejected": conversation_b}
        text_length = _calculate_example_length(example)

        for annotation in valid_annotations:
            pref = annotation.get("overall_pref", "")
            a_is_chosen = pref in ["A-is-clearly-better", "A-is-slightly-better"]
            consensus_score = (
                mean_preference_a_relative
                if a_is_chosen
                else -mean_preference_a_relative
            )

            annotations.append(
                {
                    "annotator_id": annotation["evaluator"],
                    "response_time": annotation["time_spent"],
                    "text_length": text_length,
                    "consensus_score": consensus_score,
                }
            )

    return annotations


def compute_rt_length_correlations(annotations: List[Dict]) -> Dict:
    """Compute RT-length correlations within annotators and globally."""
    return _compute_correlation(annotations, "response_time", "text_length")


def compute_rt_consensus_correlations(annotations: List[Dict]) -> Dict:
    """Compute RT-consensus correlations within annotators and globally."""
    return _compute_correlation(annotations, "response_time", "consensus_score")


def compute_length_consensus_correlations(annotations: List[Dict]) -> Dict:
    """Compute length-consensus correlations within annotators and globally."""
    return _compute_correlation(annotations, "text_length", "consensus_score")


def compute_rt_consensus_within_length_buckets(
    annotations: List[Dict], n_buckets: int
) -> Dict:
    """Compute RT-consensus correlations within annotator+length strata.

    Args:
        annotations: List of annotation records from load_multipref_annotations
        n_buckets: Number of length buckets to use for stratification

    Returns:
        Dictionary with correlation statistics from within-stratum analysis
    """
    lengths = [a["text_length"] for a in annotations]
    percentiles = np.linspace(0, 100, n_buckets + 1)
    boundaries = np.percentile(lengths, percentiles)
    boundaries[-1] += 1.0

    formatted_examples = []
    for ann in annotations:
        length = ann["text_length"]
        bucket = n_buckets - 1
        for b in range(n_buckets):
            if boundaries[b] <= length < boundaries[b + 1]:
                bucket = b
                break

        partition_id = f"{ann['annotator_id']}_{bucket}"

        formatted_examples.append(
            {
                "partition_id": partition_id,
                "rank": ann["response_time"],
                "extra": {"agreement_score": ann["consensus_score"]},
            }
        )

    return compute_within_stratum_rt_agreement_correlations(formatted_examples)


def format_correlation_macros(
    rt_length_stats: Dict,
    rt_consensus_stats: Dict,
    length_consensus_stats: Dict,
    rt_consensus_length_strat_stats: Dict,
) -> str:
    """Format correlation statistics as LaTeX macros.

    Only generates macros that are actually used in the paper.

    Args:
        rt_length_stats: Output from compute_rt_length_correlations
        rt_consensus_stats: Output from compute_rt_consensus_correlations
        length_consensus_stats: Output from compute_length_consensus_correlations
        rt_consensus_length_strat_stats: Output from compute_rt_consensus_within_length_buckets

    Returns:
        LaTeX macro definitions as string
    """
    macros = []

    def format_corr(val):
        return f"{val:+.2f}"

    macros.append(
        f"\\newcommand{{\\rtLengthWithinAnnotatorMedian}}{{{format_corr(rt_length_stats['within_annotator_median'])}}}"
    )
    macros.append(
        f"\\newcommand{{\\rtConsensusGlobalCorr}}{{{format_corr(rt_consensus_stats['global_corr'])}}}"
    )
    macros.append(
        f"\\newcommand{{\\rtConsensusWithinAnnotatorMedian}}{{{format_corr(rt_consensus_stats['within_annotator_median'])}}}"
    )
    macros.append(
        f"\\newcommand{{\\rtConsensusLengthStratMedian}}{{{format_corr(rt_consensus_length_strat_stats['stratum_rt_agreement_median_correlation'])}}}"
    )
    macros.append(
        f"\\newcommand{{\\lengthConsensusWithinAnnotatorMedian}}{{{format_corr(length_consensus_stats['within_annotator_median'])}}}"
    )

    return "\n".join(macros) + "\n"


def write_macros(latex_content: str, output_path: Path):
    """Write LaTeX macros to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_content)


def generate_dataset_stats(output_path: Path):
    """Generate dataset statistics and write LaTeX macros.

    Args:
        output_path: Path where to write the LaTeX macros file
    """
    print("Loading MultiPref annotations...")
    annotations = load_multipref_annotations()
    print(f"Loaded {len(annotations)} annotations")

    print("Computing RT-length correlations...")
    rt_length_stats = compute_rt_length_correlations(annotations)

    print("Computing RT-consensus correlations...")
    rt_consensus_stats = compute_rt_consensus_correlations(annotations)

    print("Computing length-consensus correlations...")
    length_consensus_stats = compute_length_consensus_correlations(annotations)

    print("Computing RT-consensus correlations within length buckets...")
    rt_consensus_length_strat_stats = compute_rt_consensus_within_length_buckets(
        annotations, 8
    )

    print("Formatting LaTeX macros...")
    macros = format_correlation_macros(
        rt_length_stats,
        rt_consensus_stats,
        length_consensus_stats,
        rt_consensus_length_strat_stats,
    )

    print(f"Writing macros to {output_path}...")
    write_macros(macros, output_path)
    print(f"Dataset stats saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MultiPref dataset")
    parser.add_argument(
        "--output-macros",
        type=Path,
        default=Path("paper/figures/dataset_stats.tex"),
        help="Output path for LaTeX macros",
    )
    args = parser.parse_args()
    generate_dataset_stats(args.output_macros)


if __name__ == "__main__":
    main()
