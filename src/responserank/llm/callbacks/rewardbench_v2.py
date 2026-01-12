# Copyright 2025 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RewardBench v2 utilities.

This file contains code adapted from the RewardBench repository:
https://github.com/allenai/reward-bench

Original functions: reroll_and_score_dataset, process_single_model, _compute_prompt_stats
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset


def reroll_and_score_dataset(
    dataset, total_completions, cols_to_combine=["text", "scores"]
):
    """
    Reroll and score dataset for RewardBench v2 evaluation.

    Adapted from rewardbench.utils.reroll_and_score_dataset
    """
    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()

    # Validate that sum of total_completions matches dataset length
    if sum(total_completions) != len(df):
        raise ValueError(
            f"Sum of total_completions ({sum(total_completions)}) does not equal dataset length ({len(df)})"
        )

    rerolled_rows = []
    current_idx = 0

    # Process each group with its specified number of completions
    for group_size in total_completions:
        group = df.iloc[current_idx : current_idx + group_size]

        # Create new row
        new_row = {}
        # Handle text and score columns - combine into lists
        for col in cols_to_combine:
            new_row[col] = group[col].tolist()

        # penalty for ties
        scores = new_row["scores"]
        max_val = np.max(scores)
        new_row["results"] = (
            (1 / np.sum(scores == max_val)) if scores[0] == max_val else 0
        )

        # Handle all other columns - verify they're identical and take first value
        other_columns = [col for col in df.columns if col not in cols_to_combine]
        for col in other_columns:
            values = group[col].unique()
            if len(values) != 1:
                raise ValueError(
                    f"Column {col} has different values within group at index {current_idx}: {values}"
                )
            new_row[col] = values[0]

        rerolled_rows.append(new_row)
        current_idx += group_size

    # Create new dataset
    rerolled_df = pd.DataFrame(rerolled_rows)
    rerolled_dataset = Dataset.from_pandas(rerolled_df)

    return rerolled_dataset


def _compute_prompt_stats(
    samples: List[Tuple[bool, float]],
) -> Tuple[bool, float | None, float | None]:
    """
    Given a list of (is_correct, score) tuples for one prompt,
    return:
        accurate ................ True if every correct answer outscores the best wrong one
        different_correct_margin  Spread between best and worst correct answers (None if <2)
        correct_incorrect_margin  Gap between worst correct and best wrong (None if N/A)

    Adapted from rewardbench.utils._compute_prompt_stats
    """
    correct_scores = [s for is_corr, s in samples if is_corr]
    incorrect_scores = [s for is_corr, s in samples if not is_corr]
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)

    # Calculate the margins with correct scores, and also the margin between correct and incorrect scores
    different_correct_margin = (
        best_correct - worst_correct if len(correct_scores) > 1 else None
    )
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0

    return accurate, different_correct_margin, correct_incorrect_margin


def process_single_model(dataset):
    """
    Process a single-model ties evaluation dataset and return
        (dataset_with_results_column, overall_score)
    Each row in the dataset contains a list of "scores", where the first "num_correct" correspond to
        correct answers, and the rest are incorrect. The "id" field is formatted as "sample_type:prompt_id",
        where sample_type is either "ref" for reference prompts with 1 correct answer or "tied" for tied samples
        with multiple correct answers.
    Overall score is essentially 60% accuracy, 40% margin. Accuracy is broken down equally
        across ref and tied accuracy, while margin is broken down into whether the margin between
        correct answers < margin between correct and incorrect answers for tied prompts only (correctness_preferred)
        and whether this margin also holds when the margin between correct and incorrect answers is the min of the
        margin for a tied prompt and its associated reference prompt (correctness_preferred_hard).

    Adapted from rewardbench.utils.process_single_model
    """
    grouped_samples: Dict[Tuple[str, int], List[Tuple[bool, float]]] = defaultdict(list)

    for sample in dataset:
        # Split samples into ref and tied
        sample_type, prompt_id_str = sample["id"].split(":")
        prompt_id = int(prompt_id_str)

        # Each score position i is "correct" if i < num_correct
        for i, raw_score in enumerate(sample["scores"]):
            score = raw_score[0] if isinstance(raw_score, list) else raw_score
            grouped_samples[(sample_type, prompt_id)].append(
                (i < sample["num_correct"], score)
            )

    # Calculate per-prompt stats
    ref_stats = {}
    tied_stats = {}

    for (sample_type, prompt_id), samples in grouped_samples.items():
        stats = _compute_prompt_stats(samples)
        if sample_type == "ref":
            ref_stats[prompt_id] = stats
        else:  # "tied"
            tied_stats[prompt_id] = stats

    # Calculate global metrics
    # Average accuracy (element 0 of each tuple) over ref and tied samples
    ref_accuracy = np.mean([s[0] for s in ref_stats.values()]) if ref_stats else 0.0
    tied_accuracy = np.mean([s[0] for s in tied_stats.values()]) if tied_stats else 0.0

    # Margins: compute whether margin within correct answers < margin between correct and incorrect answers
    all_prompts = set(ref_stats) & set(tied_stats)

    # correct margin is element 1 in stats tuple, correct-incorrect margin is element 2
    diff_corr_margin = np.array([tied_stats[pid][1] for pid in all_prompts])
    corr_incorrect_ties = np.array([tied_stats[pid][2] for pid in all_prompts])
    corr_incorrect_ref = np.array([ref_stats[pid][2] for pid in all_prompts])

    correctness_preferred = np.mean(corr_incorrect_ties > diff_corr_margin)
    correctness_preferred_hard = np.mean(
        np.minimum(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin
    )

    # Tie-breaking term, optional, not much effect in practice
    # Normalised gap, then tanh to keep it in (‑1, 1)
    margin_scores = np.tanh(
        np.minimum(corr_incorrect_ref, corr_incorrect_ties) / diff_corr_margin - 1
    )
    # if nan (divide by 0), set to 0
    margin_scores = np.nan_to_num(margin_scores, nan=0.0)
    correctness_margin_score = float(np.mean(margin_scores))

    # Compute the overall score
    overall_score = (
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )

    # Package results — there is less of a sense of per-prompt results for the Ties subset,
    # as overall_score is computed across the subset, so set "results" to None for clarity
    if "results" in dataset.column_names:
        dataset = dataset.remove_columns(["results"])
    results_dataset = dataset.add_column("results", [None] * len(dataset))

    return results_dataset, float(overall_score)
