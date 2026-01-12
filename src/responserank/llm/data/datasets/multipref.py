from typing import Dict, List, Tuple

import numpy as np

from datasets import Dataset, load_dataset

from .base_dataset import BaseDataset


def determine_chosen_rejected(
    preference, conversation_a, conversation_b, model_a, model_b
):
    """Determine chosen/rejected conversations based on preference.

    Args:
        preference: Preference string (e.g., "A-is-clearly-better")
        conversation_a: Conversation A (list of messages)
        conversation_b: Conversation B (list of messages)
        model_a: Model that generated conversation A
        model_b: Model that generated conversation B

    Returns:
        Tuple of (chosen_conversation, rejected_conversation, chosen_model, rejected_model)
    """
    if preference in ["A-is-clearly-better", "A-is-slightly-better"]:
        return conversation_a, conversation_b, model_a, model_b
    elif preference in ["B-is-clearly-better", "B-is-slightly-better"]:
        return conversation_b, conversation_a, model_b, model_a
    else:
        raise ValueError(f"Invalid preference: {preference}")


def extract_preference_strength(preference):
    """Extract strength from preference string.

    Args:
        preference: Preference string

    Returns:
        Ordinal preference strength (1=slight, 2=clear)
    """
    if "clearly" in preference:
        return 2
    elif "slightly" in preference:
        return 1
    else:
        raise ValueError(f"Unknown preference strength: {preference}")


def get_preference_ordinal(overall_pref, slight_preference_value):
    """Convert preference string to ordinal value.

    Args:
        overall_pref: Preference string (e.g., "A-is-clearly-better")
        slight_preference_value: Weight for slight preferences (typically 0.5)

    Returns:
        Float ordinal value where:
            +1.0 = A clearly better
            +slight_preference_value = A slightly better
            0.0 = Tie
            -slight_preference_value = B slightly better
            -1.0 = B clearly better
    """
    if overall_pref == "A-is-clearly-better":
        return 1.0
    elif overall_pref == "A-is-slightly-better":
        return slight_preference_value
    elif overall_pref == "Tie":
        return 0.0
    elif overall_pref == "B-is-slightly-better":
        return -slight_preference_value
    elif overall_pref == "B-is-clearly-better":
        return -1.0
    else:
        raise ValueError(f"Unknown preference: {overall_pref}")


def get_all_annotations(item):
    """Get all annotations from an item.

    Args:
        item: Dataset item containing annotation fields

    Returns:
        List of all annotations
    """
    normal_annotations = item.get("normal_worker_annotations", [])
    expert_annotations = item.get("expert_worker_annotations", [])
    return normal_annotations + expert_annotations


def filter_valid_annotations(annotations):
    """Filter valid annotations that have preference and time data."""
    return [
        ann
        for ann in annotations
        if ann.get("overall_pref") and ann.get("overall_pref") != "Tie"
    ]


def count_preference_strengths(annotations):
    """
    Count preference strengths from annotations.

    Args:
        annotations: List of annotation dictionaries

    Returns:
        Tuple of (pref_a_count, pref_b_count, clearly_better_a, slightly_better_a,
                 clearly_better_b, slightly_better_b, tie_count)
    """
    pref_a_count = sum(
        1
        for a in annotations
        if a.get("overall_pref", "") in ["A-is-clearly-better", "A-is-slightly-better"]
    )
    pref_b_count = sum(
        1
        for a in annotations
        if a.get("overall_pref", "") in ["B-is-clearly-better", "B-is-slightly-better"]
    )

    tie_count = sum(1 for a in annotations if a.get("overall_pref", "") in ["Tie"])

    clearly_better_a = sum(
        1 for a in annotations if a.get("overall_pref", "") == "A-is-clearly-better"
    )
    slightly_better_a = sum(
        1 for a in annotations if a.get("overall_pref", "") == "A-is-slightly-better"
    )
    clearly_better_b = sum(
        1 for a in annotations if a.get("overall_pref", "") == "B-is-clearly-better"
    )
    slightly_better_b = sum(
        1 for a in annotations if a.get("overall_pref", "") == "B-is-slightly-better"
    )

    return (
        pref_a_count,
        pref_b_count,
        clearly_better_a,
        slightly_better_a,
        clearly_better_b,
        slightly_better_b,
        tie_count,
    )


def calculate_agreement_score(
    clearly_better_a,
    slightly_better_a,
    clearly_better_b,
    slightly_better_b,
    tie,
    ties_in_agreement_total: bool,
    slight_preference_value: float,
):
    """
    Calculate normalized agreement score based on annotation distributions.

    Args:
        clearly_better_a: Count of "A-is-clearly-better"
        slightly_better_a: Count of "A-is-slightly-better"
        clearly_better_b: Count of "B-is-clearly-better"
        slightly_better_b: Count of "B-is-slightly-better"
        tie: Count of ties
        ties_in_agreement_total: Whether to include ties in total calculation

    Returns:
        Normalized agreement score as abs(weighted_a - weighted_b) / total

    Raises:
        ValueError: If total annotation count is 0
    """
    total = clearly_better_a + slightly_better_a + clearly_better_b + slightly_better_b
    if ties_in_agreement_total:
        total += tie

    if total == 0:
        return 0.0

    clearly_weight = 1.0
    slightly_weight = slight_preference_value

    weighted_a = clearly_weight * clearly_better_a + slightly_weight * slightly_better_a
    weighted_b = clearly_weight * clearly_better_b + slightly_weight * slightly_better_b

    return abs(weighted_a - weighted_b) / total


class StatedStrengthProcessor:
    """Processor that directly uses stated strengths to determine the ranking"""

    def process_dataset(self, examples, rng) -> Tuple[List[Dict], Dict]:
        for i, example in enumerate(examples):
            if example.get("extra", {}).get("preference_strength") is None:
                raise ValueError(f"Example {i} missing 'preference_strength' field.")

        processed_examples = []
        ranks = []

        for example in examples:
            preference_strength = example["extra"]["preference_strength"]
            if preference_strength == 2:
                rank = 0.0
            elif preference_strength == 1:  # slight preference
                rank = 1.0
            else:
                raise ValueError(f"Invalid preference strength: {preference_strength}")

            new_example = example.copy()
            new_example["extra"] = example["extra"].copy()
            new_example["extra"]["rank_original"] = example["rank"]
            new_example["rank"] = rank

            processed_examples.append(new_example)
            ranks.append(rank)

        info_dict = {
            "rank_mean": np.mean(ranks),
            "rank_std": np.std(ranks),
            "rank_min": np.min(ranks),
            "rank_max": np.max(ranks),
        }

        return processed_examples, info_dict


class AgreementProcessor:
    """Processor that sets rank to the negative agreement score"""

    def process_dataset(self, examples, rng) -> Tuple[List[Dict], Dict]:
        processed_examples = []
        ranks = []

        for i, example in enumerate(examples):
            extra = example.get("extra", {})

            if "agreement_score" not in extra:
                raise ValueError(
                    f"Example {i} (comparison_id: {example['comparison_id']}) "
                    f"missing required 'agreement_score' field for AgreementProcessor. "
                    f"Available extra fields: {list(extra.keys())}"
                )

            new_example = example.copy()
            new_example["extra"] = example["extra"].copy()
            new_example["extra"]["rank_original"] = example["rank"]
            new_example["rank"] = -extra["agreement_score"]

            processed_examples.append(new_example)
            ranks.append(new_example["rank"])

        info_dict = {
            "rank_mean": np.mean(ranks),
            "rank_std": np.std(ranks),
            "rank_min": np.min(ranks),
            "rank_max": np.max(ranks),
            "unique_ranks": len(set(ranks)),
        }

        return processed_examples, info_dict


class ConsensusProcessor:
    """Processor that sets rank to signed mean preference (consensus-based strength)."""

    def process_dataset(self, examples, rng) -> Tuple[List[Dict], Dict]:
        processed_examples = []
        ranks = []

        for i, example in enumerate(examples):
            extra = example.get("extra", {})

            if "mean_preference" not in extra:
                raise ValueError(
                    f"Example {i} (comparison_id: {example['comparison_id']}) "
                    f"missing required 'mean_preference' field for ConsensusProcessor. "
                    f"Available extra fields: {list(extra.keys())}"
                )

            mean_pref = extra["mean_preference"]
            original_a_is_chosen = extra["original_a_is_chosen"]

            directional_pref = mean_pref if original_a_is_chosen else -mean_pref

            new_example = example.copy()
            new_example["extra"] = example["extra"].copy()
            new_example["extra"]["rank_original"] = example["rank"]
            # Negative ranks indicate strong preferences, positive ranks weak preferences.
            new_example["rank"] = -directional_pref

            processed_examples.append(new_example)
            ranks.append(new_example["rank"])

        info_dict = {
            "rank_mean": np.mean(ranks),
            "rank_std": np.std(ranks),
            "rank_min": np.min(ranks),
            "rank_max": np.max(ranks),
            "unique_ranks": len(set(ranks)),
        }

        return processed_examples, info_dict


class MultiPrefDataset(BaseDataset):
    """Dataset implementation for AllenAI MultiPref dataset."""

    def __init__(
        self,
        train_annotation_aggregator,
        test_annotation_aggregator,
        slight_preference_value: float,
        ties_in_agreement_total: bool,
    ):
        """Initialize MultiPrefDataset.

        Args:
            train_annotation_aggregator: AnnotationAggregator for training data
            test_annotation_aggregator: AnnotationAggregator for test data
            slight_preference_value: Weight assigned to "slight" preferences when computing agreement scores
            ties_in_agreement_total: Whether to include ties in agreement total calculation
        """
        self.train_annotation_aggregator = train_annotation_aggregator
        self.test_annotation_aggregator = test_annotation_aggregator
        self.ties_in_agreement_total = ties_in_agreement_total
        self.slight_preference_value = slight_preference_value

    def load_raw_data(self):
        return load_dataset("allenai/multipref", "default")["train"]

    def create_test_split(self, dataset, rng, test_size):
        """Create a test split of comparison IDs from MultiPref dataset.

        Args:
            dataset: The dataset to get comparison IDs from
            rng: Random number generator to use
            test_size: Number of comparisons for test set

        Returns:
            Set of comparison IDs for test set
        """
        print("Creating test split...")
        all_comparison_ids = set()

        for item in dataset:
            comparison_id = item["comparison_id"]
            all_comparison_ids.add(comparison_id)

        # Convert to list for shuffling (sorted for deterministic behavior)
        comparison_ids = sorted(list(all_comparison_ids))

        rng.shuffle(comparison_ids)

        # Select test IDs
        test_comparison_ids = set(comparison_ids[:test_size])

        print(f"Created {len(test_comparison_ids)} test comparison IDs")
        return test_comparison_ids

    def get_dataset_name(self):
        return "MultiPref"

    def _compute_item_agreement_stats(self, all_annotations):
        """Compute agreement statistics for a single comparison item.

        Args:
            all_annotations: List of annotation dictionaries for one comparison

        Returns:
            Dict with computed agreement fields to add to example["extra"]
        """
        # Prepare annotations in format expected by count_preference_strengths
        formatted_annotations = []
        for annotation in all_annotations:
            formatted_annotation = {
                "overall_pref": annotation["overall_pref"],
                "time_spent": annotation["time_spent"],
            }
            formatted_annotations.append(formatted_annotation)

        # Count preference strengths
        (
            pref_a_count,
            pref_b_count,
            clearly_better_a,
            slightly_better_a,
            clearly_better_b,
            slightly_better_b,
            tie,
        ) = count_preference_strengths(formatted_annotations)

        # Calculate agreement score
        agreement_score = calculate_agreement_score(
            clearly_better_a,
            slightly_better_a,
            clearly_better_b,
            slightly_better_b,
            tie,
            ties_in_agreement_total=self.ties_in_agreement_total,
            slight_preference_value=self.slight_preference_value,
        )

        # Calculate mean preference (A-relative)
        mean_pref = sum(
            get_preference_ordinal(ann["overall_pref"], self.slight_preference_value)
            for ann in all_annotations
        ) / len(all_annotations)

        return {
            "clearly_better_a_count": clearly_better_a,
            "slightly_better_a_count": slightly_better_a,
            "clearly_better_b_count": clearly_better_b,
            "slightly_better_b_count": slightly_better_b,
            "agreement_score": agreement_score,
            "mean_preference": mean_pref,
        }

    def to_comparison_dataset(self, raw_data, rng):
        """Convert raw MultiPref data to comparison dataset."""
        comparison_dataset = []

        for item in raw_data:
            all_annotations = (
                item["normal_worker_annotations"] + item["expert_worker_annotations"]
            )

            agreement_stats = self._compute_item_agreement_stats(all_annotations)

            conversation_a = [
                {"role": "user", "content": item["text"]},
                {"role": "assistant", "content": item["completion_a"]},
            ]
            conversation_b = [
                {"role": "user", "content": item["text"]},
                {"role": "assistant", "content": item["completion_b"]},
            ]

            comparison = {
                "comparison_id": item["comparison_id"],
                "conversation_a": conversation_a,
                "conversation_b": conversation_b,
                "model_a": item["model_a"],
                "model_b": item["model_b"],
                "agreement_stats": agreement_stats,
                "annotations": [],
            }

            for annotation in all_annotations:
                ann_data = {
                    "overall_pref": annotation["overall_pref"],
                    "time_spent": annotation["time_spent"],
                    "evaluator": annotation["evaluator"],
                    "is_tie": annotation["overall_pref"] == "Tie",
                    "preference_strength": extract_preference_strength(
                        annotation["overall_pref"]
                    )
                    if annotation["overall_pref"] != "Tie"
                    else None,
                    "preference_ordinal": get_preference_ordinal(
                        annotation["overall_pref"], self.slight_preference_value
                    ),
                }
                comparison["annotations"].append(ann_data)

            comparison_dataset.append(comparison)

        return Dataset.from_list(comparison_dataset)

    def create_example_from_annotation(self, item, annotation, rng):
        """Create a standardized training example from a dataset item and annotation.

        Args:
            item: Dataset item with text, completions, models, and metadata
            annotation: Single annotation with preference and time data
            rng: Random number generator for tie-breaking (required for ties)

        Returns:
            Dictionary with standardized format
        """
        preference = annotation["overall_pref"]
        time_spent = annotation["time_spent"]

        if preference == "Tie":
            # For ties, randomly assign which completion becomes chosen/rejected
            if rng.random() < 0.5:
                chosen_completion = item["completion_a"]
                rejected_completion = item["completion_b"]
                chosen_model = item["model_a"]
                rejected_model = item["model_b"]
            else:
                chosen_completion = item["completion_b"]
                rejected_completion = item["completion_a"]
                chosen_model = item["model_b"]
                rejected_model = item["model_a"]
            preference_strength = 0  # No preference strength for ties
        else:
            chosen_completion, rejected_completion, chosen_model, rejected_model = (
                determine_chosen_rejected(
                    preference,
                    item["completion_a"],
                    item["completion_b"],
                    item["model_a"],
                    item["model_b"],
                )
            )
            preference_strength = extract_preference_strength(preference)

        # Use conversation format with implicit prompts
        chosen = [
            {"role": "user", "content": item["text"]},
            {"role": "assistant", "content": chosen_completion},
        ]
        rejected = [
            {"role": "user", "content": item["text"]},
            {"role": "assistant", "content": rejected_completion},
        ]
        example = {
            "chosen": chosen,
            "rejected": rejected,
            # Response times serve as ranks (not technically rank, but the loss only cares about the order).
            "rank": time_spent,
            "comparison_id": item["comparison_id"],
            "partition_id": -1,  # Will be set later by stratifier
            "extra": {
                "response_time": time_spent,
                "preference_strength": preference_strength,
                "overall_pref": preference,
                "is_tie": preference == "Tie",
                "evaluator": annotation.get("evaluator", "unknown"),
                "evaluator_type": "expert"
                if annotation in item.get("expert_worker_annotations", [])
                else "normal",
                "chosen_model": chosen_model,
                "rejected_model": rejected_model,
                "prompt_id": item["prompt_id"],
                "source": item["source"],
                "category": item["category"],
                "subject_study": item["subject_study"],
                "highest_level_degree": item["highest_level_degree"],
                "timestamp": annotation.get("timestamp"),
            },
        }

        return example
