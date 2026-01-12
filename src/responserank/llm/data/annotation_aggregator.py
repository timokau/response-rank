from abc import ABC, abstractmethod
from typing import Dict, List

import wandb
from datasets import load_dataset

from .datasets.multipref import (
    determine_chosen_rejected,
)


class AnnotationAggregator(ABC):
    """Abstract base class for aggregating annotations from multi-annotator data."""

    def __init__(self, bt_target: str):
        if bt_target not in ["hard", "agreement", "mean_preference"]:
            raise ValueError(
                f"bt_target must be 'hard', 'agreement', or 'mean_preference', got {bt_target}"
            )
        self.bt_target = bt_target

    @abstractmethod
    def aggregate_annotations(self, comparisons: List[Dict], rng) -> List[Dict]:
        """Aggregate annotations from multi-annotator comparisons to one per comparison.

        Args:
            comparisons: List of comparison objects with nested annotations
            rng: Random number generator

        Returns:
            List of examples with one annotation aggregated per comparison
        """
        pass

    def _create_annotation_example(self, comparison, annotation, rng):
        """Create annotation dataset example from comparison and selected annotation."""
        if annotation["is_tie"]:
            # Random assignment for ties
            if rng.random() < 0.5:
                chosen_conversation = comparison["conversation_a"]
                rejected_conversation = comparison["conversation_b"]
                chosen_model = comparison["model_a"]
                rejected_model = comparison["model_b"]
                a_is_chosen = True
            else:
                chosen_conversation = comparison["conversation_b"]
                rejected_conversation = comparison["conversation_a"]
                chosen_model = comparison["model_b"]
                rejected_model = comparison["model_a"]
                a_is_chosen = False
        else:
            chosen_conversation, rejected_conversation, chosen_model, rejected_model = (
                determine_chosen_rejected(
                    annotation["overall_pref"],
                    comparison["conversation_a"],
                    comparison["conversation_b"],
                    comparison["model_a"],
                    comparison["model_b"],
                )
            )

            a_is_chosen = chosen_conversation is comparison["conversation_a"]

        all_annotations = comparison["annotations"]

        if self.bt_target == "hard":
            bt_target_value = 1.0
        elif self.bt_target == "agreement":
            # Map agreement [0, 1] to target [0.5, 1.0]
            # Low agreement (0) -> 0.5 (neutral, minimal gradient)
            # High agreement (1) -> 1.0 (strong preference for chosen)
            bt_target_value = (
                0.5 + 0.5 * comparison["agreement_stats"]["agreement_score"]
            )
        elif self.bt_target == "mean_preference":
            # preference_ordinal is A-relative: +1 (A clearly better) to -1 (B clearly better)
            mean_pref = sum(ann["preference_ordinal"] for ann in all_annotations) / len(
                all_annotations
            )

            # Negate if B is chosen (to make it chosen-relative), then map [-1,1] -> [0,1]
            directional_pref = mean_pref if a_is_chosen else -mean_pref
            bt_target_value = (directional_pref + 1.0) / 2.0
        else:
            raise ValueError(f"Invalid bt_target: {self.bt_target}")

        example = {
            "chosen": chosen_conversation,
            "rejected": rejected_conversation,
            "comparison_id": comparison["comparison_id"],
            "rank": annotation["time_spent"],
            "partition_id": -1,
            "is_tie": annotation["is_tie"],
            "bt_target": bt_target_value,
            "extra": {
                "evaluator": annotation["evaluator"],
                "preference_strength": annotation["preference_strength"],
                "chosen_model": chosen_model,
                "rejected_model": rejected_model,
                "original_a_is_chosen": a_is_chosen,
                **comparison["agreement_stats"],
            },
        }

        if a_is_chosen:
            example["input_ids_chosen"] = comparison["input_ids_a"]
            example["attention_mask_chosen"] = comparison["attention_mask_a"]
            example["input_ids_rejected"] = comparison["input_ids_b"]
            example["attention_mask_rejected"] = comparison["attention_mask_b"]
        else:
            example["input_ids_chosen"] = comparison["input_ids_b"]
            example["attention_mask_chosen"] = comparison["attention_mask_b"]
            example["input_ids_rejected"] = comparison["input_ids_a"]
            example["attention_mask_rejected"] = comparison["attention_mask_a"]

        return example


class RandomChoiceAggregator(AnnotationAggregator):
    """Randomly aggregates one annotation per comparison by random choice."""

    def __init__(self, bt_target: str):
        super().__init__(bt_target)

    def aggregate_annotations(self, comparisons: List[Dict], rng) -> List[Dict]:
        """Convert comparison dataset to annotation dataset by random selection.

        Args:
            comparisons: List of comparison objects with nested annotations
            rng: Random number generator

        Returns:
            Annotation dataset with one example per comparison
        """
        annotation_dataset = []
        total_annotations = 0
        multi_annotation_count = 0

        for comparison in comparisons:
            annotations = comparison["annotations"]

            total_annotations += len(annotations)
            if len(annotations) > 1:
                multi_annotation_count += 1

            # Choose random annotation
            chosen_annotation = rng.choice(annotations)

            # Create annotation dataset example
            example = self._create_annotation_example(
                comparison, chosen_annotation, rng
            )
            annotation_dataset.append(example)

        min_annotations = min(
            len(comparison["annotations"]) for comparison in comparisons
        )
        max_annotations = max(
            len(comparison["annotations"]) for comparison in comparisons
        )

        print(
            f"[RandomChoiceAggregator]: min annotations per comparison: {min_annotations}"
        )
        print(
            f"[RandomChoiceAggregator]: max annotations per comparison: {max_annotations}"
        )
        print(
            f"[RandomChoiceAggregator]: comparisons with multiple annotations: {multi_annotation_count}/{len(comparisons)}"
        )

        if max_annotations < 2:
            raise ValueError(
                "RandomChoiceAggregator found no comparison with multiple annotations. "
                "All comparisons have only single annotations."
            )

        return annotation_dataset


class MultiprefBinarizedAggregator(AnnotationAggregator):
    """Aggregator that looks up preferences from human_overall_binarized dataset."""

    def __init__(self, bt_target: str):
        super().__init__(bt_target)

        print("Loading MultiprefBinarized dataset for test data aggregation")
        bin_ds = load_dataset("allenai/multipref", "human_overall_binarized")["train"]
        self.bin_lookup = {item["comparison_id"]: item for item in bin_ds}
        print(f"Loaded {len(self.bin_lookup)} binarized comparisons")

    def aggregate_annotations(self, comparisons: List[Dict], rng) -> List[Dict]:
        """Convert comparison dataset to annotation dataset using binarized preferences.

        Args:
            comparisons: List of comparison objects with nested annotations
            rng: Random number generator (unused)

        Returns:
            Annotation dataset with preferences from binarized dataset
        """
        annotation_dataset = []
        found_ids = 0
        tie_common_count = 0

        for comparison in comparisons:
            comp_id = comparison["comparison_id"]
            if comp_id in self.bin_lookup:
                found_ids += 1
                bin_item = self.bin_lookup[comp_id]

                is_tie = bin_item["tie_is_common"]
                if is_tie:
                    tie_common_count += 1

                # Determine which conversation (A or B) matches the binarized chosen response
                # by comparing the conversations
                if bin_item["chosen"] == comparison["conversation_a"]:
                    overall_pref = "A-is-clearly-better"
                elif bin_item["chosen"] == comparison["conversation_b"]:
                    overall_pref = "B-is-clearly-better"
                else:
                    raise ValueError(
                        f"Could not match binarized chosen response to conversation A or B for comparison {comp_id}"
                    )

                synthetic_annotation = {
                    "overall_pref": overall_pref,
                    "time_spent": 0,  # No response time for binarized data
                    "is_tie": is_tie,
                    "evaluator": "binarized",
                    "preference_strength": 2,  # Default to clear preference
                }

                example = self._create_annotation_example(
                    comparison, synthetic_annotation, rng
                )
                del example["rank"]
                annotation_dataset.append(example)
            else:
                raise ValueError(f"Missing comparison ID: {comp_id}")

        wandb.log(
            {
                "test_multipref_binarized/requested_ids": len(comparisons),
                "test_multipref_binarized/tie_common_count": tie_common_count,
                "test_multipref_binarized/final_examples": len(annotation_dataset),
                "test_multipref_binarized/tie_common_rate": tie_common_count
                / found_ids,
            }
        )

        print("[MultiprefBinarizedAggregator] Aggregation complete:")
        print(f"  Requested comparison IDs: {len(comparisons)}")
        print(f"  Marked as tie (tie_is_common): {tie_common_count}")
        print(f"  Final aggregated examples: {len(annotation_dataset)}")

        if not annotation_dataset:
            raise ValueError(
                "No examples found for binarized dataset after aggregation"
            )

        return annotation_dataset
