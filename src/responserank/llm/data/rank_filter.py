from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseRankFilter(ABC):
    """Base class for rank value filtering."""

    @abstractmethod
    def filter_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Filter examples by setting rank to None for outliers.

        Args:
            examples: List of example dictionaries containing rank values
            rng: Random number generator (for future use)

        Returns:
            Tuple of (modified_examples, filter_info_dict) where:
            - modified_examples: Examples with outlier ranks set to None
            - filter_info_dict: Dictionary with filtering statistics
        """
        pass


class NoOpFilter(BaseRankFilter):
    """Pass-through filter that doesn't modify any ranks."""

    def filter_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Return examples unchanged."""
        # Ensure all examples have the filtered_reason field for consistency
        for ex in examples:
            ex.setdefault("extra", {})["filtered_reason"] = None

        return examples, {"filtered_count": 0, "filtered_percentage": 0.0}


class CategoryFilter(BaseRankFilter):
    """Filter examples to keep only those matching a specific category value."""

    def __init__(
        self,
        field_name: str,
        field_value: str,
    ):
        """Initialize category filter.

        Args:
            field_name: Name of the category field to filter on (e.g., 'subject_study')
            field_value: Value to keep (e.g., 'Mathematics')
        """
        self.field_name = field_name
        self.field_value = field_value

    def filter_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Filter examples to keep only those matching the specified category.

        Sets rank to None for examples that don't match the category.
        """
        filtered_count = 0
        original_count = len(examples)
        matched_count = 0

        # Ensure all examples have the filtered_reason field for consistency
        for ex in examples:
            ex.setdefault("extra", {})["filtered_reason"] = None

        for ex in examples:
            category_value = ex.get("extra", {}).get(self.field_name)
            if category_value != self.field_value:
                ex["rank"] = None
                ex["extra"]["filtered_reason"] = f"{self.field_name}_mismatch"
                filtered_count += 1
            else:
                matched_count += 1

        filter_info = {
            "filtered_count": filtered_count,
            "matched_count": matched_count,
            "filtered_percentage": filtered_count / original_count * 100,
            "matched_percentage": matched_count / original_count * 100,
            "total_count": original_count,
            "filter_field": self.field_name,
            "filter_value": self.field_value,
        }

        return examples, filter_info


class RandomFilter(BaseRankFilter):
    """Filter that randomly removes a specified fraction of examples."""

    def __init__(self, filter_fraction: float):
        """Initialize random filter.

        Args:
            filter_fraction: Fraction of examples to filter out (0.0 to 1.0)
        """
        if not 0.0 <= filter_fraction <= 1.0:
            raise ValueError(
                f"filter_fraction must be between 0.0 and 1.0, got {filter_fraction}"
            )
        self.filter_fraction = filter_fraction

    def filter_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Randomly filter out specified fraction of examples.

        Sets rank to None for randomly selected examples.
        """
        original_count = len(examples)
        n_to_filter = int(original_count * self.filter_fraction)

        indices = list(range(original_count))
        rng.shuffle(indices)
        filter_indices = set(indices[:n_to_filter])

        filtered_count = 0
        kept_count = 0

        for i, ex in enumerate(examples):
            ex.setdefault("extra", {})["filtered_reason"] = None

            if i in filter_indices:
                ex["rank"] = None
                ex["extra"]["filtered_reason"] = "random_filter"
                filtered_count += 1
            else:
                kept_count += 1

        filter_info = {
            "filtered_count": filtered_count,
            "kept_count": kept_count,
            "filtered_percentage": filtered_count / original_count * 100,
            "kept_percentage": kept_count / original_count * 100,
            "total_count": original_count,
            "filter_fraction": self.filter_fraction,
            "actual_filtered_fraction": filtered_count / original_count,
        }

        return examples, filter_info
