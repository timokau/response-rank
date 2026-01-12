"""Dataset sampling strategies for fraction-based data selection."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List


class BaseDatasetSampler(ABC):
    """Abstract base class for dataset sampling strategies."""

    @abstractmethod
    def sample(self, examples: List[Dict], fraction: float, rng) -> List[Dict]:
        """Sample a fraction of examples from the dataset.

        Args:
            examples: List of examples to sample from
            fraction: Fraction of examples to keep (0.0 to 1.0)
            rng: Random number generator

        Returns:
            List of sampled examples
        """
        pass


class UniformDatasetSampler(BaseDatasetSampler):
    """Uniformly random sampling of examples without considering annotator distribution."""

    def sample(self, examples: List[Dict], fraction: float, rng) -> List[Dict]:
        """Sample examples uniformly at random.

        Args:
            examples: List of examples to sample from
            fraction: Fraction of examples to keep (0.0 to 1.0)
            rng: Random number generator

        Returns:
            List of sampled examples
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Fraction must be between 0.0 and 1.0, got {fraction}")

        if fraction >= 1.0:
            return examples

        n_keep = int(len(examples) * fraction)
        shuffled = examples[:]
        rng.shuffle(shuffled)

        return shuffled[:n_keep]


class AnnotatorDatasetSampler(BaseDatasetSampler):
    """Annotator-aware sampling that samples annotators first, then includes their examples."""

    def sample(self, examples: List[Dict], fraction: float, rng) -> List[Dict]:
        """Sample examples by selecting annotators first, then including their examples.

        Args:
            examples: List of examples to sample from
            fraction: Fraction of examples to keep (0.0 to 1.0)
            rng: Random number generator

        Returns:
            List of sampled examples
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Fraction must be between 0.0 and 1.0, got {fraction}")

        if fraction >= 1.0:
            return examples

        annotator_groups = defaultdict(list)
        for example in examples:
            evaluator = example["extra"]["evaluator"]
            annotator_groups[evaluator].append(example)

        annotators = list(annotator_groups.keys())
        rng.shuffle(annotators)

        target_count = int(len(examples) * fraction)
        selected_examples = []
        for annotator in annotators:
            annotator_examples = annotator_groups[annotator]

            if len(selected_examples) + len(annotator_examples) > target_count:
                remaining_needed = target_count - len(selected_examples)
                partial_examples = annotator_examples[:remaining_needed]
                selected_examples.extend(partial_examples)

                break
            else:
                selected_examples.extend(annotator_examples)

        return selected_examples
