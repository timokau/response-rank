import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from math import isclose
from typing import Dict, List, Tuple

import numpy as np
import wandb


def split_partition_avoiding_ties(
    ranks: List[float],
    rel_tol: float,
    target_size: int,
) -> List[List[int]]:
    """
    Split indices to avoid rank ties using round-robin assignment for max distance.

    Args:
        ranks: List of rank values (floats)
        rel_tol: Relative tolerance for float comparison with isclose
        target_size: Target partition size (positive int), or -1 for auto (partition count = max tie group size)

    Returns:
        List of sub-partitions, each containing indices with no tied ranks
    """
    if target_size != -1 and target_size < 1:
        raise ValueError(f"target_size must be positive or -1, got {target_size}")

    if not ranks:
        return []

    n = len(ranks)
    if n == 1:
        return [[0]]

    # Group consecutive equal ranks
    sorted_indices = sorted(range(n), key=lambda i: ranks[i])
    groups = []
    i = 0
    while i < n:
        j = i + 1
        # Find all indices with ranks close to the current one
        while j < n and isclose(
            ranks[sorted_indices[i]],
            ranks[sorted_indices[j]],
            rel_tol=rel_tol,
            abs_tol=rel_tol,
        ):
            j += 1
        groups.append(sorted_indices[i:j])
        i = j

    # Round-robin assignment for maximum within-partition distance
    if target_size == -1:
        max_group_size = max(len(g) for g in groups)
        num_partitions = max_group_size
    else:
        num_partitions = math.ceil(n / target_size)

    sorted_items = sorted(range(n), key=lambda i: ranks[i])
    partitions = [[] for _ in range(num_partitions)]

    for i, idx in enumerate(sorted_items):
        partitions[i % num_partitions].append(idx)

    return partitions


def _compute_partition_statistics(examples: List[Dict], examples_modified: int) -> Dict:
    """Compute statistics about partition sizes."""
    final_partition_ids = [ex["partition_id"] for ex in examples]
    stratum_counts = Counter(final_partition_ids)
    partition_sizes = list(stratum_counts.values())
    non_singleton_sizes = [size for size in partition_sizes if size > 1]

    singleton_examples = sum(1 for size in partition_sizes if size == 1)
    non_singleton_examples = len(examples) - singleton_examples

    size_counts = Counter(partition_sizes)
    table_data = [[size, count] for size, count in sorted(size_counts.items())]
    size_table = wandb.Table(data=table_data, columns=["partition_size", "count"])

    return {
        "partition_count": len(stratum_counts),
        "partition_size_min": min(stratum_counts.values()) if stratum_counts else 0,
        "partition_size_max": max(stratum_counts.values()) if stratum_counts else 0,
        "partition_size_mean": np.mean(partition_sizes),
        "partition_sizes": size_table,
        "nonsingleton_partition_size_mean": np.mean(non_singleton_sizes)
        if non_singleton_sizes
        else 0.0,
        "examples_modified": examples_modified,
        "singleton_partitions": sum(1 for size in partition_sizes if size == 1),
        "non_singleton_partitions": sum(1 for size in partition_sizes if size > 1),
        "singleton_examples": singleton_examples,
        "non_singleton_examples": non_singleton_examples,
        "non_singleton_fraction": non_singleton_examples / len(examples),
    }


class BasePartitioner(ABC):
    """Base class for partitioning data after filtering and before sampling."""

    @abstractmethod
    def partition_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Partition examples by modifying their partition_id values.

        Args:
            examples: List of example dictionaries with partition_id field
            rng: Random number generator

        Returns:
            Tuple of (modified_examples, partition_info_dict) where:
            - modified_examples: Examples with potentially modified partition_id values
            - partition_info_dict: Dictionary with partitioning statistics
        """
        pass


class RoundRobinPartitioner(BasePartitioner):
    """Partitioner that splits partitions to avoid ties using round-robin assignment."""

    def __init__(self, target_size: int):
        self._target_size = target_size

    def partition_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Split partitions to avoid tied ranks within each partition."""
        partition_groups = defaultdict(list)
        for idx, ex in enumerate(examples):
            partition_groups[ex["partition_id"]].append(idx)

        examples_modified = 0
        next_partition_id = max(ex["partition_id"] for ex in examples) + 1

        for partition_id, indices in partition_groups.items():
            if len(indices) <= 1:
                continue

            ranks = [examples[idx]["rank"] for idx in indices]

            subpartitions = split_partition_avoiding_ties(
                ranks=ranks,
                rel_tol=1e-9,
                target_size=self._target_size,
            )

            if len(subpartitions) > 1:
                for subpartition_indices in subpartitions[1:]:
                    for local_idx in subpartition_indices:
                        global_idx = indices[local_idx]
                        examples[global_idx]["partition_id"] = next_partition_id
                        examples_modified += 1
                    next_partition_id += 1

        return examples, _compute_partition_statistics(examples, examples_modified)


class RandomPartitioner(BasePartitioner):
    """Partitioner that creates randomly shuffled partitions of fixed size.

    Note: Resulting partitions may have ties.
    """

    def __init__(self, partition_size):
        self.partition_size = partition_size

    def partition_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Create randomly shuffled partitions of the specified size."""
        if rng is None:
            raise ValueError("RandomPartitioner requires rng parameter")

        filtered_indices = []
        valid_indices = []
        for i, ex in enumerate(examples):
            if ex.get("rank") is None:
                filtered_indices.append(i)
            else:
                valid_indices.append(i)

        rng.shuffle(valid_indices)

        current_partition_id = 0

        for i in range(0, len(valid_indices), self.partition_size):
            batch_indices = valid_indices[i : i + self.partition_size]
            for idx in batch_indices:
                examples[idx]["partition_id"] = current_partition_id
            current_partition_id += 1

        for idx in filtered_indices:
            examples[idx]["partition_id"] = current_partition_id
            current_partition_id += 1

        return examples, _compute_partition_statistics(
            examples, examples_modified=len(examples)
        )


class NoOpPartitioner(BasePartitioner):
    """Pass-through partitioner that doesn't modify partition assignments."""

    def partition_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Return examples unchanged."""
        return examples, _compute_partition_statistics(examples, examples_modified=0)
