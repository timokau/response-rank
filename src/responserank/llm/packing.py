"""Partition Packing Algorithms for Fixed-Size Batch Generation"""

import heapq
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


class BasePacker(ABC):
    """Abstract base class for partition packing algorithms."""

    @abstractmethod
    def pack(
        self, partition_to_indices: Dict[int, List[int]], batch_size: int, seed: int
    ) -> List[List[int]]:
        """Pack partitions into fixed-size batches.

        Args:
            partition_to_indices: Mapping from partition_id to list of indices
            batch_size: Target batch size
            seed: Random seed

        Returns:
            List of batches (each batch is a list of indices)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this packing algorithm."""
        pass


class FlattenAndSlicePacker(BasePacker):
    """Pack by flattening all partitions and slicing into fixed-size batches."""

    @property
    def name(self) -> str:
        return "Flatten and Slice"

    def pack(
        self, partition_to_indices: Dict[int, List[int]], batch_size: int, seed: int
    ) -> List[List[int]]:
        rng = random.Random(seed)

        all_indices = []
        partitions = list(partition_to_indices.keys())
        # Shuffle partition order
        rng.shuffle(partitions)

        # Flatten all indices while keeping partitions consecutive
        for pid in partitions:
            indices = partition_to_indices[pid].copy()
            # Shuffle within partition to randomize splitting position
            rng.shuffle(indices)
            all_indices.extend(indices)

        # Slice into batches
        batches = []
        for i in range(0, len(all_indices), batch_size):
            batch = all_indices[i : i + batch_size]
            batches.append(batch)

        return batches


class LargestFirstBestFitPacker(BasePacker):
    """Largest-first ordering with best-fit placement.

    - Heap-based dynamic ordering (largest fragments processed first)
    - Best-fit placement for intact partitions
    - Split fragments are placed into the batch with largest remaining capacity.
    """

    @property
    def name(self) -> str:
        return "Largest First Best Fit"

    def pack(
        self, partition_to_indices: Dict[int, List[int]], batch_size: int, seed: int
    ) -> List[List[int]]:
        rng = random.Random(seed)

        all_elements = []
        for indices in partition_to_indices.values():
            all_elements.extend(indices)

        num_batches = (len(all_elements) + batch_size - 1) // batch_size

        batches = [[] for _ in range(num_batches)]
        batch_remaining = [batch_size] * num_batches

        # Total capacity should equal needed capacity. We want at most one small batch.
        remainder = len(all_elements) % batch_size
        if remainder != 0:
            batch_remaining[-1] = remainder

        max_fragment_heap = []
        for pid, indices in partition_to_indices.items():
            indices_copy = indices.copy()
            # Shuffle for random split location
            rng.shuffle(indices_copy)
            # This is a min heap, so negative size
            heapq.heappush(max_fragment_heap, (-len(indices_copy), pid, indices_copy))

        while max_fragment_heap:
            neg_size, pid, indices = heapq.heappop(max_fragment_heap)
            size = -neg_size

            # Try to place entire fragment using best fit
            best_idx = None
            best_waste = float("inf")

            for i, remaining in enumerate(batch_remaining):
                if remaining >= size:
                    waste = remaining - size
                    if waste < best_waste:
                        best_waste = waste
                        best_idx = i

            if best_idx is not None:
                # Place entire fragment in best-fitting batch
                batches[best_idx].extend(indices)
                batch_remaining[best_idx] -= size
            else:
                # Fragment doesn't fit entirely. Split and place.
                largest_capacity_batch_idx = None
                max_remaining = 0

                for i, remaining in enumerate(batch_remaining):
                    if remaining > max_remaining:
                        max_remaining = remaining
                        largest_capacity_batch_idx = i

                if largest_capacity_batch_idx is not None and max_remaining > 0:
                    # Place what fits, re-add remainder to heap
                    fit_part = indices[:max_remaining]
                    remainder = indices[max_remaining:]

                    batches[largest_capacity_batch_idx].extend(fit_part)
                    batch_remaining[largest_capacity_batch_idx] = 0
                    heapq.heappush(max_fragment_heap, (-len(remainder), pid, remainder))

        return batches


def analyze_partition_cohesion(
    batches: List[List[int]],
    partition_to_indices: Dict[int, List[int]],
    batch_size: int,
) -> Dict:
    """Analyze how well partitions stay together in the generated batches.

    Args:
        batches: List of batches (each batch is a list of indices)
        partition_to_indices: Mapping from partition_id to list of indices
        batch_size: Target batch size for calculating theoretical minimum fragmentation

    Returns:
        Dictionary with cohesion statistics
    """
    index_to_partition = {}
    for pid, indices in partition_to_indices.items():
        for idx in indices:
            index_to_partition[idx] = pid

    batch_partitions = []
    for batch_idx, batch in enumerate(batches):
        partition_map = defaultdict(list)
        for idx in batch:
            pid = index_to_partition[idx]
            partition_map[pid].append(idx)
        batch_partitions.append(dict(partition_map))

    # Analyze per-partition distribution
    intact_partitions = 0
    total_splits = 0
    all_cohesion_scores = []
    all_fragment_sizes = []
    ideal_fragmentation_count = 0
    total_min_fragments = 0
    total_actual_fragments = 0

    for pid, original_indices in partition_to_indices.items():
        # Find all batches containing this partition
        fragment_sizes = []

        for batch_idx, partition_map in enumerate(batch_partitions):
            if pid in partition_map:
                fragment_sizes.append(len(partition_map[pid]))

        # Calculate stats for this partition
        num_splits = len(fragment_sizes)
        is_intact = num_splits == 1
        if is_intact:
            intact_partitions += 1
        total_splits += num_splits

        # Calculate cohesion score: fraction of pairs that stay together
        total_pairs = len(original_indices) * (len(original_indices) - 1) // 2
        same_batch_pairs = sum(size * (size - 1) // 2 for size in fragment_sizes)
        cohesion_score = same_batch_pairs / total_pairs if total_pairs > 0 else 1.0
        all_cohesion_scores.append(cohesion_score)
        all_fragment_sizes.extend(fragment_sizes)

        partition_size = len(original_indices)
        min_fragments_needed = (
            math.ceil(partition_size / batch_size) if batch_size > 0 else 1
        )

        total_min_fragments += min_fragments_needed
        total_actual_fragments += num_splits

        if num_splits == min_fragments_needed:
            ideal_fragmentation_count += 1

    num_partitions = len(partition_to_indices)
    intact_ratio = intact_partitions / num_partitions if num_partitions > 0 else 0
    avg_splits = total_splits / num_partitions if num_partitions > 0 else 0
    avg_cohesion = (
        sum(all_cohesion_scores) / len(all_cohesion_scores)
        if all_cohesion_scores
        else 0
    )

    ideal_fragmentation_ratio = ideal_fragmentation_count / num_partitions
    fragment_overhead = (
        (total_actual_fragments - total_min_fragments) / total_min_fragments * 100
    )

    return {
        "num_partitions": num_partitions,
        "num_batches": len(batches),
        "intact_partitions": intact_partitions,
        "intact_ratio": intact_ratio,
        "avg_splits_per_partition": avg_splits,
        "avg_cohesion_score": avg_cohesion,
        "avg_fragment_size": np.mean(all_fragment_sizes) if all_fragment_sizes else 0,
        "max_fragment_size": max(all_fragment_sizes) if all_fragment_sizes else 0,
        "ideal_fragmentation_ratio": ideal_fragmentation_ratio,
        "fragment_overhead_pct": fragment_overhead,
    }


def pack_and_analyze(
    packer: BasePacker,
    partition_to_indices: Dict[int, List[int]],
    batch_size: int,
    seed: int,
) -> Tuple[List[List[int]], Dict]:
    """Pack partitions using the given algorithm and return results with timing.

    Args:
        packer: Packing algorithm to use
        partition_to_indices: Mapping from partition_id to list of indices
        batch_size: Target batch size
        seed: Random seed

    Returns:
        Tuple of (batches, cohesion_stats)
    """
    batches = packer.pack(partition_to_indices, batch_size, seed)

    cohesion_stats = analyze_partition_cohesion(
        batches, partition_to_indices, batch_size
    )

    return (
        batches,
        cohesion_stats,
    )
