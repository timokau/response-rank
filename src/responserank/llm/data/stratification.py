import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from math import isclose
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans


def compute_within_stratum_rt_agreement_correlations(examples: List[Dict]) -> Dict:
    """Compute RT-agreement correlations within each stratum/partition.

    Args:
        examples: List of examples with partition_id, rank (RT), and agreement info

    Returns:
        Dictionary with correlation statistics for wandb logging
    """
    partitions = defaultdict(list)
    for ex in examples:
        partitions[ex["partition_id"]].append(ex)

    correlations = []
    partition_sizes = []
    significant_count = 0
    invalid_count = 0

    for partition_id, partition_examples in partitions.items():
        ranks = []
        agreements = []

        for ex in partition_examples:
            rt = ex.get("rank")
            agreement = ex["extra"]["agreement_score"]

            # Skip examples with filtered ranks (None)
            if rt is not None:
                ranks.append(rt)
                agreements.append(agreement)

        with warnings.catch_warnings(
            action="ignore", category=stats.ConstantInputWarning
        ):
            # Returns nan in cases where spearman correlation is not defined (e.g., constant input)
            corr, p_val = stats.spearmanr(ranks, agreements)
        if not np.isnan(corr):
            correlations.append(corr)
            partition_sizes.append(len(ranks))
            if p_val < 0.05:
                significant_count += 1
        else:
            invalid_count += 1

    if len(correlations) > 0:
        mean_corr = np.mean(correlations)
        median_corr = np.median(correlations)
        std_corr = np.std(correlations)
        min_corr = np.min(correlations)
        max_corr = np.max(correlations)
        mean_partition_size = np.mean(partition_sizes)
    else:
        print(
            "Warning: Cannot compute correlations, likely because no comparisons have ranks."
        )
        mean_corr = np.nan
        median_corr = np.nan
        std_corr = np.nan
        min_corr = np.nan
        max_corr = np.nan
        mean_partition_size = np.nan

    return {
        "stratum_rt_agreement_mean_correlation": mean_corr,
        "stratum_rt_agreement_median_correlation": median_corr,
        "stratum_rt_agreement_std_correlation": std_corr,
        "stratum_rt_agreement_min_correlation": min_corr,
        "stratum_rt_agreement_max_correlation": max_corr,
        "stratum_rt_agreement_num_valid_partitions": len(correlations),
        "stratum_rt_agreement_num_significant": significant_count,
        "stratum_rt_agreement_num_invalid": invalid_count,
        "stratum_rt_agreement_mean_partition_size": mean_partition_size,
        "stratum_total_partitions": len(partitions),
    }


def _compute_concordance_for_examples(
    examples: List[Dict], rel_tol: float = 1e-9
) -> Dict:
    """Compute concordance statistics for a single group of examples.

    Args:
        examples: List of examples with rank and agreement info
        rel_tol: Relative tolerance for tie detection

    Returns:
        Dictionary with concordance counts (concordant, discordant, ties)
    """
    n = len(examples)
    if n < 2:
        return {
            "concordant": 0,
            "discordant": 0,
            "ties": 0,
            "is_singleton": True,
        }

    total_concordant = 0
    total_discordant = 0
    total_ties = 0

    # Form all pairs within this group
    for i in range(n):
        for j in range(i + 1, n):
            ex1, ex2 = examples[i], examples[j]

            rank1, rank2 = ex1["rank"], ex2["rank"]
            agree1, agree2 = (
                ex1["extra"]["agreement_score"],
                ex2["extra"]["agreement_score"],
            )

            rank_tied = isclose(rank1, rank2, rel_tol=rel_tol, abs_tol=rel_tol)
            agree_tied = isclose(agree1, agree2, rel_tol=rel_tol, abs_tol=rel_tol)

            if rank_tied or agree_tied:
                total_ties += 1
            else:
                rank_order = 1 if rank1 < rank2 else -1  # Lower rank = faster/better
                agree_order = 1 if agree1 > agree2 else -1  # Higher agreement = better

                if rank_order == agree_order:
                    total_concordant += 1
                else:
                    total_discordant += 1

    return {
        "concordant": total_concordant,
        "discordant": total_discordant,
        "ties": total_ties,
        "is_singleton": False,
    }


def compute_pooled_concordance(examples: List[Dict], rel_tol: float = 1e-9) -> Dict:
    """Compute pooled concordance rate across all partitions.

    Args:
        examples: List of examples with partition_id, rank, and agreement info
        rel_tol: Relative tolerance for tie detection (same as tie splitting)

    Returns:
        Dictionary with concordance statistics
    """
    partitions = defaultdict(list)
    for ex in examples:
        partitions[ex["partition_id"]].append(ex)

    total_concordant = 0
    total_discordant = 0
    total_ties = 0
    partitions_used = 0
    singleton_partitions = 0

    for partition_examples in partitions.values():
        stats = _compute_concordance_for_examples(partition_examples, rel_tol)

        if stats["is_singleton"]:
            singleton_partitions += 1
        else:
            partitions_used += 1
            total_concordant += stats["concordant"]
            total_discordant += stats["discordant"]
            total_ties += stats["ties"]

    total_comparable_pairs = total_concordant + total_discordant

    if total_comparable_pairs > 0:
        pooled_concordance = total_concordant / total_comparable_pairs
        tie_rate = total_ties / (total_comparable_pairs + total_ties)
    else:
        print(
            "Warning: Cannot compute concordance, likely because no comparisons have ranks."
        )
        pooled_concordance = np.nan
        tie_rate = np.nan

    return {
        "stratum_pooled_concordance": pooled_concordance,
        "stratum_total_concordant_pairs": total_concordant,
        "stratum_total_discordant_pairs": total_discordant,
        "stratum_total_tied_pairs": total_ties,
        "stratum_total_comparable_pairs": total_comparable_pairs,
        "stratum_total_pairs": total_concordant + total_discordant + total_ties,
        "stratum_partitions_used": partitions_used,
        "stratum_singleton_partitions": singleton_partitions,
        "stratum_tie_rate": tie_rate,
    }


def _group_by_annotator(examples: List[Dict]) -> Dict[str, List[int]]:
    """
    Group examples by annotator/evaluator ID.

    Args:
        examples: List of example dictionaries

    Returns:
        Dictionary mapping annotator ID to list of example indices
    """
    annotator_to_indices = defaultdict(list)
    for i, ex in enumerate(examples):
        annotator = ex["extra"]["evaluator"]
        annotator_to_indices[annotator].append(i)
    return annotator_to_indices


def _calculate_example_length(example: Dict) -> int:
    if "prompt" in example:
        # Legacy format
        return (
            len(example["prompt"]) + len(example["chosen"]) + len(example["rejected"])
        )
    else:
        # Implicit format: sum all message content in chosen and rejected conversations
        total_length = 0

        assert isinstance(example["chosen"], list)
        for message in example["chosen"]:
            total_length += len(message["content"])

        if isinstance(example["rejected"], list):
            for message in example["rejected"]:
                total_length += len(message["content"])

        return total_length


def _compute_global_length_buckets(
    examples: List[Dict], n_buckets: int
) -> Tuple[List[Optional[int]], Dict[str, object]]:
    """Assign dataset-level text length buckets.

    Args:
        examples: Raw examples that include rank metadata.
        n_buckets: Number of buckets to slice the global length distribution into.

    Returns:
        Tuple of (bucket_ids, stats) where bucket_ids matches example order and
        stats captures percentile boundaries and bucket counts for logging.
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be greater than zero")

    lengths = [_calculate_example_length(ex) for ex in examples]

    filtered_indices: List[int] = []
    valid_indices: List[int] = []
    for i, ex in enumerate(examples):
        if ex.get("rank") is None:
            filtered_indices.append(i)
        else:
            valid_indices.append(i)

    if not valid_indices:
        raise ValueError("Cannot create length buckets without valid ranks")

    valid_lengths = np.array([lengths[i] for i in valid_indices], dtype=float)

    percentiles = np.linspace(0, 100, n_buckets + 1)
    boundaries = np.percentile(valid_lengths, percentiles)
    boundaries[-1] += 1.0

    bucket_ids: List[Optional[int]] = [None] * len(examples)
    bucket_counts = [0] * n_buckets

    for global_idx in valid_indices:
        length = lengths[global_idx]
        bucket = n_buckets - 1
        for b in range(n_buckets):
            lower = boundaries[b]
            upper = boundaries[b + 1]
            if lower <= length < upper:
                bucket = b
                break

        bucket_ids[global_idx] = bucket
        bucket_counts[bucket] += 1

    stats = {
        "boundaries": boundaries.tolist(),
        "bucket_counts": bucket_counts,
        "valid_examples": len(valid_indices),
        "filtered_examples": len(filtered_indices),
    }

    return bucket_ids, stats


def _compute_base_partitions(
    examples: List[Dict], partition_clusters: int, min_cluster_size: int
) -> List[int]:
    """
    Base partition computation logic.
    Groups by evaluator and clusters by text length.
    """
    evaluator_to_indices = _group_by_annotator(examples)

    lengths = np.array([_calculate_example_length(ex) for ex in examples])

    partition_ids = [None] * len(examples)
    current_pid = 0
    small_bucket: List[int] = []
    for evaluator, inds in evaluator_to_indices.items():
        if len(inds) < min_cluster_size:
            small_bucket.extend(inds)
            continue

        eval_lens = lengths[inds].reshape(-1, 1)
        if np.std(eval_lens) < 1e-6:
            for i in inds:
                partition_ids[i] = current_pid
            current_pid += 1
            continue

        max_clusters = len(inds) // min_cluster_size
        n_clusters = min(partition_clusters, max_clusters, len(np.unique(eval_lens)))
        n_clusters = max(1, n_clusters)

        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
            labels = kmeans.fit_predict(eval_lens)

            counts = np.bincount(labels)
            centers = kmeans.cluster_centers_.flatten()
            order = np.argsort(centers)

            for c in order:
                if counts[c] < min_cluster_size:
                    candidates = {
                        o: abs(centers[c] - centers[o])
                        for o in order
                        if o != c and counts[o] + counts[c] >= min_cluster_size
                    }
                    target = (
                        min(candidates, key=candidates.get)
                        if candidates
                        else int(np.argmax(counts))
                    )
                    labels[labels == c] = target
                    counts[target] += counts[c]
                    counts[c] = 0

            uniq = np.unique(labels)
            remap = {old: new for new, old in enumerate(uniq)}
            labels = np.array([remap[label] for label in labels])
        else:
            labels = np.zeros(len(inds), dtype=int)

        for idx, lab in zip(inds, labels):
            partition_ids[idx] = current_pid + lab
        current_pid += int(labels.max()) + 1

    if small_bucket:
        for idx in small_bucket:
            partition_ids[idx] = current_pid
        current_pid += 1

    for i, pid in enumerate(partition_ids):
        if pid is None:
            partition_ids[i] = current_pid
            current_pid += 1

    return partition_ids


def _split_large_partitions(
    partition_ids: List[int], max_partition_size: int
) -> List[int]:
    """Split partitions that exceed max_partition_size into smaller chunks."""
    pid_to_inds: Dict[int, List[int]] = defaultdict(list)
    for i, pid in enumerate(partition_ids):
        pid_to_inds[pid].append(i)

    next_pid = max(pid_to_inds.keys()) + 1
    for pid, inds in list(pid_to_inds.items()):
        if len(inds) <= max_partition_size:
            continue
        for start in range(0, len(inds), max_partition_size):
            chunk = inds[start : start + max_partition_size]
            new_pid = pid if start == 0 else next_pid
            for idx in chunk:
                partition_ids[idx] = new_pid
            if start != 0:
                next_pid += 1

    return partition_ids


def _log_partition_statistics(
    partition_ids: List[int],
    min_cluster_size: int,
    function_name: str,
    max_partition_size: Optional[int] = None,
):
    """Log partition statistics."""
    counts = Counter(partition_ids)
    sizes = list(counts.values())
    total = len(sizes)

    print(f"\n[{function_name}] Partition statistics:")
    print(f"  Total partitions: {total}")
    print(f"  Minimum partition size: {min(sizes)}")
    print(f"  Maximum partition size: {max(sizes)}")
    print(f"  Average partition size: {sum(sizes) / total:.2f}")

    smalls = sum(1 for sz in sizes if sz < min_cluster_size)
    print(f"  Partitions below min_cluster_size ({min_cluster_size}): {smalls}")

    if max_partition_size is not None:
        oversized = sum(1 for sz in sizes if sz > max_partition_size)
        print(
            f"  Partitions above max_partition_size ({max_partition_size}): {oversized}"
        )
    print()


class BaseStratifier(ABC):
    """Abstract base class for stratification methods."""

    @abstractmethod
    def compute_partitions(self, examples: List[Dict], rng) -> List[int]:
        """Compute partition IDs for examples.

        Args:
            examples: List of example dictionaries
            rng: Random number generator (not used by all subclasses)

        Returns:
            List of partition IDs
        """
        pass


class GlobalPartitionStratifier(BaseStratifier):
    """Stratifier that assigns all examples to a single global partition."""

    def compute_partitions(self, examples: List[Dict], rng) -> List[int]:
        filtered_indices = []
        valid_indices = []
        for i, ex in enumerate(examples):
            if ex.get("rank") is None:
                filtered_indices.append(i)
            else:
                valid_indices.append(i)

        partition_ids = [None] * len(examples)

        for idx in valid_indices:
            partition_ids[idx] = 0

        current_pid = 1
        for idx in filtered_indices:
            partition_ids[idx] = current_pid
            current_pid += 1

        if filtered_indices:
            print(
                f"  Filtered examples in singleton partitions: {len(filtered_indices)}"
            )

        _log_partition_statistics(partition_ids, 1, "GlobalPartitionStratifier")

        return partition_ids


class AnnotatorStratifier(BaseStratifier):
    """Stratifier that groups examples by annotator ID only."""

    def compute_partitions(self, examples: List[Dict], rng) -> List[int]:
        """
        Compute partitions based on annotator ID only.

        Each annotator gets its own partition.
        """
        if not examples:
            raise ValueError("Examples list cannot be empty")

        annotator_to_indices = _group_by_annotator(examples)
        partition_ids = [None] * len(examples)
        current_pid = 0

        for annotator, indices in annotator_to_indices.items():
            for idx in indices:
                partition_ids[idx] = current_pid
            current_pid += 1

        _log_partition_statistics(partition_ids, 1, "AnnotatorStratifier")

        print("\n  Per-annotator statistics:")
        for annotator, indices in annotator_to_indices.items():
            print(f"    {annotator}: n={len(indices)}")

        return partition_ids


class AnnotatorLengthBucketStratifier(BaseStratifier):
    """Stratifier that groups examples by annotator and global length buckets."""

    def __init__(self, n_buckets: int):
        self.n_buckets = n_buckets

    def compute_partitions(self, examples: List[Dict], rng) -> List[int]:
        if not examples:
            raise ValueError("Examples list cannot be empty")

        bucket_ids, bucket_stats = _compute_global_length_buckets(
            examples, self.n_buckets
        )

        filtered_indices = [
            i for i, ex in enumerate(examples) if ex.get("rank") is None
        ]
        evaluator_to_indices = _group_by_annotator(examples)

        annotator_bucket_sizes: Dict[str, Dict[int, int]] = defaultdict(dict)

        partition_ids: List[Optional[int]] = [None] * len(examples)
        current_pid = 0

        for evaluator, indices in evaluator_to_indices.items():
            bucket_to_indices: Dict[int, List[int]] = defaultdict(list)
            for idx in indices:
                bucket = bucket_ids[idx]
                if bucket is None:
                    continue
                bucket_to_indices[bucket].append(idx)

            for bucket in sorted(bucket_to_indices.keys()):
                member_indices = bucket_to_indices[bucket]
                for idx in member_indices:
                    partition_ids[idx] = current_pid
                annotator_bucket_sizes[evaluator][bucket] = len(member_indices)
                current_pid += 1

        for idx in filtered_indices:
            partition_ids[idx] = current_pid
            current_pid += 1

        print("\n[AnnotatorLengthBucketStratifier] Global length bucket statistics:")
        boundaries_str = ", ".join(f"{b:.1f}" for b in bucket_stats["boundaries"])
        print(f"  Boundaries: [{boundaries_str}]")
        print(f"  Bucket counts: {bucket_stats['bucket_counts']}")
        print(
            f"  Valid examples: {bucket_stats['valid_examples']},"
            f" filtered examples: {bucket_stats['filtered_examples']}"
        )

        print("\n  Per-annotator bucket usage:")
        for evaluator in sorted(annotator_bucket_sizes.keys()):
            bucket_counts = annotator_bucket_sizes[evaluator]
            if bucket_counts:
                detail = ", ".join(
                    f"bucket {bucket}: {bucket_counts[bucket]}"
                    for bucket in sorted(bucket_counts)
                )
            else:
                detail = "no valid examples"
            print(f"    {evaluator}: {detail}")

        if filtered_indices:
            print(
                f"  Filtered examples assigned to singleton partitions: {len(filtered_indices)}"
            )

        _log_partition_statistics(partition_ids, 1, "AnnotatorLengthBucketStratifier")

        return partition_ids
