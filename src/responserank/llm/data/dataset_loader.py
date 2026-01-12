from collections import Counter

import numpy as np

from responserank.llm.data.prepare_dataset import (
    prepare_datasets,
)


def analyze_dataset_statistics(train_dataset, dataset_type: str):
    """Analyze and print statistics for a dataset."""
    if "partition_id" in train_dataset.column_names:
        partition_ids = train_dataset["partition_id"]

        # Health check: verify no partition_ids of -1 remain after partitioning
        invalid_partition_count = sum(1 for pid in partition_ids if pid == -1)
        if invalid_partition_count > 0:
            raise ValueError(
                f"Found {invalid_partition_count} examples with partition_id = -1 out of {len(partition_ids)} total examples. "
                "This indicates a bug in the partitioning pipeline - all examples should have valid partition IDs assigned."
            )

        print_partition_statistics(partition_ids)


def print_partition_statistics(partition_ids):
    partition_counts = Counter(partition_ids)
    partition_sizes = list(partition_counts.values())
    unique_partitions = len(partition_counts)

    print("\nPartition Statistics:")
    print(f"  Total unique partitions: {unique_partitions}")
    print("  Partition size statistics:")
    print(f"    Minimum size: {min(partition_sizes)}")
    print(f"    Maximum size: {max(partition_sizes)}")
    print(f"    Average size: {np.mean(partition_sizes):.2f}")
    print(f"    Median size: {np.median(partition_sizes):.2f}")

    min_size_threshold = 4
    small_partitions = sum(1 for size in partition_sizes if size < min_size_threshold)
    print(
        f"    Partitions with fewer than {min_size_threshold} examples: {small_partitions} ({small_partitions / len(partition_counts) * 100:.2f}%)"
    )

    print("\n  Example partition sizes:")
    for pid, count in sorted(partition_counts.items())[:10]:
        print(f"    Partition {pid}: {count} examples")
    if len(partition_counts) > 10:
        print(f"    ... and {len(partition_counts) - 10} more partitions")


def prepare_datasets_wrapper(
    args,
    rng,
    stratifier,
    train_processor,
    test_processor,
    train_filter,
    test_filter,
    train_partitioner,
    test_partitioner,
    dataset_sampler,
    train_rank_transform,
    test_rank_transform,
    dataset,
    tokenizer,
):
    """
    Run the dataset preparation script

    Args:
        args: Arguments object with dataset configuration
        rng: Random number generator to use for dataset preparation
        stratifier: Stratifier object for partitioning data
        train_processor: Annotation processor object for processing train dataset annotations
        test_processor: Optional annotation processor object for processing test dataset annotations
        train_filter: Filter object for filtering train dataset rank values
        test_filter: Filter object for filtering test dataset rank values
        train_partitioner: Partitioner object for reorganizing train dataset partitions
        test_partitioner: Partitioner object for reorganizing test dataset partitions
        dataset_sampler: Dataset sampler object for fraction sampling
        train_rank_transform: Rank transformation to apply to training data after partitioning
        test_rank_transform: Rank transformation to apply to test data after partitioning
        dataset: Dataset implementation
    """
    print("Running dataset preparation...")
    train_dataset, _, test_dataset = prepare_datasets(
        dataset=dataset,
        rng=rng,
        fraction=args.experiment.fraction,
        test_size=args.experiment.test_size,
        stratifier=stratifier,
        train_processor=train_processor,
        test_processor=test_processor,
        train_filter=train_filter,
        test_filter=test_filter,
        train_partitioner=train_partitioner,
        test_partitioner=test_partitioner,
        dataset_sampler=dataset_sampler,
        train_rank_transform=train_rank_transform,
        test_rank_transform=test_rank_transform,
        shuffle_ranks=args.experiment.shuffle_ranks,
        invert_ranks=args.experiment.invert_ranks,
        rank_filter_after_fraction_sampling=args.experiment.rank_filter_after_fraction_sampling,
        filter_ties_before_aggregation=args.experiment.filter_ties_before_aggregation,
        filter_ties_after_aggregation=args.experiment.filter_ties_after_aggregation,
        tokenizer=tokenizer,
        max_length=args.experiment.max_length,
    )

    analyze_dataset_statistics(
        train_dataset,
        args.experiment.dataset_type,
    )

    return train_dataset, test_dataset
