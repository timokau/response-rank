import argparse
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import wandb
from datasets import Dataset
from trl.data_utils import apply_chat_template, is_conversational
from trl.trainer.reward_trainer import _tokenize as trl_tokenize

from responserank.llm.data.data_analysis import analyze_response_times
from responserank.llm.data.stratification import (
    compute_pooled_concordance,
    compute_within_stratum_rt_agreement_correlations,
)


def apply_aggreation(comparison_dataset, aggregator, rng):
    """Convert comparison dataset to annotation dataset via aggregation.

    Args:
        comparison_dataset: Dataset of comparisons with nested annotations
        aggregator: Aggregator to select one annotation per comparison
        rng: Random number generator

    Returns:
        Annotation dataset with one example per comparison
    """
    all_comparisons = comparison_dataset.to_list()

    # Aggregate to create annotation dataset
    annotation_dataset = aggregator.aggregate_annotations(all_comparisons, rng)

    return Dataset.from_list(annotation_dataset)


def compute_bt_target_stats(dataset_or_examples):
    """Compute statistics about bt_target distribution.

    Args:
        dataset_or_examples: Dataset or list of examples with bt_target field

    Returns:
        Dict of statistics for wandb logging
    """
    if isinstance(dataset_or_examples, Dataset):
        examples = dataset_or_examples.to_list()
    else:
        examples = dataset_or_examples

    bt_targets = [ex["bt_target"] for ex in examples]

    contradictory_count = sum(1 for t in bt_targets if t < 0.5)
    weak_signal_count = sum(1 for t in bt_targets if 0.4 < t < 0.6)
    strong_signal_count = sum(1 for t in bt_targets if t > 0.75)

    return {
        "mean": float(np.mean(bt_targets)),
        "min": float(np.min(bt_targets)),
        "max": float(np.max(bt_targets)),
        "std": float(np.std(bt_targets)),
        "median": float(np.median(bt_targets)),
        "contradictory_count": contradictory_count,
        "contradictory_fraction": contradictory_count / len(bt_targets),
        "weak_signal_count": weak_signal_count,
        "weak_signal_fraction": weak_signal_count / len(bt_targets),
        "strong_signal_count": strong_signal_count,
        "strong_signal_fraction": strong_signal_count / len(bt_targets),
    }


class AnnotationProcessor(ABC):
    """Abstract base class for processing annotations into training examples."""

    @abstractmethod
    def process_dataset(self, examples, rng) -> Tuple[List[Dict], Dict]:
        """Process standardized examples into training examples.

        Args:
            examples: List of standardized examples from dataset
            rng: Random number generator

        Returns:
            Tuple of (examples, info_dict) where:
            - examples: List of processed training examples
            - info_dict: Dictionary with processor-specific metrics for logging
        """
        pass


class RealRTProcessor(AnnotationProcessor):
    """Processor for real response time data."""

    def process_dataset(self, examples, rng) -> Tuple[List[Dict], Dict]:
        """Process standardized examples with real response times."""
        original_ranks = [ex["rank"] for ex in examples]
        info_dict = {
            "real_rank_mean": np.mean(original_ranks),
            "real_rank_std": np.std(original_ranks),
            "real_rank_min": np.min(original_ranks),
            "real_rank_max": np.max(original_ranks),
        }

        preference_strengths = [
            ex["extra"].get("preference_strength")
            for ex in examples
            if "preference_strength" in ex.get("extra", {})
        ]
        if preference_strengths:
            strength_counts = Counter(preference_strengths)
            info_dict.update(
                {
                    "preference_strength_1_count": strength_counts.get(1, 0),
                    "preference_strength_2_count": strength_counts.get(2, 0),
                }
            )

        return examples, info_dict


class NoRankProcessor(AnnotationProcessor):
    def process_dataset(self, examples, rng) -> Tuple[List[Dict], Dict]:
        return examples, {}


def _shuffle_ranks(examples, rng):
    """Shuffle rank values across examples while preserving statistics.

    Args:
        examples: List of standardized examples
        rng: Random number generator
    """
    if not examples:
        return

    print(f"Shuffling rank values for {len(examples)} examples")

    original_rank_mean = np.mean([ex["rank"] for ex in examples])
    original_rank_std = np.std([ex["rank"] for ex in examples])

    all_ranks = [ex["rank"] for ex in examples]
    rng.shuffle(all_ranks)

    for i, ex in enumerate(examples):
        ex["rank"] = all_ranks[i]
        # Also update response_time in extra if present
        if "response_time" in ex["extra"]:
            ex["extra"]["response_time"] = all_ranks[i]

    # Verify statistics are preserved
    shuffled_rank_mean = np.mean([ex["rank"] for ex in examples])
    shuffled_rank_std = np.std([ex["rank"] for ex in examples])

    print(
        f"  Original rank stats: mean={original_rank_mean:.2f}, std={original_rank_std:.2f}"
    )
    print(
        f"  Shuffled rank stats: mean={shuffled_rank_mean:.2f}, std={shuffled_rank_std:.2f}"
    )
    print(f"  Difference in means: {abs(original_rank_mean - shuffled_rank_mean):.6f}")
    assert np.allclose(shuffled_rank_mean, original_rank_mean, rtol=1e-10, atol=1e-10)
    assert np.allclose(shuffled_rank_std, original_rank_std, rtol=1e-10, atol=1e-10)


def invert_ranks(examples):
    """Invert rank values for examples.

    Args:
        examples: List of standardized examples
    """
    if not examples:
        return

    print(f"Inverting rank values for {len(examples)} examples")

    original_rank_mean = np.mean([ex["rank"] for ex in examples])
    original_rank_std = np.std([ex["rank"] for ex in examples])

    for ex in examples:
        ex["rank"] = 1.0 / (ex["rank"] + 1.0)
        # Also update response_time in extra if present
        if "response_time" in ex["extra"]:
            ex["extra"]["response_time"] = ex["rank"]

    inverted_rank_mean = np.mean([ex["rank"] for ex in examples])
    inverted_rank_std = np.std([ex["rank"] for ex in examples])

    print(
        f"  Original rank stats: mean={original_rank_mean:.2f}, std={original_rank_std:.2f}"
    )
    print(
        f"  Inverted rank stats: mean={inverted_rank_mean:.4f}, std={inverted_rank_std:.4f}"
    )


def apply_rank_transform_to_partitions(examples, rank_transform, rng):
    """Apply rank transform to each partition separately.

    Args:
        examples: List of examples with partition_id and rank fields
        rank_transform: RankTransform instance to apply
        rng: Random number generator

    Returns:
        Dict of statistics from rank transform application
    """
    # Group examples by partition_id
    partitions = defaultdict(list)
    for ex in examples:
        partitions[ex["partition_id"]].append(ex)

    # Apply transform to each partition independently
    all_stats = []
    partitions_processed = 0

    for partition_id, partition_examples in partitions.items():
        if len(partition_examples) < 2:
            # For single-item partitions, set rank_pre_transform without transformation
            for ex in partition_examples:
                if "rank" in ex:
                    ex["extra"]["rank_pre_transform"] = ex["rank"]
            continue

        stats = rank_transform.transform(partition_examples, rng)
        all_stats.append(stats)
        partitions_processed += 1

    # Aggregate statistics across partitions
    if not all_stats:
        return {"partitions_processed": 0}

    # Average numeric statistics across partitions
    aggregated_stats = {"partitions_processed": partitions_processed}
    for key in all_stats[0].keys():
        assert isinstance(all_stats[0][key], (int, float))
        values = [stats[key] for stats in all_stats if key in stats]
        if values:
            aggregated_stats[f"{key}_mean"] = np.mean(values)
            aggregated_stats[f"{key}_std"] = np.std(values)

    return aggregated_stats


def maybe_apply_chat_template_passthrough(
    example,
    tokenizer,
    tools: Optional[List[Dict]],
):
    """Apply chat template while preserving all fields in the example.

    Args:
        example: Preference example containing chosen/rejected conversations.
        tokenizer: Tokenizer whose chat template should be applied.
        tools: Optional tool definitions to forward to the chat template helper.

    Returns:
        Example with chat template applied while retaining auxiliary metadata.
    """
    if not is_conversational(example):
        return example

    template_result = apply_chat_template(example, tokenizer, tools)

    # apply_chat_template only keeps known fields. It is important that we pass through rank and partition information.
    # https://github.com/huggingface/trl/blob/eee9ec94efbbadb3652aa428827b052b58f36ac7/trl/data_utils.py#L151
    result = example.copy()
    result.update(template_result)

    return result


def tokenize_comparison_dataset(comparison_dataset, tokenizer):
    """
    Tokenize comparison dataset by temporarily renaming columns for TRL compatibility.

    Args:
        comparison_dataset: Dataset with comparison objects containing completion_a/completion_b
        tokenizer: Tokenizer to use

    Returns:
        Dataset with tokenized fields added: input_ids_a/b, attention_mask_a/b
    """
    print(
        f"Tokenizing comparison dataset with {len(comparison_dataset)} comparisons..."
    )

    def prepare_for_tokenization(comparison):
        # Apply chat template to get templated strings. Chosen/rejected is arbitrary here for TRL compatibility.
        preference_example = {
            "chosen": comparison["conversation_a"],
            "rejected": comparison["conversation_b"],
        }

        templated_example = maybe_apply_chat_template_passthrough(
            preference_example,
            tokenizer,
            tools=None,
        )

        # Add temporary fields for TRL tokenization
        comparison_copy = comparison.copy()
        comparison_copy["chosen"] = templated_example["chosen"]
        comparison_copy["rejected"] = templated_example["rejected"]

        return comparison_copy

    dataset_with_temp_fields = comparison_dataset.map(
        prepare_for_tokenization, desc="Preparing comparisons for tokenization"
    )

    fn_kwargs = {"tokenizer": tokenizer}
    tokenized_dataset = dataset_with_temp_fields.map(
        trl_tokenize, batched=True, fn_kwargs=fn_kwargs, desc="Tokenizing comparisons"
    )

    # Rename tokenized fields to match completion_a/completion_b and clean up temporary fields
    def rename_tokenized_fields(example):
        # Rename from chosen/rejected to completion_a/completion_b
        example["input_ids_a"] = example["input_ids_chosen"]
        example["attention_mask_a"] = example["attention_mask_chosen"]
        example["input_ids_b"] = example["input_ids_rejected"]
        example["attention_mask_b"] = example["attention_mask_rejected"]

        # Clean up temporary fields
        del example["chosen"]
        del example["rejected"]
        del example["input_ids_chosen"]
        del example["attention_mask_chosen"]
        del example["input_ids_rejected"]
        del example["attention_mask_rejected"]

        return example

    final_dataset = tokenized_dataset.map(
        rename_tokenized_fields, desc="Renaming tokenized fields"
    )

    return final_dataset


def tokenize_chosen_rejected(dataset, tokenizer):
    """
    Tokenize HF Dataset examples, applying conversational format and chat template.
    Preserves original conversational format alongside tokenized versions.

    Pre-tokenizing works around this TRL bug with custom collators:
    https://github.com/huggingface/trl/issues/3101

    Args:
        dataset: HF Dataset with conversational format examples
        tokenizer: Tokenizer to use

    Returns:
        HF Dataset with tokenized fields added
    """
    print(f"Tokenizing {len(dataset)} examples...")

    def preserve_original_and_tokenize(example):
        # Store original conversational format for stratifiers
        example["chosen_messages"] = example["chosen"].copy()
        example["rejected_messages"] = example["rejected"].copy()

        # Apply chat template (this creates templated strings in chosen/rejected)
        template_result = maybe_apply_chat_template_passthrough(
            example, tokenizer, tools=None
        )
        example.update(template_result)

        return example

    dataset = dataset.map(
        preserve_original_and_tokenize,
        desc="Applying chat template",
    )

    fn_kwargs = {"tokenizer": tokenizer}
    dataset = dataset.map(
        trl_tokenize,
        batched=True,
        fn_kwargs=fn_kwargs,
        desc="Tokenizing",
    )

    def restore_original_format(example):
        example["chosen"] = example["chosen_messages"]
        example["rejected"] = example["rejected_messages"]
        del example["chosen_messages"]
        del example["rejected_messages"]
        return example

    # Restore original format (conversational, no chat template) for stratifiers while keeping tokenized fields
    dataset = dataset.map(restore_original_format, desc="Restoring original format")

    return dataset


def filter_max_length(dataset, max_length):
    """
    Filter HF Dataset to remove comparisons exceeding max_length.

    Args:
        dataset: HF Dataset with input_ids_a and input_ids_b fields (comparison dataset)
        max_length: Maximum sequence length allowed

    Returns:
        HF Dataset with overlong comparisons filtered out
    """
    # Filter out sequences that exceed max_length (following TRL implementation)
    # https://github.com/huggingface/trl/blob/640a9f39164ce3f9bbac7d325c1149ba023314e6/trl/trainer/reward_trainer.py#L221
    original_size = len(dataset)
    dataset = dataset.filter(
        lambda x: len(x["input_ids_a"]) <= max_length
        and len(x["input_ids_b"]) <= max_length,
        desc="Filtering examples exceeding max_length",
    )
    filtered_count = original_size - len(dataset)
    print(
        f"Filtered {filtered_count}/{original_size} comparisons exceeding max_length={max_length}"
    )

    wandb.log(
        {
            "max_length_filtering/pre_filter_count": original_size,
            "max_length_filtering/post_filter_count": len(dataset),
        }
    )

    return dataset


def prepare_datasets(
    dataset,
    rng,
    fraction,
    test_size,
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
    shuffle_ranks,
    invert_ranks,
    rank_filter_after_fraction_sampling,
    filter_ties_before_aggregation,
    filter_ties_after_aggregation,
    tokenizer,
    max_length,
):
    """
    Orchestrate dataset preparation with standardized examples.

    Args:
        dataset: Dataset implementation (BaseDataset subclass)
        rng: Random number generator
        fraction: Fraction of data to use for training
        test_size: Number of comparisons for test set
        stratifier: Stratifier for partitioning data
        train_processor: Processor for training data
        test_processor: Processor for test data
        train_filter: Filter for training data rank values
        test_filter: Filter for test data rank values
        train_partitioner: Partitioner for reorganizing training data partitions
        test_partitioner: Partitioner for reorganizing test data partitions
        dataset_sampler: Sampler for dataset fractions
        shuffle_ranks: Whether to shuffle rank values
        invert_ranks: Whether to invert rank values
        rank_filter_after_fraction_sampling: Whether to apply rank filter after fraction sampling
    """
    print(f"Loading {dataset.get_dataset_name()} dataset…")
    raw_data = dataset.load_raw_data()

    comparison_dataset = dataset.to_comparison_dataset(raw_data, rng)
    print(f"Created comparison dataset with {len(comparison_dataset)} comparisons")

    print(
        f"Filtering tie annotations within comparisons={filter_ties_before_aggregation}..."
    )
    filtered_comparison_dataset = comparison_dataset
    if filter_ties_before_aggregation:
        # Filter tie annotations within each comparison
        filtered_comparisons = []
        total_annotations_before = 0
        total_annotations_after = 0

        for comparison in comparison_dataset:
            total_annotations_before += len(comparison["annotations"])
            filtered_annotations = [
                ann for ann in comparison["annotations"] if not ann["is_tie"]
            ]
            total_annotations_after += len(filtered_annotations)

            if filtered_annotations:  # Keep comparison if it has non-tie annotations
                filtered_comparison = comparison.copy()
                filtered_comparison["annotations"] = filtered_annotations
                filtered_comparisons.append(filtered_comparison)

        filtered_comparison_dataset = Dataset.from_list(filtered_comparisons)
        print(
            f"Pre-aggregation tie filtering: {total_annotations_before} -> {total_annotations_after} annotations"
        )
        print(f"Comparisons after filtering: {len(filtered_comparison_dataset)}")
    else:
        total_annotations = sum(len(comp["annotations"]) for comp in comparison_dataset)
        print(
            f"No filtering applied: {total_annotations} annotations in {len(comparison_dataset)} comparisons"
        )

    print("Tokenizing comparison dataset before aggregation")
    tokenized_comparison_dataset = tokenize_comparison_dataset(
        filtered_comparison_dataset, tokenizer
    )

    print("Filtering by max_length after tokenization...")
    length_filtered_comparison_dataset = filter_max_length(
        tokenized_comparison_dataset, max_length
    )

    print("Computing pre-aggregation tie statistics")
    total_annotations = 0
    total_ties = 0
    ties_per_comparison = []

    for comparison in length_filtered_comparison_dataset:
        annotations = comparison["annotations"]
        total_annotations += len(annotations)

        comparison_ties = sum(1 for ann in annotations if ann["is_tie"])
        total_ties += comparison_ties
        ties_per_comparison.append(comparison_ties)

    avg_ties_per_comparison = total_ties / len(length_filtered_comparison_dataset)
    tie_fraction = total_ties / total_annotations

    tie_counts = Counter(ties_per_comparison)
    tie_distribution_data = [
        [ties, count] for ties, count in sorted(tie_counts.items())
    ]
    tie_distribution_table = wandb.Table(
        data=tie_distribution_data, columns=["ties_per_comparison", "count"]
    )

    wandb.log(
        {
            "pre_aggregation_stats/total_annotations": total_annotations,
            "pre_aggregation_stats/total_ties": total_ties,
            "pre_aggregation_stats/tie_fraction": tie_fraction,
            "pre_aggregation_stats/avg_ties_per_comparison": avg_ties_per_comparison,
            "pre_aggregation_stats/tie_distribution": tie_distribution_table,
        }
    )

    print(
        f"Pre-aggregation stats: {total_ties}/{total_annotations} ties ({tie_fraction:.3f}), avg {avg_ties_per_comparison:.2f} ties/comparison"
    )

    print("Splitting into train and test comparison sets...")
    unique_comparison_ids = set(length_filtered_comparison_dataset["comparison_id"])
    print(f"Total unique comparison_ids: {len(unique_comparison_ids)}")

    comparison_ids_list = sorted(list(unique_comparison_ids))
    rng.shuffle(comparison_ids_list)
    test_ids = set(comparison_ids_list[:test_size])
    train_ids = unique_comparison_ids - test_ids
    print(
        f"Selected {len(test_ids)} test comparison IDs, {len(train_ids)} train comparison IDs"
    )

    # Split comparison dataset
    train_comparison_dataset = length_filtered_comparison_dataset.filter(
        lambda x: x["comparison_id"] in train_ids,
        desc="Selecting train comparisons",
    )
    test_comparison_dataset = length_filtered_comparison_dataset.filter(
        lambda x: x["comparison_id"] in test_ids,
        desc="Selecting test comparisons",
    )

    print("Aggregating train annotations per comparison")
    train_annotation_dataset = apply_aggreation(
        train_comparison_dataset,
        dataset.train_annotation_aggregator,
        rng,
    )
    print(
        f"Created train annotation dataset with {len(train_annotation_dataset)} examples"
    )

    train_bt_target_stats = compute_bt_target_stats(train_annotation_dataset)
    wandb.log({f"bt_target/train/{k}": v for k, v in train_bt_target_stats.items()})
    print(
        f"Train bt_target stats: mean={train_bt_target_stats['mean']:.3f}, "
        f"range=[{train_bt_target_stats['min']:.3f}, {train_bt_target_stats['max']:.3f}], "
        f"contradictory={train_bt_target_stats['contradictory_count']}"
    )

    print("Processing train and test data separately")
    train_data_pre_tie_filter = train_annotation_dataset

    # Apply tie filtering to train data only (test is filtered separately)
    if filter_ties_after_aggregation:
        print("Filtering ties from train data")
        pre_tie_count = len(train_data_pre_tie_filter)
        train_data = train_data_pre_tie_filter.filter(
            lambda x: not x["is_tie"], desc="Filtering tie examples from train data"
        )
        post_tie_count = len(train_data)
        filtered_tie_count = pre_tie_count - post_tie_count
        print(f"Train tie filtering: {pre_tie_count} -> {post_tie_count} examples")

        wandb.log(
            {
                "train_tie_filtering/pre_filter_count": pre_tie_count,
                "train_tie_filtering/post_filter_count": post_tie_count,
                "train_tie_filtering/filtered_ties": filtered_tie_count,
                "train_tie_filtering/tie_rate": filtered_tie_count / pre_tie_count,
            }
        )
    else:
        train_data = train_data_pre_tie_filter

    train_examples_list = train_data.to_list()
    processed_examples, train_processor_info = train_processor.process_dataset(
        train_examples_list, rng
    )
    train_ds = Dataset.from_list(processed_examples)

    wandb.log(
        {f"processor/train/{key}": value for key, value in train_processor_info.items()}
    )

    train_examples = train_ds.to_list()

    print(f"Split data: {len(train_examples)} train, {len(test_ids)} test comparisons")

    if shuffle_ranks:
        _shuffle_ranks(train_examples, rng)

    if invert_ranks:
        invert_ranks(train_examples)

    if not rank_filter_after_fraction_sampling:
        train_examples, train_filter_info = train_filter.filter_examples(
            train_examples, rng
        )
        wandb.log(
            {f"filter/train/{key}": value for key, value in train_filter_info.items()}
        )

    original_count = len(train_examples)
    analyze_annotator_distribution(train_examples, "full")
    train_examples = dataset_sampler.sample(train_examples, fraction, rng)
    analyze_annotator_distribution(train_examples, "fraction")

    if fraction < 1.0:
        print(
            f"Applied fraction {fraction}: kept {len(train_examples)}/{original_count} examples"
        )

    if rank_filter_after_fraction_sampling:
        train_examples, train_filter_info = train_filter.filter_examples(
            train_examples, rng
        )
        wandb.log(
            {f"filter/train/{key}": value for key, value in train_filter_info.items()}
        )

    print("Applying stratification")
    partition_ids = stratifier.compute_partitions(train_examples, rng=rng)
    for i, ex in enumerate(train_examples):
        ex["partition_id"] = partition_ids[i]

    stratum_correlations = compute_within_stratum_rt_agreement_correlations(
        train_examples
    )
    wandb.log(
        {f"processor/train/{key}": val for key, val in stratum_correlations.items()}
    )

    # Mean correlation (above) is a problematic metric (mean is not meaningful over correlations, does not take partition size into account). This is an alternative.
    pooled_concordance_stats = compute_pooled_concordance(train_examples)
    wandb.log(
        {f"processor/train/{key}": val for key, val in pooled_concordance_stats.items()}
    )

    partition_ids = [ex["partition_id"] for ex in train_examples]
    stratum_counts = Counter(partition_ids)
    stratum_sizes = list(stratum_counts.values())

    if stratum_sizes:
        non_singleton_examples = sum(size for size in stratum_sizes if size > 1)
        non_singleton_sizes = [size for size in stratum_sizes if size > 1]
        total_examples = len(train_examples)
        non_singleton_fraction = (
            non_singleton_examples / total_examples if total_examples > 0 else 0.0
        )

        wandb.log(
            {
                "stratification/stratum_count": len(stratum_sizes),
                "stratification/stratum_size_min": min(stratum_sizes),
                "stratification/stratum_size_max": max(stratum_sizes),
                "stratification/stratum_size_mean": np.mean(stratum_sizes),
                "stratification/stratum_size_median": np.median(stratum_sizes),
                "stratification/stratum_size_std": np.std(stratum_sizes),
                "stratification/stratum_size_distribution": wandb.Histogram(
                    stratum_sizes
                ),
                "stratification/non_singleton_fraction": non_singleton_fraction,
                "stratification/singleton_examples": total_examples
                - non_singleton_examples,
                "stratification/non_singleton_stratum_size_mean": np.mean(
                    non_singleton_sizes
                )
                if non_singleton_sizes
                else 0.0,
            }
        )

    train_examples, train_partitioner_info = train_partitioner.partition_examples(
        train_examples, rng
    )
    wandb.log(
        {
            f"partitioner/train/{key}": value
            for key, value in train_partitioner_info.items()
        }
    )

    print("Applying rank transformations")
    train_transform_stats = apply_rank_transform_to_partitions(
        train_examples, train_rank_transform, rng
    )

    for key, value in train_transform_stats.items():
        print(f"rank_transform/train/{key}: {value}")

    wandb.log(
        {
            f"rank_transform/train/{key}": value
            for key, value in train_transform_stats.items()
        }
    )

    print("Building final train Dataset with partition_id…")
    train_dataset = Dataset.from_list(train_examples)

    print("Preparing test Dataset…")
    print("Aggregating test annotations per comparison")
    test_aggregated = apply_aggreation(
        test_comparison_dataset,
        dataset.test_annotation_aggregator,
        rng,
    )
    print(f"Created test annotation dataset with {len(test_aggregated)} examples")

    test_bt_target_stats = compute_bt_target_stats(test_aggregated)
    wandb.log({f"bt_target/test/{k}": v for k, v in test_bt_target_stats.items()})
    print(
        f"Test bt_target stats: mean={test_bt_target_stats['mean']:.3f}, "
        f"range=[{test_bt_target_stats['min']:.3f}, {test_bt_target_stats['max']:.3f}], "
        f"contradictory={test_bt_target_stats['contradictory_count']}"
    )

    # Filter ties after aggregation if configured
    if filter_ties_after_aggregation:
        print("Filtering ties from test data after aggregation")
        pre_filter_count = len(test_aggregated)
        test_aggregated = test_aggregated.filter(
            lambda x: not x["is_tie"], desc="Filtering tie examples from test data"
        )
        post_filter_count = len(test_aggregated)
        filtered_tie_count = pre_filter_count - post_filter_count
        print(
            f"Test post-aggregation tie filtering: {pre_filter_count} -> {post_filter_count} examples"
        )

        wandb.log(
            {
                "test_tie_filtering/pre_filter_count": pre_filter_count,
                "test_tie_filtering/post_filter_count": post_filter_count,
                "test_tie_filtering/filtered_ties": filtered_tie_count,
                "test_tie_filtering/tie_rate": filtered_tie_count / pre_filter_count
                if pre_filter_count > 0
                else 0.0,
            }
        )

    test_examples_list = test_aggregated.to_list()
    processed_examples, test_processor_info = test_processor.process_dataset(
        test_examples_list, rng
    )
    test_ds = Dataset.from_list(processed_examples)

    test_examples = test_ds.to_list()

    wandb.log(
        {f"processor/test/{key}": value for key, value in test_processor_info.items()}
    )

    if not rank_filter_after_fraction_sampling:
        test_examples, test_filter_info = test_filter.filter_examples(
            test_examples, rng
        )
        wandb.log(
            {f"filter/test/{key}": value for key, value in test_filter_info.items()}
        )

    test_examples, test_partitioner_info = test_partitioner.partition_examples(
        test_examples, rng
    )
    wandb.log(
        {
            f"partitioner/test/{key}": value
            for key, value in test_partitioner_info.items()
        }
    )

    print("Applying rank transformations to test data")
    test_transform_stats = apply_rank_transform_to_partitions(
        test_examples, test_rank_transform, rng
    )

    for key, value in test_transform_stats.items():
        print(f"rank_transform/test/{key}: {value}")

    wandb.log(
        {
            f"rank_transform/test/{key}": value
            for key, value in test_transform_stats.items()
        }
    )

    if rank_filter_after_fraction_sampling:
        test_examples, test_filter_info = test_filter.filter_examples(
            test_examples, rng
        )
        wandb.log(
            {f"filter/test/{key}": value for key, value in test_filter_info.items()}
        )
    test_dataset = Dataset.from_list(test_examples)

    # Analyze response times if available (either in extra or as rank)
    if train_dataset and "extra_response_time" in train_dataset.column_names:
        analyze_response_times(train_dataset, output_dir="stats")

    print(
        f"\nDataset stats:\n  Train examples: {len(train_dataset)}\n  Test examples: {len(test_dataset) if test_dataset else 0}"
    )
    return train_dataset, None, test_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare MultiPref datasets for training"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--partition_clusters", type=int, default=8)
    parser.add_argument(
        "--mode",
        type=str,
        default="real",
        choices=["real", "simulated"],
        help="Mode: 'real' for actual response times, 'simulated' for preference-strength based times",
    )
    parser.add_argument(
        "--partition-method",
        type=str,
        default="random",
        choices=["length", "random", "rt", "std"],
        help="Method to create partitions (for simulated mode)",
    )
    parser.add_argument(
        "--partition-size",
        type=int,
        default=4,
        help="Size of each partition for random partitioning (for simulated mode)",
    )
    parser.add_argument(
        "--std-window",
        type=float,
        default=1.0,
        help="Standard deviation window size for std-based partitioning",
    )
    parser.add_argument(
        "--aggregation-method",
        type=str,
        default="scaled",
        choices=["averaged", "scaled"],
        help="Method to aggregate preference strengths into response times (for simulated mode)",
    )
    parser.add_argument(
        "--rt-noise",
        action="store_true",
        help="Enable noise in simulated response times",
    )
    args = parser.parse_args()
    if not (0.0 < args.fraction <= 1.0):
        parser.error("Fraction must be between 0.0 and 1.0")
    return args


def analyze_annotator_distribution(examples: List[Dict], label: str):
    """Analyze and log annotator distribution statistics.

    Args:
        examples: List of examples with evaluator field
        label: Label for the analysis (e.g., "full", "fraction")
    """
    annotator_counts = defaultdict(int)
    for example in examples:
        evaluator = example.get("extra", {}).get("evaluator", "unknown")
        annotator_counts[evaluator] += 1

    counts = list(annotator_counts.values())
    n_annotators = len(annotator_counts)
    total_examples = len(examples)

    mean_examples = np.mean(counts)
    median_examples = np.median(counts)
    min_examples = np.min(counts)
    max_examples = np.max(counts)
    single_example_annotators = sum(1 for count in counts if count == 1)

    wandb.log(
        {
            f"annotator_dist/{label}/total_examples": total_examples,
            f"annotator_dist/{label}/total_annotators": n_annotators,
            f"annotator_dist/{label}/mean_examples_per_annotator": mean_examples,
            f"annotator_dist/{label}/median_examples_per_annotator": median_examples,
            f"annotator_dist/{label}/min_examples_per_annotator": min_examples,
            f"annotator_dist/{label}/max_examples_per_annotator": max_examples,
            f"annotator_dist/{label}/single_example_annotators": single_example_annotators,
            f"annotator_dist/{label}/examples_per_annotator_histogram": wandb.Histogram(
                counts
            ),
        }
    )
