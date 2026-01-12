from datasets import Dataset, load_dataset


def _load_rewardbench_dataset(dataset_name: str, split: str) -> Dataset:
    """Generic function to load RewardBench datasets from HuggingFace."""
    return load_dataset(dataset_name, split=split)


def _convert_to_trainer_format(example) -> dict:
    """Convert raw RewardBench format to TRL RewardTrainer conversational format."""
    return {
        "chosen": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]},
        ],
        "rejected": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["rejected"]},
        ],
        "subset": example["subset"],
        "comparison_id": example["id"],
    }


def _print_dataset_info(dataset: Dataset, dataset_name: str) -> None:
    """Print dataset information and sample for debugging."""
    print(f"{dataset_name} eval set ready: {len(dataset)} comparisons")

    # Print a sample to verify the format
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample example:")
        print(f"  Chosen conversation: {sample['chosen']}")
        print(f"  Rejected conversation: {sample['rejected']}")
        print(f"  Subset: {sample['subset']}")


def prepare_rewardbench_eval_ds() -> Dataset:
    """
    Load the RewardBench dataset directly from Hugging Face.

    Returns
    -------
    Dataset
        Dataset formatted for reward model training in conversational format
    """
    # Load the "filtered" split from RewardBench
    rb_ds = _load_rewardbench_dataset("allenai/reward-bench", "filtered")

    # Convert to conversational format expected by the reward trainer
    pair_ds = rb_ds.map(_convert_to_trainer_format, desc="Converting to trainer format")

    # Remove original prompt column as it's now included in conversations
    pair_ds = pair_ds.remove_columns(["prompt"])

    _print_dataset_info(pair_ds, "RewardBench")

    return pair_ds
