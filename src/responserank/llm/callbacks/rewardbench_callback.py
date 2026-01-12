import logging
from typing import Dict, Tuple

import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl.trainer.reward_trainer import _tokenize as trl_tokenize
from trl.trainer.utils import RewardDataCollatorWithPadding

from responserank.llm.callbacks.rewardbench_v1 import (
    EXAMPLE_COUNTS,
    SUBSET_MAPPING,
    calculate_scores_per_section,
)
from responserank.llm.callbacks.rewardbench_v2 import (
    process_single_model,
    reroll_and_score_dataset,
)
from responserank.llm.data.prepare_dataset import maybe_apply_chat_template_passthrough
from responserank.llm.data.prepare_rewardbench_dataset import (
    prepare_rewardbench_eval_ds,
)


class RewardDataCollatorWithSubsets(RewardDataCollatorWithPadding):
    """
    Custom collator that preserves subset information for RewardBench v1 evaluation.
    Extends TRL's standard collator to keep the 'subset' field in batches.
    """

    def __call__(self, features):
        subsets = [f["subset"] for f in features]
        batch = super().__call__(features)
        batch["subsets"] = subsets
        return batch


logger = logging.getLogger(__name__)


def calculate_rewardbench_v1_scores(
    subset_accuracies: Dict[str, float],
) -> Dict[str, float]:
    """
    Calculate official RewardBench v1 weighted scores from subset accuracies.

    Args:
        subset_accuracies: Dict mapping subset names to accuracy values

    Returns:
        Dict with section scores, overall score, and individual subset scores
    """
    # Calculate official section scores using RewardBench methodology
    section_scores = calculate_scores_per_section(
        EXAMPLE_COUNTS, SUBSET_MAPPING, subset_accuracies
    )

    # Calculate overall score as average of section scores
    overall_score = sum(section_scores.values()) / len(section_scores)

    results = {"overall": overall_score}
    results.update(section_scores)
    results.update(subset_accuracies)  # Include individual subset scores

    return results


def _prepare_rewardbench_v2_dataset(tokenizer) -> Tuple[Dataset, list, list]:
    """
    Load and prepare RewardBench v2 dataset for evaluation.

    Args:
        tokenizer: Tokenizer to use for chat template application

    Returns:
        Tuple of (unrolled_dataset, total_completions, num_correct)
    """
    raw_dataset = load_dataset("allenai/reward-bench-2", split="test")

    logger.info(f"Loaded RewardBench v2 dataset: {len(raw_dataset)} prompts")

    # Extract metadata before unrolling
    total_completions = raw_dataset["total_completions"]
    num_correct = raw_dataset["num_correct"]

    # Unroll every response in chosen and rejected to a new row
    # IMPORTANT: The order matters! First num_correct responses are chosen, rest are rejected
    # This is how reroll_and_score_dataset determines correctness
    def unroll_responses(example, idx):
        rows = []

        # Combine chosen and rejected responses IN ORDER
        # The scoring assumes first num_correct responses are the correct ones
        all_responses = example["chosen"] + example["rejected"]

        for i, response in enumerate(all_responses):
            # Create new row with response as raw text in "input" field
            new_row = {
                "prompt": example["prompt"],
                "input": response,  # Raw response text
                "subset": example["subset"],
                "id": example["id"],
                "original_idx": idx,
                "response_idx": i,
            }
            rows.append(new_row)

        return rows

    unrolled_data = []
    for idx, example in enumerate(raw_dataset):
        unrolled_data.extend(unroll_responses(example, idx))

    unrolled_dataset = Dataset.from_list(unrolled_data)

    # Apply chat template to create "text" field from prompt + input
    def apply_chat_template(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["input"]},
        ]
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return example

    unrolled_dataset = unrolled_dataset.map(
        apply_chat_template, desc="Applying chat template to RewardBench v2 dataset"
    )

    unrolled_dataset = unrolled_dataset.remove_columns(
        ["input", "original_idx", "response_idx"]
    )

    logger.info(f"Unrolled to {len(unrolled_dataset)} individual responses")

    return unrolled_dataset, total_completions, num_correct


def _tokenize_rewardbench_v1_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    filter_max_len: bool,
    desc_prefix: str = "RewardBench",
) -> Tuple[Dataset, list]:
    """
    Tokenization logic for RewardBench v1 datasets (chosen/rejected pairs).

    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length for filtering (if enabled)
        filter_max_len: Whether to filter sequences exceeding max_length
        desc_prefix: Description prefix for progress bars

    Returns:
        Tuple of (filtered_dataset, kept_indices) where kept_indices
        maps new dataset indices to original dataset indices.
    """
    dataset = dataset.map(
        maybe_apply_chat_template_passthrough,
        fn_kwargs={"tokenizer": tokenizer, "tools": None},
        desc=f"Applying chat template to {desc_prefix} dataset",
    )

    dataset = dataset.map(
        trl_tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        desc=f"Tokenizing {desc_prefix} dataset",
    )

    original_size = len(dataset)

    if filter_max_len:
        # Track which indices are kept after filtering
        kept_indices = []

        def filter_with_index_tracking(example, idx):
            keep = (
                len(example["input_ids_chosen"]) <= max_length
                and len(example["input_ids_rejected"]) <= max_length
            )
            if keep:
                kept_indices.append(idx)
            return keep

        dataset = dataset.filter(filter_with_index_tracking, with_indices=True)
        filtered_size = len(dataset)
        logger.info(
            f"{desc_prefix} dataset filtered: {original_size} -> {filtered_size} samples"
        )
    else:
        kept_indices = list(range(original_size))
        logger.info(
            f"{desc_prefix} dataset tokenized: {original_size} samples (no filtering)"
        )

    return dataset, kept_indices


def _prepare_rewardbench_v2_for_inference(dataset: Dataset) -> Dataset:
    """
    Prepare RewardBench v2 dataset for inference by keeping only the text field.
    This follows the reward-bench repository approach for simple inference.

    Args:
        dataset: Dataset with "text" field from chat template

    Returns:
        Dataset with only necessary fields for inference
    """
    columns_to_keep = ["text", "subset", "id", "prompt"]
    columns_to_remove = [
        col for col in dataset.column_names if col not in columns_to_keep
    ]

    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    return dataset


def _compute_flat_rewards_v1(model, batch, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute and flatten reward logits for chosen and rejected (v1 format)."""
    rewards_chosen = model(
        input_ids=batch["input_ids_chosen"].to(device),
        attention_mask=batch["attention_mask_chosen"].to(device),
        return_dict=True,
    )["logits"]

    rewards_rejected = model(
        input_ids=batch["input_ids_rejected"].to(device),
        attention_mask=batch["attention_mask_rejected"].to(device),
        return_dict=True,
    )["logits"]

    # Our RM should output exactly one scalar per sequence
    assert rewards_chosen.size(-1) == 1, (
        f"Expected reward model to output 1 value per sequence, got chosen shape {rewards_chosen.shape}"
    )
    assert rewards_rejected.size(-1) == 1, (
        f"Expected reward model to output 1 value per sequence, got rejected shape {rewards_rejected.shape}"
    )

    rewards_chosen_flat = rewards_chosen.squeeze(-1)
    rewards_rejected_flat = rewards_rejected.squeeze(-1)

    return rewards_chosen_flat, rewards_rejected_flat


def _compute_rewards_from_text_v2(model, texts, tokenizer, device) -> torch.Tensor:
    """
    Compute reward scores directly from text (following reward-bench approach).

    Args:
        model: The reward model
        texts: List of conversation texts
        tokenizer: Tokenizer for the model
        device: Device to run inference on

    Returns:
        Tensor of reward scores
    """
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=2048,
    )

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        logits = outputs["logits"]

    # Reward models should output exactly one scalar per sequence
    assert logits.size(-1) == 1, (
        f"Expected reward model to output 1 value per sequence, got shape {logits.shape}"
    )

    rewards_flat = logits.squeeze(-1)
    return rewards_flat


class RewardBenchEvaluationCallback(TrainerCallback):
    """
    Callback that evaluates the model on RewardBench after each evaluation step.
    """

    def __init__(self, filter_max_len: bool):
        self.rewardbench_dataset = None
        self.data_collator = None
        self._loaded = False
        self.filter_max_len = filter_max_len

    def _load_rewardbench_dataset(self):
        """Load RewardBench dataset once during first evaluation."""
        if self._loaded:
            return

        logger.info("Loading RewardBench dataset for evaluation...")
        self.rewardbench_dataset = prepare_rewardbench_eval_ds()

        self._loaded = True
        logger.info(
            f"RewardBench dataset loaded: {len(self.rewardbench_dataset)} comparisons"
        )

    def _evaluate_rewardbench(self, model, batch_size) -> Dict[str, float]:
        """Evaluate model on RewardBench dataset with per-subset tracking."""
        model.eval()

        dataloader = torch.utils.data.DataLoader(
            self.rewardbench_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

        correct_predictions = 0
        total_predictions = 0
        subset_results = {}

        with torch.no_grad():
            for batch in dataloader:
                rewards_chosen_flat, rewards_rejected_flat = _compute_flat_rewards_v1(
                    model, batch, model.device
                )

                correct = (rewards_chosen_flat > rewards_rejected_flat).cpu().numpy()
                correct_predictions += correct.sum()
                total_predictions += len(correct)

                batch_subsets = batch["subsets"]

                for i, subset in enumerate(batch_subsets):
                    if subset not in subset_results:
                        subset_results[subset] = {"correct": 0, "total": 0}
                    subset_results[subset]["correct"] += correct[i]
                    subset_results[subset]["total"] += 1

        subset_accuracies = {}
        for subset, counts in subset_results.items():
            subset_accuracies[subset] = counts["correct"] / counts["total"]

        results = calculate_rewardbench_v1_scores(subset_accuracies)

        # We previously reported overall accuracy, so let's keep that for comparability
        results["overall_accuracy"] = correct_predictions / total_predictions

        return results

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Run RewardBench evaluation after each regular evaluation."""
        # The huggingface Trainer passes these kwargs to on_evaluate, even though I cannot find much documentation about hit.
        # https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/trainer_callback.py#L554
        tokenizer = kwargs["processing_class"]
        model = kwargs["model"]
        if not self._loaded:
            self._load_rewardbench_dataset()
            self.rewardbench_dataset, kept_indices = _tokenize_rewardbench_v1_dataset(
                self.rewardbench_dataset,
                tokenizer,
                args.max_length,
                self.filter_max_len,
                desc_prefix="RewardBench",
            )

            self.data_collator = RewardDataCollatorWithSubsets(tokenizer)

        logger.info("Running RewardBench evaluation...")
        results = self._evaluate_rewardbench(model, args.per_device_eval_batch_size)

        wandb_metrics = {
            "train/epoch": state.epoch,
            "train/global_step": state.global_step,
            "eval/rewardbench/overall/accuracy": results["overall_accuracy"],
            # Not technically an accuracy, but close enough and that makes it easier to filter on wandb.
            "eval/rewardbench/overall/accuracy_score": results["overall"],
        }

        sections = ["Chat", "Chat Hard", "Safety", "Reasoning"]
        for section in sections:
            if section in results:
                section_key = section.lower().replace(" ", "_")
                wandb_metrics[f"eval/rewardbench/section/{section_key}_accuracy"] = (
                    results[section]
                )

        for subset, accuracy in results.items():
            if subset not in ["overall", "overall_accuracy"] + sections:
                wandb_metrics[
                    f"eval_detailled/rewardbench/subset/{subset}_accuracy"
                ] = accuracy

        wandb.log(wandb_metrics)

        logger.info(f"RewardBench overall accuracy: {results['overall_accuracy']:.4f}")
        logger.info(f"RewardBench official score: {results['overall']:.4f}")
        for section in sections:
            if section in results:
                logger.info(f"RewardBench {section}: {results[section]:.4f}")

        for subset, accuracy in results.items():
            if subset not in ["overall", "overall_accuracy"] + sections:
                logger.info(f"RewardBench {subset}: {accuracy:.4f}")


class RewardBenchV2EvaluationCallback(TrainerCallback):
    """
    Callback that evaluates the model on RewardBench v2 after each evaluation step.
    """

    def __init__(self):
        self.rewardbench_dataset = None
        self.data_collator = None
        self._loaded = False
        self.subset_list = None
        self.total_completions = None
        self.num_correct = None

    def _load_rewardbench_v2_dataset(self, tokenizer):
        """Load RewardBench v2 dataset once during first evaluation."""
        if self._loaded:
            return

        logger.info("Loading RewardBench v2 dataset for evaluation...")

        self.rewardbench_dataset, self.total_completions, self.num_correct = (
            _prepare_rewardbench_v2_dataset(tokenizer)
        )
        self.subsets = np.unique(self.rewardbench_dataset["subset"])

        self._loaded = True
        logger.info(
            f"RewardBench v2 dataset loaded: {len(self.rewardbench_dataset)} responses"
        )

    def _evaluate_rewardbench_v2(
        self, model, tokenizer, batch_size
    ) -> Dict[str, float]:
        """Evaluate model on RewardBench v2 dataset using simple text-based inference."""
        model.eval()

        # Use simple default collation for text data (like reward-bench repo)
        from torch.utils.data.dataloader import default_collate

        dataloader = torch.utils.data.DataLoader(
            self.rewardbench_dataset,
            batch_size=batch_size,
            collate_fn=default_collate,
            shuffle=False,
        )

        scores = []

        for batch in dataloader:
            batch_scores_tensor = _compute_rewards_from_text_v2(
                model, batch["text"], tokenizer, model.device
            )
            batch_scores = batch_scores_tensor.cpu().numpy().tolist()
            scores.extend(batch_scores)

        # Add scores to dataset and reroll back to per-prompt format
        scored_dataset = self.rewardbench_dataset.add_column("scores", scores)

        rerolled_dataset = reroll_and_score_dataset(
            scored_dataset, self.total_completions, cols_to_combine=["text", "scores"]
        )
        rerolled_dataset = rerolled_dataset.add_column("num_correct", self.num_correct)

        results = {}

        for subset in self.subsets:
            subset_dataset = rerolled_dataset.filter(
                lambda example: example["subset"] == subset
            )

            if subset.lower() == "ties":
                # Special handling for ties subset
                ties_dataset_with_results, overall_score = process_single_model(
                    subset_dataset
                )
                results[subset] = overall_score
            else:
                num_correct = sum(subset_dataset["results"])
                num_total = len(subset_dataset["results"])
                accuracy = num_correct / num_total
                results[subset] = accuracy

        # Calculate overall score by subset average
        overall_score = sum(results.values()) / len(results)
        results["overall"] = overall_score

        return results

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Run RewardBench v2 evaluation after each regular evaluation."""
        tokenizer = kwargs["processing_class"]
        model = kwargs["model"]

        if not self._loaded:
            self._load_rewardbench_v2_dataset(tokenizer)

            self.rewardbench_dataset = _prepare_rewardbench_v2_for_inference(
                self.rewardbench_dataset
            )

        logger.info("Running RewardBench v2 evaluation...")
        results = self._evaluate_rewardbench_v2(
            model, tokenizer, args.per_device_eval_batch_size
        )

        wandb_metrics = {
            "train/epoch": state.epoch,
            "train/global_step": state.global_step,
            "eval/rewardbench_v2/overall/accuracy_score": results["overall"],
        }

        for section, score in results.items():
            if section != "overall":
                wandb_metrics[f"eval/rewardbench_v2/section/{section}_score"] = score

        wandb.log(wandb_metrics)

        logger.info(f"RewardBench v2 overall score: {results['overall']:.4f}")
        for section, score in results.items():
            if section != "overall":
                logger.info(f"RewardBench v2 {section}: {score:.4f}")
