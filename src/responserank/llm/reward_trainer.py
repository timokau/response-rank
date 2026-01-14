"""Custom RewardTrainer for ResponseRank training."""

import logging

import torch
import torch.nn as nn
from trl.trainer.reward_trainer import RewardTrainer
from trl.trainer.utils import RewardDataCollatorWithPadding

from responserank.llm.losses import compute_responserank_loss_sum

logger = logging.getLogger(__name__)


class PassthroughRewardDataCollatorWithPadding(RewardDataCollatorWithPadding):
    """
    Data collator for reward modeling that passes through ranking values and metadata.
    """

    def __call__(self, features):
        batch = super().__call__(features)

        # Add rank to batch if it exists in the features
        if all("rank" in f for f in features):
            batch["rank"] = torch.tensor(
                [0.0 if f["rank"] is None else f["rank"] for f in features],
                dtype=torch.float32,
            )

        # Add partition IDs for partitioned ranking loss
        if all("partition_id" in f for f in features):
            batch["partition_id"] = torch.tensor(
                [f["partition_id"] for f in features], dtype=torch.long
            )

        if not all("bt_target" in f for f in features):
            raise ValueError(
                "bt_target field is mandatory but missing in some features"
            )
        batch["bt_target"] = torch.tensor(
            [f["bt_target"] for f in features], dtype=torch.float32
        )

        return batch


class ResponseRankRewardTrainer(RewardTrainer):
    """
    Reward Trainer that incorporates response time information into the loss calculation.
    Uses Plackett-Luce ranking loss with anchor points and partitions.

    This trainer requires partition-aware batch sampling. A sampler must always be provided,
    and all datasets must include partition_id information. Additionally, datasets must be
    pre-tokenized with input_ids_chosen/rejected and attention_mask_chosen/rejected fields.
    """

    def __init__(
        self,
        rng,
        model,
        args,
        data_collator,
        train_dataset,
        eval_dataset,
        processing_class,
        model_init,
        compute_metrics,
        callbacks,
        optimizers,
        preprocess_logits_for_metrics,
        peft_config,
        rr_loss_weight: float,
        sampler,
        divide_by_len: bool,
        accumulation_aware_scaling: bool,
        allow_ties: bool,
        max_length: int,
    ):
        self.rng = rng

        # Store RR parameters
        self.rr_loss_weight = rr_loss_weight
        self.sampler = sampler
        self.divide_by_len = divide_by_len
        self.accumulation_aware_scaling = accumulation_aware_scaling
        self.allow_ties = allow_ties

        # Initialize the parent class
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # TRL injects an accuracy metric if compute_metric is None, so we need to wrap it after calling the TRL init
        compute_metrics_orig = self.compute_metrics

        def compute_metrics_with_loss_components(eval_pred):
            """Add pref_loss and responserank_loss to eval metrics"""
            metrics = {}

            if len(self._eval_pref_losses) > 0:
                metrics["pref_loss"] = sum(self._eval_pref_losses) / len(
                    self._eval_pref_losses
                )
                self._eval_pref_losses = []

            if len(self._eval_responserank_losses) > 0:
                metrics["responserank_loss"] = sum(
                    self._eval_responserank_losses
                ) / len(self._eval_responserank_losses)
                self._eval_responserank_losses = []

            if compute_metrics_orig is not None:
                original_metrics = compute_metrics_orig(eval_pred)
                metrics.update(original_metrics)

            return metrics

        self.compute_metrics = compute_metrics_with_loss_components

        self._eval_pref_losses = []
        self._eval_responserank_losses = []
        self._in_evaluation = False
        self._eval_without_rr = False

        missing_train_cols = self._check_for_missing_rr_columns(
            train_dataset, self.rr_loss_weight
        )
        missing_eval_cols = self._check_for_missing_rr_columns(
            eval_dataset, self.rr_loss_weight
        )
        if len(missing_train_cols) > 0:
            raise AssertionError(
                f"Missing required columns in train dataset: {missing_train_cols}"
            )
        if len(missing_eval_cols) > 0:
            self._eval_without_rr = True
            logger.info(
                f"Missing rr columns in eval dataset: {missing_eval_cols}. RR loss will be skipped during evaluation."
            )

    def _check_for_missing_rr_columns(self, dataset, rr_loss_weight):
        if rr_loss_weight <= 0:
            return []

        missing_columns = []
        if "rank" not in dataset.column_names:
            missing_columns.append("rank")
        if "partition_id" not in dataset.column_names:
            missing_columns.append("partition_id")
        return missing_columns

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        """
        Compute loss including response time information.
        """
        # Get model outputs as usual
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]

        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]

        # Extract scalar rewards
        # The RM is an AutoModelForSequenceClassification with num_labels=1,
        # which outputs tensors of shape (batch_size, num_labels) = (batch_size, 1).
        assert rewards_chosen.shape[1:] == (1,), (
            f"Expected scalar rewards, got shape {rewards_chosen.shape}"
        )
        assert rewards_rejected.shape[1:] == (1,), (
            f"Expected scalar rewards, got shape {rewards_rejected.shape}"
        )
        # Remove dimension 1, resulting in a flat tensor of length batch_size
        rewards_chosen_flat = rewards_chosen.squeeze(1)
        rewards_rejected_flat = rewards_rejected.squeeze(1)

        utility_diff = rewards_chosen_flat - rewards_rejected_flat

        bt_targets = inputs["bt_target"].to(utility_diff.device)
        if self.accumulation_aware_scaling and num_items_in_batch is not None:
            pref_loss = nn.functional.binary_cross_entropy_with_logits(
                utility_diff, bt_targets, reduction="sum"
            )
            pref_loss = pref_loss / num_items_in_batch
        else:
            pref_loss = nn.functional.binary_cross_entropy_with_logits(
                utility_diff, bt_targets, reduction="mean"
            )
        if self._in_evaluation:
            self._eval_pref_losses.append(pref_loss.detach().cpu().item())

        if self.rr_loss_weight > 0 and not (
            self._in_evaluation and self._eval_without_rr
        ):
            ranks = inputs["rank"].to(utility_diff.device)
            partition_ids = inputs["partition_id"].to(utility_diff.device)
            rr_loss_sum = compute_responserank_loss_sum(
                utility_diff, ranks, partition_ids, allow_ties=self.allow_ties
            )
            if self.divide_by_len:
                if self.accumulation_aware_scaling and num_items_in_batch is not None:
                    if torch.is_tensor(num_items_in_batch):
                        num_items_in_batch = num_items_in_batch.to(rr_loss_sum.device)
                    rr_loss = rr_loss_sum / num_items_in_batch
                else:
                    rr_loss = rr_loss_sum / len(utility_diff)
            else:
                rr_loss = rr_loss_sum
            if self._in_evaluation:
                self._eval_responserank_losses.append(rr_loss.detach().cpu().item())
            total_loss = (
                1 - self.rr_loss_weight
            ) * pref_loss + self.rr_loss_weight * rr_loss
        else:
            rr_loss = None
            total_loss = pref_loss

        if return_outputs:
            outputs = {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
            return total_loss, outputs
        return total_loss

    def get_train_dataloader(self):
        """
        Override to create a partition-aware dataloader for training.
        All RT loss types require partition-aware sampling.
        """
        if self.train_dataset is None:
            raise ValueError("train_dataset is required")

        if self.sampler is None:
            raise ValueError("sampler must be provided for all RR loss types")

        sampler_seed = self.rng.randint(0, 2**32 - 1)

        return self.sampler.create_dataloader(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            seed=sampler_seed,
            collate_fn=self.data_collator,
        )

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # If we do not use responserank in evaluation, eval_loss differs from train_loss. So it is not meaningful to track it.
        if (
            self._eval_responserank_losses
            and hasattr(output, "metrics")
            and f"{metric_key_prefix}_loss" in output.metrics
        ):
            del output.metrics[f"{metric_key_prefix}_loss"]

        return output

    def evaluate(self, *args, **kwargs):
        self._eval_pref_losses = []
        self._eval_responserank_losses = []
        self._in_evaluation = True
        try:
            # The super class calls compute_loss to report the evaluation loss. It does
            # call model.eval(), however, which triggers
            # UserWarning: None of the inputs have requires_grad=True. Gradients will be None
            self.model.eval()
            return super().evaluate(*args, **kwargs)
        finally:
            self._in_evaluation = False
