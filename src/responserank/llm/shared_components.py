import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer

from responserank.llm.callbacks.rewardbench_callback import (
    RewardBenchEvaluationCallback,
    RewardBenchV2EvaluationCallback,
)
from responserank.llm.reward_trainer import (
    PassthroughRewardDataCollatorWithPadding,
    ResponseRankRewardTrainer,
)


def train_reward_model(
    args,
    train_dataset,
    test_dataset,
    rng,
    tokenizer,
    learning_rate,
    warmup_ratio,
    logging_steps,
    eval_strategy,
    gradient_checkpointing,
    bf16,
    save_strategy,
    eval_on_start,
    eval_metric_namespace,
    run_name,
    sampler,
):
    """Train the reward model with RR loss"""
    print(f"Loading model: {args.experiment.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.experiment.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        num_labels=1,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Create custom data collator that handles response times
    data_collator = PassthroughRewardDataCollatorWithPadding(tokenizer)

    # Create training config
    output_dir = f"reward_models/{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = RewardConfig(
        per_device_train_batch_size=args.experiment.batch_size,
        per_device_eval_batch_size=args.experiment.batch_size,
        gradient_accumulation_steps=args.experiment.gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        bf16=bf16,
        learning_rate=learning_rate,
        report_to="wandb",
        run_name=run_name,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        disable_tqdm=True,
        save_strategy=save_strategy,
        warmup_ratio=warmup_ratio,
        num_train_epochs=args.experiment.epochs,
        eval_on_start=eval_on_start,
        output_dir=output_dir,
        seed=args.seed if args.model_seed is None else args.model_seed,
        max_length=args.experiment.max_length,
        max_grad_norm=args.experiment.max_grad_norm,
        weight_decay=args.experiment.weight_decay,
        adam_beta1=args.experiment.adam_beta1,
        adam_beta2=args.experiment.adam_beta2,
        adam_epsilon=args.experiment.adam_epsilon,
        disable_dropout=args.experiment.disable_dropout,
        lr_scheduler_type=args.experiment.lr_scheduler_type,
        center_rewards_coefficient=args.experiment.center_rewards_coefficient,
    )

    trainer = ResponseRankRewardTrainer(
        rng,
        model=model,
        args=config,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        model_init=None,
        compute_metrics=None,
        callbacks=[
            RewardBenchV2EvaluationCallback(),
        ]
        + (
            [
                RewardBenchEvaluationCallback(
                    filter_max_len=args.experiment.rewardbenchv1_filter_max_len,
                )
            ]
            if args.experiment.eval_on_rewardbench_v1
            else []
        ),
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
        use_rr_loss=args.experiment.use_rr_loss,
        sampler=sampler,
        divide_by_len=args.experiment.divide_by_len,
        accumulation_aware_scaling=args.experiment.accumulation_aware_scaling,
        allow_ties=args.experiment.allow_ties,
        max_length=args.experiment.max_length,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Final evaluation on the test set
    print(f"Final evaluation on test dataset (size: {len(test_dataset)})...")
    metrics = trainer.evaluate()
    trainer.log_metrics(eval_metric_namespace, metrics)
    trainer.save_metrics(eval_metric_namespace, metrics)

    # Optionally save the model
    # print(f"Saving model to {output_dir}")
    # trainer.save_model(output_dir)

    print("Training complete!")
    return trainer, metrics


def train_baseline_reward_model(
    args,
    train_dataset,
    test_dataset,
    tokenizer,
    learning_rate,
    warmup_ratio,
    logging_steps,
    eval_strategy,
    gradient_checkpointing,
    bf16,
    save_strategy,
    eval_on_start,
    eval_metric_namespace,
    run_name,
):
    """Train a baseline reward model without response time information"""
    # Load the model and tokenizer
    print(f"Loading model: {args.experiment.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.experiment.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        num_labels=1,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Create training config
    output_dir = f"reward_models/{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = RewardConfig(
        per_device_train_batch_size=args.experiment.batch_size,
        per_device_eval_batch_size=args.experiment.batch_size,
        gradient_accumulation_steps=args.experiment.gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        bf16=bf16,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        report_to="wandb",
        run_name=run_name,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        disable_tqdm=True,
        save_strategy=save_strategy,
        num_train_epochs=args.experiment.epochs,
        eval_on_start=eval_on_start,
        output_dir=output_dir,
        seed=args.seed if args.model_seed is None else args.model_seed,
        max_length=args.experiment.max_length,
        max_grad_norm=args.experiment.max_grad_norm,
        weight_decay=args.experiment.weight_decay,
        adam_beta1=args.experiment.adam_beta1,
        adam_beta2=args.experiment.adam_beta2,
        adam_epsilon=args.experiment.adam_epsilon,
        disable_dropout=args.experiment.disable_dropout,
        lr_scheduler_type=args.experiment.lr_scheduler_type,
        center_rewards_coefficient=args.experiment.center_rewards_coefficient,
    )

    # Create standard trainer
    trainer = RewardTrainer(
        model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=config,
        callbacks=[
            RewardBenchV2EvaluationCallback(),
        ]
        + (
            [
                RewardBenchEvaluationCallback(
                    filter_max_len=args.experiment.rewardbenchv1_filter_max_len,
                )
            ]
            if args.experiment.eval_on_rewardbench_v1
            else []
        ),
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Final evaluation on the test set
    print(f"Final evaluation on test dataset (size: {len(test_dataset)})...")
    metrics = trainer.evaluate()
    trainer.log_metrics(eval_metric_namespace, metrics)
    trainer.save_metrics(eval_metric_namespace, metrics)

    # Save model
    # print(f"Saving model to {output_dir}")
    # trainer.save_model(output_dir)

    return trainer, metrics


def set_seed(seed: int) -> tuple[random.Random, random.Random]:
    """Set all random seeds for reproducibility and return training and dataset RNG instances."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    main_rng = random.Random(seed)
    # Use a derived RNG to make training independent of dataset generation (which may use caching)
    dataset_rng = random.Random(main_rng.randint(0, 2**32 - 1))

    return main_rng, dataset_rng
