"""
Entry point for training reward models with ResponseRank or BT loss.

Hydra configuration is defined in conf/llm/experiment/ with experiment-specific
presets that compose settings.

Usage (run from code/ directory):
    python -m responserank.llm.train experiment=bt
    python -m responserank.llm.train experiment=rr_agree
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from responserank.llm.data.dataset_loader import prepare_datasets_wrapper
from responserank.llm.shared_components import (
    set_seed,
    train_baseline_reward_model,
    train_reward_model,
)
from responserank.llm.wandb_utils import wandb_init_claim


def termination_handler(signum, frame):
    signal_name = signal.Signals(signum).name
    exit_code = 128 + signum
    print(f"Received {signal_name} (signal {signum}). Finishing wandb run...")
    wandb.run.summary["status"] = "stopped"
    wandb.finish(exit_code=exit_code)
    exit(exit_code)


def run_experiment(cfg: DictConfig):
    if cfg.experiment.fraction <= 0.0 or cfg.experiment.fraction > 1.0:
        raise ValueError("Fraction must be between 0.0 (exclusive) and 1.0 (inclusive)")
    if cfg.experiment.rt_loss_weight < 0.0 or cfg.experiment.rt_loss_weight > 1.0:
        raise ValueError("RT loss weight must be between 0.0 and 1.0 (inclusive)")

    # Set seed for reproducibility and get RNGs
    main_rng, dataset_rng = set_seed(cfg.seed)

    wandb_project = f"{cfg.wandb.project_base}-multipref-v{cfg.version}"

    # Compute a hash that identifies this experimental condition (excluding seed and fraction)
    hash_relevant_config = OmegaConf.to_container(cfg, resolve=True)
    del hash_relevant_config["seed"]
    del hash_relevant_config["experiment"]["fraction"]
    config_serialized = json.dumps(hash_relevant_config, sort_keys=True).encode()
    config_hash = hashlib.sha256(config_serialized).hexdigest()

    # Create experiment group ID (egid) and experiment ID (eid)
    dry = "dry-" if cfg.dry_run else ""
    egid = f"{dry}{cfg.experiment.name}-v{cfg.experiment.base_version}.{cfg.experiment.version}-{config_hash[:8]}"
    eid = f"{egid}-{cfg.experiment.fraction}"
    run_name = f"{eid}-{cfg.seed}"

    sampler = None
    if not cfg.experiment.baseline:
        sampler = instantiate(cfg.experiment.sampler)

    skip_existing_check = (
        os.getenv("SKIP_EXISTING_RUN_CHECK", "false").lower() == "true"
    ) or cfg.dry_run
    run = wandb_init_claim(
        project=wandb_project,
        run_name=run_name,
        skip_existing_check=skip_existing_check,
        group=egid,
    )
    if run is None:
        print(
            f"Run '{run_name}' already exists in project '{wandb_project}'. Skipping experiment."
        )
        return

    wandb.run.summary["status"] = "running"

    if sampler is not None:
        sampler.wandb_run = (
            run  # Pass wandb run to sampler (cannot do it at initialization time)
        )

    git_clean = (
        subprocess.run(["git", "diff", "--quiet"], capture_output=True).returncode == 0
    )
    if not git_clean:
        git_diff = subprocess.run(
            ["git", "diff", "--no-color"], capture_output=True, text=True
        ).stdout

        artifact = wandb.Artifact("git_diff", type="diff")
        with artifact.new_file("git_diff.patch", mode="w") as f:
            f.write(git_diff)
        wandb.log_artifact(artifact)

    hydra_config = HydraConfig.get()

    wandb.config.update(
        {
            "hydra_cfg": OmegaConf.to_container(cfg, resolve=True),
            "hydra_cfg_hash": config_hash,
            "egid": egid,
            "eid": eid,
            "git_clean": git_clean,
            "hydra_runtime_choices": OmegaConf.to_container(
                hydra_config.runtime.choices
            ),
            "hydra_runtime_overrides": dict(
                [kv.split("=", maxsplit=1) for kv in hydra_config.overrides.task]
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
    )

    # Set dataset_type for baseline mode (needed by shared dataset loading function)
    if cfg.experiment.baseline:
        cfg.experiment.dataset_type = "real_rt"

    stratifier = instantiate(cfg.experiment.stratifier)
    train_processor = instantiate(cfg.experiment.train_processor)
    test_processor = (
        instantiate(cfg.experiment.test_processor)
        if cfg.experiment.test_processor._target_ is not None
        else None
    )
    train_filter = instantiate(cfg.experiment.train_filter)
    test_filter = instantiate(cfg.experiment.test_filter)
    train_partitioner = instantiate(cfg.experiment.train_partitioner)
    test_partitioner = instantiate(cfg.experiment.test_partitioner)
    dataset_sampler = instantiate(cfg.experiment.dataset_sampler)
    train_rank_transform = instantiate(cfg.experiment.train_rank_transform)
    test_rank_transform = instantiate(cfg.experiment.test_rank_transform)
    dataset = instantiate(cfg.experiment.dataset)

    hf_token_path = Path.home() / ".hf_token"
    if hf_token_path.exists():
        print("Loading HF token from file")
        os.environ["HF_TOKEN"] = hf_token_path.read_text().strip()

    print(f"Loading tokenizer: {cfg.experiment.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.experiment.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load or prepare datasets
    train_dataset, test_dataset = prepare_datasets_wrapper(
        args=cfg,
        rng=dataset_rng,
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
        dataset=dataset,
        tokenizer=tokenizer,
    )

    if cfg.dry_run:
        print(f"\n{'=' * 80}")
        print("Dry run mode. Exiting without training")
        print("=" * 80)
        wandb.run.summary["status"] = "finished"
        return

    eval_metric_namespace = "eval"

    # Train the reward model
    if cfg.experiment.baseline:
        trainer, metrics = train_baseline_reward_model(
            cfg,
            train_dataset,
            test_dataset,
            tokenizer=tokenizer,
            learning_rate=cfg.experiment.learning_rate,
            warmup_ratio=cfg.experiment.warmup_ratio,
            logging_steps=cfg.experiment.logging_steps,
            eval_strategy=cfg.experiment.eval_strategy,
            gradient_checkpointing=cfg.experiment.gradient_checkpointing,
            bf16=cfg.experiment.bf16,
            save_strategy=cfg.experiment.save_strategy,
            eval_on_start=cfg.experiment.eval_on_start,
            eval_metric_namespace=eval_metric_namespace,
            run_name=run_name,
        )
    else:
        trainer, metrics = train_reward_model(
            cfg,
            train_dataset,
            test_dataset,
            main_rng,
            tokenizer=tokenizer,
            learning_rate=cfg.experiment.learning_rate,
            warmup_ratio=cfg.experiment.warmup_ratio,
            logging_steps=cfg.experiment.logging_steps,
            eval_strategy=cfg.experiment.eval_strategy,
            gradient_checkpointing=cfg.experiment.gradient_checkpointing,
            bf16=cfg.experiment.bf16,
            save_strategy=cfg.experiment.save_strategy,
            eval_on_start=cfg.experiment.eval_on_start,
            eval_metric_namespace=eval_metric_namespace,
            run_name=run_name,
            sampler=sampler,
        )

    output_dir = trainer.args.output_dir

    wandb.log({"train_examples": len(train_dataset)})
    wandb.log({"test_examples": len(test_dataset)})

    for key, value in metrics.items():
        wandb.log({key: value})

    print(f"All done! Model saved to: {output_dir}")
    wandb.run.summary["status"] = "finished"


@hydra.main(version_base="1.3", config_path="../../../conf/llm", config_name="config")
def main(cfg: DictConfig):
    signal.signal(signal.SIGTERM, termination_handler)
    signal.signal(signal.SIGINT, termination_handler)
    signal.signal(signal.SIGHUP, termination_handler)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)

    exit_code = 0
    try:
        run_experiment(cfg)
    except Exception as e:
        print(f"Training failed with exception: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        wandb.run.summary["status"] = "failed"
        exit_code = 1
        raise
    finally:
        wandb.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
