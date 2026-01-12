"""Utilities for WandB integration."""

import random
import time
from typing import List, Optional

import wandb
from wandb.apis.public.runs import Run


def find_runs_with_name(project: str, run_name: str) -> List[Run]:
    """Find all runs with the given name in the project.

    Args:
        project: WandB project name
        run_name: Name of the run to check for

    Returns:
        List of Run objects with the given name
    """
    api = wandb.Api()
    runs = api.runs(project, filters={"display_name": run_name})
    return list(runs)


def wandb_init_claim(
    project: str,
    run_name: str,
    skip_existing_check: bool,
    *,
    group: Optional[str],
) -> Optional[Run]:
    """
    Claim a unique run name in a wandb project using optimistic concurrency control.

    Args:
        project: WandB project name
        run_name: Name of the run to claim
        skip_existing_check: If True, allow claiming even when a run already exists
        group: Optional group name for the run

    Returns:
        wandb.Run object if successful, None if run already exists
    """
    while True:
        successful_or_running = [
            run
            for run in find_runs_with_name(project, run_name)
            if run.state in ["finished", "running"]
        ]

        if len(successful_or_running) > 0 and not skip_existing_check:
            return None
        elif len(successful_or_running) > 0 and skip_existing_check:
            print(
                f"Run '{run_name}' already exists in project '{project}', but SKIP_EXISTING_RUN_CHECK=true. Proceeding anyway."
            )

        this_run = wandb.init(project=project, name=run_name, group=group)

        # Check again in case of races (optimistic concurrency control)
        successful_or_running = [
            run
            for run in find_runs_with_name(project, run_name)
            if run.state in ["finished", "running"]
        ]
        if len(successful_or_running) > 1 and not skip_existing_check:
            wandb.finish(exit_code=1)
            api = wandb.Api()
            run = api.run(f"{this_run.entity}/{this_run.project}/{this_run.id}")
            run.delete()
            # Backoff time should be unique, even if two runs are configured to use the same seed.
            unseeded_rng = random.Random(x=None)
            backoff_time = unseeded_rng.uniform(0, 5)
            print(f"Run collision. Backing off {backoff_time:.1f}s.")
            time.sleep(backoff_time)
        else:
            return this_run
