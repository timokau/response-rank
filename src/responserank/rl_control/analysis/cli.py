"""Generate paper plots for RL control experiments."""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib import rc

from responserank.llm.analysis.joint_perf_plot import calculate_ci95_half_width
from responserank.llm.analysis.plotting_utils import (
    save_figure_reproducible,
    setup_plot_style,
)

SEEDS_EXPECTED = 5
BT_COLOR = "#E74C3C"  # Matches BT (baseline) color in multipref plots
RR_COLOR = "#8D6E63"  # Matches RR-Agree (ours) color in multipref plots
RT_CURVE_COLOR = "#2ca02c"  # Green for RT variant in training curves
BT_CURVE_COLOR = "#1f77b4"  # Blue for BT baseline in training curves


@dataclass
class RLRunInfo:
    """Run information parsed from the name."""

    env: str
    noise: float
    is_rt: bool
    seed: int


def parse_run_name(run_name: str) -> RLRunInfo:
    """Parse RL run name to extract experiment parameters.

    Expected format: RL_ppo_{env}_{seed}_comparative_{seed}[_noise_{level}]_nfeedback_5000[_rt1.0]

    Omitted noise level implies 0.0.
    """
    if run_name.startswith("RL_"):
        match = re.match(r"RL_ppo_([A-Za-z0-9]+-v\d+)_(\d+)_", run_name)
    else:
        match = re.match(r"ppo_([A-Za-z0-9]+-v\d+)_(\d+)_", run_name)
    if not match:
        raise ValueError(f"Cannot parse env/seed from: {run_name}")

    env = match.group(1)
    seed = int(match.group(2))

    noise_match = re.search(r"_noise_([0-9.]+)", run_name)
    # Default 0.0 if noise level not present in run name
    noise = float(noise_match.group(1)) if noise_match else 0.0

    is_rt = run_name.endswith("_rt1.0")

    return RLRunInfo(env=env, noise=noise, is_rt=is_rt, seed=seed)


def parse_rew_model_run_name(run_name: str) -> RLRunInfo:
    """Parse reward model run name (ppo_ prefix, no RL_ prefix).

    Expected format: ppo_{env}_{seed}_comparative_{seed}[_noise_{level}]_nfeedback_5000[_rt1.0]
    """
    match = re.match(r"ppo_([A-Za-z0-9]+-v\d+)_(\d+)_", run_name)
    if not match:
        raise ValueError(f"Cannot parse env/seed from reward model run: {run_name}")

    env = match.group(1)
    seed = int(match.group(2))

    noise_match = re.search(r"_noise_([0-9.]+)", run_name)
    noise = float(noise_match.group(1)) if noise_match else 0.0

    is_rt = run_name.endswith("_rt1.0")

    return RLRunInfo(env=env, noise=noise, is_rt=is_rt, seed=seed)


def _load_and_parse_rl_csv(path: Path) -> pd.DataFrame:
    """Load RL CSV and parse run_name into structured columns."""
    df = pd.read_csv(path)

    rows = []
    for _, row in df.iterrows():
        run_name = row["run_name"]
        info = parse_run_name(run_name)
        rows.append(
            {
                "run_name": run_name,
                "env": info.env,
                "noise": info.noise,
                "is_rt": info.is_rt,
                "seed": info.seed,
                "global_step_history": row["global_step_history"],
                "reward_history": row["reward_history"],
            }
        )

    return pd.DataFrame(rows)


def load_rl_cache(cache_dir: Path) -> pd.DataFrame:
    """Load cached RL data from CSV files."""
    mujoco_path = cache_dir / "mujoco_runs.csv"
    highway_path = cache_dir / "highway_runs.csv"

    if not mujoco_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {mujoco_path}\n"
            f"Run with --collect to fetch data from wandb first, "
            f"or copy CSV files to {cache_dir}/"
        )

    mujoco_df = _load_and_parse_rl_csv(mujoco_path)

    if highway_path.exists():
        highway_df = _load_and_parse_rl_csv(highway_path)
        return pd.concat([mujoco_df, highway_df], ignore_index=True)

    return mujoco_df


def compute_summary_stats(runs_df: pd.DataFrame) -> dict:
    """Compute summary stats for analysis.

    Returns dict with structure:
        {env: {noise: [...],
               max_rr: [...], max_rr_std: [...], max_rr_ci95: [...],
               max_bt: [...], max_bt_std: [...], max_bt_ci95: [...],
               final_rr: [...], final_rr_std: [...], final_rr_ci95: [...],
               final_bt: [...], final_bt_std: [...], final_bt_ci95: [...]}}
    """
    result = {}

    for env in sorted(runs_df["env"].unique()):
        env_df = runs_df[runs_df["env"] == env]
        noise_levels = sorted(env_df["noise"].unique())

        max_rt, max_bt = [], []
        max_rt_std, max_bt_std = [], []
        max_rt_ci95, max_bt_ci95 = [], []
        final_rt, final_bt = [], []
        final_rt_std, final_bt_std = [], []
        final_rt_ci95, final_bt_ci95 = [], []

        for noise in noise_levels:
            noise_df = env_df[env_df["noise"] == noise]

            for (
                is_rt,
                max_list,
                max_std_list,
                max_ci95_list,
                final_list,
                final_std_list,
                final_ci95_list,
            ) in [
                (
                    True,
                    max_rt,
                    max_rt_std,
                    max_rt_ci95,
                    final_rt,
                    final_rt_std,
                    final_rt_ci95,
                ),
                (
                    False,
                    max_bt,
                    max_bt_std,
                    max_bt_ci95,
                    final_bt,
                    final_bt_std,
                    final_bt_ci95,
                ),
            ]:
                variant_df = noise_df[noise_df["is_rt"] == is_rt]

                if variant_df.empty:
                    raise ValueError()

                max_rewards, final_rewards = [], []
                for _, seed_row in variant_df.iterrows():
                    rewards = json.loads(seed_row["reward_history"])
                    if not rewards:
                        seed = seed_row["seed"]
                        print(
                            f"Warning! Empty rewards history for {env=}, {noise=}, {is_rt=}, {seed=}"
                        )
                    if rewards:
                        max_rewards.append(max(rewards))
                        final_rewards.append(rewards[-1])

                num_seeds = len(max_rewards)
                if num_seeds < SEEDS_EXPECTED:
                    print(
                        f"Warning! Num seeds for {env=} ({noise=}, {is_rt=}): {len(max_rewards)}"
                    )
                max_list.append(np.mean(max_rewards))
                max_std_list.append(np.std(max_rewards))
                max_ci95_list.append(calculate_ci95_half_width(max_rewards))
                final_list.append(np.mean(final_rewards))
                final_std_list.append(np.std(final_rewards))
                final_ci95_list.append(calculate_ci95_half_width(final_rewards))

        result[env] = {
            "noise": noise_levels,
            "max_rr": max_rt,
            "max_rr_std": max_rt_std,
            "max_rr_ci95": max_rt_ci95,
            "max_bt": max_bt,
            "max_bt_std": max_bt_std,
            "max_bt_ci95": max_bt_ci95,
            "final_rr": final_rt,
            "final_rr_std": final_rt_std,
            "final_rr_ci95": final_rt_ci95,
            "final_bt": final_bt,
            "final_bt_std": final_bt_std,
            "final_bt_ci95": final_bt_ci95,
        }

    return result


def generate_combined_reward_bars(
    data: dict,
    output_path: Path,
) -> None:
    """Generate 2-row bar chart with max reward (top) and final reward (bottom)."""
    rc("text", usetex=True)
    rc("font", family="serif")
    setup_plot_style()

    fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.0))
    bar_width = 0.35

    rows = [
        ("max_bt", "max_rr", "Max Reward"),
        ("final_bt", "final_rr", "Final Reward"),
    ]

    for row_idx, (bt_key, rr_key, ylabel) in enumerate(rows):
        for col_idx, (env_name, env_data) in enumerate(data.items()):
            ax = axes[row_idx, col_idx]
            noise_levels = env_data["noise"]
            x_positions = np.arange(len(noise_levels))

            ax.bar(
                x_positions - bar_width / 2,
                env_data[bt_key],
                bar_width,
                yerr=env_data[bt_key + "_ci95"],
                label="BT",
                color=BT_COLOR,
                capsize=2,
            )
            ax.bar(
                x_positions + bar_width / 2,
                env_data[rr_key],
                bar_width,
                yerr=env_data[rr_key + "_ci95"],
                label="RR",
                color=RR_COLOR,
                capsize=2,
            )

            if row_idx == 1:
                ax.set_xlabel("Noise", fontsize=8.0)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=8.0)
            if row_idx == 0:
                ax.set_title(env_name, fontsize=8.0)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"{n:g}" for n in noise_levels], fontsize=8.0)
            spine_color = ax.spines["left"].get_edgecolor()
            ax.tick_params(
                axis="y",
                which="major",
                direction="inout",
                length=4,
                width=1.0,
                labelsize=8.0,
                left=True,
                colors=spine_color,
                labelcolor="black",
            )
            ax.tick_params(axis="x", labelsize=8.0)
            ax.grid(True, alpha=0.3, axis="y")

    for col_idx in range(4):
        ymin = min(axes[0, col_idx].get_ylim()[0], axes[1, col_idx].get_ylim()[0])
        ymax = max(axes[0, col_idx].get_ylim()[1], axes[1, col_idx].get_ylim()[1])
        axes[0, col_idx].set_ylim(ymin, ymax)
        axes[1, col_idx].set_ylim(ymin, ymax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=8.0,
        frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93], w_pad=0.3, h_pad=0.5, pad=0.2)
    save_figure_reproducible(fig, str(output_path))
    plt.close(fig)

    print(f"Saved: {output_path}")


def generate_combined_reward_lines(
    data: dict,
    output_path: Path,
) -> None:
    """Generate 2-row line plot with max reward (top) and final reward (bottom)."""
    rc("text", usetex=True)
    rc("font", family="serif")
    setup_plot_style()

    fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.0))

    rows = [
        ("max_bt", "max_rr", "Max Reward"),
        ("final_bt", "final_rr", "Final Reward"),
    ]

    for row_idx, (bt_key, rr_key, ylabel) in enumerate(rows):
        for col_idx, (env_name, env_data) in enumerate(data.items()):
            ax = axes[row_idx, col_idx]
            noise_levels = env_data["noise"]

            bt_mean = np.array(env_data[bt_key])
            bt_ci95 = np.array(env_data[bt_key + "_ci95"])
            rr_mean = np.array(env_data[rr_key])
            rr_ci95 = np.array(env_data[rr_key + "_ci95"])

            ax.plot(
                noise_levels,
                bt_mean,
                marker="^",
                linestyle="--",
                color=BT_COLOR,
                label="BT",
                markersize=4,
                linewidth=1.5,
            )
            ax.fill_between(
                noise_levels,
                bt_mean - bt_ci95,
                bt_mean + bt_ci95,
                color=BT_COLOR,
                alpha=0.2,
            )
            ax.plot(
                noise_levels,
                rr_mean,
                marker="o",
                linestyle="-",
                color=RR_COLOR,
                label="RR",
                markersize=4,
                linewidth=1.5,
            )
            ax.fill_between(
                noise_levels,
                rr_mean - rr_ci95,
                rr_mean + rr_ci95,
                color=RR_COLOR,
                alpha=0.2,
            )

            if row_idx == 1:
                ax.set_xlabel("Noise", fontsize=8.0)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=8.0)
            if row_idx == 0:
                ax.set_title(env_name, fontsize=8.0)
            ax.set_xticks(noise_levels)
            ax.set_xticklabels([f"{n:g}" for n in noise_levels], fontsize=8.0)
            spine_color = ax.spines["left"].get_edgecolor()
            ax.tick_params(
                axis="y",
                which="major",
                direction="inout",
                length=4,
                width=1.0,
                labelsize=8.0,
                left=True,
                colors=spine_color,
                labelcolor="black",
            )
            ax.tick_params(axis="x", labelsize=8.0)
            ax.grid(True, alpha=0.3)

    for col_idx in range(4):
        ymin = min(axes[0, col_idx].get_ylim()[0], axes[1, col_idx].get_ylim()[0])
        ymax = max(axes[0, col_idx].get_ylim()[1], axes[1, col_idx].get_ylim()[1])
        axes[0, col_idx].set_ylim(ymin, ymax)
        axes[1, col_idx].set_ylim(ymin, ymax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=8.0,
        frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93], w_pad=0.3, h_pad=0.5, pad=0.2)
    save_figure_reproducible(fig, str(output_path))
    plt.close(fig)

    print(f"Saved: {output_path}")


def aggregate_training_curves(
    variant_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate training curves across seeds via interpolation.

    Returns:
        Tuple of (step_grid, mean_reward, std_reward) arrays.
    """
    seed_curves = []
    max_step = 0

    for _, row in variant_df.iterrows():
        steps = json.loads(row["global_step_history"])
        rewards = json.loads(row["reward_history"])
        max_step = max(max_step, max(steps))
        seed_curves.append((np.array(steps), np.array(rewards)))

    if len(seed_curves) == 0:
        raise ValueError("No training histories found in variant_df")

    step_grid = np.arange(0, int(max_step) + 1, 1000)

    # Linearly interpolate each seed's rewards onto the common step grid
    interpolated = [
        np.interp(step_grid, steps, rewards) for steps, rewards in seed_curves
    ]
    aligned_rewards = np.stack(interpolated)

    return step_grid, np.mean(aligned_rewards, axis=0), np.std(aligned_rewards, axis=0)


def generate_training_curves(runs_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate training curve plots with mean +/- std shaded bands.

    Layout: 2x2 grid per environment (one subplot per noise level).
    """
    rc("text", usetex=True)
    rc("font", family="serif")
    setup_plot_style()

    for env in sorted(runs_df["env"].unique()):
        env_df = runs_df[runs_df["env"] == env]
        noise_levels = sorted(env_df["noise"].unique())
        assert len(noise_levels) == 4

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes = axes.flatten()

        global_ymin, global_ymax = float("inf"), float("-inf")

        for idx, noise in enumerate(noise_levels):
            ax = axes[idx]

            for is_rt, color, label in [
                (True, RT_CURVE_COLOR, "RT"),
                (False, BT_CURVE_COLOR, "Non-RT"),
            ]:
                variant_df = env_df[
                    (env_df["noise"] == noise) & (env_df["is_rt"] == is_rt)
                ]

                assert not variant_df.empty
                steps, mean_reward, std_reward = aggregate_training_curves(variant_df)

                ax.plot(steps, mean_reward, color=color, label=label, linewidth=2)
                ax.fill_between(
                    steps,
                    mean_reward - std_reward,
                    mean_reward + std_reward,
                    color=color,
                    alpha=0.2,
                )

                global_ymin = min(global_ymin, np.min(mean_reward - std_reward))
                global_ymax = max(global_ymax, np.max(mean_reward + std_reward))

            ax.set_title(f"Noise: {noise:g}", fontsize=12)
            ax.set_xlabel("Global Steps", fontsize=10)
            ax.set_ylabel("Mean Reward", fontsize=10)
            ax.tick_params(axis="both", which="major", labelsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        for ax in axes:
            ax.set_ylim(global_ymin, global_ymax)

        fig.suptitle(
            f"RL Training Curves for {env}: RT vs Non-RT Comparison", fontsize=16
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)

        output_path = output_dir / f"rl_training_curves_{env}.pdf"
        save_figure_reproducible(fig, str(output_path))
        plt.close(fig)
        print(f"Saved: {output_path}")


def generate_rl_control_table(
    data: dict, output_path: Path, metric: str, title: str
) -> None:
    """Generate LaTeX table for RL control results.

    Args:
        data: Summary stats from compute_summary_stats
        output_path: Path to write the table
        metric: Either "max" or "final"
        title: Title for the metric column (e.g., "Max Reward" or "Final Reward")
    """
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "\\textbf{Environment / noise} & \\textbf{BT} & \\textbf{RR} \\\\",
        "\\midrule",
    ]

    for env_idx, (env_name, env_data) in enumerate(data.items()):
        noise_levels = env_data["noise"]
        bt_means = env_data[f"{metric}_bt"]
        bt_stds = env_data[f"{metric}_bt_std"]
        bt_ci95s = env_data[f"{metric}_bt_ci95"]
        rr_means = env_data[f"{metric}_rr"]
        rr_stds = env_data[f"{metric}_rr_std"]
        rr_ci95s = env_data[f"{metric}_rr_ci95"]

        for i, noise in enumerate(noise_levels):
            rr_is_best = rr_means[i] > bt_means[i]
            bt_mean_str = (
                f"\\mathbf{{{bt_means[i]:.1f}}}"
                if not rr_is_best
                else f"{bt_means[i]:.1f}"
            )
            rr_mean_str = (
                f"\\mathbf{{{rr_means[i]:.1f}}}" if rr_is_best else f"{rr_means[i]:.1f}"
            )
            bt_cell = (
                f"${bt_mean_str} \\pm {bt_stds[i]:.1f}$ [$\\pm$ {bt_ci95s[i]:.1f}]"
            )
            rr_cell = (
                f"${rr_mean_str} \\pm {rr_stds[i]:.1f}$ [$\\pm$ {rr_ci95s[i]:.1f}]"
            )

            lines.append(f"{env_name} / noise={noise:g} & {bt_cell} & {rr_cell} \\\\")

        if env_idx < len(data) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {output_path}")


def load_rew_model_data(cache_dir: Path) -> pd.DataFrame:
    """Load reward model accuracy data from CSV files.

    Returns DataFrame with final validation accuracy per run.
    """
    mujoco_path = cache_dir / "mujoco_rew_model_data.csv"
    highway_path = cache_dir / "highway_rew_model_data.csv"

    if not mujoco_path.exists():
        raise FileNotFoundError(
            f"Reward model cache not found: {mujoco_path}\n"
            f"Copy CSV files to {cache_dir}/"
        )

    dfs = []
    for path in [mujoco_path, highway_path]:
        if not path.exists():
            continue

        df = pd.read_csv(path)

        # Get final accuracy per run (last row per run_id)
        final_rows = df.groupby("run_id").last().reset_index()

        rows = []
        for _, row in final_rows.iterrows():
            run_name = row["run_name"]
            info = parse_rew_model_run_name(run_name)
            rows.append(
                {
                    "run_name": run_name,
                    "env": info.env,
                    "noise": info.noise,
                    "is_rt": info.is_rt,
                    "seed": info.seed,
                    "val_accuracy": row["val_accuracy"],
                }
            )

        dfs.append(pd.DataFrame(rows))

    return pd.concat(dfs, ignore_index=True)


def compute_rew_model_stats(rew_model_df: pd.DataFrame) -> dict:
    """Compute reward model accuracy stats grouped by (env, noise, is_rt).

    Returns dict with structure:
        {env: {noise: [...],
               rr_mean: [...], rr_std: [...], rr_ci95: [...],
               bt_mean: [...], bt_std: [...], bt_ci95: [...]}}
    """
    result = {}

    for env in sorted(rew_model_df["env"].unique()):
        env_df = rew_model_df[rew_model_df["env"] == env]
        noise_levels = sorted(env_df["noise"].unique())

        rr_means, rr_stds, rr_ci95s = [], [], []
        bt_means, bt_stds, bt_ci95s = [], [], []

        for noise in noise_levels:
            noise_df = env_df[env_df["noise"] == noise]

            for is_rt, mean_list, std_list, ci95_list in [
                (True, rr_means, rr_stds, rr_ci95s),
                (False, bt_means, bt_stds, bt_ci95s),
            ]:
                variant_df = noise_df[noise_df["is_rt"] == is_rt]
                accuracies = variant_df["val_accuracy"].values

                mean_list.append(np.mean(accuracies))
                std_list.append(np.std(accuracies))
                ci95_list.append(calculate_ci95_half_width(accuracies))

        result[env] = {
            "noise": noise_levels,
            "rr_mean": rr_means,
            "rr_std": rr_stds,
            "rr_ci95": rr_ci95s,
            "bt_mean": bt_means,
            "bt_std": bt_stds,
            "bt_ci95": bt_ci95s,
        }

    return result


def generate_rew_model_accuracy_table(stats: dict, output_path: Path) -> None:
    """Generate LaTeX table for reward model validation accuracies.

    Format: mean +- std [+- CI].
    """
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "\\textbf{Environment / noise} & ",
        "\\textbf{RR} & ",
        "\\textbf{BT} \\\\",
        "\\midrule",
    ]

    for env_idx, (env_name, env_data) in enumerate(stats.items()):
        noise_levels = env_data["noise"]
        rr_means = env_data["rr_mean"]
        rr_stds = env_data["rr_std"]
        rr_ci95s = env_data["rr_ci95"]
        bt_means = env_data["bt_mean"]
        bt_stds = env_data["bt_std"]
        bt_ci95s = env_data["bt_ci95"]

        for i, noise in enumerate(noise_levels):
            rr_is_best = rr_means[i] > bt_means[i]
            rr_mean_str = (
                f"\\mathbf{{{rr_means[i]:.3f}}}" if rr_is_best else f"{rr_means[i]:.3f}"
            )
            bt_mean_str = (
                f"\\mathbf{{{bt_means[i]:.3f}}}"
                if not rr_is_best
                else f"{bt_means[i]:.3f}"
            )
            rr_cell = (
                f"${rr_mean_str} \\pm {rr_stds[i]:.3f}$ [$\\pm$ {rr_ci95s[i]:.3f}]"
            )
            bt_cell = (
                f"${bt_mean_str} \\pm {bt_stds[i]:.3f}$ [$\\pm$ {bt_ci95s[i]:.3f}]"
            )

            lines.append(f"{env_name} / noise={noise:g} & {rr_cell} & {bt_cell} \\\\")

        if env_idx < len(stats) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {output_path}")


def collect_rl_control_data(
    cache_dir: Path, mujoco_project: str, highway_project: str
) -> None:
    """Fetch RL control data from wandb and save to CSV cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    mujoco_runs = fetch_rl_runs(
        api,
        project=mujoco_project,
        run_filter={"display_name": {"$regex": "^RL_ppo_.*nfeedback_5000"}},
    )
    mujoco_path = cache_dir / "mujoco_runs.csv"
    mujoco_runs.to_csv(mujoco_path, index=False)
    print(f"Saved {len(mujoco_runs)} MuJoCo runs to {mujoco_path}")

    highway_runs = fetch_rl_runs(
        api,
        project=highway_project,
        run_filter={"display_name": {"$regex": "^RL_.*nfeedback_5000"}},
    )
    highway_path = cache_dir / "highway_runs.csv"
    highway_runs.to_csv(highway_path, index=False)
    print(f"Saved {len(highway_runs)} Highway runs to {highway_path}")


def collect_reward_model_control_data(
    cache_dir: Path, mujoco_project: str, highway_project: str
) -> None:
    """Fetch RL control data from wandb and save to CSV cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    mujoco_runs = fetch_reward_model_runs(
        api,
        project=mujoco_project,
        run_filter={"display_name": {"$regex": "^ppo_.*nfeedback_5000"}},
    )
    mujoco_path = cache_dir / "mujoco_rew_model_data.csv"
    mujoco_runs.to_csv(mujoco_path, index=False)
    print(f"Saved RM data for {len(mujoco_runs)} MuJoCo runs to {mujoco_path}")

    highway_runs = fetch_reward_model_runs(
        api,
        project=highway_project,
        run_filter={"display_name": {"$regex": "^ppo_.*nfeedback_5000"}},
    )
    highway_path = cache_dir / "highway_rew_model_data.csv"
    highway_runs.to_csv(highway_path, index=False)
    print(f"Saved RM data for {len(highway_runs)} Highway runs to {highway_path}")


def fetch_rl_runs(api, project: str, run_filter: dict) -> pd.DataFrame:
    """Fetch RL training runs with history data from wandb."""
    runs = api.runs(project, filters=run_filter)

    records = []
    for run in runs:
        if "ensemble" in run.name:
            continue

        info = parse_run_name(run.name)

        step_col = "global_step"
        # We report dedicated eval reward (10 episodes, as per stable baseline default) for
        # all envs instead of merge. The merge env has very short episodes (~15 steps) and
        # high variance in the resulting rewards. The 10 episode eval estimate is too noisy.
        # Therefor we use mean reward (sliding window over the last 100 training episodes)
        # instead. If we were to re-run this experiment, we should use more eval episodes.
        reward_col = (
            "rollout/ep_rew_mean" if "merge" in info.env else "eval/mean_reward"
        )

        history = run.history(keys=[reward_col, step_col], pandas=True)

        steps = history[step_col].dropna().tolist()
        rewards = history[reward_col].dropna().tolist()

        records.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "env": info.env,
                "noise": info.noise,
                "is_rt": info.is_rt,
                "seed": info.seed,
                "global_step_history": json.dumps(steps),
                "reward_history": json.dumps(rewards),
            }
        )

    return pd.DataFrame(records)


def fetch_reward_model_runs(api, project: str, run_filter: dict) -> pd.DataFrame:
    """Fetch RL training runs with history data from wandb."""
    runs = api.runs(project, filters=run_filter)

    records = []
    for run in runs:
        if "ensemble" in run.name:
            continue

        info = parse_run_name(run.name)
        val_acc_column = "val_accuracy"

        summary = run.summary._json_dict

        records.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "env": info.env,
                "noise": info.noise,
                "is_rt": info.is_rt,
                "seed": info.seed,
                "val_accuracy": summary.get(val_acc_column, 0.0),
            }
        )

    return pd.DataFrame(records)


def generate_all_plots(data_dir: Path, out_dir: Path) -> None:
    """Generate all RL control plots and tables."""
    runs_df = load_rl_cache(data_dir)
    data = compute_summary_stats(runs_df)

    generate_combined_reward_bars(
        data,
        out_dir / "rl_control_combined_reward_bars.pdf",
    )
    generate_combined_reward_lines(
        data,
        out_dir / "rl_control_combined_reward_lines.pdf",
    )
    generate_rl_control_table(
        data,
        out_dir / "rl_control_max_reward_table.tex",
        metric="max",
        title="Max Reward",
    )
    generate_rl_control_table(
        data,
        out_dir / "rl_control_final_reward_table.tex",
        metric="final",
        title="Final Reward",
    )
    generate_training_curves(runs_df, out_dir)

    rew_model_df = load_rew_model_data(data_dir)
    rew_model_stats = compute_rew_model_stats(rew_model_df)
    generate_rew_model_accuracy_table(
        rew_model_stats,
        out_dir / "rl_control_rew_model_accuracy_table.tex",
    )


def main():
    parser = argparse.ArgumentParser(description="RL control analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    collect_parser = subparsers.add_parser(
        "collect", help="Fetch data from wandb and save to cache"
    )
    collect_parser.add_argument(
        "cache_dir",
        type=Path,
        help="Directory for cached CSV files",
    )
    collect_parser.add_argument(
        "--mujoco-project",
        type=str,
        required=True,
        help="wandb project name for MuJoCo runs",
    )
    collect_parser.add_argument(
        "--highway-project",
        type=str,
        required=True,
        help="wandb project name for Highway runs",
    )

    generate_parser = subparsers.add_parser(
        "generate", help="Generate all plots and tables from cached data"
    )
    generate_parser.add_argument(
        "cache_dir",
        type=Path,
        help="Directory containing cached CSV files",
    )
    generate_parser.add_argument(
        "figures_dir",
        type=Path,
        help="Directory to save figures",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "collect":
        collect_rl_control_data(
            args.cache_dir, args.mujoco_project, args.highway_project
        )
        collect_reward_model_control_data(
            args.cache_dir, args.mujoco_project, args.highway_project
        )
    elif args.command == "generate":
        generate_all_plots(args.cache_dir, args.figures_dir)


if __name__ == "__main__":
    main()
