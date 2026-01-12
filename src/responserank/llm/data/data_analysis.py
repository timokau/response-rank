import os

import numpy as np
import pandas as pd


def analyze_response_times(
    dataset,
    output_dir="plots",
    response_time_col="extra_response_time",
    preference_strength_col="extra_preference_strength",
):
    """Analyze response time statistics and distributions

    Args:
        dataset: Dataset containing response time data
        output_dir: Directory to save plots and statistics
        response_time_col: Name of the column containing response times (default: extra_response_time)
        preference_strength_col: Name of the column containing preference strength (default: extra_preference_strength)

    Returns:
        Dictionary of statistics
    """
    if response_time_col not in dataset.column_names:
        print(f"No {response_time_col} column in dataset, skipping analysis")
        return {}

    response_times = dataset[response_time_col]
    # Use ordinal values: 1=slight, 2=clear
    strength_map = {1: [], 2: []}
    strength_labels = {1: "Slight", 2: "Clear"}

    if preference_strength_col in dataset.column_names:
        for rt, strength in zip(response_times, dataset[preference_strength_col]):
            if strength in strength_map:
                strength_map[strength].append(rt)

    stats = {
        "Overall RT - Mean": np.mean(response_times),
        "Overall RT - Median": np.median(response_times),
        "Overall RT - Min": np.min(response_times),
        "Overall RT - Max": np.max(response_times),
        "Overall RT - Std Dev": np.std(response_times),
    }

    for strength, times in strength_map.items():
        if times:
            label = strength_labels.get(strength, f"Strength {strength}")
            stats.update(
                {
                    f"{label} Preference RT - Mean": np.mean(times),
                    f"{label} Preference RT - Median": np.median(times),
                    f"{label} Preference RT - Count": len(times),
                }
            )

    # Check correlation between preference strength and response time
    clear_mean = np.mean(strength_map[2]) if strength_map[2] else 0
    slight_mean = np.mean(strength_map[1]) if strength_map[1] else 0

    if clear_mean and slight_mean:
        stats["RT Ratio (Slight/Clear)"] = slight_mean / clear_mean

    print("\n===== Response Time Statistics =====")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    print("=" * 37 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        if strength_map[2]:  # Clear preference (strength=2)
            plt.hist(strength_map[2], alpha=0.5, label="Clear Preference", bins=20)
        if strength_map[1]:  # Slight preference (strength=1)
            plt.hist(strength_map[1], alpha=0.5, label="Slight Preference", bins=20)
        plt.legend(loc="upper right")
        plt.xlabel("Response Time (seconds)")
        plt.ylabel("Frequency")
        plt.title("Response Time Distribution by Preference Strength")
        plot1_path = f"{output_dir}/rt_by_strength.png"
        plt.savefig(plot1_path)
        plt.close()

        # Normalized version (fraction instead of frequency)
        plt.figure(figsize=(10, 6))
        if strength_map[2]:  # Clear preference (strength=2)
            plt.hist(
                strength_map[2],
                alpha=0.5,
                label="Clear Preference",
                bins=20,
                density=True,
            )
        if strength_map[1]:  # Slight preference (strength=1)
            plt.hist(
                strength_map[1],
                alpha=0.5,
                label="Slight Preference",
                bins=20,
                density=True,
            )
        plt.legend(loc="upper right")
        plt.xlabel("Response Time (seconds)")
        plt.ylabel("Fraction")
        plt.title("Response Time Distribution by Preference Strength (Normalized)")
        plot1_norm_path = f"{output_dir}/rt_by_strength_normalized.png"
        plt.savefig(plot1_norm_path)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(response_times, bins=30, alpha=0.7, color="blue")
        plt.xlabel("Response Time (seconds)")
        plt.ylabel("Frequency")
        plt.title("Overall Response Time Distribution")
        plt.axvline(
            np.mean(response_times),
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {np.mean(response_times):.2f}s",
        )
        plt.axvline(
            np.median(response_times),
            color="green",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {np.median(response_times):.2f}s",
        )
        plt.legend()
        plot2_path = f"{output_dir}/rt_overall.png"
        plt.savefig(plot2_path)
        plt.close()

        print(f"Saved histograms to {output_dir}/")

        stats_df = pd.DataFrame([stats])
        stats_path = f"{output_dir}/rt_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved statistics to {stats_path}")

    except ImportError:
        print("Matplotlib not available, skipping histogram generation")

    return stats
