"""Rank transformation classes."""

import numpy as np
from scipy.stats import kendalltau, rankdata


class RankTransform:
    """Base class for rank transformations applied after processing."""

    def transform(self, examples, rng):
        """Transform ranks in-place. Returns statistics dict."""
        raise NotImplementedError


class NoOpRankTransform(RankTransform):
    """No-operation rank transform that leaves ranks unchanged."""

    def transform(self, examples, rng):
        """No-op transform that returns empty statistics dict."""
        # Add rank_pre_transform for consistency with other transforms
        for ex in examples:
            if "rank" in ex:
                ex["extra"]["rank_pre_transform"] = ex["rank"]
        return {}


class PartialShuffleTransform(RankTransform):
    """Apply noise by shuffling a random subset of items among themselves.

    This transform selects a random subset of items (size = noise_level * n)
    and shuffles their values among themselves, while the rest stay unchanged.
    This allows large jumps for the selected items while keeping most items
    unchanged at low noise levels.

    Example (n=10, noise_level=0.3):
    - Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    - Select 3 items randomly: indices [2, 5, 8] (values [3, 6, 9])
    - Shuffle among themselves: [9, 3, 6]
    - Result: [1, 2, 9, 4, 5, 3, 7, 8, 6, 10]

    Properties:
    - Exact endpoints: noise_level=0 preserves original, noise_level=1 fully shuffles
    - Allows rare large jumps: at low noise, few items can jump anywhere
    - No smooth locality: items either stay or participate in shuffle

    Args:
        noise_level: Float in [0,1]. Fraction of items to shuffle among themselves.
                    0 = no shuffling, 1 = full shuffle
    """

    def __init__(self, noise_level: float):
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError(f"noise_level must be in [0,1], got {noise_level}")
        self.noise_level = noise_level

    def transform(self, examples, rng):
        """Apply partial shuffle to a random subset of examples.

        Selects subset of items and shuffles their values among themselves.
        """
        n = len(examples)
        current_values = np.array([ex["rank"] for ex in examples])

        for i, ex in enumerate(examples):
            ex["extra"]["rank_pre_transform"] = current_values[i]

        n_shuffle = int(round(self.noise_level * n))
        if n_shuffle == 0:
            noised_values = current_values.copy()
        else:
            indices = rng.sample(range(n), n_shuffle)

            subset_values = current_values[indices].copy()
            rng.shuffle(subset_values)

            noised_values = current_values.copy()
            noised_values[indices] = subset_values

        for i, ex in enumerate(examples):
            ex["rank"] = noised_values[i]

        current_positions = rankdata(current_values, method="ordinal") - 1
        noised_positions = rankdata(noised_values, method="ordinal") - 1
        kendall_tau, _ = kendalltau(current_positions, noised_positions)

        return {
            "transform/kendall_tau": kendall_tau,
        }


class RandomRankTransform(RankTransform):
    """Assign completely random ranks, ignoring all existing rank information."""

    def transform(self, examples, rng):
        """Assign random ranks to all examples."""
        n = len(examples)
        current_ranks = np.array([ex["rank"] for ex in examples])

        for i, ex in enumerate(examples):
            ex["extra"]["rank_pre_transform"] = current_ranks[i]

        random_ranks = np.arange(n)
        rng.shuffle(random_ranks)

        for i, ex in enumerate(examples):
            ex["rank"] = random_ranks[i]

        current_positions = rankdata(current_ranks, method="ordinal") - 1
        kendall_tau, _ = kendalltau(current_positions, random_ranks)

        return {
            "transform/kendall_tau": kendall_tau,
        }
