from typing import Callable, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import torch

from responserank.synthetic.losses import compute_pl_log_likelihood_of_ranking
from responserank.synthetic.util import extract_item_features_from_pandas_df


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        float: Pearson correlation coefficient.
    """
    return np.corrcoef(x, y)[0, 1]


def compute_item_utilities(
    df: pd.DataFrame, uf: Callable
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and compute utilities for item pairs in a dataframe.

    Args:
        df: DataFrame containing paired item features
        uf: Utility function that predicts utilities from features

    Returns:
        Tuple containing (u1_pred, u2_pred) - predicted utilities for both items
    """
    x1 = extract_item_features_from_pandas_df(df, "x1")
    x2 = extract_item_features_from_pandas_df(df, "x2")

    u1_pred = uf(x1)
    u2_pred = uf(x2)

    return u1_pred, u2_pred


def _get_true_utilities(df: pd.DataFrame) -> np.ndarray:
    """Helper function to extract and combine true utilities from dataframe."""
    return np.concatenate([df["u1"].to_numpy(), df["u2"].to_numpy()])


def _get_predicted_utilities(df: pd.DataFrame, uf: Callable) -> np.ndarray:
    """Helper function to compute and combine predicted utilities."""
    u1_pred, u2_pred = compute_item_utilities(df, uf)
    return np.concatenate([u1_pred, u2_pred])


def _get_utility_differences(
    df: pd.DataFrame, uf: Callable
) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to compute true and predicted utility differences.

    Args:
        df: DataFrame containing paired item features and true utilities
        uf: Utility function that predicts utilities

    Returns:
        Tuple containing (true_diffs, predicted_diffs)
    """
    # Compute true differences
    udiff_true = df["u2"].to_numpy() - df["u1"].to_numpy()

    # Compute predicted differences
    u1_pred, u2_pred = compute_item_utilities(df, uf)
    udiff_pred = u2_pred - u1_pred

    return udiff_true, udiff_pred


def _compute_bt_probabilities(udiff: np.ndarray) -> np.ndarray:
    """Convert utility differences to Bradley-Terry choice probabilities.

    Args:
        udiff: Array of utility differences (u2 - u1)

    Returns:
        Array of probabilities using the Bradley-Terry model
    """
    return 1 / (1 + np.exp(-udiff))


def pearson_utility_correlation(
    df: pd.DataFrame,
    uf: Callable,
) -> float:
    """Measure Pearson correlation between true and predicted utilities.

    This metric measures how well a model can predict the true utilities directly.

    Args:
        df: DataFrame containing true utilities and features
        uf: Utility function that predicts utilities

    Returns:
        float: Pearson correlation coefficient
    """
    utotal = _get_true_utilities(df)
    utotalprime = _get_predicted_utilities(df, uf)
    return pearson_correlation(utotal, utotalprime)


def pearson_difference_correlation(df: pd.DataFrame, uf: Callable) -> float:
    """Measure Pearson correlation between true and predicted utility differences.

    This metric measures how well a model can predict the direction and magnitude
    of utility differences.

    Args:
        df: DataFrame containing true utilities and features
        uf: Utility function that predicts utilities

    Returns:
        float: Pearson correlation coefficient
    """
    udiff_true, udiff_pred = _get_utility_differences(df, uf)
    return pearson_correlation(udiff_true, udiff_pred)


def pearson_distance_correlation(df: pd.DataFrame, uf: Callable) -> float:
    """Compute Pearson Distance Correlation (PDC) between true and predicted utility differences.

    PDC is the Pearson correlation between the absolute true utility differences and
    absolute predicted utility differences. This metric measures how well a model learns
    the magnitude/strength of preferences, independent of their direction.

    Args:
        df: DataFrame containing true utilities and features
        uf: Utility function that predicts utilities

    Returns:
        float: Pearson Distance Correlation coefficient
    """
    udiff_true, udiff_pred = _get_utility_differences(df, uf)
    udist_true = np.abs(udiff_true)
    udist_pred = np.abs(udiff_pred)
    return pearson_correlation(udist_true, udist_pred)


def pearson_utility_rank_correlation(df: pd.DataFrame) -> float:
    """Measure Pearson correlation between true utilities and their ranks.

    This metric quantifies the relationship between utility values and their ordinal ranks.
    It measures to what extent utilities increase linearly with their rank positions.
    A perfect correlation (1.0) indicates that utilities are perfectly linearly related
    to their ranks, meaning equal steps in rank correspond to equal steps in utility.
    Lower correlations indicate that some rank transitions involve larger utility
    differences than others, which can happen when utility values form clusters
    with gaps between them.

    Args:
        df: DataFrame containing true utilities

    Returns:
        float: Pearson correlation coefficient between true utilities and their ranks
    """
    utotal = _get_true_utilities(df)
    return pearson_correlation(utotal, np.argsort(np.argsort(utotal)))


def spearman_utility_correlation(df: pd.DataFrame, uf: Callable) -> float:
    """Measure Spearman rank correlation between true and predicted utilities.

    This metric measures how well a model ranks the true utilities directly.

    Args:
        df: DataFrame containing true utilities and features
        uf: Utility function that predicts utilities

    Returns:
        float: Spearman rank correlation coefficient
    """
    utotal = _get_true_utilities(df)
    utotalprime = _get_predicted_utilities(df, uf)
    spearman = scipy.stats.spearmanr(utotal, utotalprime)
    return spearman.correlation  # type: ignore


def spearman_difference_correlation(df: pd.DataFrame, uf: Callable) -> float:
    """Measure Spearman rank correlation between true and predicted utility differences.

    This metric measures how well a model ranks utility differences, considering
    both direction and magnitude.

    Args:
        df: DataFrame containing true utilities and features
        uf: Utility function that predicts utilities

    Returns:
        float: Spearman rank correlation coefficient
    """
    udiff_true, udiff_pred = _get_utility_differences(df, uf)
    spearman = scipy.stats.spearmanr(udiff_true, udiff_pred)
    return spearman.correlation  # type: ignore


def spearman_distance_correlation(df: pd.DataFrame, uf: Callable) -> float:
    """Measure Spearman rank correlation between absolute true and predicted utility differences.

    This metric measures how well a model ranks the magnitude of utility differences,
    independent of their direction.

    Args:
        df: DataFrame containing true utilities and features
        uf: Utility function that predicts utilities

    Returns:
        float: Spearman rank correlation coefficient
    """
    udiff_true, udiff_pred = _get_utility_differences(df, uf)
    udist_true = np.abs(udiff_true)
    udist_pred = np.abs(udiff_pred)

    spearman = scipy.stats.spearmanr(udist_true, udist_pred)
    return spearman.correlation  # type: ignore


def true_calibration_error_bt_probs(df: pd.DataFrame, uf: Callable) -> float:
    """Compute MSE between true and predicted Bradley-Terry choice probabilities.

    Args:
        df: DataFrame containing true utilities and features
        uf: Utility function that predicts utilities

    Returns:
        float: Mean squared error between true and predicted probabilities

    Example:
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'u1': [0.0, 1.0, -1.0],  # true utilities for option 1
        ...     'u2': [1.0, -1.0, 0.0],  # true utilities for option 2
        ...     'x1_0': [0.1, 0.2, 0.3], # features for option 1
        ...     'x2_0': [0.4, 0.5, 0.6]  # features for option 2
        ... })
        >>> # Simple utility function that returns feature values
        >>> def dummy_uf(x): return x[:, 0]
        >>> # True probabilities: [0.269, 0.881, 0.377]
        >>> # (from 1/(1 + exp(-(u2-u1))))
        >>> # Predicted probabilities: [0.575, 0.575, 0.575]
        >>> # (from 1/(1 + exp(-(x2-x1))))
        >>> mse = true_calibration_error_bt_probs(df, dummy_uf)
        >>> print(f"{mse:.3f}")  # Should be around 0.085
        0.085
    """
    udiff_true, udiff_pred = _get_utility_differences(df, uf)

    # Compute probabilities using Bradley-Terry model
    probs_true = _compute_bt_probabilities(udiff_true)
    probs_pred = _compute_bt_probabilities(udiff_pred)

    # Compute MSE
    return np.mean((probs_true - probs_pred) ** 2)


def expected_calibration_error_bt_probs(
    df: pd.DataFrame, uf: Callable, num_bins: int = 10
) -> float:
    """Compute Expected Calibration Error for Bradley-Terry probabilities.

    Args:
        df: DataFrame containing true utilities and features
        uf: Utility function that predicts utilities
        num_bins: Number of bins to use for calibration assessment

    Returns:
        float: Expected Calibration Error

    Example:
        >>> df = pd.DataFrame({
        ...     'u1': [1.0, 2.0], 'u2': [2.0, 1.0],
        ...     'x1_0': [0.1, 0.2], 'x2_0': [0.3, 0.4],
        ...     'y': [1, 0]
        ... })
        >>> def dummy_uf(x): return x[:, 0]  # dummy utility function
        >>> ece = expected_calibration_error_bt_probs(df, dummy_uf)
    """
    _, udiff_pred = _get_utility_differences(df, uf)

    # Compute predicted probabilities using Bradley-Terry model
    probs = _compute_bt_probabilities(udiff_pred)

    # Get actual outcomes
    y = df["y"].values

    # Create bins and compute ECE
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0

    for bin_start, bin_end in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Find predictions in this bin
        mask = (probs >= bin_start) & (probs < bin_end)
        if not np.any(mask):
            continue

        # Get mean predicted probability in bin
        bin_prob = np.mean(probs[mask])
        # Get actual accuracy in bin
        bin_acc = np.mean(y[mask])
        # Weight by fraction of samples in bin
        weight = np.sum(mask) / len(probs)

        # Add weighted absolute difference to ECE
        ece += weight * np.abs(bin_acc - bin_prob)

    return ece


def log_likelihood_pl_ranking(df: pd.DataFrame, uf: Callable) -> float:
    """Measure log-likelihood of true preference ranking under the Plackett-Luce model.

    This metric evaluates how well the model represents preference strengths by
    calculating the log-likelihood of the true preference ranking under the
    Plackett-Luce choice model. Higher log-likelihood values indicate better
    alignment between predicted utility differences and the true ranking of preferences.

    Args:
        df: DataFrame containing true utilities and features
        uf: Utility function that predicts utilities

    Returns:
        float: Log-likelihood of the true preference ranking
    """
    udiff_true, udiff_pred = _get_utility_differences(df, uf)

    udiff_true_tensor = torch.tensor(udiff_true, dtype=torch.float32)
    udiff_pred_tensor = torch.tensor(udiff_pred, dtype=torch.float32)

    # Sort in descending order to put the item with the highest true preference
    # strength first, which matches the intended order of the worths.
    ranking_true = torch.argsort(udiff_true_tensor, descending=True)
    item_worths = torch.exp(udiff_pred_tensor)

    log_likelihood_tensor = compute_pl_log_likelihood_of_ranking(
        item_worths=item_worths, ranking=ranking_true
    )
    return log_likelihood_tensor.item()
