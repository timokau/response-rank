"""
This module provides a collection of utility functions used throughout the codebase.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn


def extract_item_features_from_pandas_df(df, prefix):
    """Extracts the data from a pandas dataframe and returns it as a tensor

    Example:
    --------

    Create a dataframe with columns x1_0, x1_1, x2_0, x2_1, y
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x1_0': [1, 2, 3], 'x1_1': [4, 5, 6], 'x2_0': [7, 8, 9], 'x2_1': [10, 11, 12], 'y': [13, 14, 15]})
    >>> df
       x1_0  x1_1  x2_0  x2_1   y
    0     1     4     7    10  13
    1     2     5     8    11  14
    2     3     6     9    12  15
    >>> extract_item_features_from_pandas_df(df, 'x1')
    array([[1, 4],
           [2, 5],
           [3, 6]])
    """
    # Raise an error if the prefix is not in the columns
    columns = sorted([c for c in df.columns if c.startswith(f"{prefix}_")])
    if len(columns) == 0:
        raise ValueError(f"No columns with prefix {prefix} found in the dataframe.")
    return df[columns].to_numpy()


def evaluate_decision_accuracy(df):
    """Evaluates the decision accuracy of the true utilities stored in the dataframe.

    Example:
    --------
    Create a dataframe with columns u1, u2, y
    >>> import pandas as pd
    >>> df = pd.DataFrame({'u1': [1, 2, 3], 'u2': [4, 5, 6], 'y': [0, 1, 0]})
    >>> df
       u1  u2  y
    0   1   4  0
    1   2   5  1
    2   3   6  0

    As you can see, in 2 out of three cases (rows 0 and 2), the decision follows the utility function.
    >>> bool(np.isclose(evaluate_decision_accuracy(df), 2/3))
    True
    """
    y_pred = df["u1"] > df["u2"]
    return np.mean(y_pred == df["y"])


def evaluate_utility_function(df, uf):
    """Evaluates the utility function accuracy on the given dataframe.

    Example:
    --------
    Create a dataframe with columns x1_0, x2_0, y
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x1_0': [1, 2, 3], 'x2_0': [4, 5, 6], 'y': [0, 1, 0]})
    >>> df
       x1_0  x2_0  y
    0     1     4  0
    1     2     5  1
    2     3     6  0

    Define a utility function
    >>> def uf(x):
    ...     return x.sum(axis=-1)

    Note that the utility is equal to the first feature in the input, so the
    decision should be based on the first feature. This is the case for the
    first and the third row.
    >>> bool(np.isclose(evaluate_utility_function(df, uf), 2/3))
    True
    """
    x1 = extract_item_features_from_pandas_df(df, "x1")
    x2 = extract_item_features_from_pandas_df(df, "x2")
    u1prime = uf(x1)
    u2prime = uf(x2)

    y_pred = np.where(u1prime > u2prime, 1, np.where(u1prime < u2prime, 0, 0.5))
    return np.mean(y_pred == df["y"])


class UtilityFunction(nn.Module):
    """A flexible utility function that takes the item features as input and returns a scalar utility.

    Example:
    --------
    >>> uf = UtilityFunction(2, [10, 5], dropout_rate=None)
    >>> x = th.tensor([[1, 2], [3, 4]], dtype=th.float32)
    >>> uf(x).shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        num_item_features: int,
        hidden_layers: Sequence[int],
        *,
        dropout_rate: Optional[float],
    ):
        """Create a feed-forward utility network.

        Args:
            num_item_features: Number of features per item.
            hidden_layers: Sizes of the hidden layers to include in order.
            dropout_rate: Optional dropout probability. Pass ``None`` to disable dropout.
        """
        super(UtilityFunction, self).__init__()
        layers = []
        in_features = num_item_features
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x, apply_dropout=True):
        if not apply_dropout:
            self.model.eval()
        else:
            self.model.train()
        return self.model(x)


def module_to_numpy_uf(module):
    """Converts a PyTorch module to a numpy utility function.

    The numpy utility function takes a numpy array as input and returns a numpy
    array, with one less dimension.

    Example
    -------
    >>> module = nn.Linear(2, 1)
    >>> module.weight.data = th.tensor([[1, 2]], dtype=th.float32)
    >>> module.bias.data = th.tensor([3], dtype=th.float32)
    >>> uf = module_to_numpy_uf(module)
    >>> x = np.array([[1, 2], [3, 4]])

    We expect results to be 1 * 1 + 2 * 2 + 3 = 8 and 3 * 1 + 4 * 2 + 3 = 14.

    >>> module(th.tensor(x, dtype=th.float32)).detach()
    tensor([[ 8.],
            [14.]])
    >>> uf(x)
    array([ 8., 14.], dtype=float32)
    """
    return (
        lambda x: module(th.tensor(x, dtype=th.float32))
        .detach()
        .numpy()
        .reshape(*x.shape[:-1])
    )


def convert_to_serializable(obj):
    """Convert objects to JSON-serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def get_log_normal_mean_sd(mean=0, sd=1):
    """Given the mean and sd of a lognormal distribution, return the mean and sd of the underlying normal distribution.

    Example:
    --------
    >>> (log_mean, log_sd) = get_log_normal_mean_sd(2, 1)
    >>> (log_mean.round(2), log_sd.round(2))
    (np.float64(0.58), np.float64(0.47))
    >>> dist = np.random.RandomState(0).lognormal(log_mean, log_sd, size=2000)
    >>> np.mean(dist).round(1)
    np.float64(2.0)
    >>> np.std(dist).round(1)
    np.float64(1.0)

    The function can also generate multiple means and sds at once.
    >>> means = np.array([2, 2, 3])
    >>> sds = np.array([1, 1, 2])
    >>> log_means, log_sds = get_log_normal_mean_sd(means, sds)
    >>> log_means.round(2)
    array([0.58, 0.58, 0.91])
    >>> log_sds.round(2)
    array([0.47, 0.47, 0.61])

    The function raises a ValueError for non-positive inputs:
    >>> get_log_normal_mean_sd(0, 1)
    Traceback (most recent call last):
        ...
    ValueError: Input mean and sd must be positive
    """
    if np.any(mean <= 0) or np.any(sd <= 0):
        raise ValueError("Input mean and sd must be positive")
    log_mean = np.log(mean**2 / np.sqrt(mean**2 + sd**2))
    log_sd = np.sqrt(np.log(1 + sd**2 / mean**2))
    return log_mean, log_sd


def dataset_to_dataframe(x1s, x2s, u1s, u2s, ys, ts, partition_ids):
    num_x_features = x1s.shape[1]
    num_x_digits = len(str(num_x_features - 1))

    actual_partition_ids = (
        np.zeros(len(ys), dtype=int) if partition_ids is None else partition_ids
    )

    return pd.DataFrame(
        {
            **{f"x1_{i:0{num_x_digits}}": x1s[:, i] for i in range(num_x_features)},
            **{f"x2_{i:0{num_x_digits}}": x2s[:, i] for i in range(num_x_features)},
            "u1": u1s,
            "u2": u2s,
            "y": ys,
            "t": ts,
            "partition_id": actual_partition_ids,
        }
    )


def reorder_dataframe_such_that_udiff_always_positive(df):
    """Reorder the dataframe such that the difference between u1 and u2 is always positive.

    If the difference between u1 and u2 is negative, u1 and u2 and the
    corresponding attributes x1 and x2 are swapped.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'x1_0': [1, 2, 3],
    ...     'x1_1': [4, 5, 6],
    ...     'x2_0': [7, 8, 9],
    ...     'x2_1': [10, 11, 12],
    ...     'u1': [1, 3, 6],
    ...     'u2': [2, 4, 5],
    ... })
    >>> reorder_dataframe_such_that_udiff_always_positive(df)
       x1_0  x1_1  x2_0  x2_1  u1  u2
    0     7    10     1     4   2   1
    1     8    11     2     5   4   3
    2     3     6     9    12   6   5
    """
    df = df.copy()
    x1_columns = [col for col in df.columns if col.startswith("x1_")]
    x2_columns = [col for col in df.columns if col.startswith("x2_")]

    mask = df["u1"] < df["u2"]

    df.loc[mask, ["u1", "u2"]] = df.loc[mask, ["u2", "u1"]].values

    for x1_col, x2_col in zip(x1_columns, x2_columns):
        df.loc[mask, [x1_col, x2_col]] = df.loc[mask, [x2_col, x1_col]].values

    return df


def shuffle_utilities(df, random_state):
    """
    Shuffle the u1 and u2 values in a pandas DataFrame df, but do not maintain the sign of u1 - u2.

    Parameters:
    df : pandas DataFrame
        The DataFrame containing at least the columns 'u1' and 'u2'. May contain other columns which will remain unaffected.
    random_state : numpy RandomState
        A numpy RandomState object used for reproducibility of the shuffling.

    Returns:
    pandas DataFrame
        A new DataFrame with shuffled 'u1' and 'u2' values.

    Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {
    ...     'id': [1, 2, 3, 4, 5],
    ...     'u1': [2, 3, 4, 6, 7],
    ...     'u2': [1, 4, 5, 5, 8],
    ... }
    >>> df = pd.DataFrame(data)
    >>> random_state = np.random.RandomState(2)
    >>> shuffle_utilities(df, random_state)
       id  u1  u2
    0   1   4   5
    1   2   7   8
    2   3   3   4
    3   4   6   5
    4   5   2   1
    """
    df = df.copy()
    idx = df.index.values
    u1_u2 = df[["u1", "u2"]].values  # Extract u1 and u2
    perm = random_state.permutation(len(u1_u2))
    shuffled_u1_u2 = u1_u2[perm]
    # Assign back to the dataframe at the correct indices
    df.loc[idx, ["u1", "u2"]] = shuffled_u1_u2
    return df


def shuffle_utilities_maintaining_sign(df, random_state):
    """
    Shuffle the u1 and u2 values in a pandas DataFrame df, but only within the same sign category of u1 - u2.
    The pairs of u1 and u2 are kept together during shuffling.

    Parameters:
    df : pandas DataFrame
        The DataFrame containing at least the columns 'u1' and 'u2'. May contain other columns which will remain unaffected.
    random_state : numpy RandomState
        A numpy RandomState object used for reproducibility of the shuffling.

    Returns:
    pandas DataFrame
        A new DataFrame with shuffled 'u1' and 'u2' values within the same sign category.

    Detailed Explanation:
    - Compute the difference between 'u1' and 'u2' for each row.
    - Determine the sign of this difference (positive, negative, or zero).
    - Group the rows based on the sign of the difference.
    - Shuffle the 'u1' and 'u2' pairs within each group, so that each pair stays together.
    - Other columns remain unchanged.

    Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {
    ...     'id': [1, 2, 3, 4, 5],
    ...     'u1': [2, 3, 4, 6, 7],
    ...     'u2': [1, 4, 5, 5, 8],
    ... }
    >>> df = pd.DataFrame(data)
    >>> signs_pre = np.sign(df['u1'] - df['u2'])
    >>> signs_pre
    0    1
    1   -1
    2   -1
    3    1
    4   -1
    dtype: int64
    >>> random_state = np.random.RandomState(2)
    >>> shuffle_utilities_maintaining_sign(df, random_state)
       id  u1  u2
    0   1   2   1
    1   2   7   8
    2   3   4   5
    3   4   6   5
    4   5   3   4
    >>> signs_post = np.sign(df['u1'] - df['u2'])
    >>> all(signs_pre == signs_post)
    True
    """

    df = df.copy()
    # Compute the sign of u1 - u2
    df["sign_category"] = np.sign(df["u1"] - df["u2"])

    # For each sign category, shuffle u1 and u2 values within that category
    grouped = df.groupby("sign_category")
    for sign, group in grouped:
        idx = group.index.values
        u1_u2 = group[["u1", "u2"]].values  # Extract u1 and u2
        perm = random_state.permutation(len(u1_u2))
        shuffled_u1_u2 = u1_u2[perm]
        # Assign back to the dataframe at the correct indices
        df.loc[idx, ["u1", "u2"]] = shuffled_u1_u2

    # Drop the temporary 'sign_category' column
    df = df.drop(columns=["sign_category"])

    return df
