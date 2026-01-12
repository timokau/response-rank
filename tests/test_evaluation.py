import numpy as np
import pandas as pd

from responserank.synthetic.evaluation import (
    pearson_utility_correlation,
    spearman_utility_correlation,
)


def test_pearson_utility_correlation():
    # Create sample data
    df = pd.DataFrame(
        {
            "x1_0": [1, 2, 3],
            "x1_1": [0, 0, 0],
            "x2_0": [0, 0, 0],
            "x2_1": [1, 2, 3],
            "u1": [0.1, 0.4, 0.7],
            "u2": [0.2, 0.5, 0.8],
        }
    )

    # Define a simple utility function
    def mock_uf(x):
        return np.array([sum(item) for item in x])

    # Calculate correlation
    correlation = pearson_utility_correlation(df, mock_uf)

    # Verify correlation is between -1 and 1
    assert -1 <= correlation <= 1

    # Test perfect correlation case
    def perfect_uf(x):
        mapping = {
            (1, 0): 0.1,
            (2, 0): 0.4,
            (3, 0): 0.7,
            (0, 1): 0.2,
            (0, 2): 0.5,
            (0, 3): 0.8,
        }
        return np.array([mapping[tuple(item)] for item in x])

    perfect_correlation = pearson_utility_correlation(df, perfect_uf)
    rank_correlation = spearman_utility_correlation(df, perfect_uf)
    assert np.isclose(perfect_correlation, 1.0)
    assert np.isclose(rank_correlation, 1.0)

    # Test inverse correlation case
    def inverse_uf(x):
        mapping = {
            (1, 0): 0.8,
            (2, 0): 0.5,
            (3, 0): 0.2,
            (0, 1): 0.7,
            (0, 2): 0.4,
            (0, 3): 0.1,
        }
        return np.array([mapping[tuple(item)] for item in x])

    inverse_correlation = pearson_utility_correlation(df, inverse_uf)
    assert inverse_correlation < 0
