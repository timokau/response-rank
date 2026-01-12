import numpy as np
import pytest

from responserank.synthetic.synthetic_data import (
    HyperbolicUdiffRtRel,
    StochasticTrialGenerator,
    UniformItemGenerator,
    generate_preference_dataset,
)
from responserank.synthetic.util import (
    UtilityFunction,
    dataset_to_dataframe,
    evaluate_utility_function,
    extract_item_features_from_pandas_df,
    module_to_numpy_uf,
)


def test_dataset_to_dataframe():
    x1s = np.array([[1, 2], [3, 4]])
    x2s = np.array([[5, 6], [7, 8]])
    u1s = np.array([3, 7])
    u2s = np.array([11, 15])
    ys = np.array([0, 1])
    ts = np.array([1.0, 2.0])

    df = dataset_to_dataframe(x1s, x2s, u1s, u2s, ys, ts, None)

    assert df.shape == (2, 9)
    assert list(df.columns) == [
        "x1_0",
        "x1_1",
        "x2_0",
        "x2_1",
        "u1",
        "u2",
        "y",
        "t",
        "partition_id",
    ]
    assert np.array_equal(df["x1_0"].values, [1, 3])
    assert np.array_equal(df["x2_1"].values, [6, 8])
    assert np.array_equal(df["u1"].values, [3, 7])
    assert np.array_equal(df["y"].values, [0, 1])
    assert np.array_equal(df["t"].values, [1.0, 2.0])


def test_dataframe_conversion():
    random_state = np.random.RandomState(0)

    num_features = 101
    udiff_rt_rel = HyperbolicUdiffRtRel(min_rt=0, max_rt=10)
    uf = module_to_numpy_uf(
        UtilityFunction(num_features, hidden_layers=[10], dropout_rate=None)
    )
    item_generator = UniformItemGenerator(num_features, uf, utility_sd=None)
    trial_generator = StochasticTrialGenerator(
        udiff_rt_rel,
        response_time_sd=num_features,
        deterministic_choice=False,
        deterministic_rt=False,
    )
    x1s, x2s, u1s, u2s, ys, ts, scaling_factor, partition_ids = (
        generate_preference_dataset(
            item_generator,
            trial_generator,
            random_state,
            3,
            num_partitions=1,
            partition_rt_variability=0.0,
        )
    )
    df = dataset_to_dataframe(x1s, x2s, u1s, u2s, ys, ts, partition_ids)
    x1s_extracted = extract_item_features_from_pandas_df(df, "x1")
    x2s_extracted = extract_item_features_from_pandas_df(df, "x2")
    np.testing.assert_equal(x1s, x1s_extracted)
    np.testing.assert_equal(x2s, x2s_extracted)
    np.testing.assert_equal(u1s, df["u1"].values)
    np.testing.assert_equal(u2s, df["u2"].values)
    np.testing.assert_equal(ys, df["y"].values)
    np.testing.assert_equal(ts, df["t"].values)


def test_can_reproduce_decisions():
    random_state = np.random.RandomState(0)
    num_item_features = 100
    num_comparisons = 100
    true_response_time_sd = 1

    uf = module_to_numpy_uf(
        UtilityFunction(num_item_features, hidden_layers=[10], dropout_rate=None)
    )

    udiff_rt_rel = HyperbolicUdiffRtRel(min_rt=0, max_rt=10)
    item_generator = UniformItemGenerator(num_item_features, uf, utility_sd=1)
    trial_generator = StochasticTrialGenerator(
        udiff_rt_rel,
        response_time_sd=true_response_time_sd,
        deterministic_choice=True,
        deterministic_rt=False,
    )
    x1s, x2s, u1s, u2s, ys, ts, scaling_factor, partition_ids = (
        generate_preference_dataset(
            item_generator,
            trial_generator,
            random_state,
            num_comparisons,
            num_partitions=1,
            partition_rt_variability=0.0,
        )
    )
    df = dataset_to_dataframe(x1s, x2s, u1s, u2s, ys, ts, partition_ids)

    # Make sure the choices always follow u1s > u2s
    np.testing.assert_equal(ys, u1s > u2s)

    # Now make sure we can reproduce this with the utility function
    u1s_reproduced = uf(x1s)
    u2s_reproduced = uf(x2s)
    ys_reproduced = u1s_reproduced > u2s_reproduced
    np.testing.assert_equal(ys, ys_reproduced)

    accuracy_of_true_uf = evaluate_utility_function(df, uf)
    assert accuracy_of_true_uf == pytest.approx(1.0)
