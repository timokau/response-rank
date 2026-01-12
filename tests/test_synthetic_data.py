import numpy as np
import pytest

from responserank.synthetic.synthetic_data import (
    HyperbolicUdiffRtRel,
    StochasticTrialGenerator,
    UniformItemGenerator,
    generate_preference_dataset,
)


def test_dataset_statistics():
    def utility_function(x):
        return np.sum(x, axis=-1)

    n = 100_000
    udiff_rt_rel = HyperbolicUdiffRtRel(min_rt=0, max_rt=10)
    item_generator = UniformItemGenerator(2, utility_function, utility_sd=1)
    trial_generator = StochasticTrialGenerator(
        udiff_rt_rel,
        response_time_sd=1,
        deterministic_choice=True,
        deterministic_rt=False,
    )
    x1, x2, u1, u2, y, t, scaling_factor, partition_ids = generate_preference_dataset(
        item_generator,
        trial_generator,
        np.random.RandomState(0),
        n,
        num_partitions=1,
        partition_rt_variability=0.0,
    )
    assert x1.shape == (n, 2)
    assert x2.shape == (n, 2)
    assert u1.shape == (n,)
    assert u2.shape == (n,)
    assert y.shape == (n,)
    assert t.shape == (n,)
    assert scaling_factor > 0
    assert partition_ids.shape == (n,)

    # Test that the utilities are scaled correctly
    assert np.concatenate([u1, u2]).std() == pytest.approx(1)
    assert u1.std() == pytest.approx(1, abs=0.05)
    assert u2.std() == pytest.approx(1, abs=0.05)

    # Test that response times are generated properly
    response_time_means = udiff_rt_rel.udiff_to_rt(np.abs(u1 - u2))
    assert (t - response_time_means).std() == pytest.approx(1.0, abs=0.05)
    assert (t - response_time_means).mean() == pytest.approx(0.0, abs=0.05)


def test_multiple_partitions():
    """Test that partitioning works correctly with multiple partitions."""

    def utility_function(x):
        return np.sum(x, axis=-1)

    n = 10_000
    num_partitions = 5
    partition_rt_variability = 0.3

    random_state = np.random.RandomState(42)
    udiff_rt_rel = HyperbolicUdiffRtRel(min_rt=0, max_rt=10)
    item_generator = UniformItemGenerator(2, utility_function, utility_sd=1)
    trial_generator = StochasticTrialGenerator(
        udiff_rt_rel,
        response_time_sd=0.1,  # Low variability to better observe partition effects
        deterministic_choice=True,
        deterministic_rt=False,
    )

    # Generate dataset with multiple partitions
    x1, x2, u1, u2, y, t, scaling_factor, partition_ids = generate_preference_dataset(
        item_generator,
        trial_generator,
        random_state,
        n,
        num_partitions=num_partitions,
        partition_rt_variability=partition_rt_variability,
    )

    # Test partition_ids shape and range
    assert partition_ids.shape == (n,)
    assert np.min(partition_ids) == 0
    assert np.max(partition_ids) == num_partitions - 1

    # Each partition should have approximately the same number of items
    for partition_id in range(num_partitions):
        partition_mask = partition_ids == partition_id
        expected_count = n / num_partitions
        # Allow 10% deviation from expected count due to random assignment
        assert np.sum(partition_mask) == pytest.approx(expected_count, rel=0.1)

    # Test that response times vary by partition
    # First capture baseline response times for comparison
    expected_rts = udiff_rt_rel.udiff_to_rt(np.abs(u1 - u2))

    # Calculate mean RT ratio per partition
    partition_rt_ratios = []
    for partition_id in range(num_partitions):
        partition_mask = partition_ids == partition_id
        if np.sum(partition_mask) > 0:  # Ensure partition has data
            partition_rt_ratio = np.mean(
                t[partition_mask] / expected_rts[partition_mask]
            )
            partition_rt_ratios.append(partition_rt_ratio)

    # Test that partition multipliers have expected variability
    # Standard deviation of ratios should be close to the specified variability
    assert np.std(partition_rt_ratios) > 0
    assert np.std(partition_rt_ratios) == pytest.approx(
        partition_rt_variability, abs=0.1
    )
