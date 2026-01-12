import torch

from responserank.synthetic.losses import (
    RTRankAnchoredLoss,
    compute_pl_log_likelihood_of_ranking,
)


def test_responserank_anchored_pl_log_likelihood():
    """Test the Plackett-Luce log-likelihood calculation with anchoring."""
    # Create a simple test case with 3 comparisons
    u1 = torch.tensor([5.0, 2.0, 3.0])
    u2 = torch.tensor([2.0, 4.0, 1.0])
    y = torch.tensor([1, 0, 1])
    rt = torch.tensor([0.1, 0.2, 0.3])
    partition_ids = torch.zeros(3, dtype=torch.long)  # All in same partition

    loss_fn = RTRankAnchoredLoss(
        unreduce=False,
        misclassification_penalty=False,
        use_strata=True,
        worth_transform="exp",
    )

    utility_diffs, response_times = loss_fn._reorder_based_on_preference(u1, u2, y, rt)

    # Expected utility diffs after reordering (should be all positive)
    expected_udiffs = torch.tensor(
        [
            u1[0] - u2[0],  # y=1, so u1 - u2 = 5.0 - 2.0 = 3.0
            u2[1] - u1[1],  # y=0, so u2 - u1 = 4.0 - 2.0 = 2.0
            u1[2] - u2[2],  # y=1, so u1 - u2 = 3.0 - 1.0 = 2.0
        ]
    )
    assert torch.allclose(utility_diffs, expected_udiffs)

    # Expected ranking order based on response times
    expected_ranking = torch.tensor([0, 1, 2])  # Sorted by rt in ascending order
    actual_ranking = torch.argsort(response_times)
    assert torch.allclose(actual_ranking, expected_ranking)

    # Step 1: Compute a manual Plackett-Luce log-likelihood
    # Get the augmented utility diffs with anchor
    anchor_udiff = torch.tensor([0.0])
    augmented_udiffs = torch.cat([utility_diffs, anchor_udiff])

    # Transform to worths using exp
    augmented_worths = torch.exp(augmented_udiffs)

    # Create augmented ranking with anchor at the end
    augmented_ranking = torch.cat([actual_ranking, torch.tensor([len(actual_ranking)])])

    # Use the utility function to compute PL log-likelihood
    pl_log_likelihood = compute_pl_log_likelihood_of_ranking(
        augmented_worths, augmented_ranking
    )

    # Manually compute each probability in the Plackett-Luce model:
    # 1. Probability of choosing item 0 first
    p1 = augmented_worths[0] / (
        augmented_worths[0]
        + augmented_worths[1]
        + augmented_worths[2]
        + augmented_worths[3]
    )

    # 2. Probability of choosing item 1 second
    p2 = augmented_worths[1] / (
        augmented_worths[1] + augmented_worths[2] + augmented_worths[3]
    )

    # 3. Probability of choosing item 2 third
    p3 = augmented_worths[2] / (augmented_worths[2] + augmented_worths[3])

    # 4. Probability of choosing anchor (item 3) last
    p4 = augmented_worths[3] / augmented_worths[3]  # Should be 1.0

    # Compute log-likelihood manually
    manual_pl_log_likelihood = (
        torch.log(p1) + torch.log(p2) + torch.log(p3) + torch.log(p4)
    )

    # Check if the manual calculation matches the function's calculation
    assert torch.allclose(manual_pl_log_likelihood, pl_log_likelihood), (
        f"Manual PL log-likelihood {manual_pl_log_likelihood.item():.6f} "
        f"does not match computed value {pl_log_likelihood.item():.6f}"
    )

    # Step 2: Verify that the loss function's _compute_partition_log_likelihood
    # returns the same value as our manual calculation
    partition_log_likelihood, _, _ = loss_fn._compute_partition_log_likelihood(
        u1, u2, y, rt
    )

    assert torch.allclose(partition_log_likelihood, pl_log_likelihood), (
        f"Partition log-likelihood {partition_log_likelihood.item():.6f} "
        f"does not match direct PL computation {pl_log_likelihood.item():.6f}"
    )

    # Step 3: Verify that the full loss function gives the negative of this log-likelihood
    loss = loss_fn(u1, u2, y, rt, partition_ids)
    expected_loss = -partition_log_likelihood

    assert torch.allclose(loss, expected_loss), (
        f"Loss {loss.item():.6f} does not match expected value {expected_loss.item():.6f}"
    )
