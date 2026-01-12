import torch
import torch.nn as nn

from responserank.llm.losses import (
    compute_pl_log_likelihood_of_ranking,
    compute_responserank_loss_sum,
    rev_cumsum,
)


def test_rev_cumsum():
    """Test the rev_cumsum function from responserank.llm.losses."""
    # Basic 1D functionality
    t = torch.tensor([1, 2, 3, 4])
    result = rev_cumsum(t, 0)
    # [1+2+3+4, 2+3+4, 3+4, 4] = [10, 9, 7, 4]
    expected = torch.tensor([10, 9, 7, 4])
    assert torch.equal(result, expected)

    # 2D tensor along axis 1 (rows)
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = rev_cumsum(t, 1)
    # Rev cumsum along axis 1:
    # Row 0: [1+2+3, 2+3, 3] = [6, 5, 3]
    # Row 1: [4+5+6, 5+6, 6] = [15, 11, 6]
    expected = torch.tensor([[6, 5, 3], [15, 11, 6]])
    assert torch.equal(result, expected)

    t = torch.tensor([5])
    result = rev_cumsum(t, 0)
    expected = torch.tensor([5])
    assert torch.equal(result, expected)


def test_compute_pl_log_likelihood_of_ranking():
    """Test the compute_pl_log_likelihood_of_ranking function from responserank.llm.losses."""
    # Basic case
    # Item worths: [2.0, 1.0, 4.0]
    # Ranking: [2, 0, 1] (item 2 first, then 0, then 1)
    # worths[ranking]: [4.0, 2.0, 1.0]
    # Rev cumsum: [7.0, 3.0, 1.0]
    # Likelihoods: [4.0/7.0, 2.0/3.0, 1.0/1.0]
    # Log-likelihood: log(4/7) + log(2/3) + log(1) ≈ -0.9651
    item_worths = torch.tensor([2.0, 1.0, 4.0])
    log_item_worths = torch.log(item_worths)
    ranking = torch.tensor([2, 0, 1])
    result = compute_pl_log_likelihood_of_ranking(log_item_worths, ranking)
    expected = (
        torch.log(torch.tensor(4.0 / 7.0))
        + torch.log(torch.tensor(2.0 / 3.0))
        + torch.log(torch.tensor(1.0))
    )
    assert torch.allclose(result, expected)

    # Equal worths case
    # Item worths: [2.0, 2.0, 2.0]
    # Any ranking should give same likelihood
    # Rev cumsum: [6.0, 4.0, 2.0]
    # Likelihoods: [2.0/6.0, 2.0/4.0, 2.0/2.0] = [1/3, 1/2, 1]
    # Log-likelihood: log(1/3) + log(1/2) + log(1) = log(1/6)
    item_worths = torch.tensor([2.0, 2.0, 2.0])
    ranking_1 = torch.tensor([0, 1, 2])
    ranking_2 = torch.tensor([2, 1, 0])
    result_1 = compute_pl_log_likelihood_of_ranking(item_worths, ranking_1)
    result_2 = compute_pl_log_likelihood_of_ranking(item_worths, ranking_2)
    expected = (
        torch.log(torch.tensor(1.0 / 3.0))
        + torch.log(torch.tensor(1.0 / 2.0))
        + torch.log(torch.tensor(1.0))
    )
    assert torch.allclose(result_1, expected)
    assert torch.allclose(result_2, expected)

    # Single item case
    # Log-likelihood: log(1) = 0
    item_worths = torch.tensor([3.0])
    ranking = torch.tensor([0])
    result = compute_pl_log_likelihood_of_ranking(item_worths, ranking)
    expected = torch.tensor(0.0)
    assert torch.allclose(result, expected)


def test_compute_responserank_loss_sum():
    """Test the compute_responserank_loss_sum function from responserank.llm.losses."""
    # Basic case: single partition with 2 items
    # Expected: shorter RT should rank higher, leading to consistent ranking
    utility_diff = torch.tensor([1.0, -0.5])
    # both in the same partition
    partition_ids = torch.tensor([0, 0])
    # first item has shorter RT, consistent with predicted udiff
    rt_1 = torch.tensor([0.5, 1.0])
    # inconsistent with predicted udiff
    rt_2 = torch.tensor([1.0, 0.5])

    result_1_sum = compute_responserank_loss_sum(
        utility_diff, rt_1, partition_ids, allow_ties=False
    )
    result_1 = result_1_sum / len(utility_diff)
    result_2_sum = compute_responserank_loss_sum(
        utility_diff, rt_2, partition_ids, allow_ties=False
    )
    result_2 = result_2_sum / len(utility_diff)
    assert isinstance(result_1, torch.Tensor)
    assert result_1.ndim == 0  # scalar loss
    assert result_1.item() > 0  # loss should be positive

    assert result_2.item() > result_1.item()  # inconsistent should have higher loss

    # Multiple partitions case
    # Partition 0: items 0,1 - utility_diff: [1.0, -0.5], rt: [0.5, 1.0]
    # Partition 1: items 2,3 - utility_diff: [0.8, -0.2], rt: [0.3, 0.7]
    utility_diff_multi = torch.tensor([1.0, -0.5, 0.8, -0.2])
    rt_multi = torch.tensor([0.5, 1.0, 0.3, 0.7])
    partition_ids_multi = torch.tensor([0, 0, 1, 1])

    result_multi_sum = compute_responserank_loss_sum(
        utility_diff_multi,
        rt_multi,
        partition_ids_multi,
        allow_ties=False,
    )
    result_multi = result_multi_sum / len(utility_diff_multi)
    assert isinstance(result_multi, torch.Tensor)
    assert result_multi.ndim == 0
    assert result_multi.item() > 0

    # Single item per partition (edge case)
    # Each item is in its own partition, so ranking is trivial
    utility_diff_single = torch.tensor([1.0, -0.5])
    rt_single = torch.tensor([0.5, 1.0])
    partition_ids_single = torch.tensor([0, 1])  # different partitions

    result_single_sum = compute_responserank_loss_sum(
        utility_diff_single,
        rt_single,
        partition_ids_single,
        allow_ties=False,
    )
    result_single = result_single_sum / len(utility_diff_single)
    assert isinstance(result_single, torch.Tensor)
    assert result_single.ndim == 0
    assert result_single.item() > 0


def test_responserank_single_element_equals_bt():
    """Test theoretical property: ResponseRank loss with single comparison equals BT."""

    test_udiffs = [-2.0, -1.0, 0.0, 1.0, 2.0]

    for utility_diff_val in test_udiffs:
        utility_diff_tensor = torch.tensor([utility_diff_val])
        rt_tensor = torch.tensor([0.5])  # RT value doesn't matter for single element
        partition_ids = torch.tensor([0])  # Single partition

        responserank_loss_sum = compute_responserank_loss_sum(
            utility_diff_tensor,
            rt_tensor,
            partition_ids,
            allow_ties=False,
        )
        responserank_loss = responserank_loss_sum / len(utility_diff_tensor)
        bt_loss = -nn.functional.logsigmoid(utility_diff_tensor).mean()

        assert torch.allclose(responserank_loss, bt_loss, atol=1e-6), (
            f"ResponseRank loss ({responserank_loss.item():.6f}) != BT loss ({bt_loss.item():.6f}) "
            f"for utility_diff={utility_diff_val}"
        )


def test_responserank_plackett_luce_formula():
    """Test theoretical responserank decomposition into BT and rank factor.

    For ranking [q1, q2, anchor] where q1 ranks higher than q2,
    we verify that the theoretical formula:

    log(BT(q1)) + log(BT(q2)) - log(1 + exp(u2) / (exp(u1) + 1))

    equals log P_PL. The last part can be seen as the influence of the
    response time, encouraging u1 > u2.
    """
    import torch

    test_cases = [
        # (u1, u2, rt1, rt2) - rt determines ranking
        (1.0, -0.5, 0.5, 1.0),  # q1 ranks higher (shorter RT)
        (2.0, -1.0, 0.3, 0.8),  # q1 ranks higher
        (0.5, 0.2, 0.4, 0.9),  # q1 ranks higher
        (-0.3, -0.8, 0.2, 0.7),  # q1 ranks higher
        (0.0, 1.0, 0.5, 1.5),  # q1 ranks higher
        # Cases where q2 ranks higher
        (1.0, -0.5, 1.0, 0.5),  # q2 ranks higher (shorter RT)
        (2.0, -1.0, 0.8, 0.3),  # q2 ranks higher
        (0.5, 0.2, 0.9, 0.4),  # q2 ranks higher
    ]

    for i, (u1, u2, rt1, rt2) in enumerate(test_cases):
        utility_diff = torch.tensor([u1, u2])
        rt = torch.tensor([rt1, rt2])
        partition_ids = torch.tensor([0, 0])  # Same partition

        responserank_loss = compute_responserank_loss_sum(
            utility_diff, rt, partition_ids, allow_ties=False
        )

        # Determine which item ranks higher based on RT
        if rt1 < rt2:
            # q1 ranks higher: [q1, q2, anchor]
            u_first = u1
            u_second = u2
        else:
            # q2 ranks higher: [q2, q1, anchor]
            u_first = u2
            u_second = u1

        # Compute theoretical formula for the actual ranking
        bt_first = 1 / (1 + torch.exp(-torch.tensor(u_first)))
        bt_second = 1 / (1 + torch.exp(-torch.tensor(u_second)))
        rank_correction = torch.log(
            1
            + torch.exp(torch.tensor(u_second)) / (torch.exp(torch.tensor(u_first)) + 1)
        )
        theoretical_log_pl = (
            torch.log(bt_first) + torch.log(bt_second) - rank_correction
        )

        expected = -theoretical_log_pl

        diff = abs(responserank_loss.item() - expected.item())

        assert diff < 1e-5, (
            f"ResponseRank without division doesn't match theory: diff={diff:.6f}"
        )
