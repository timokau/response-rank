import logging

import torch

logger = logging.getLogger(__name__)


def _detect_ties_in_ranks(ranks: torch.Tensor, rel_tol: float = 1e-9) -> bool:
    """
    Detect if there are ties in rank values.

    Args:
        ranks: Tensor of rank values
        rel_tol: Relative tolerance for float comparison

    Returns:
        True if ties are detected, False otherwise
    """
    if len(ranks) <= 1:
        return False

    sorted_ranks = torch.sort(ranks)[0]

    for i in range(len(sorted_ranks) - 1):
        if torch.isclose(sorted_ranks[i], sorted_ranks[i + 1], rtol=rel_tol):
            return True

    return False


def rev_cumsum(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Computes the cumulative sum of the tensor in reverse order.
    """
    return torch.flip(torch.cumsum(torch.flip(tensor, [axis]), axis), [axis])


def compute_pl_log_likelihood_of_ranking(log_item_worths, ranking):
    """
    Computes the Plackett-Luce log-likelihood for the given ranking.

    Args:
        log_item_worths: Log of item worths (utilities in log-space)
        ranking: Ranking indices

    Returns:
        Log-likelihood of the ranking
    """
    # Sort log-worths according to ranking
    log_worths_ranked = torch.gather(log_item_worths, -1, ranking)

    # Compute log-sum-exp of remaining items at each position
    # For each position i, we need log(sum(exp(log_worths[j]))) for j >= i
    log_denominators = []
    for i in range(log_worths_ranked.shape[-1]):
        remaining_log_worths = log_worths_ranked[..., i:]
        # log sum exp for numerical stability
        log_denominator = torch.logsumexp(remaining_log_worths, dim=-1)
        log_denominators.append(log_denominator)

    log_denominators = torch.stack(log_denominators, dim=-1)

    log_result = torch.sum(log_worths_ranked - log_denominators, dim=-1)

    return log_result


def compute_responserank_loss_sum(utility_diff, ranks, partition_ids, allow_ties: bool):
    """
    Compute ResponseRank loss sum with anchoring (a fixed reference point at 0 utility).

    This method:
    1. Assumes the preferred option always comes first (utility_diff may still be negative as it is a prediction)
    2. Adds a virtual anchor with utility difference of 0 to each partition
    3. Computes the Plackett-Luce ranking log-likelihood with the anchor

    Args:
        utility_diff: Tensor of utility differences (chosen - rejected)
        ranks: Tensor of ranking values (can be response times or pre-computed ranks)
        partition_ids: Tensor of partition IDs for grouping comparisons
        allow_ties: If False, raises ValueError when ties are detected in rankings

    Returns:
        loss_sum: Scalar loss sum value (negative log-likelihood sum)

    Raises:
        ValueError: When ties are detected in rankings and allow_ties=False
    """
    assert partition_ids is not None

    unique_partitions = torch.unique(partition_ids)
    joint_log_likelihood = 0.0

    total_items = 0
    anchor_udiff = torch.zeros(1, device=utility_diff.device, dtype=utility_diff.dtype)
    for partition in unique_partitions:
        partition_mask = partition_ids == partition
        n = int(partition_mask.sum())
        assert n > 0

        # Extract data for this partition
        partition_udiff = utility_diff[partition_mask]
        partition_ranks = ranks[partition_mask]

        if not allow_ties and _detect_ties_in_ranks(partition_ranks):
            raise ValueError(
                f"Ties detected in ranking for partition {partition.item()}: "
                f"ranks={partition_ranks.tolist()}. "
                f"Set allow_ties=True to allow tied rankings."
            )

        # Sort by rank values for ranking
        # Lower values mean stronger preference
        ranking = torch.argsort(partition_ranks)

        # Add virtual anchor with utility diff of 0
        augmented_udiff = torch.cat([partition_udiff, anchor_udiff])
        anchor_idx = torch.tensor(
            [len(ranking)], device=ranking.device, dtype=ranking.dtype
        )
        augmented_ranking = torch.cat([ranking, anchor_idx])

        # Compute Plackett-Luce log likelihood
        log_likelihood = compute_pl_log_likelihood_of_ranking(
            augmented_udiff, augmented_ranking
        )

        joint_log_likelihood += log_likelihood
        total_items += n

    return -joint_log_likelihood
