from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from responserank.synthetic.metric_tracker import MetricTracker
from responserank.synthetic.synthetic_data import UdiffRtRel


class UtilityLoss:
    """Base class for all utility-based loss functions.

    This abstract class defines the interface for loss functions used to train models
    on pairwise comparisons, assuming a utility function.

    All concrete loss implementations should implement the loss_from_utilities method.
    """

    def __init__(self):
        self.metric_tracker = MetricTracker(metric_prefix="", backend=None)

    def set_metric_tracker(self, metric_tracker: MetricTracker):
        self.metric_tracker = metric_tracker

    def loss_from_utilities(
        self,
        u1: torch.Tensor,
        u2: torch.Tensor,
        y: torch.Tensor,
        rt: torch.Tensor,
        partition_ids=None,
    ) -> torch.Tensor:
        """Compute loss for utilities given choice, response times, and partitioning.

        Args:
            u1: Utility values for first options
            u2: Utility values for second options
            y: Whether or not the first option (corresponding to u1) is preferred (0 or 1)
            rt: Response times
            partition_ids: Optional partition identifiers for stratified learning

        Returns:
            Computed loss tensor
        """
        raise NotImplementedError()

    def __call__(
        self,
        u1: torch.Tensor,
        u2: torch.Tensor,
        y: torch.Tensor,
        rt: torch.Tensor,
        partition_ids=None,
    ) -> torch.Tensor:
        """Call method that forwards to loss_from_utilities.

        Args:
            u1: Utility values for first options
            u2: Utility values for second options
            y: Whether or not the first option (corresponding to u1) is preferred (0 or 1)
            rt: Response times
            partition_ids: Optional partition identifiers for stratified learning

        Returns:
            Computed loss tensor
        """
        return self.loss_from_utilities(u1, u2, y, rt, partition_ids)


class DifferenceBasedLoss(UtilityLoss):
    """Base class for loss functions that only require utility differences.

    This abstract class simplifies implementation of losses that work with
    the difference between utilities rather than the raw utility values.
    Subclasses need only implement loss_from_difference.
    """

    def loss_from_utilities(self, u1, u2, y, rt, partition_ids=None):
        return self.loss_from_difference(u1 - u2, y, rt, partition_ids)

    def loss_from_difference(
        self,
        utility_diff: torch.Tensor,
        y: torch.Tensor,
        rt: torch.Tensor,
        partition_ids=None,
    ) -> torch.Tensor:
        """Compute loss from utility differences.

        Args:
            utility_diff: Utility differences (u1 - u2)
            y: Whether or not the first option (corresponding to u1) is preferred (0 or 1)
            rt: Response times
            partition_ids: Optional partition identifiers for stratified learning

        Returns:
            Computed loss tensor
        """
        raise NotImplementedError()


def smoothen_labels(y: torch.Tensor, smoothing: float) -> torch.Tensor:
    """
    Applies label smoothing to the target labels.

    Label smoothing moves binary labels (0 or 1) slightly toward the uniform
    distribution.

    Args:
        y: The target labels tensor.
        smoothing: The smoothing factor.

    Returns:
        The smoothed target labels tensor.

    Example:
        >>> import torch
        >>> y = torch.tensor([0.0, 1.0])
        >>> smoothen_labels(y, 0.1)
        tensor([0.0500, 0.9500])
    """
    return (1 - smoothing) * y + smoothing * 0.5


class BTLoss(DifferenceBasedLoss):
    """Bradley-Terry loss based on utility differences.

    This loss implements the Bradley-Terry model of pairwise comparisons,
    using binary cross-entropy between the sigmoid of utility differences
    and the observed choices.

    Optionally supports label smoothing.
    """

    def __init__(self, smoothing: Optional[float]):
        """Initialize the Bradley-Terry loss.

        Args:
            smoothing: Optional label smoothing factor. Pass ``None`` to disable
                label smoothing.
        """
        super().__init__()
        self.smoothing = smoothing

    def loss_from_difference(self, utility_diff, y, rt=None, partition_ids=None):
        y = y.to(torch.float32)
        choice_probabilities = torch.sigmoid(utility_diff)

        if self.smoothing is not None:
            y = smoothen_labels(y, self.smoothing)

        loss = F.binary_cross_entropy(choice_probabilities, y, reduction="none")
        return loss

    def __call__(self, u1, u2, y, rt=None, partition_ids=None):
        return super().__call__(u1, u2, y, rt, partition_ids)


class DeterministicRTLoss(DifferenceBasedLoss):
    """Deterministic response time loss based on utility differences.

    This loss assumes a deterministic relationship between utility differences
    and response times, where stronger preferences lead to faster responses.
    It uses MSE loss between predicted utility differences and those derived
    from response times via an assumed utility-RT relationship.
    """

    def __init__(self, assumed_udiff_rt_rel: UdiffRtRel):
        """Initialize the deterministic RT loss.

        Args:
            assumed_udiff_rt_rel: The utility difference to response time relationship object
                that defines the mapping between utility differences and response times
        """
        super().__init__()
        self.assumed_udiff_rt_rel = assumed_udiff_rt_rel

    def loss_from_difference(self, utility_diff, y, rt, partition_ids=None):
        y = y.to(torch.float32)

        target_udiff = self.assumed_udiff_rt_rel.rt_to_udiff(rt)
        correct_sign_for_utility_diff = torch.where(y > 0.5, 1, -1)
        target_udiff = target_udiff * correct_sign_for_utility_diff

        loss = F.mse_loss(utility_diff, target_udiff, reduction="none")
        return loss


def rev_cumsum(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Computes the cumulative sum of the tensor in reverse order.

    Args:
        tensor: The input tensor to compute reverse cumulative sum on.
        axis: The axis along which to compute the reverse cumulative sum.

    Returns:
        A tensor with the same shape as the input containing the reverse
        cumulative sum along the specified axis.

    Example:
        >>> import torch
        >>> t = torch.tensor([1, 2, 3, 4])
        >>> rev_cumsum(t, 0)
        tensor([10,  9,  7,  4])
    """
    # Flip tensor, compute regular cumsum, then flip back
    return torch.flip(torch.cumsum(torch.flip(tensor, [axis]), axis), [axis])


def compute_pl_log_likelihood_of_ranking(
    item_worths: torch.Tensor, ranking: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Plackett-Luce log-likelihood for the given ranking.

    Args:
        item_worths: A 1D or batched tensor of item worth values (non-negative).
        ranking: A 1D or batched tensor of indices representing the ranking.
            Each index refers to the position in item_worths.

    Returns:
        A tensor containing the sum of log probabilities for the Plackett-Luce model.

    Example:
        >>> import torch
        >>> worths = torch.tensor([2.0, 1.0, 4.0])
        >>> order = torch.tensor([2, 0, 1])

        Higher worths should lead to a higher likelihood of the item being
        ranked in a higher position. The order tensor represents the ranking.
        It maps the position of the item in the ranking to the index of the
        item in the worths tensor. order[0] is the index of the item ranked
        first, order[1] is the index of the item ranked second, and so on.

        Mapping this to item worths sorted by order:
        - The first item has worth worths[order[0]] = worths[2] = 4.0.
        - The second item has worth worths[order[1]] = worths[0] = 2.0.
        - The third item has worth worths[order[2]] = worths[1] = 1.0.

        This results in the following probabilities:
        - First item: 4.0 / (4.0 + 2.0 + 1.0) = 0.5714
        - Second item: 2.0 / (2.0 + 1.0) = 0.6667
        - Third item: 1.0 / 1.0 = 1.0

        The log likelihood of the ranking is therefore:
        log(0.5714) + log(0.6667) + log(1.0) = -0.9651

        >>> compute_pl_log_likelihood_of_ranking(worths, order)
        tensor(-0.9651)
    """
    assert (item_worths >= 0).all(), "Item worths must be non-negative."
    item_worths_sorted = torch.gather(item_worths, -1, ranking)
    plackett_luce_item_likelihoods = item_worths_sorted / rev_cumsum(
        item_worths_sorted, -1
    )
    return torch.sum(torch.log(plackett_luce_item_likelihoods), dim=-1)


def compute_misclassification_penalty(
    utility_diffs: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """
    Computes a penalty for misclassified choices based on utility differences.

    The penalty is computed as the sum of absolute utility differences for all
    misclassified choices (where the sign of the utility difference doesn't match
    the expected sign based on the choice).

    Args:
        utility_diffs: A tensor containing utility differences (u1 - u2)
        y: A tensor containing the true choices (0 or 1)

    Returns:
        A scalar tensor containing the sum of penalties for misclassified choices.

    Example:
        >>> import torch
        >>> udiffs = torch.tensor([1.0, -2.0, 0.5])
        >>> y = torch.tensor([0, 1, 1])

        First and second choices are misclassified with utility differences 1.0 and -2.0.
        >>> compute_misclassification_penalty(udiffs, y)
        tensor(3.)
    """
    # Convert y to expected sign (+1 for y=1, -1 for y=0)
    expected_sign = 2 * y - 1

    # Actual sign of utility differences
    actual_sign = torch.sign(utility_diffs)

    # Identify misclassifications (where signs don't match)
    misclassified = (expected_sign * actual_sign) < 0

    # Compute penalty as sum of absolute utility differences for misclassified cases
    penalty = torch.where(
        misclassified, torch.abs(utility_diffs), torch.zeros_like(utility_diffs)
    )

    return penalty.sum()


class BaseRTRankLoss(UtilityLoss):
    """Base class for RT-rank losses implementing shared functionality.

    Implements the common methods for computing joint log-likelihood across partitions
    and converting it to a loss value with optional penalty for misclassifications.

    The core of ResponseRank is to convert pairwise comparisons with their response times
    into a ranking problem, then use a Plackett-Luce model to learn from this ranking.
    Child classes should implement the _compute_partition_log_likelihood method
    that defines how rankings are created from response times and utilities.
    """

    def __init__(
        self,
        unreduce: bool,
        misclassification_penalty: bool,
        use_strata: bool,
        worth_transform="exp",
        mean_reduce: bool = False,
    ):
        super().__init__()
        self.unreduce = unreduce
        self.misclassification_penalty = misclassification_penalty
        self.use_strata = use_strata
        self.worth_transform = worth_transform
        self.mean_reduce = mean_reduce
        if mean_reduce and unreduce:
            raise ValueError("mean_reduce and unreduce are mutually exclusive")

    def loss_from_utilities(self, u1, u2, y, rt, partition_ids=None):
        """Computes loss from utilities, choices, and response times.

        Args:
            u1: Utility values for first options
            u2: Utility values for second options
            y: Choices (0 or 1)
            rt: Response times
            partition_ids: Optional partition identifiers for stratified learning

        Returns:
            Computed loss value
        """
        if self.use_strata:
            assert partition_ids is not None, (
                "Partition IDs must be provided when use_strata is True."
            )
            return self._compute_joint_likelihood_loss(u1, u2, y, rt, partition_ids)
        else:
            # For the non-stratified case, create a uniform partition ID array
            # This allows us to use the same joint likelihood function for both cases
            dummy_partition_ids = torch.zeros_like(y, dtype=torch.long)
            return self._compute_joint_likelihood_loss(
                u1, u2, y, rt, dummy_partition_ids
            )

    def _compute_partition_log_likelihood(self, u1, u2, y, rt):
        """Compute log likelihood for a partition of data points.

        This is an abstract method that must be implemented by child classes.

        Args:
            u1: Utilities for first options in the partition
            u2: Utilities for second options in the partition
            y: Choices in the partition
            rt: Response times in the partition

        Returns:
            Tuple of (log_likelihood, utility_diffs, y)
        """
        raise NotImplementedError(
            "Child classes must implement _compute_partition_log_likelihood"
        )

    def _compute_joint_log_likelihood(self, u1, u2, y, rt, partition_ids):
        """Compute joint log-likelihood across all partitions by summing individual log-likelihoods.

        Args:
            u1: Utilities for first options
            u2: Utilities for second options
            y: Choices
            rt: Response times
            partition_ids: Partition identifiers

        Returns:
            Tuple of (joint_log_likelihood, total_comparisons) or (None, 0) if no valid partitions
        """
        # Convert to tensor if not already
        partition_ids = torch.as_tensor(partition_ids)

        # Get unique partitions
        unique_partitions = torch.unique(partition_ids)

        # Initialize joint log likelihood
        joint_log_likelihood = 0.0
        valid_partitions = 0
        total_comparisons = 0
        all_partition_likelihoods = []

        # Process each partition
        for partition in unique_partitions:
            # Get indices for this partition
            partition_mask = partition_ids == partition
            n_comparisons = int(partition_mask.sum())

            # Skip partitions with insufficient data
            if n_comparisons <= 1:
                print(
                    f"Warning: Skipping stratum {partition.item()} with insufficient data."
                )
                continue

            # Extract data for this partition
            partition_u1 = u1[partition_mask]
            partition_u2 = u2[partition_mask]
            partition_y = y[partition_mask]
            partition_rt = rt[partition_mask]

            # Calculate log-likelihood for this partition
            partition_log_likelihood, _, _ = self._compute_partition_log_likelihood(
                partition_u1, partition_u2, partition_y, partition_rt
            )

            joint_log_likelihood += partition_log_likelihood
            valid_partitions += 1
            total_comparisons += n_comparisons

            # Track metrics
            partition_id = partition.item()
            self.metric_tracker.track_epoch_scalar(
                f"stratum_{partition_id}_log_likelihood",
                partition_log_likelihood.detach().numpy(),
            )
            all_partition_likelihoods.append(partition_log_likelihood.detach().numpy())

        # Check if we have any valid partitions
        if valid_partitions == 0:
            return None, 0

        # Track metrics
        self.metric_tracker.track_epoch_scalar(
            "joint_log_likelihood", joint_log_likelihood.detach().numpy()
        )
        self.metric_tracker.track_epoch_scalar("num_valid_strata", valid_partitions)
        self.metric_tracker.track_epoch_scalar("total_comparisons", total_comparisons)

        if len(all_partition_likelihoods) > 1:
            all_likes_array = np.array(all_partition_likelihoods)
            self.metric_tracker.track_epoch_scalar(
                "stratum_likelihood_variance", np.var(all_likes_array)
            )

        return joint_log_likelihood, total_comparisons

    def _compute_joint_likelihood_loss(self, u1, u2, y, rt, partition_ids):
        """Compute loss based on joint likelihood and global misclassification penalty.

        Args:
            u1: Utilities for first options
            u2: Utilities for second options
            y: Choices
            rt: Response times
            partition_ids: Partition identifiers

        Returns:
            Computed loss value

        Raises:
            ValueError: If no valid strata are found in the data
        """
        # First compute the joint log-likelihood
        joint_log_likelihood, total_comparisons = self._compute_joint_log_likelihood(
            u1, u2, y, rt, partition_ids
        )

        # Handle case with no valid strata
        if joint_log_likelihood is None:
            raise ValueError(
                "No valid strata found. Ensure that each stratum has at least two data points."
            )

        # Convert to loss (negative log-likelihood)
        loss = -joint_log_likelihood

        if self.mean_reduce:
            loss = loss / total_comparisons

        # Add misclassification penalty if needed
        if self.misclassification_penalty:
            utility_diffs = u1 - u2
            penalty = compute_misclassification_penalty(utility_diffs, y)
            self.metric_tracker.track_epoch_scalar(
                "misclassification_penalty", penalty.detach().numpy()
            )
            loss += penalty

        if self.unreduce:
            # "Unreduce" the loss to number of comparisons
            loss = loss.repeat_interleave(u1.shape[-1])

        return loss

    def __call__(self, u1s, u2s, ys, rts, partition_ids=None):
        """Call method forwarding to loss_from_utilities."""
        return self.loss_from_utilities(u1s, u2s, ys, rts, partition_ids)


class RTRankAnchoredLoss(BaseRTRankLoss):
    """Unidirectional response time ranking loss with an anchor element.

    This variant reorders comparisons based on preference, then sorts by response time,
    and adds a virtual anchor with utility difference of 0 to the ranking.
    """

    def _reorder_based_on_preference(self, u1, u2, y, rt):
        """Reorders each comparison so preferred element comes first.

        Args:
            u1: Tensor of utilities for first option
            u2: Tensor of utilities for second option
            y: Tensor of choices (0 or 1)
            rt: Tensor of response times

        Returns:
            Tuple of (utility_diffs, response_times)
        """
        batch_size = u1.shape[0]

        utility_diffs = torch.zeros(batch_size)
        response_times = torch.zeros(batch_size)

        for i in range(batch_size):
            if y[i] == 1:
                # First option preferred; the diff *should* be positive, but doesn't have to be if
                # the prediction is wrong.
                utility_diffs[i] = u1[i] - u2[i]
                response_times[i] = rt[i]
            else:
                utility_diffs[i] = u2[i] - u1[i]
                response_times[i] = rt[i]

        return utility_diffs, response_times

    def _compute_partition_log_likelihood(self, u1, u2, y, rt):
        """Compute log likelihood for a partition of data points with anchoring."""
        # Step 1: Reorder based on preference
        # Ensures preferred option has positive ground truth utility difference; but the predicted
        # difference may or may not be positive.
        utility_diffs, response_times = self._reorder_based_on_preference(u1, u2, y, rt)

        # Step 2: Sort items by response time to create ranking
        # For anchored ranking, we want shorter response times at beginning of ranking
        indices = torch.argsort(response_times)
        true_ranking = indices

        # Step 3: Add virtual anchor with utility difference of 0
        anchor_udiff = torch.tensor([0.0])
        augmented_udiffs = torch.cat([utility_diffs, anchor_udiff])
        augmented_ranking = torch.cat([true_ranking, torch.tensor([len(true_ranking)])])

        # Step 4: Transform all utilities to worths
        if self.worth_transform == "exp":
            augmented_worths = torch.exp(augmented_udiffs)
        elif self.worth_transform == "sigmoid":
            augmented_worths = torch.sigmoid(augmented_udiffs)
        else:
            raise ValueError(
                f"Unknown worth transform: {self.worth_transform}. Use 'exp' or 'sigmoid'."
            )

        # Step 5: Compute Plackett-Luce log likelihood of this ranking
        plackett_luce_ranking_log_likelihoods = compute_pl_log_likelihood_of_ranking(
            augmented_worths, augmented_ranking
        )

        self.metric_tracker.track_epoch_scalar(
            "plackett_luce_ranking_log_likelihoods",
            plackett_luce_ranking_log_likelihoods.detach().numpy(),
        )

        return plackett_luce_ranking_log_likelihoods, utility_diffs, y
