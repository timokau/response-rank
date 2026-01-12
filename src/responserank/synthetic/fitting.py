import copy
import logging
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import kendalltau
from sklearn.feature_selection import mutual_info_regression

from responserank.synthetic.evaluation import (
    expected_calibration_error_bt_probs,
    log_likelihood_pl_ranking,
    pearson_difference_correlation,
    pearson_distance_correlation,
    pearson_utility_correlation,
    pearson_utility_rank_correlation,
    spearman_difference_correlation,
    spearman_distance_correlation,
    spearman_utility_correlation,
    true_calibration_error_bt_probs,
)
from responserank.synthetic.losses import (
    BTLoss,
    UtilityLoss,
)
from responserank.synthetic.metric_tracker import MetricTracker
from responserank.synthetic.util import (
    evaluate_utility_function,
    extract_item_features_from_pandas_df,
    module_to_numpy_uf,
    shuffle_utilities,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaseUtilityFunctionFitter:
    """Base class for utility function fitting from pairwise comparisons."""

    def __init__(
        self,
        utility_function,
        metric_tracker: MetricTracker,
        learning_rate=0.01,
        l2_reg=0.01,
        loss: UtilityLoss = BTLoss(smoothing=None),
        seed=None,
    ):
        """Initialize the utility function fitter.

        Args:
            utility_function: Neural network or other callable model that maps features to utilities
            metric_tracker: Tracker object for logging metrics during training
            learning_rate: Learning rate for Adam optimizer
            l2_reg: L2 regularization strength (weight decay)
            loss: Loss function to optimize (default: Bradley-Terry loss)
            seed: Random seed
        """
        if seed is not None:
            set_seed(seed)
        self.uf = utility_function
        self.optimizer = optim.AdamW(
            self.uf.parameters(), lr=learning_rate, weight_decay=l2_reg
        )
        self.loss = loss
        self.metric_tracker = metric_tracker

    def log_general_metrics(self, u1, u2, y, true_response_time, partition_ids):
        all_utilities = torch.cat([u1, u2])
        self.metric_tracker.track_epoch_scalar(
            "train_logit_mean", all_utilities.mean().item()
        )
        self.metric_tracker.track_epoch_scalar(
            "train_logit_sd", all_utilities.std().item()
        )
        self.metric_tracker.track_epoch_scalar(
            "train_logit_difference", torch.abs(u1 - u2).mean().item()
        )
        u_span = torch.max(all_utilities) - torch.min(all_utilities)
        self.metric_tracker.track_epoch_scalar("train_logit_span", u_span.item())
        correctly_classified = (u1 > u2).float() == y
        self.metric_tracker.track_epoch_scalar(
            "train_classification_accuracy",
            correctly_classified.to(torch.float32).mean().item(),
        )
        self.metric_tracker.track_epoch_scalar(
            "train_abs_logit_diff_rt_correlation",
            np.corrcoef(torch.abs(u1 - u2).detach().numpy(), true_response_time)[0, 1],
        )

        # Track per-partition metrics
        unique_partitions = np.unique(partition_ids)

        for partition_id in unique_partitions:
            partition_mask = partition_ids == partition_id

            # Track correlation between utility difference and RT within each partition
            partition_u1 = u1[torch.as_tensor(partition_mask)]
            partition_u2 = u2[torch.as_tensor(partition_mask)]
            partition_rt = true_response_time[partition_mask]

            if len(partition_rt) >= 2:  # Need at least 2 points for correlation
                partition_corr = np.corrcoef(
                    torch.abs(partition_u1 - partition_u2).detach().numpy(),
                    partition_rt,
                )[0, 1]

                self.metric_tracker.track_epoch_scalar(
                    f"train_abs_logit_diff_rt_correlation_partition_{int(partition_id)}",
                    partition_corr,
                )

    def prepare_data(self, df):
        x1 = extract_item_features_from_pandas_df(df, "x1")
        x2 = extract_item_features_from_pandas_df(df, "x2")
        y = df["y"].values
        t = torch.tensor(df["t"].values, dtype=torch.float32)
        partition_ids = torch.tensor(df["partition_id"].values, dtype=torch.int64)

        return x1, x2, y, t, partition_ids

    def compute_utilities(self, x1, x2):
        u1 = self.uf(torch.tensor(x1, dtype=torch.float32)).flatten()
        u2 = self.uf(torch.tensor(x2, dtype=torch.float32)).flatten()
        return u1, u2

    def get_dataset_metrics(self, test_df):
        """Return metrics that are properties of the dataset, not the learned model."""
        metrics = {}
        y = test_df["y"].values
        true_response_time = test_df["t"].values
        u1, u2 = test_df["u1"].values, test_df["u2"].values

        true_preference = (u1 > u2).astype(np.int32)  # 1 if u1>u2, 0 if u1<u2
        confusion_matrix = np.zeros((2, 2), dtype=np.int32)
        for true, choice in zip(true_preference, y):
            confusion_matrix[int(true), int(choice)] += 1

        metrics["test_confusion_matrix"] = {
            "type": "confusion_matrix",
            "value": confusion_matrix,
            "dataset_metric": True,
        }

        y_uint8 = y.astype(np.uint8)
        metrics["test_y"] = {
            "type": "histogram",
            "value": {"x": y_uint8, "bins": 2},
        }
        metrics["test_true_rt"] = {
            "type": "histogram",
            "value": {"x": true_response_time, "bins": 64},
        }

        # Track response time histograms per partition
        partition_ids = test_df["partition_id"].unique()
        for partition_id in partition_ids:
            partition_mask = test_df["partition_id"] == partition_id
            partition_rt = test_df.loc[partition_mask, "t"].values
            metrics[f"test_true_rt_partition_{int(partition_id)}"] = {
                "type": "histogram",
                "value": {"x": partition_rt, "bins": 32},
                "dataset_metric": True,
            }

        # Track number of samples per partition individually
        partition_counts = test_df["partition_id"].value_counts()
        for partition_id, count in partition_counts.items():
            metrics[f"test_partition_{int(partition_id)}_size"] = {
                "type": "scalar",
                "value": int(count),
                "dataset_metric": True,
            }

        true_accuracy = np.mean((u1 > u2) == y)
        metrics["test_dataset_accuracy"] = {
            "type": "histogram",
            "value": {"x": np.array([true_accuracy]), "range": (0, 1)},
        }

        logit_abs_diff = np.abs(u1 - u2)
        mutual_info = mutual_info_regression(
            logit_abs_diff.reshape(-1, 1), true_response_time
        )[0].item()
        metrics["test_abs_logit_diff_rt_mutual_info"] = {
            "type": "histogram",
            "value": {"x": np.array([mutual_info])},
        }

        rank_correlation = kendalltau(
            -np.argsort(np.argsort(true_response_time)),
            np.argsort(np.argsort(logit_abs_diff)),
        ).statistic  # type: ignore
        metrics["test_abs_logit_diff_rt_kendall_tau"] = {
            "type": "histogram",
            "value": {"x": np.array([rank_correlation]), "range": (0, 1)},
        }
        # Rank linearity measures correlation between utilities and their ranks
        # Range is [0, 1] because ranks are ordered to match utilities by definition
        metrics["rank_linearity"] = {
            "type": "histogram",
            "value": {
                "x": np.array([pearson_utility_rank_correlation(test_df)]),
                "range": (0, 1),
            },
        }

        delta_u = u2 - u1
        metrics["test_delta_u_true"] = {
            "type": "histogram",
            "value": {"x": delta_u, "bins": 64},
        }

        choice_probs = 1 / (1 + np.exp(delta_u))
        metrics["test_choice_probs_true"] = {
            "type": "histogram",
            "value": {"x": choice_probs, "bins": 64, "range": [0, 1]},
        }

        return metrics

    def log_dataset_metrics(self, test_df):
        """Log metrics that are properties of the dataset."""
        metrics = self.get_dataset_metrics(test_df)
        for name, metric in metrics.items():
            self.metric_tracker.track_summary_metric(
                name, metric["value"], metric["type"], dataset_metric=True
            )

    def log_summary_metrics(self, test_df, fitted_numpy_uf):
        # Compare with metrics on a dataset that has shuffled utilities as a sanity check / baseline.
        test_df_shuf = shuffle_utilities(test_df, random_state=np.random.RandomState(0))

        self.metric_tracker.track_summary_metric(
            "pearson_utility_correlation",
            pearson_utility_correlation(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "pearson_utility_correlation_shuffled",
            pearson_utility_correlation(test_df_shuf, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "spearman_utility_correlation",
            spearman_utility_correlation(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "choice_accuracy",
            evaluate_utility_function(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "pearson_difference_correlation",
            pearson_difference_correlation(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "pearson_difference_correlation_shuffled",
            pearson_difference_correlation(test_df_shuf, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "log_likelihood_pl_ranking",
            log_likelihood_pl_ranking(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "spearman_difference_correlation",
            spearman_difference_correlation(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "pearson_distance_correlation",
            pearson_distance_correlation(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "pearson_distance_correlation_shuffled",
            pearson_distance_correlation(test_df_shuf, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "spearman_distance_correlation",
            spearman_distance_correlation(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )

        u1_pred = fitted_numpy_uf(extract_item_features_from_pandas_df(test_df, "x1"))
        u2_pred = fitted_numpy_uf(extract_item_features_from_pandas_df(test_df, "x2"))
        delta_u_pred = u2_pred - u1_pred
        self.metric_tracker.track_summary_metric(
            "test_delta_u_pred",
            {"x": delta_u_pred, "bins": 64},
            "histogram",
            dataset_metric=False,
        )
        choice_probs = 1 / (1 + np.exp(delta_u_pred))
        self.metric_tracker.track_summary_metric(
            "test_choice_probs_pred",
            {"x": choice_probs, "bins": 64, "range": [0, 1]},
            "histogram",
            dataset_metric=False,
        )

        self.metric_tracker.track_summary_metric(
            "expected_calibration_error_bt_probs",
            expected_calibration_error_bt_probs(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )
        self.metric_tracker.track_summary_metric(
            "true_calibration_error_bt_probs",
            true_calibration_error_bt_probs(test_df, fitted_numpy_uf),
            "scalar",
            dataset_metric=False,
        )

    def fit(
        self,
        train_df,
        test_df,
        num_epochs=200,
        *,
        early_stopping_patience: Optional[int],
    ):
        self.log_dataset_metrics(test_df)
        self.uf.train()
        x1, x2, y, t, partition_ids = self.prepare_data(train_df)
        test_x1, test_x2, test_y, test_t, test_partition_ids = self.prepare_data(
            test_df
        )

        best_loss = float("inf")
        patience_counter = 0
        best_uf = None

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            u1, u2 = self.compute_utilities(x1, x2)
            self.log_general_metrics(u1, u2, y, train_df["t"].values, partition_ids)
            loss = self.compute_loss(u1, u2, y, train_df)
            self.metric_tracker.track_epoch_scalar("loss", loss.item())
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.uf.eval()
                test_u1, test_u2 = self.compute_utilities(test_x1, test_x2)
                test_y_tensor = torch.as_tensor(test_y, dtype=torch.float32)
                test_losses = self.loss(
                    test_u1, test_u2, test_y_tensor, test_t, test_partition_ids
                )
                test_loss = test_losses.mean()
                self.metric_tracker.track_epoch_scalar("test_loss", test_loss.item())
                test_correctly_classified = (test_u1 > test_u2).float() == test_y_tensor
                self.metric_tracker.track_epoch_scalar(
                    "test_classification_accuracy",
                    test_correctly_classified.float().mean().item(),
                )
                udiff_pred = (test_u2 - test_u1).numpy()
                udiff_true = test_df["u2"].values - test_df["u1"].values
                pdc = np.corrcoef(np.abs(udiff_true), np.abs(udiff_pred))[0, 1]
                self.metric_tracker.track_epoch_scalar(
                    "test_pearson_distance_correlation", pdc
                )
                self.uf.train()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_uf = copy.deepcopy(self.uf)
            else:
                patience_counter += 1

            if (
                early_stopping_patience is not None
                and patience_counter >= early_stopping_patience
            ):
                logger.debug(f"Early stopping at epoch {epoch}")
                break

            self.metric_tracker.finalize_epoch()

        result = best_uf if best_uf is not None else self.uf
        fitted_numpy_uf = module_to_numpy_uf(result)
        self.log_summary_metrics(test_df, fitted_numpy_uf)

        return result

    def compute_loss(self, u1, u2, y, df):
        loss_name = self.loss.__class__.__name__.lower()
        self.loss.set_metric_tracker(self.metric_tracker.with_prefix(f"{loss_name}_"))
        _, _, _, t, partition_ids = self.prepare_data(df)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        losses = self.loss(u1, u2, y_tensor, t, partition_ids)
        loss = losses.mean()
        self.metric_tracker.track_epoch_scalar(f"loss_{loss_name}", loss.mean().item())
        return loss.mean()


BTUtilityFunctionFitter = partial(
    BaseUtilityFunctionFitter, loss=BTLoss(smoothing=None)
)
