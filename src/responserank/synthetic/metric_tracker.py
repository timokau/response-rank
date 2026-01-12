"""
Metric tracking infrastructure for experiment monitoring and evaluation.

This module provides classes for tracking metrics during training and evaluation.
It supports tracking both per-epoch metrics (like loss values or accuracy) and
summary metrics (aggregated statistics at the end of an experiment).
"""

from typing import Optional

import numpy as np


class MetricTracker:
    def __init__(self, metric_prefix: str, backend: Optional["MemoryBackend"]):
        """Create a metric tracker.

        Args:
            metric_prefix: Metric name prefix to apply when logging.
            backend: Backend implementation. Pass ``None`` to use
                :class:`MemoryBackend`.
        """
        self.metric_prefix = metric_prefix
        self.backend = MemoryBackend() if backend is None else backend

    def _mname(self, metric_name):
        return f"{self.metric_prefix}{metric_name}"

    def with_prefix(self, prefix):
        return MetricTracker(self._mname(prefix), self.backend)

    def track_epoch_metric(self, metric_name, value, metric_type):
        self.backend.track_epoch_metric(self._mname(metric_name), value, metric_type)

    def track_epoch_scalar(self, metric_name, value):
        self.backend.track_epoch_metric(self._mname(metric_name), value, "scalar")

    def track_epoch_list(self, metric_name, value):
        self.track_epoch_scalar(f"{metric_name}_mean", np.mean(value).item())

    def track_summary_metric(
        self, metric_name, value, metric_type, dataset_metric: bool
    ):
        self.backend.track_summary_metric(
            self._mname(metric_name), value, metric_type, dataset_metric=dataset_metric
        )

    def finalize_epoch(self):
        self.backend.finalize_epoch()

    def get_epoch_metrics(self):
        return self.backend.get_epoch_metrics()

    def get_summary_metrics(self):
        return self.backend.get_summary_metrics()


class MemoryBackend:
    def __init__(self):
        self.epoch_metrics = []
        self.summary_metrics = {}
        self.current_epoch_metric = {}

    def track_epoch_metric(self, metric_name, value, metric_type):
        self.current_epoch_metric[metric_name] = {"value": value, "type": metric_type}

    def track_summary_metric(
        self, metric_name, value, metric_type, dataset_metric: bool
    ):
        # Some sanity checks
        if metric_type == "histogram":
            assert "x" in value, f"Missing 'x' key for histogram {metric_name}"
            # Ensure value is a list or a numpy array
            x = value["x"]
            assert isinstance(x, (list, np.ndarray)), (
                f"Histogram must be a list, but got {type(x)} for {metric_name}: {x}"
            )
        self.summary_metrics[metric_name] = {
            "value": value,
            "type": metric_type,
            "dataset_metric": dataset_metric,
        }

    def finalize_epoch(self):
        self.epoch_metrics.append(self.current_epoch_metric)
        self.current_epoch_metric = {}

    def get_epoch_metrics(self):
        return self.epoch_metrics

    def get_summary_metrics(self):
        return self.summary_metrics
