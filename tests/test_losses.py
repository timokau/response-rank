import torch

from responserank.synthetic.losses import (
    BTLoss,
    DeterministicRTLoss,
)
from responserank.synthetic.metric_tracker import MetricTracker
from responserank.synthetic.synthetic_data import HyperbolicUdiffRtRel


def test_bt_loss():
    metric_tracker = MetricTracker(metric_prefix="", backend=None)
    bt_loss = BTLoss(smoothing=None)
    bt_loss.set_metric_tracker(metric_tracker)

    # Test basic functionality
    u1 = torch.tensor([1.0, 1.0, 2.0, 0.0])
    u2 = torch.tensor([1.0, 0.0, 1.0, 2.0])
    y = torch.tensor([1, 0, 1, 0])
    bt_loss(u1, u2, y)
    metric_tracker.finalize_epoch()
    metrics = metric_tracker.get_epoch_metrics()
    assert isinstance(metrics, list), (
        f"Expected metrics to be a list, but got {type(metrics)}"
    )
    assert len(metrics) > 0, "Metrics list is empty"

    # Test extreme cases
    u1 = torch.tensor([100.0, -100.0])
    u2 = torch.tensor([0.0, 0.0])
    y = torch.tensor([1, 0])
    bt_loss(u1, u2, y)
    metric_tracker.finalize_epoch()
    metrics = metric_tracker.get_epoch_metrics()
    bt_loss(u1, u2, y)
    metric_tracker.finalize_epoch()

    # Test shift invariance
    u1 = torch.tensor([1.0, 2.0, 3.0])
    u2 = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([1, 0, 1])
    loss1 = bt_loss(u1, u2, y)

    shift = 10.0
    u1_shifted = u1 + shift
    u2_shifted = u2 + shift
    loss2 = bt_loss(u1_shifted, u2_shifted, y)

    assert torch.allclose(loss1, loss2, atol=1e-7)

    # Symmetry
    u1 = torch.tensor([1.0])
    u2 = torch.tensor([0.0])
    loss1 = bt_loss(u1, u2, torch.tensor([1]))
    loss2 = bt_loss(u2, u1, torch.tensor([0]))
    assert torch.allclose(loss1, loss2)

    # Check that loss increases when prediction gets worse
    y = torch.tensor([1])
    u1_good = torch.tensor([1.0])
    u2 = torch.tensor([0.0])
    u1_bad = torch.tensor([0.0])
    loss_good = bt_loss(u1_good, u2, y)
    loss_bad = bt_loss(u1_bad, u2, y)
    assert loss_bad > loss_good


def test_deterministic_rt_loss():
    # Test basic functionality
    det_rt_loss = DeterministicRTLoss(assumed_udiff_rt_rel=HyperbolicUdiffRtRel(0, 10))
    u1 = torch.tensor([1.0, 2.0, 3.0])
    u2 = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([1, 1, 0])
    t = torch.tensor([0.6, 0.7, 0.8])

    losses = det_rt_loss(u1, u2, y, t)

    assert losses.shape == (3,)
    assert torch.all(losses >= 0)
