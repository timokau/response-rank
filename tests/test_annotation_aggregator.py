import numpy as np
import pytest

from responserank.llm.data.annotation_aggregator import (
    RandomChoiceAggregator,
)
from responserank.llm.data.prepare_dataset import compute_bt_target_stats


def test_random_choice_aggregator():
    """Test RandomChoiceAggregator aggregates one example per comparison."""
    comparisons = [
        {
            "comparison_id": "comp1",
            "conversation_a": [{"role": "assistant", "content": "A"}],
            "conversation_b": [{"role": "assistant", "content": "B"}],
            "model_a": "model_a",
            "model_b": "model_b",
            "input_ids_a": [1, 2, 3],
            "input_ids_b": [4, 5, 6],
            "attention_mask_a": [1, 1, 1],
            "attention_mask_b": [1, 1, 1],
            "agreement_stats": {},
            "annotations": [
                {
                    "overall_pref": "A-is-clearly-better",
                    "time_spent": 1.0,
                    "evaluator": "ann1",
                    "is_tie": False,
                    "preference_strength": 2,
                    "preference_ordinal": 1.0,
                },
                {
                    "overall_pref": "A-is-slightly-better",
                    "time_spent": 2.0,
                    "evaluator": "ann2",
                    "is_tie": False,
                    "preference_strength": 1,
                    "preference_ordinal": 0.5,
                },
            ],
        },
        {
            "comparison_id": "comp2",
            "conversation_a": [{"role": "assistant", "content": "C"}],
            "conversation_b": [{"role": "assistant", "content": "D"}],
            "model_a": "model_a",
            "model_b": "model_b",
            "input_ids_a": [7, 8, 9],
            "input_ids_b": [10, 11, 12],
            "attention_mask_a": [1, 1, 1],
            "attention_mask_b": [1, 1, 1],
            "agreement_stats": {},
            "annotations": [
                {
                    "overall_pref": "A-is-clearly-better",
                    "time_spent": 1.5,
                    "evaluator": "ann1",
                    "is_tie": False,
                    "preference_strength": 2,
                    "preference_ordinal": 1.0,
                },
                {
                    "overall_pref": "B-is-slightly-better",
                    "time_spent": 1.8,
                    "evaluator": "ann3",
                    "is_tie": False,
                    "preference_strength": 1,
                    "preference_ordinal": -0.5,
                },
            ],
        },
    ]

    aggregator = RandomChoiceAggregator(bt_target="hard")
    rng = np.random.RandomState(42)

    selected = aggregator.aggregate_annotations(comparisons, rng)

    # Should have one example per comparison
    assert len(selected) == 2

    comparison_ids = {ex["comparison_id"] for ex in selected}
    assert comparison_ids == {"comp1", "comp2"}

    assert all(ex["bt_target"] == 1.0 for ex in selected)


def _make_base_comparison(comparison_id, agreement_score):
    return {
        "comparison_id": comparison_id,
        "conversation_a": [{"role": "assistant", "content": "A"}],
        "conversation_b": [{"role": "assistant", "content": "B"}],
        "model_a": "model_a",
        "model_b": "model_b",
        "input_ids_a": [1, 2, 3],
        "input_ids_b": [4, 5, 6],
        "attention_mask_a": [1, 1, 1],
        "attention_mask_b": [1, 1, 1],
        "agreement_stats": {"agreement_score": agreement_score},
        "annotations": [],
    }


def _make_annotation(
    *,
    evaluator: str,
    time_spent: float,
    overall_pref: str = "A-is-clearly-better",
    preference_strength: int = 2,
    preference_ordinal: float = 1.0,
    is_tie: bool = False,
):
    return {
        "overall_pref": overall_pref,
        "time_spent": time_spent,
        "evaluator": evaluator,
        "is_tie": is_tie,
        "preference_strength": preference_strength,
        "preference_ordinal": preference_ordinal,
    }


def test_bt_target_mean_preference():
    """Test bt_target='mean_preference' correctly averages preferences and adjusts for chosen option."""
    # 3-1 majority for A: mean = (1.0 + 1.0 + 1.0 + (-1.0)) / 4 = 0.5
    comparisons = [_make_base_comparison("comp1", agreement_score=0.5)]
    comparisons[0]["annotations"] = [
        _make_annotation(
            evaluator="ann1",
            time_spent=1.0,
            overall_pref="A-is-clearly-better",
            preference_ordinal=1.0,
        ),
        _make_annotation(
            evaluator="ann2",
            time_spent=2.0,
            overall_pref="A-is-clearly-better",
            preference_ordinal=1.0,
        ),
        _make_annotation(
            evaluator="ann3",
            time_spent=3.0,
            overall_pref="A-is-clearly-better",
            preference_ordinal=1.0,
        ),
        _make_annotation(
            evaluator="ann4",
            time_spent=4.0,
            overall_pref="B-is-clearly-better",
            preference_ordinal=-1.0,
        ),
    ]

    aggregator = RandomChoiceAggregator(bt_target="mean_preference")
    rng = np.random.RandomState(42)

    selected = aggregator.aggregate_annotations(comparisons, rng)

    assert len(selected) == 1
    example = selected[0]

    # Mean preference = 0.5 (favors A)
    # If A is chosen: target = (0.5 + 1) / 2 = 0.75
    # If B is chosen: target = (-0.5 + 1) / 2 = 0.25
    if example["chosen"] == comparisons[0]["conversation_a"]:
        assert example["bt_target"] == pytest.approx(0.75)
    else:
        assert example["bt_target"] == pytest.approx(0.25)


def test_compute_bt_target_stats_hard_mode():
    """Test compute_bt_target_stats with hard targets (all 1.0)."""
    examples = [{"bt_target": 1.0} for _ in range(10)]
    stats = compute_bt_target_stats(examples)

    assert stats["mean"] == pytest.approx(1.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(1.0)
    assert stats["std"] == pytest.approx(0.0)
    assert stats["contradictory_count"] == 0
    assert stats["contradictory_fraction"] == pytest.approx(0.0)


def test_compute_bt_target_stats_agreement_mode():
    """Test compute_bt_target_stats with agreement targets (range 0.5-1.0)."""
    examples = [
        {"bt_target": 1.0},  # Perfect agreement
        {"bt_target": 0.75},  # High agreement
        {"bt_target": 0.5},  # No agreement
    ]
    stats = compute_bt_target_stats(examples)

    assert stats["mean"] == pytest.approx(0.75)
    assert stats["min"] == pytest.approx(0.5)
    assert stats["max"] == pytest.approx(1.0)
    assert stats["contradictory_count"] == 0
    assert stats["contradictory_fraction"] == pytest.approx(0.0)
