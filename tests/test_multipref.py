import random

from responserank.llm.data.datasets.multipref import (
    AgreementProcessor,
    calculate_agreement_score,
    count_preference_strengths,
    determine_chosen_rejected,
    extract_preference_strength,
)


def test_determine_chosen_rejected():
    """Test preference determination logic"""
    comp_a, comp_b = "Response A", "Response B"
    model_a, model_b = "model_1", "model_2"

    chosen, rejected, chosen_model, rejected_model = determine_chosen_rejected(
        "A-is-clearly-better", comp_a, comp_b, model_a, model_b
    )
    assert chosen == comp_a
    assert rejected == comp_b
    assert chosen_model == model_a
    assert rejected_model == model_b

    chosen, rejected, chosen_model, rejected_model = determine_chosen_rejected(
        "A-is-slightly-better", comp_a, comp_b, model_a, model_b
    )
    assert chosen == comp_a
    assert rejected == comp_b

    chosen, rejected, chosen_model, rejected_model = determine_chosen_rejected(
        "B-is-clearly-better", comp_a, comp_b, model_a, model_b
    )
    assert chosen == comp_b
    assert rejected == comp_a
    assert chosen_model == model_b
    assert rejected_model == model_a


def test_extract_preference_strength():
    """Test preference strength extraction"""
    assert extract_preference_strength("A-is-clearly-better") == 2
    assert extract_preference_strength("B-is-clearly-better") == 2
    assert extract_preference_strength("A-is-slightly-better") == 1
    assert extract_preference_strength("B-is-slightly-better") == 1


def test_count_preference_strengths():
    """Test preference strength counting"""
    annotations = [
        {"overall_pref": "A-is-clearly-better"},
        {"overall_pref": "A-is-slightly-better"},
        {"overall_pref": "B-is-clearly-better"},
    ]

    (
        pref_a_count,
        pref_b_count,
        clearly_better_a,
        slightly_better_a,
        clearly_better_b,
        slightly_better_b,
        tie_count,
    ) = count_preference_strengths(annotations)

    assert pref_a_count == 2
    assert pref_b_count == 1
    assert clearly_better_a == 1
    assert slightly_better_a == 1
    assert clearly_better_b == 1
    assert slightly_better_b == 0
    assert tie_count == 0


def test_calculate_agreement_score():
    """Test agreement score calculation"""
    # Test strong consensus for A
    score = calculate_agreement_score(4, 0, 0, 0, 0, False, 0.5)
    assert score == 1.0  # abs(4 - 0) / 4 = 1.0

    # Test complete disagreement
    score = calculate_agreement_score(2, 0, 2, 0, 0, False, 0.5)
    assert score == 0.0  # abs(2 - 2) / 4 = 0.0

    # Test with mixed weights
    score = calculate_agreement_score(2, 2, 0, 0, 0, False, 0.5)
    assert score == 0.75  # abs(2*1 + 2*0.5 - 0) / 4 = 3/4 = 0.75


def test_agreement_processor():
    """Test AgreementProcessor"""
    processor = AgreementProcessor()
    rng = random.Random(42)

    # Create test examples with required count fields
    examples = [
        {
            "comparison_id": "comp1",
            "extra": {
                "clearly_better_a_count": 4,
                "slightly_better_a_count": 0,
                "clearly_better_b_count": 0,
                "slightly_better_b_count": 0,
                "agreement_score": calculate_agreement_score(4, 0, 0, 0, 0, False, 0.5),
            },
            "rank": 0.5,  # Original rank
        },
        {
            "comparison_id": "comp2",
            "extra": {
                "clearly_better_a_count": 2,
                "slightly_better_a_count": 0,
                "clearly_better_b_count": 2,
                "slightly_better_b_count": 0,
                "agreement_score": calculate_agreement_score(2, 0, 2, 0, 0, False, 0.5),
            },
            "rank": 0.8,
        },
        {
            "comparison_id": "comp3",
            "extra": {
                "clearly_better_a_count": 0,
                "slightly_better_a_count": 2,
                "clearly_better_b_count": 0,
                "slightly_better_b_count": 2,
                "agreement_score": calculate_agreement_score(0, 2, 0, 2, 0, False, 0.5),
            },
            "rank": 0.3,
        },
    ]

    processed, info = processor.process_dataset(examples, rng)

    # Check first example (strong consensus for A)
    assert processed[0]["rank"] == -1.0  # -abs(4 - 0) / 4 = -1.0
    assert processed[0]["extra"]["rank_original"] == 0.5
    assert processed[0]["extra"]["agreement_score"] == 1.0

    # Check second example (split between A and B)
    assert processed[1]["rank"] == 0.0  # -abs(2 - 2) / 4 = 0
    assert processed[1]["extra"]["rank_original"] == 0.8
    assert processed[1]["extra"]["agreement_score"] == 0.0

    # Check third example (split between slight preferences)
    assert processed[2]["rank"] == 0.0  # -abs(1 - 1) / 4 = 0
    assert processed[2]["extra"]["rank_original"] == 0.3
    assert processed[2]["extra"]["agreement_score"] == 0.0

    # Check statistics
    assert info["rank_mean"] == (-1.0 + 0.0 + 0.0) / 3
    assert info["rank_min"] == -1.0
    assert info["rank_max"] == 0.0
    assert info["unique_ranks"] == 2
