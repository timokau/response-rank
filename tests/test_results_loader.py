"""Tests for the results_loader module."""

import numpy as np
import pandas as pd

from responserank.llm.analysis.results_loader import (
    get_epoch_summaries,
    matches_any_pattern,
)


class TestResultsLoader:
    """Tests for the results_loader module."""

    def test_get_epoch_summaries_explicit_epochs(self):
        """Test that only rows with explicit epoch values are included."""
        # Create data where metrics are logged at different steps than epochs
        data = {
            "_step": list(range(10)),
            "train/epoch": [
                np.nan,  # Step 0: no epoch
                np.nan,  # Step 1: no epoch
                0.0,  # Step 2: epoch 0 logged
                np.nan,  # Step 3: no epoch (metric logged here)
                np.nan,  # Step 4: no epoch
                1.0,  # Step 5: epoch 1 logged
                np.nan,  # Step 6: no epoch
                2.0,  # Step 7: epoch 2 logged
                np.nan,  # Step 8: no epoch
                2.5,  # Step 9: fractional epoch (cast to 2)
            ],
            "eval/accuracy": [
                np.nan,  # Step 0
                np.nan,  # Step 1
                0.50,  # Step 2: logged with epoch 0
                0.52,  # Step 3: logged without epoch (ignored)
                np.nan,  # Step 4
                0.60,  # Step 5: logged with epoch 1
                0.62,  # Step 6: logged without epoch (ignored)
                np.nan,  # Step 7: epoch 2 but no accuracy
                0.70,  # Step 8: logged without epoch (ignored)
                0.75,  # Step 9: logged with fractional epoch 2.5
            ],
            "eval/rewardbench_accuracy": [
                np.nan,  # Step 0
                np.nan,  # Step 1
                np.nan,  # Step 2: epoch 0 but no rb accuracy
                0.51,  # Step 3: logged without epoch (ignored)
                np.nan,  # Step 4
                0.61,  # Step 5: logged with epoch 1
                np.nan,  # Step 6
                0.71,  # Step 7: logged with epoch 2
                np.nan,  # Step 8
                np.nan,  # Step 9: no rb accuracy
            ],
        }

        df = pd.DataFrame(data)
        result = get_epoch_summaries(df)

        # Check epochs - only 0, 1, and 2 should be present
        assert sorted(result.keys()) == [0, 1, 2]

        # Epoch 0: only data from step 2 (where epoch 0.0 is explicitly logged)
        assert result[0]["eval/accuracy"] == 0.50
        assert "eval/rewardbench_accuracy" not in result[0]  # No rb accuracy at step 2

        # Epoch 1: only data from step 5 (where epoch 1.0 is explicitly logged)
        assert result[1]["eval/accuracy"] == 0.60
        assert result[1]["eval/rewardbench_accuracy"] == 0.61

        # Epoch 2: combines data from steps 7 and 9 (both have epoch 2)
        # Last values: accuracy from step 9, rb_accuracy from step 7
        assert result[2]["eval/accuracy"] == 0.75  # From step 9
        assert result[2]["eval/rewardbench_accuracy"] == 0.71  # From step 7

    def test_get_epoch_summaries_edge_cases(self):
        """Test edge cases and column filtering."""
        # Test 1: Empty/missing data
        assert get_epoch_summaries(pd.DataFrame()) == {}
        assert (
            get_epoch_summaries(
                pd.DataFrame({"_step": [0, 1], "eval/accuracy": [0.5, 0.6]})
            )
            == {}
        )
        assert (
            get_epoch_summaries(
                pd.DataFrame(
                    {
                        "_step": [0, 1],
                        "train/epoch": [np.nan, np.nan],
                        "eval/accuracy": [0.5, 0.6],
                    }
                )
            )
            == {}
        )

        # Test 2: Column filtering
        df = pd.DataFrame(
            {
                "_step": [0, 1, 2, 3],
                "train/epoch": [0.0, 0.0, 1.0, 1.0],
                "eval/accuracy": [0.5, 0.6, 0.7, 0.8],
                "eval/rewardbench_accuracy": [0.51, 0.61, 0.71, 0.81],
                "train/loss": [1.0, 0.9, 0.8, 0.7],
            }
        )

        # Filter to only eval/accuracy
        result = get_epoch_summaries(df, columns=["eval/accuracy"])
        assert sorted(result.keys()) == [0, 1]
        assert result[0] == {"eval/accuracy": 0.6}
        assert result[1] == {"eval/accuracy": 0.8}

        # Verify unfiltered includes all columns
        result_all = get_epoch_summaries(df)
        assert len(result_all[0]) > 1  # More than just eval/accuracy

    def test_matches_any_pattern(self):
        patterns = {r"baseline.*", r"rr.*", r"experiment_\d+"}

        assert matches_any_pattern("baseline_v1", patterns) is True
        assert matches_any_pattern("rr_agreement", patterns) is True
        assert matches_any_pattern("experiment_123", patterns) is True
        assert matches_any_pattern("nomatch", patterns) is False
        assert matches_any_pattern("test", set()) is False
