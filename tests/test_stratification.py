import numpy as np

from responserank.llm.data.stratification import (
    AnnotatorStratifier,
    GlobalPartitionStratifier,
)


class TestGlobalPartitionStratifier:
    def test_single_partition(self):
        """Test that all examples go to single partition."""
        examples = [
            {"rank": 0.0},
            {"rank": 0.0},
            {"rank": 1.0},
            {"rank": 1.0},
            {"rank": 2.0},
        ]

        stratifier = GlobalPartitionStratifier()
        rng = np.random.RandomState(42)
        partition_ids = stratifier.compute_partitions(examples, rng)

        # All examples should be in partition 0
        assert all(pid == 0 for pid in partition_ids)
        assert len(set(partition_ids)) == 1


class TestAnnotatorStratifier:
    """Test cases for the AnnotatorStratifier."""

    def test_basic_functionality(self):
        """Test that each annotator gets its own partition."""
        examples = [
            {
                "prompt": "a",
                "chosen": "b",
                "rejected": "c",
                "extra": {"evaluator": "ann1"},
            },
            {
                "prompt": "d",
                "chosen": "e",
                "rejected": "f",
                "extra": {"evaluator": "ann1"},
            },
            {
                "prompt": "g",
                "chosen": "h",
                "rejected": "i",
                "extra": {"evaluator": "ann2"},
            },
            {
                "prompt": "j",
                "chosen": "k",
                "rejected": "l",
                "extra": {"evaluator": "ann3"},
            },
            {
                "prompt": "m",
                "chosen": "n",
                "rejected": "o",
                "extra": {"evaluator": "ann3"},
            },
        ]

        stratifier = AnnotatorStratifier()
        rng = np.random.RandomState(42)
        partition_ids = stratifier.compute_partitions(examples, rng)

        # Should have 3 partitions (one per annotator)
        assert len(set(partition_ids)) == 3

        # Examples from same annotator should be in same partition
        assert partition_ids[0] == partition_ids[1]  # ann1
        assert partition_ids[3] == partition_ids[4]  # ann3

        # Different annotators should have different partitions
        assert partition_ids[0] != partition_ids[2]
        assert partition_ids[0] != partition_ids[3]
        assert partition_ids[2] != partition_ids[3]
