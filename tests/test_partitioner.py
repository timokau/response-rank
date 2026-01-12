from responserank.llm.data.partitioner import split_partition_avoiding_ties


class TestSplitPartitionAvoidingTies:
    """Test cases for the split_partition_avoiding_ties function."""

    def test_no_ties(self):
        """Test with no tied ranks - should return single partition."""
        result = split_partition_avoiding_ties(
            [0.0, 1.0, 2.0], rel_tol=1e-9, target_size=-1
        )
        assert len(result) == 1

    def test_complex_ties_auto(self):
        """Test auto target_size with complex tie pattern: [0.0, 0.0, 0.0, 1.0, 1.0, 2.0]."""
        ranks = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0]
        result = split_partition_avoiding_ties(ranks, rel_tol=1e-9, target_size=-1)

        # Should have 3 partitions (max duplicates = 3 for rank 0.0)
        assert len(result) == 3

        # Each partition should have unique ranks
        for partition in result:
            ranks_in_partition = [ranks[i] for i in partition]
            assert len(set(ranks_in_partition)) == len(ranks_in_partition)

        # All indices should be covered
        all_indices = []
        for partition in result:
            all_indices.extend(partition)
        assert sorted(all_indices) == [0, 1, 2, 3, 4, 5]

    def test_all_same_rank(self):
        """Test when all ranks are the same."""
        result = split_partition_avoiding_ties(
            [1.0, 1.0, 1.0], rel_tol=1e-9, target_size=-1
        )

        # Should have 3 partitions (one per element)
        assert len(result) == 3

        # Each partition should have exactly one element
        for partition in result:
            assert len(partition) == 1

        # All indices should be covered
        all_indices = []
        for partition in result:
            all_indices.extend(partition)
        assert sorted(all_indices) == [0, 1, 2]

    def test_spread_distribution(self):
        """Test that distribution is spread across partitions."""
        ranks = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        result = split_partition_avoiding_ties(ranks, rel_tol=1e-9, target_size=-1)

        # Should have 4 partitions (max duplicates = 4 for rank 0.0)
        assert len(result) == 4

        # Check partitions are spread (each should have 2 elements)
        sizes = [len(partition) for partition in result]
        assert all(size == 2 for size in sizes)

        # Each partition should have unique ranks
        for partition in result:
            ranks_in_partition = [ranks[i] for i in partition]
            assert len(set(ranks_in_partition)) == len(ranks_in_partition)
