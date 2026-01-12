"""Tests for PassthroughRewardDataCollatorWithPadding to verify metadata passthrough."""

import torch

from responserank.llm.reward_trainer import (
    PassthroughRewardDataCollatorWithPadding,
)


class MockTokenizer:
    def pad(self, features, padding, pad_to_multiple_of=None, return_tensors="pt"):
        """Mock pad method that converts lists to tensors without actual padding."""
        input_ids = torch.tensor([f["input_ids"] for f in features])
        attention_mask = torch.tensor([f["attention_mask"] for f in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def test_response_time_association_after_sampling():
    """Test that response times are correctly associated with comparisons after sampling."""
    tokenizer = MockTokenizer()
    collator = PassthroughRewardDataCollatorWithPadding(tokenizer)

    features = [
        {
            "input_ids_chosen": [1, 2, 3],
            "attention_mask_chosen": [1, 1, 1],
            "input_ids_rejected": [4, 5, 6],
            "attention_mask_rejected": [1, 1, 1],
            "rank": 1.5,
            "partition_id": 0,
            "bt_target": 1.0,
        },
        {
            "input_ids_chosen": [7, 8, 9],
            "attention_mask_chosen": [1, 1, 1],
            "input_ids_rejected": [10, 11, 12],
            "attention_mask_rejected": [1, 1, 1],
            "rank": 2.3,
            "partition_id": 1,
            "bt_target": 1.0,
        },
        {
            "input_ids_chosen": [13, 14, 15],
            "attention_mask_chosen": [1, 1, 1],
            "input_ids_rejected": [16, 17, 18],
            "attention_mask_rejected": [1, 1, 1],
            "rank": 0.8,
            "partition_id": 0,
            "bt_target": 1.0,
        },
    ]

    batch = collator(features)

    expected_input_ids_chosen = torch.tensor([[1, 2, 3], [7, 8, 9], [13, 14, 15]])
    assert torch.equal(batch["input_ids_chosen"], expected_input_ids_chosen), (
        f"input_ids_chosen mismatch: expected {expected_input_ids_chosen}, "
        f"got {batch['input_ids_chosen']}"
    )

    expected_input_ids_rejected = torch.tensor([[4, 5, 6], [10, 11, 12], [16, 17, 18]])
    assert torch.equal(batch["input_ids_rejected"], expected_input_ids_rejected), (
        f"input_ids_rejected mismatch: expected {expected_input_ids_rejected}, "
        f"got {batch['input_ids_rejected']}"
    )

    expected_response_times = torch.tensor([1.5, 2.3, 0.8], dtype=torch.float32)
    assert torch.equal(batch["rank"], expected_response_times), (
        f"rank mismatch: expected {expected_response_times}, got {batch['rank']}"
    )

    expected_partition_ids = torch.tensor([0, 1, 0], dtype=torch.long)
    assert torch.equal(batch["partition_id"], expected_partition_ids), (
        f"partition_id mismatch: expected {expected_partition_ids}, "
        f"got {batch['partition_id']}"
    )
