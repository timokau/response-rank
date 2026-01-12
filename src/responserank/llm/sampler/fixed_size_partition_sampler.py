import random
from collections import defaultdict
from typing import Dict, Iterator, List

from torch.utils.data import Dataset, Sampler

from responserank.llm.packing import BasePacker, pack_and_analyze


class FixedSizePartitionBatchSampler(Sampler[List[int]]):
    """Sampler that respects partitions but guarantees fixed-size batches.

    Uses configurable packing algorithms to optimize partition cohesion while
    ensuring consistently sized batches for efficient training.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        partition_key: str,
        drop_last: bool,
        seed: int,
        packer: BasePacker,
        wandb_run=None,
    ) -> None:
        super().__init__(data_source=None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.partition_key = partition_key
        self.drop_last = drop_last
        self.seed = seed
        self.packer = packer
        self.wandb_run = wandb_run

        self.partition_to_indices: Dict[int, List[int]] = defaultdict(list)
        partition_ids = dataset[partition_key]
        for idx, pid in enumerate(partition_ids):
            self.partition_to_indices[int(pid)].append(idx)

        print(
            f"[FixedSizePartitionBatchSampler] Found {len(self.partition_to_indices)} partitions "
            f"(min size={min(len(v) for v in self.partition_to_indices.values())}, "
            f"max size={max(len(v) for v in self.partition_to_indices.values())}), "
            f"using {self.packer.name} packing algorithm."
        )

        self.batches, cohesion_stats = pack_and_analyze(
            self.packer, self.partition_to_indices, batch_size, seed
        )

        # Check all batches are the right size except possibly the last one
        for i, batch in enumerate(self.batches):
            if len(batch) != self.batch_size and (
                i != len(self.batches) - 1 or self.drop_last
            ):
                print(
                    f"WARNING: Batch {i} has size {len(batch)}, expected {self.batch_size}"
                )

        if drop_last:
            self.batches = [batch for batch in self.batches if len(batch) == batch_size]

        print(f"Generated {len(self.batches)} batches using {self.packer.name}")

        print("\n" + "=" * 60)
        print(f"PARTITION PACKING ANALYSIS ({self.packer.name})")
        print("=" * 60)
        print(f"Total partitions: {cohesion_stats['num_partitions']}")
        print(f"Total batches: {cohesion_stats['num_batches']}")
        print("-" * 60)
        print(
            f"Intact partitions: {cohesion_stats['intact_partitions']} "
            f"({cohesion_stats['intact_ratio']:.1%} of all partitions)"
        )
        print(
            f"Average fragments per partition: {cohesion_stats['avg_splits_per_partition']:.2f}"
        )
        print(f"Average cohesion score: {cohesion_stats['avg_cohesion_score']:.3f}")
        print(
            f"  Ideally fragmented partitions: {cohesion_stats.get('ideal_fragmentation_ratio', 0):.1%} "
        )
        print(
            f"  Fragment overhead: {cohesion_stats.get('fragment_overhead_pct', 0):.1f}%"
        )
        print("-" * 60)
        print("Fragment size statistics:")
        print(f"  Average: {cohesion_stats['avg_fragment_size']:.1f}")
        print(f"  Max: {cohesion_stats['max_fragment_size']}")
        print("=" * 60 + "\n")

        if self.wandb_run is not None:
            prefix = "packing"

            wandb_stats = {
                f"{prefix}/algorithm": self.packer.name,
                f"{prefix}/num_partitions": cohesion_stats["num_partitions"],
                f"{prefix}/num_batches": cohesion_stats["num_batches"],
                f"{prefix}/intact_ratio": cohesion_stats["intact_ratio"],
                f"{prefix}/avg_fragments_per_partition": cohesion_stats[
                    "avg_splits_per_partition"
                ],
                f"{prefix}/avg_cohesion_score": cohesion_stats["avg_cohesion_score"],
                f"{prefix}/avg_fragment_size": cohesion_stats["avg_fragment_size"],
                f"{prefix}/max_fragment_size": cohesion_stats["max_fragment_size"],
                f"{prefix}/ideal_fragmentation_ratio": cohesion_stats[
                    "ideal_fragmentation_ratio"
                ],
                f"{prefix}/fragment_overhead_pct": cohesion_stats[
                    "fragment_overhead_pct"
                ],
            }

            self.wandb_run.log(wandb_stats)

    def __iter__(self) -> Iterator[List[int]]:
        """Return an iterator over the pre-generated batches."""
        batches = self.batches.copy()
        # Use a fresh random instance each time to ensure consistent shuffling
        epoch_rng = random.Random(self.seed)
        epoch_rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.batches)
