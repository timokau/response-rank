"""Base class for partition-aware samplers that can be instantiated via Hydra."""

from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler as TorchRandomSampler

from .fixed_size_partition_sampler import FixedSizePartitionBatchSampler


class BaseSampler(ABC):
    """Abstract base class for samplers that can create DataLoaders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the sampler for logging purposes."""
        pass

    @abstractmethod
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        seed: int,
        collate_fn,
        **kwargs,
    ) -> DataLoader:
        """Create a DataLoader with the appropriate sampler configuration.

        Args:
            dataset: The dataset to sample from
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            collate_fn: Collation function for the DataLoader
            **kwargs: Additional arguments for specific samplers

        Returns:
            DataLoader configured with the appropriate sampler
        """
        pass


class FixedSizeSampler(BaseSampler):
    """Sampler that maintains fixed batch sizes while respecting partitions."""

    def __init__(
        self,
        drop_last: bool,
        packer,
    ):
        """Initialize the fixed-size sampler.

        Args:
            drop_last: Whether to drop the last incomplete batch
            packer: Packing algorithm to use
        """
        self.drop_last = drop_last
        self.packer = packer
        self.wandb_run = None  # Will be set externally

    @property
    def name(self) -> str:
        return "fixed_size"

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        seed: int,
        collate_fn,
        **kwargs,
    ) -> DataLoader:
        """Create a DataLoader with fixed-size partition-aware sampling."""
        sampler = FixedSizePartitionBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            partition_key="partition_id",
            drop_last=self.drop_last,
            seed=seed,
            packer=self.packer,
            wandb_run=self.wandb_run,
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )


class RandomSampler(BaseSampler):
    """Sampler that uses PyTorch's standard RandomSampler, ignoring partitions entirely."""

    def __init__(self, drop_last: bool):
        """Initialize the random sampler.

        Args:
            drop_last: Whether to drop the last incomplete batch
        """
        self.drop_last = drop_last

    @property
    def name(self) -> str:
        return "random"

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        seed: int,
        collate_fn,
        **kwargs,
    ) -> DataLoader:
        """Create a DataLoader with standard random sampling, ignoring partitions."""
        # Use PyTorch's standard RandomSampler which ignores partition information
        sampler = TorchRandomSampler(
            dataset, generator=torch.Generator().manual_seed(seed)
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
            num_workers=0,
        )
