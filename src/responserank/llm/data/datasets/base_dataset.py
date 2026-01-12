from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Abstract base class for datasets.

    All datasets must return examples in standardized format with these core fields:
    - prompt: str - The input prompt
    - chosen: str - Preferred response
    - rejected: str - Non-preferred response
    - rank: float - Ranking value (lower = stronger preference, e.g. response time)
    - comparison_id: str - Unique identifier for this comparison
    - partition_id: int - Stratification partition ID
    - extra: dict - Dataset-specific metadata fields
    """

    @abstractmethod
    def load_raw_data(self):
        """Load the raw dataset and return it.

        Returns:
            Raw dataset object (e.g., HuggingFace Dataset)
        """
        pass

    @abstractmethod
    def create_test_split(self, dataset, rng, test_size):
        """Create test split and return set of test comparison IDs.

        Args:
            dataset: Raw dataset object
            rng: Random number generator
            test_size: Number of comparisons for test set

        Returns:
            Set of comparison IDs for test set
        """
        pass

    @abstractmethod
    def to_comparison_dataset(self, raw_data, rng):
        """Convert raw dataset to comparison dataset format.

        Args:
            raw_data: Raw dataset object
            rng: Random number generator

        Returns:
            Dataset of comparisons with nested annotations
        """
        pass

    @abstractmethod
    def get_dataset_name(self):
        """Return human-readable dataset name for logging.

        Returns:
            String name of the dataset
        """
        pass
