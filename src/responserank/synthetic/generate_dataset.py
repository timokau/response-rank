import logging
import os
from pathlib import Path

import hydra
import hydra.core
import hydra.core.hydra_config
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from responserank.synthetic.synthetic_data import generate_preference_dataset
from responserank.synthetic.util import (
    UtilityFunction,
    dataset_to_dataframe,
    module_to_numpy_uf,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_true_uf(num_features, true_utility_hidden_layers, random_state):
    uf = UtilityFunction(
        num_features,
        hidden_layers=true_utility_hidden_layers,
        dropout_rate=None,
    )
    fixed_nn_utility = module_to_numpy_uf(uf)

    all_xs = random_state.rand(1000, num_features)
    all_utilities = fixed_nn_utility(all_xs)
    mean_utility = np.mean(all_utilities)
    mean_value_first_feature = np.mean(all_xs[:, 0])

    def true_utility(x):
        normalized_nn_utility = fixed_nn_utility(x) / mean_utility
        normalized_first_feature = x[:, 0] / mean_value_first_feature
        return normalized_nn_utility + 2 * normalized_first_feature

    return true_utility


@hydra.main(
    config_path="../../../conf/synthetic",
    config_name="generate_dataset_config",
    version_base="1.3",
)
def generate_dataset_main(cfg: DictConfig):
    logger.info(f"Generating dataset with config: {cfg}")

    random_state = np.random.RandomState(cfg.random_seed)

    true_utility = instantiate(
        cfg.dataset.true_utility,
        num_features=cfg.dataset.item_generator.num_features,
        random_state=random_state,
    )

    item_generator = instantiate(
        cfg.dataset.item_generator, utility_function=true_utility
    )
    trial_generator = instantiate(cfg.dataset.trial_generator)

    x1, x2, u1, u2, y, t, _, partition_ids = generate_preference_dataset(
        item_generator=item_generator,
        trial_generator=trial_generator,
        random_state=random_state,
        num_comparisons=cfg.dataset.num_train_samples + cfg.dataset.num_test_samples,
        num_partitions=cfg.dataset.num_partitions,
        partition_rt_variability=cfg.dataset.partition_rt_variability,
    )

    df = dataset_to_dataframe(x1, x2, u1, u2, y, t, partition_ids)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    df_file = output_dir / f"{dataset_name}.csv"
    df.to_csv(df_file, index=False)

    logger.info(f"Dataset saved to {df_file}")
    print(df_file)


if __name__ == "__main__":
    generate_dataset_main()
