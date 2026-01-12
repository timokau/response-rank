import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, cast

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from responserank.synthetic.fitting import (
    BaseUtilityFunctionFitter,
    BTUtilityFunctionFitter,
)
from responserank.synthetic.losses import (
    DeterministicRTLoss,
    RTRankAnchoredLoss,
)
from responserank.synthetic.metric_tracker import MetricTracker
from responserank.synthetic.util import (
    UtilityFunction,
    convert_to_serializable,
    extract_item_features_from_pandas_df,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../../conf/synthetic",
    config_name="run_learner_config",
    version_base="1.3",
)
def run_learner(cfg: DictConfig):
    start_time = time.time()
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(hydra_output_dir)

    logger.info(f"Starting run_learner with config: {cfg}")
    train_df = pd.read_csv(cfg.train_df_file)
    test_df = pd.read_csv(cfg.test_df_file)

    if train_df["y"].isna().any():
        logger.error(
            f"Found {train_df['y'].isna().sum()} NaN values in training dataset choice column."
        )
        raise ValueError("NaN values in training dataset 'y' column")
    if test_df["y"].isna().any():
        logger.error(
            f"Found {test_df['y'].isna().sum()} NaN values in test dataset choice column."
        )
        raise ValueError("NaN values in test dataset 'y' column")

    subset_size = int(len(train_df) * cfg.dataset_fraction)
    logger.info(
        f"Using {subset_size}/{len(train_df)} training samples ({cfg.dataset_fraction:.2f} fraction)"
    )
    train_df = train_df.iloc[:subset_size]

    dataset_name = Path(cfg.train_df_file).parent.name

    learner_cfg = cfg.learner

    x1 = extract_item_features_from_pandas_df(train_df, "x1")
    extract_item_features_from_pandas_df(train_df, "x2")
    num_features = x1.shape[1]

    if learner_cfg.permute_response_times:
        # Permute response times in both train and test datasets
        train_df["t"] = np.random.permutation(train_df["t"].to_numpy())
        test_df["t"] = np.random.permutation(test_df["t"].to_numpy())
        logger.info("Response times have been permuted.")

    # Create MetricTracker instance
    learner_name = HydraConfig.get().job.override_dirname
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Appease pyright, which believes cfg_dict is None.
    cfg_dict = cast(Dict[str, Any], cfg_dict)
    cfg_dict["dataset_name"] = dataset_name
    cfg_dict["learner_name"] = learner_name
    metric_tracker = MetricTracker(metric_prefix="", backend=None)

    if learner_cfg.learner_type == "bt":
        utility_function = UtilityFunction(
            num_features,
            hidden_layers=learner_cfg.hidden_layers,
            dropout_rate=None,
        )
        fitter = BTUtilityFunctionFitter(
            utility_function,
            metric_tracker=metric_tracker,
            learning_rate=learner_cfg.learning_rate,
        )
    elif learner_cfg.learner_type in [
        "rr",
        "rr_stratified",
        "rr_pooled",
    ]:
        utility_function = UtilityFunction(
            num_features,
            hidden_layers=learner_cfg.hidden_layers,
            dropout_rate=None,
        )
        rt_loss = RTRankAnchoredLoss(
            unreduce=learner_cfg.unreduce,
            misclassification_penalty=learner_cfg.misclassification_penalty,
            use_strata=learner_cfg.use_strata,
            worth_transform=learner_cfg.worth_transform,
            mean_reduce=learner_cfg.mean_reduce,
        )

        fitter = BaseUtilityFunctionFitter(
            utility_function,
            metric_tracker=metric_tracker,
            learning_rate=learner_cfg.learning_rate,
            l2_reg=learner_cfg.l2_reg,
            loss=rt_loss,
        )
    else:  # RT regression variants
        utility_function = UtilityFunction(
            num_features,
            hidden_layers=learner_cfg.hidden_layers,
            dropout_rate=None,
        )
        assumed_udiff_rt_rel = instantiate(learner_cfg.udiff_rt_rel)
        rt_loss = DeterministicRTLoss(assumed_udiff_rt_rel=assumed_udiff_rt_rel)

        fitter = BaseUtilityFunctionFitter(
            utility_function,
            metric_tracker=metric_tracker,
            learning_rate=learner_cfg.learning_rate,
            l2_reg=learner_cfg.l2_reg,
            loss=rt_loss,
        )

    fitter.fit(
        train_df,
        test_df,
        num_epochs=learner_cfg.num_epochs,
        early_stopping_patience=None,
    )

    results = {
        "summary_metrics": metric_tracker.get_summary_metrics(),
        "epoch_metrics": metric_tracker.get_epoch_metrics(),
    }

    # Convert results to JSON serializable format
    serializable_results = convert_to_serializable(results)

    # Save results to file
    output_file = hydra_output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    duration = time.time() - start_time
    logger.info(
        f"Completed learner {learner_name} in {duration:.2f} seconds. Results saved to {output_file}"
    )


if __name__ == "__main__":
    run_learner()
