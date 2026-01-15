#!/usr/bin/env bash

# This is a quick demo to run a reduced version of the entire experimental
# pipeline.

CONFIG="num_trials=2 dataset_sizes=[0.5,1.0]"

# Main experiments
python -m responserank.synthetic.experiment_runner dataset=deterministic_all $CONFIG
python -m responserank.synthetic.experiment_runner dataset=stochastic $CONFIG
python -m responserank.synthetic.experiment_runner dataset=drift_diffusion $CONFIG

# Ablations
python -m responserank.synthetic.experiment_runner dataset=deterministic_all_no_variability $CONFIG
python -m responserank.synthetic.experiment_runner dataset=stochastic_no_variability $CONFIG
python -m responserank.synthetic.experiment_runner dataset=drift_diffusion_no_variability $CONFIG

echo ""
echo "Experiments finished."
echo "⚠️ Note: Use num_trials=2 only for testing. The paper results use num_trials=100 for statistical reliability."
echo "⚠️ Note: Used a reduced set of dataset sizes."
