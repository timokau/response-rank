# ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning

This repository contains the implementation of [ResponseRank](https://openreview.net/pdf?id=3JlBQRvod7), a novel approach for preference learning that leverages strength information to improve utility estimation. We compare the method to a loss based on the standard Bradley-Terry model that only considers choice outcomes.

## Installation

```bash
# After optional venv setup
pip install -r requirements-lock.txt
pip install --no-deps -e .
wandb login # LLM experiments require wandb access
# Additionally requires a working LaTeX installation to render text in figures
```

## Reproducing experiments

### Regenerate figures and tables (without re-running)

All three experiment types consist of three main steps: Running the experiments, collecting the results, and generating the artifacts (plots, tables, and statistics). This repository includes the cached results used in the paper, so you can skip the first two steps and reproduce the artifacts as follows:

```bash
# Synthetic
python -m responserank.synthetic.analysis.cli generate results/synthetic paper/figures

# LLM (MultiPref)
python -m responserank.llm.analysis.cli generate results/llm paper/figures

# RL control
python -m responserank.rl_control.analysis.cli generate results/rl_control paper/figures
```

Cached data lives in `results/synthetic/`, `results/llm/`, and `results/rl_control/`. The following sections describe how to reproduce the results from scratch.

### Synthetic experiments

No GPU required, experiments run in parallel on a single node. Results are stored locally in the ./outputs directory. See the [paper](https://openreview.net/pdf?id=3JlBQRvod7) for descriptions of the datasets and learners.

```bash
# Main experiments
python -m responserank.synthetic.experiment_runner dataset=deterministic_all
python -m responserank.synthetic.experiment_runner dataset=stochastic
python -m responserank.synthetic.experiment_runner dataset=drift_diffusion

# No-variability ablations
python -m responserank.synthetic.experiment_runner dataset=deterministic_all_no_variability
python -m responserank.synthetic.experiment_runner dataset=stochastic_no_variability
python -m responserank.synthetic.experiment_runner dataset=drift_diffusion_no_variability

# Reduced version (2 trials, fewer dataset sizes) of the entire pipeline above. Use for testing purposes.
# Equivalently apply `num_trials=2 dataset_sizes=[0.5,1.0]` to all experiment_runner calls above.
bash run_full_mini.sh
```

#### Collect results and generate figures

```bash
# Find latest experiment directories and collect results (optional, pre-collected results are in ./results)
python -m responserank.synthetic.analysis.cli find outputs/experiment_runner results/synthetic
python -m responserank.synthetic.analysis.cli collect results/synthetic \
    --paths-file results/synthetic/experiment_paths.json

# Generate figures
python -m responserank.synthetic.analysis.cli generate results/synthetic paper/figures
```

### LLM experiments

Train LLM-based reward models on real human preference data. Requires GPU (tested on A100-80G/H100). ~1.5h on H100, ~3.25h on A100 at full dataset (fraction 1.0). Each run is executed independently and saves results to wandb.

```bash
huggingface-cli login # Or set HF_TOKEN. Llama requires accepting Meta's license on HuggingFace.
python -m responserank.llm.train experiment=bt  # baseline
python -m responserank.llm.train experiment=rr_agree  # ResponseRank with agreement
# Others: ls conf/llm/experiment/*.yaml

# Override parameters
python -m responserank.llm.train experiment=rr_agree seed=43 experiment.fraction=0.1 experiment.train_partitioner.target_size=4

# Dry run (verify config without training)
python -m responserank.llm.train experiment=rr_agree dry_run=true
```

Results go to wandb; MultiPref downloads automatically.

#### Cluster execution

For full reproduction, you'll likely want to run experiments on a GPU cluster.
The training command can be wrapped in your cluster's job submission system.

Example SLURM job array script (`run_experiment.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=responserank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

cd /path/to/code
source venv/bin/activate  # Make sure the venv is set up
python -m responserank.llm.train experiment=$1 experiment.fraction=$2 seed=$SLURM_ARRAY_TASK_ID
```

Submit two experiment configurations, each over two dataset sizes and with 2 seeds (42, 43):

```bash
for exp in bt rr_agree; do
  for frac in 0.1 1.0; do
    sbatch --array=42-43 run_experiment.sh $exp $frac
  done
done
```

#### Collect results and generate figures

```bash
# If you ran new experiments, manually update results/llm/registry.yaml first.
# Fetch data from wandb (optional, pre-fetched results are in ./results)
python -m responserank.llm.analysis.cli collect results/llm \
    --project-name rr-llm \
    --registry results/llm/registry.yaml

# Generate figures and tables from cached data
python -m responserank.llm.analysis.cli generate results/llm paper/figures
```

### RL control experiments

The RL control experiments (MuJoCo and Highway environments) can be executed using the [responserank_extension](https://github.com/ymetz/multi-type-feedback/tree/responserank_extension) branch of ymetz/multi-type-feedback. See that repository's README for installation and replication instructions. This repository only contains the analysis code.

#### Collect results and generate figures

```bash
wandb login # Needed to fetch experiment results
# Fetch data from wandb (optional, pre-fetched results are in ./results)
python -m responserank.rl_control.analysis.cli collect results/rl_control \
    --mujoco-project rr-rl-control-mujoco \
    --highway-project rr-rl-control-highway

# Generate figures and tables from cached data
python -m responserank.rl_control.analysis.cli generate results/rl_control paper/figures
```

## Project structure

The main package is `src/responserank/` with three submodules: `synthetic/` (synthetic experiments), `llm/` (LLM reward model training on MultiPref), and `rl_control/` (analysis scripts; experiments run via external repo). Hydra configs live in `conf/` with `synthetic/` (dataset and learner configs) and `llm/` (experiment presets like bt.yaml, rr_agree.yaml). Tests are in `tests/`.

## Development

```bash
bash ./run_tests          # run tests
bash ./run_linters --fix  # lint and fix
```

## Citation

```bibtex
@inproceedings{kaufmann2025responserank,
  title={ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning},
  author={Kaufmann, Timo and Metz, Yannick and Keim, Daniel and H{\"u}llermeier, Eyke},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
