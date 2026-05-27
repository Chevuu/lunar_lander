# Lunar Lander

This project contains a notebook-based exercise for evolving a Lunar Lander controller with genetic programming. 

## Requirements

- Python lower than 3.10

## Install

Create and activate a virtual environment with a Python version lower than 3.10, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

Open `solution.ipynb` and run the notebook cells in order.

## Automated experiments

Most experiments do not need to run the whole notebook. Use `run_experiment.py`
to run the required training/evaluation steps and log the output:

```bash
.venv/bin/python run_experiment.py --mode train --experiment-name baseline
```

Each run creates a timestamped folder in `experiment_runs/` containing:

- `run_config.json` and `requested_config.json` with the hyperparameters
- `metrics.json` with train/test scores
- `generation_history.csv` with best fitness per generation
- `best_tree.txt` and `best_tree.pkl`
- `best_lander.gif` unless `--no-video` is used

Useful examples:

```bash
# Fast smoke run
.venv/bin/python run_experiment.py --mode train --pop-size 8 --max-gens 1 --n-jobs 1 --coeff-opt-steps 0

# Full configured run with a fixed seed
.venv/bin/python run_experiment.py --mode train --seed 42 --pop-size 128 --max-gens 35 --num-episodes 15

# Hyperparameter sweep only
.venv/bin/python run_experiment.py --mode sweep --n-trials 20 --sweep-gens 10 --sweep-episodes 5

# Sweep first, then train once with the best sweep parameters
.venv/bin/python run_experiment.py --mode sweep-train --n-trials 20 --sweep-gens 10 --sweep-episodes 5
```

You can also keep settings in a JSON file and run that from the terminal or a
notebook cell:

```json
{
  "mode": "train",
  "experiment_name": "baseline_seed_42",
  "seed": 42,
  "pop_size": 64,
  "max_gens": 20,
  "num_episodes": 10,
  "n_jobs": 1,
  "video": true
}
```

```bash
.venv/bin/python run_experiment.py --config-json experiment_config.json
```

For the most reproducible runs, use `--n-jobs 1`. Parallel fitness evaluation is
faster, but random tree initialization can be less deterministic across worker
processes.
