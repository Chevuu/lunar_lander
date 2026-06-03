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
- `metrics.json` with train/test scores, score standard deviation, and survival/crash rates
- `generation_history.csv` with fitness, tree size, score standard deviation, and survival/crash rates per generation
- `best_tree.txt` and optimized `best_tree.pkl`
- `best_lander_before_optimization.gif` and optimized `best_lander.gif` unless `--no-video` is used
- `generation_artifacts/` with an optimized pickled model plus before/after optimization GIFs every 10 generations

Useful examples:

```bash
# Fast smoke run
.venv/bin/python run_experiment.py --mode train --pop-size 8 --max-gens 1 --n-jobs 1 --coeff-opt-steps 0

# Full configured run with a fixed seed
.venv/bin/python run_experiment.py --mode train --seed 42 --pop-size 128 --max-gens 35 --num-episodes 15

# Default 20-generation run with generation 10 and 20 artifacts
.venv/bin/python run_experiment.py --mode train --experiment-name default_20gens --max-gens 20

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

### How `run_experiment.py` works

`run_experiment.py` is the main script for repeatable runs outside the notebook.
It creates a new timestamped directory for every run, stores the requested
configuration, trains the genetic programming population, evaluates the final
best model, and writes the metrics/artifacts needed for analysis and the
presentation.

In `train` mode, the script:

1. Loads defaults from `config.py`, optionally overridden by CLI arguments or
   `--config-json`.
2. Sets the runtime seed and training hyperparameters.
3. Builds the GP leaf nodes and evolves the population for `--max-gens`.
4. Evaluates every individual over `--num-episodes` episodes per generation.
5. Records generation-level fitness, survival/crash, and variance metrics in
   `generation_history.csv`.
6. Saves optimized model checkpoints and before/after optimization GIFs every
   `--artifact-interval` generations. The default interval is `10`.
7. Selects the best evolutionary tree, optimizes its constants, evaluates it on
   the test episodes, and saves the optimized final model.

In `sweep` mode, the script uses Optuna to search hyperparameters and writes the
trial results. In `sweep-train` mode, it first runs the sweep and then trains one
final model with the best found parameters.

The final model files are:

- `best_tree.pkl`: optimized final model.
- `best_tree.txt`: readable representation of the optimized final model.
- `best_lander_before_optimization.gif`: final best tree before coefficient
  optimization.
- `best_lander.gif`: final best tree after coefficient optimization.

The periodic checkpoint files are stored in `generation_artifacts/`:

- `generation_0010_tree.pkl`: optimized model from generation 10.
- `generation_0010_before_optimization.gif`: raw evolutionary tree before
  coefficient optimization.
- `generation_0010_after_optimization.gif`: same tree after coefficient
  optimization.

### `generation_history.csv` columns

Each row summarizes the whole population at one generation. Generation `0` is
the initialized population before any evolutionary step. Later rows are after
each generation has been evaluated.

An episode is counted as crashed when the environment terminates and the final
reward is `<= -100`, which is the Lunar Lander crash penalty. Any episode that
does not meet that crash condition is counted as survived. An agent is counted
as survived only if it has zero crashed episodes during its fitness evaluation.

| Column | Meaning |
| --- | --- |
| `generation` | Generation index. `0` is the initial population. |
| `best_fitness` | Highest fitness value in the population for this generation. |
| `mean_fitness` | Average fitness across all agents in the population. |
| `std_fitness` | Standard deviation of fitness across the population. Lower values mean agents are more consistent with each other. |
| `best_tree_size` | Number of nodes in the best-fitness tree. |
| `mean_tree_size` | Average number of nodes across all trees in the population. |
| `population_size` | Number of agents/trees evaluated in the generation. |
| `survived_agents` | Number of agents with zero crashed episodes. |
| `crashed_agents` | Number of agents with at least one crashed episode. |
| `agent_survival_rate` | `survived_agents / population_size`. |
| `agent_crash_rate` | `crashed_agents / population_size`. |
| `total_episodes` | Total evaluated episodes in this generation, normally `population_size * num_episodes`. |
| `survived_episodes` | Number of episodes not classified as crashes. |
| `crashed_episodes` | Number of episodes classified as crashes. |
| `episode_survival_rate` | `survived_episodes / total_episodes`. |
| `episode_crash_rate` | `crashed_episodes / total_episodes`. |
| `mean_episode_score` | Average episode reward across every episode played by every agent in the generation. |
| `std_episode_score` | Standard deviation of those episode rewards. Lower values mean episode outcomes are more consistent. |

Fitness is the sum of rewards collected across the training episodes for one
agent. Because each agent is evaluated over multiple episodes, `mean_episode_score`
is useful for understanding typical per-episode behavior, while `best_fitness`
shows which individual the evolutionary process is selecting.
