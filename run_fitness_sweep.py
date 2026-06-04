"""
Sweeps shaped fitness weights using Optuna TPE.

GP hyperparameters are fixed at the best-known values (from config.py /
experiment_configs/shaped_fitness_sweep.json). Only the four fitness shaping
weights are searched:
  crash_penalty_weight, landing_bonus_weight, std_penalty_weight, time_penalty_weight

RANDOM_SEEDS=True and fitness_mode="shaped" are forced for every trial.

Phase 1 – sweep:   40 Optuna trials, each running 25 gens / 12 episodes.
Phase 2 – train:   full run at 75 gens / 20 episodes with the best weights.

Usage:
    python run_fitness_sweep.py
    python run_fitness_sweep.py --config-json experiment_configs/shaped_fitness_sweep.json
    python run_fitness_sweep.py --n-trials 5 --sweep-gens 10   # quick test
"""
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

import config
from run_experiment import (
    apply_sweep_params,
    args_to_dict,
    run_training,
    save_training_outputs,
    write_json,
)
from train import env


# ---------------------------------------------------------------------------
# Search space bounds
# ---------------------------------------------------------------------------
CRASH_PENALTY_WEIGHT_MIN  = 0.0
CRASH_PENALTY_WEIGHT_MAX  = 100.0
LANDING_BONUS_WEIGHT_MIN  = 0.0
LANDING_BONUS_WEIGHT_MAX  = 200.0
STD_PENALTY_WEIGHT_MIN    = 0.0
STD_PENALTY_WEIGHT_MAX    = 3.0
TIME_PENALTY_WEIGHT_MIN   = 0.0
TIME_PENALTY_WEIGHT_MAX   = 0.3


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
_BAR_WIDTH = 28

def _render_bar(label, n_done, n_total, start_time, extra=""):
    fraction = min(1.0, n_done / max(n_total, 1))
    filled   = int(round(_BAR_WIDTH * fraction))
    bar      = "#" * filled + "-" * (_BAR_WIDTH - filled)
    elapsed  = time.time() - start_time
    line = (
        f"{label} [{bar}] {n_done:>3}/{n_total}"
        f"  {fraction * 100:>5.1f}%"
        f"  elapsed {elapsed:>6.1f}s"
    )
    if extra:
        line += "  " + extra
    sys.stdout.write("\r" + line)
    sys.stdout.flush()


def _clear_bar():
    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_json_defaults(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config-json", default=None)
    pre_args, _ = pre.parse_known_args()
    d = load_json_defaults(pre_args.config_json)

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config-json", default=pre_args.config_json)

    # Experiment identity
    p.add_argument("--experiment-name", default=d.get("experiment_name", "shaped_fitness_weight_sweep"))
    p.add_argument("--results-dir",     default=d.get("results_dir", "experiment_runs"))
    p.add_argument("--seed",            type=int,   default=d.get("seed", config.SEED))
    p.add_argument("--n-jobs",          type=int,   default=d.get("n_jobs", config.N_JOBS))
    p.add_argument("--verbose",         default=d.get("verbose", config.VERBOSE), action=argparse.BooleanOptionalAction)

    # GP hyperparameters — fixed; not swept
    p.add_argument("--pop-size",         type=int,   default=d.get("pop_size", config.POP_SIZE))
    p.add_argument("--max-gens",         type=int,   default=d.get("max_gens", 75))
    p.add_argument("--max-tree-size",    type=int,   default=d.get("max_tree_size", config.MAX_TREE_SIZE))
    p.add_argument("--num-constants",    type=int,   default=d.get("num_constants", config.NUM_CONSTANTS))
    p.add_argument("--num-episodes",     type=int,   default=d.get("num_episodes", 20))
    p.add_argument("--episode-duration", type=int,   default=d.get("episode_duration", config.EPISODE_DURATION))
    p.add_argument("--coeff-lr",         type=float, default=d.get("coeff_lr", config.COEFF_LR))
    p.add_argument("--coeff-opt-steps",  type=int,   default=d.get("coeff_opt_steps", config.COEFF_OPT_STEPS))
    p.add_argument("--batch-size",       type=int,   default=d.get("batch_size", config.BATCH_SIZE))
    p.add_argument("--gamma",            type=float, default=d.get("gamma", config.GAMMA))
    p.add_argument("--grad-clip",        type=float, default=d.get("grad_clip", config.GRAD_CLIP))
    p.add_argument("--elite-count",      type=int,   default=d.get("elite_count", config.ELITE_COUNT))

    # Evaluation
    p.add_argument("--test-episodes",          type=int, default=d.get("test_episodes", config.TEST_EPISODES))
    p.add_argument("--test-duration",          type=int, default=d.get("test_duration", config.TEST_EPISODE_DURATION))
    p.add_argument("--validation-episodes",    type=int, default=d.get("validation_episodes", config.TEST_EPISODES))
    p.add_argument("--validation-seed-offset", type=int, default=d.get("validation_seed_offset", 10_000))
    p.add_argument("--test-seed-offset",       type=int, default=d.get("test_seed_offset", 20_000))
    p.add_argument("--video",                  default=d.get("video", True), action=argparse.BooleanOptionalAction)
    p.add_argument("--video-duration",         type=int, default=d.get("video_duration", config.TEST_EPISODE_DURATION))
    p.add_argument("--video-seed-offset",      type=int, default=d.get("video_seed_offset", 30_000))
    p.add_argument("--artifact-interval",      type=int, default=d.get("artifact_interval", 10))

    # Sweep control
    p.add_argument("--n-trials",       type=int,  default=d.get("n_trials", 40))
    p.add_argument("--sweep-gens",     type=int,  default=d.get("sweep_gens", 25))
    p.add_argument("--sweep-episodes", type=int,  default=d.get("sweep_episodes", 12))
    p.add_argument("--resume-sweep",   default=d.get("resume_sweep", False), action=argparse.BooleanOptionalAction)

    # These are forced to fixed values; exposed so run_training / save_training_outputs see them
    p.add_argument("--fitness-mode",          default="shaped")
    p.add_argument("--random-seeds",          default=True,  action=argparse.BooleanOptionalAction)
    p.add_argument("--crash-penalty",         default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--parsimony",             default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--time-pressure",         default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--crash-penalty-weight",  type=float, default=0.0)
    p.add_argument("--landing-bonus-weight",  type=float, default=0.0)
    p.add_argument("--std-penalty-weight",    type=float, default=0.0)
    p.add_argument("--time-penalty-weight",   type=float, default=0.0)

    # run_training / save_training_outputs expect a mode attribute
    p.add_argument("--mode", default="sweep-train")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_fitness_weight_sweep(args, run_dir):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise SystemExit("optuna is required. Install with: pip install optuna") from exc

    sweep_csv  = run_dir / "sweep_trials.csv"
    fieldnames = [
        "trial", "score",
        "survival_rate", "crash_rate", "landing_rate",
        "crash_penalty_weight", "landing_bonus_weight",
        "std_penalty_weight", "time_penalty_weight",
        "seed",
    ]

    sweep_start = time.time()
    best_score  = float("-inf")
    n_completed = 0

    # Draw initial empty bar before first trial starts
    _render_bar("Sweep", 0, args.n_trials, sweep_start)

    def objective(trial):
        nonlocal best_score, n_completed

        trial_args = copy.copy(args)
        trial_args.seed            = args.seed + trial.number
        trial_args.verbose         = False
        trial_args.fitness_mode    = "shaped"
        trial_args.random_seeds    = True
        trial_args.max_gens        = args.sweep_gens
        trial_args.num_episodes    = args.sweep_episodes

        trial_args.crash_penalty_weight = trial.suggest_float(
            "crash_penalty_weight", CRASH_PENALTY_WEIGHT_MIN, CRASH_PENALTY_WEIGHT_MAX,
        )
        trial_args.landing_bonus_weight = trial.suggest_float(
            "landing_bonus_weight", LANDING_BONUS_WEIGHT_MIN, LANDING_BONUS_WEIGHT_MAX,
        )
        trial_args.std_penalty_weight = trial.suggest_float(
            "std_penalty_weight", STD_PENALTY_WEIGHT_MIN, STD_PENALTY_WEIGHT_MAX,
        )
        trial_args.time_penalty_weight = trial.suggest_float(
            "time_penalty_weight", TIME_PENALTY_WEIGHT_MIN, TIME_PENALTY_WEIGHT_MAX,
        )

        _, _, _, _, evaluations = run_training(trial_args)
        val   = evaluations["validation"]["optimized"]
        score = val["mean_score"]

        if score > best_score:
            best_score = score
        n_completed += 1

        row = {
            "trial":                 trial.number,
            "score":                 score,
            "survival_rate":         val["survival_rate"],
            "crash_rate":            val["crash_rate"],
            "landing_rate":          val["landing_rate"],
            "crash_penalty_weight":  trial_args.crash_penalty_weight,
            "landing_bonus_weight":  trial_args.landing_bonus_weight,
            "std_penalty_weight":    trial_args.std_penalty_weight,
            "time_penalty_weight":   trial_args.time_penalty_weight,
            "seed":                  trial_args.seed,
        }
        write_header = not sweep_csv.exists()
        with open(sweep_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # Clear bar, print completed trial as a permanent line, redraw bar
        _clear_bar()
        print(
            f"  Trial {trial.number + 1:>2}/{args.n_trials}"
            f"  score={score:>8.2f}"
            f"  surv={val['survival_rate']:.2f}"
            f"  crash={val['crash_rate']:.2f}"
            f"  land={val['landing_rate']:.2f}"
            f"  [crash_w={trial_args.crash_penalty_weight:>6.1f}"
            f"  land_w={trial_args.landing_bonus_weight:>6.1f}"
            f"  std_w={trial_args.std_penalty_weight:.2f}"
            f"  time_w={trial_args.time_penalty_weight:.3f}]"
        )
        _render_bar(
            "Sweep", n_completed, args.n_trials, sweep_start,
            extra=f"best {best_score:>8.2f}  last surv {val['survival_rate']:.2f}",
        )

        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name="fitness_weight_sweep",
        storage=f"sqlite:///{run_dir / 'sweep.db'}",
        load_if_exists=args.resume_sweep,
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.n_trials)

    # Finish bar on its own line
    sys.stdout.write("\n")
    sys.stdout.flush()

    best = {
        "value":     float(study.best_value),
        "params":    dict(study.best_params),
        "sweep_csv": str(sweep_csv),
        "sweep_db":  str(run_dir / "sweep.db"),
    }
    write_json(run_dir / "best_sweep_params.json", best)
    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    run_dir = (
        Path(args.results_dir)
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.experiment_name.replace(' ', '_')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "requested_config.json", args_to_dict(args))

    print(f"Run directory: {run_dir}")
    print()
    print("Phase 1 -- Sweeping shaped fitness weights")
    print(f"  Trials:      {args.n_trials}")
    print(f"  Gens/trial:  {args.sweep_gens}  |  Episodes/trial: {args.sweep_episodes}")
    print("  Search space:")
    print(f"    crash_penalty_weight  [{CRASH_PENALTY_WEIGHT_MIN:.0f}, {CRASH_PENALTY_WEIGHT_MAX:.0f}]")
    print(f"    landing_bonus_weight  [{LANDING_BONUS_WEIGHT_MIN:.0f}, {LANDING_BONUS_WEIGHT_MAX:.0f}]")
    print(f"    std_penalty_weight    [{STD_PENALTY_WEIGHT_MIN:.1f}, {STD_PENALTY_WEIGHT_MAX:.1f}]")
    print(f"    time_penalty_weight   [{TIME_PENALTY_WEIGHT_MIN:.1f}, {TIME_PENALTY_WEIGHT_MAX:.1f}]")
    print()

    sweep_result = run_fitness_weight_sweep(args, run_dir)

    print()
    print(f"Best fitness weights (validation mean_score = {sweep_result['value']:.2f}):")
    for k, v in sweep_result["params"].items():
        print(f"  {k}: {v:.4f}")
    print()

    print("Phase 2 -- Full training with best weights")
    print(f"  Gens: {args.max_gens}  |  Episodes: {args.num_episodes}  |  random_seeds=True")
    print()

    args = apply_sweep_params(args, sweep_result)
    args.fitness_mode  = "shaped"
    args.random_seeds  = True

    raw_best, best, evo, coeff_loss, evaluations = run_training(args, run_dir)
    save_training_outputs(args, run_dir, raw_best, best, evo, coeff_loss, evaluations, sweep_result)

    test = evaluations["test"]["optimized"]
    print()
    print(f"Experiment complete. Results in {run_dir}")
    print(f"  Test mean score:   {test['mean_score']:.2f}")
    print(f"  Test survival:     {test['survival_rate']:.0%}")
    print(f"  Test crash rate:   {test['crash_rate']:.0%}")
    print(f"  Test landing rate: {test['landing_rate']:.0%}")

    env.close()

    print()
    print("Generating plots...")
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "plot_run.py"), run_dir.name],
        check=False,
    )


if __name__ == "__main__":
    main()
