"""
Usage: venv/bin/python finalize_run.py <run_name>

Run this after killing a training run early. Loads the best checkpoint,
runs test evaluation, writes metrics.json / summary / GIFs, and generates plots.
"""

import csv
import json
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import config
from plot_run import generate_plots
from run_experiment import evaluate_tree, render_gif, update_runtime_config, write_json
from train import env


def load_args(run_dir):
    with open(run_dir / "requested_config.json") as f:
        return Namespace(**json.load(f))


def load_csv_rows(run_dir):
    csv_rows = []
    csv_path = run_dir / "generation_history.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            csv_rows = [{"generation": int(r["generation"]), **r} for r in csv.DictReader(f)]
    return csv_rows


def row_fitness(row):
    if "best_fitness" in row:
        return float(row["best_fitness"])
    return float(row.get("fitness", 0.0))


def find_best_checkpoint(run_dir, csv_rows):
    best_validation = run_dir / "checkpoints" / "best_validation_tree.pkl"
    if best_validation.exists():
        return best_validation

    best_training = run_dir / "checkpoints" / "best_training_tree.pkl"
    if best_training.exists():
        return best_training

    pkls = sorted((run_dir / "generation_artifacts").glob("generation_*_tree.pkl"))
    if not pkls:
        sys.exit("No checkpoint pkl files found in generation_artifacts/")

    if not csv_rows:
        return pkls[-1]

    fitness_by_gen = {r["generation"]: row_fitness(r) for r in csv_rows}

    def checkpoint_fitness(path):
        gen = int(path.stem.split("_")[1])
        return fitness_by_gen.get(gen, float("-inf"))

    return max(pkls, key=checkpoint_fitness)


def reconstruct_artifacts(run_dir, csv_rows):
    artifact_dir = run_dir / "generation_artifacts"
    fitness_by_gen = {r["generation"]: row_fitness(r) for r in csv_rows}
    artifacts = []
    for pkl_path in sorted(artifact_dir.glob("generation_*_tree.pkl")):
        gen = int(pkl_path.stem.split("_")[1])
        with open(pkl_path, "rb") as f:
            tree = pickle.load(f)
        artifact = {
            "generation": gen,
            "training_fitness": fitness_by_gen.get(gen, 0.0),
            "tree_size": len(tree),
            "model_is_optimized": True,
            "coefficient_optimization_last_loss": None,
            "model_path": str(pkl_path),
        }
        raw_gif = artifact_dir / f"generation_{gen:04d}_before_optimization.gif"
        opt_gif = artifact_dir / f"generation_{gen:04d}_after_optimization.gif"
        if raw_gif.exists():
            artifact["raw_gif"] = {"path": str(raw_gif), "frames": None, "reward": None}
        if opt_gif.exists():
            artifact["optimized_gif"] = {"path": str(opt_gif), "frames": None, "reward": None}
            artifact["gif"] = artifact["optimized_gif"]
        artifacts.append(artifact)
    return artifacts


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python finalize_run.py <run_name>")

    run_name = sys.argv[1]
    run_dir = Path("experiment_runs") / run_name
    if not run_dir.exists():
        sys.exit(f"Run not found: {run_dir}")

    args = load_args(run_dir)
    update_runtime_config(args)

    csv_rows = load_csv_rows(run_dir)
    best_pkl = find_best_checkpoint(run_dir, csv_rows)
    print(f"Loading best tree from {best_pkl.name}...")
    with open(best_pkl, "rb") as f:
        best = pickle.load(f)
    print(f"  fitness: {best.fitness:.3f}, size: {len(best)}")

    print("Running test evaluation...")
    evaluation = evaluate_tree(best, args.test_episodes, args.test_duration, args.seed + 10_000)
    print(f"  mean score: {evaluation['mean_score']:.2f}, survival rate: {evaluation['survival_rate']:.1%}")

    with open(run_dir / "best_tree.txt", "w") as f:
        f.write(json.dumps(best.get_readable_repr(), indent=2) + "\n")
    with open(run_dir / "best_tree.pkl", "wb") as f:
        pickle.dump(best, f)

    video_info = None
    if args.video:
        print("Rendering GIFs...")
        opt_gif_path = run_dir / "best_lander.gif"
        opt_info = render_gif(best, opt_gif_path, seed=args.seed + args.video_seed_offset, duration=args.video_duration)
        opt_info["path"] = str(opt_gif_path)

        video_info = {"path": str(opt_gif_path), "optimized_gif": opt_info}
        if getattr(args, "save_raw_gifs", False):
            raw_gif_path = run_dir / "best_lander_before_optimization.gif"
            raw_info = render_gif(best, raw_gif_path, seed=args.seed + args.video_seed_offset, duration=args.video_duration)
            raw_info["path"] = str(raw_gif_path)
            video_info["raw_gif"] = raw_info

    generation_artifacts = reconstruct_artifacts(run_dir, csv_rows)

    write_json(run_dir / "metrics.json", {
        "best_training_fitness": float(best.fitness),
        "best_training_fitness_before_optimization": float(best.fitness),
        "best_model_is_optimized": True,
        "coefficient_optimization_last_loss": None,
        "test": evaluation,
        "video": video_info,
        "generation_artifacts": generation_artifacts,
        "sweep": None,
    })

    with open(run_dir / "summary.txt", "w") as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Best checkpoint: {best_pkl.name}\n")
        f.write(f"Best training fitness: {best.fitness:.3f}\n")
        f.write(f"Test mean score: {evaluation['mean_score']:.3f}\n")
        f.write(f"Test survival rate: {evaluation['survival_rate']:.3f}\n")
        f.write(f"Checkpoints: {len(generation_artifacts)}\n")

    if not (run_dir / "run_config.json").exists():
        write_json(run_dir / "run_config.json", vars(args))

    print("Generating plots...")
    generate_plots(run_name)

    actual_gens = csv_rows[-1]["generation"] if csv_rows else 0
    import re
    new_name = re.sub(r"\d+gens", f"{actual_gens}gens", run_name)
    if new_name != run_name:
        new_run_dir = run_dir.parent / new_name
        run_dir.rename(new_run_dir)
        print(f"Renamed to {new_name}")
        run_dir = new_run_dir

    print(f"Done. Results written to {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
