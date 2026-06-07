"""
Generate the presentation comparison plots requested for the Lunar Lander deck.

Default usage:
    .venv/bin/python plot_presentation_comparisons.py

When the original baseline run finishes:
    .venv/bin/python plot_presentation_comparisons.py \
        --baseline experiment_runs/20260607_015412_baseline_original_final_val4_s1 \
        --baseline-label "Original baseline"
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")

DEFAULT_IMPROVED = "experiment_runs/20260605_001126_final_val4_s1"
DEFAULT_BASELINE = "experiment_runs/20260605_003533_final_validation_set_size-2_s1"
DEFAULT_OUT = "comparisons/presentation_plots"

BASELINE_COLOR = "#6b7280"
IMPROVED_COLOR = "#2563eb"
BASELINE_SURVIVED = "#34d399"
BASELINE_CRASHED = "#f87171"
IMPROVED_SURVIVED = "#059669"
IMPROVED_CRASHED = "#dc2626"


@dataclass
class RunData:
    path: Path
    label: str
    history: pd.DataFrame
    metrics: dict
    config: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create selected Lunar Lander comparison plots.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--improved", default=DEFAULT_IMPROVED)
    parser.add_argument("--baseline-label", default="Placeholder baseline")
    parser.add_argument("--improved-label", default="Final model")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--same-episode-count", type=int, default=20)
    parser.add_argument("--same-episode-seed", type=int, default=20260607)
    parser.add_argument("--same-episode-duration", type=int, default=500)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear the output directory before writing plots.",
    )
    return parser.parse_args()


def resolve_run(path_text: str) -> Path:
    path = Path(path_text)
    if not path.exists():
        candidate = Path("experiment_runs") / path_text
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise SystemExit(f"Run not found: {path_text}")
    return path


def load_run(path_text: str, label: str) -> RunData:
    path = resolve_run(path_text)
    history_path = path / "generation_history.csv"
    metrics_path = path / "metrics.json"
    config_path = path / "run_config.json"
    if not history_path.exists():
        raise SystemExit(f"Missing generation_history.csv in {path}")
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics.json in {path}")

    return RunData(
        path=path,
        label=label,
        history=pd.read_csv(history_path),
        metrics=json.loads(metrics_path.read_text()),
        config=json.loads(config_path.read_text()) if config_path.exists() else {},
    )


def prepare_out(out_dir: str, keep_existing: bool) -> Path:
    out = Path(out_dir)
    if out.exists() and not keep_existing:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save(fig: plt.Figure, out: Path, name: str, dpi: int) -> None:
    path = out / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


def style_axes(ax, title: str, xlabel: str = "Generation", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_two_lines(
    baseline: RunData,
    improved: RunData,
    out: Path,
    dpi: int,
    column: str,
    title: str,
    ylabel: str,
    filename: str,
    scale: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for run, color in [(baseline, BASELINE_COLOR), (improved, IMPROVED_COLOR)]:
        if column not in run.history.columns:
            continue
        ax.plot(
            run.history["generation"],
            pd.to_numeric(run.history[column], errors="coerce") * scale,
            label=run.label,
            color=color,
            linewidth=2.3,
        )
    if scale == 100:
        ax.set_ylim(-3, 103)
    style_axes(ax, title, ylabel=ylabel)
    ax.legend(frameon=False)
    save(fig, out, filename, dpi)


def plot_validation_selection(improved: RunData, out: Path, dpi: int) -> None:
    df = improved.history.copy()
    if "validation_best_mean" not in df.columns or df["validation_best_mean"].dropna().empty:
        return

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.plot(
        df["generation"],
        df["best_lander_fitness"],
        color=IMPROVED_COLOR,
        linewidth=1.8,
        label="best training candidate mean score",
    )
    ax.plot(
        df["generation"],
        df["validation_best_mean"],
        color=IMPROVED_SURVIVED,
        linewidth=2.1,
        label="best validation candidate mean score",
    )
    ax.plot(
        df["generation"],
        df["validation_best_mean"].cummax(),
        color="#111827",
        linewidth=2.4,
        linestyle="--",
        label="best validation so far",
    )

    selected_gen = improved.metrics.get("best_model_generation")
    if selected_gen is not None:
        ax.axvline(selected_gen, color="#dc2626", linestyle=":", linewidth=2)
        ax.annotate(
            f"selected gen {selected_gen}",
            xy=(selected_gen, df["validation_best_mean"].max()),
            xytext=(8, -18),
            textcoords="offset points",
            color="#dc2626",
            fontsize=10,
        )

    style_axes(
        ax,
        f"Validation-Based Model Selection ({improved.label})",
        ylabel="Mean score / episode",
    )
    ax.legend(frameon=False)
    save(fig, out, "04_validation_selection_curve.png", dpi)


def load_tree(run: RunData):
    model_path = run.path / "best_tree.pkl"
    if not model_path.exists():
        raise SystemExit(f"Missing best_tree.pkl in {run.path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def select_action(tree, observation: np.ndarray) -> int:
    with torch.no_grad():
        sample = torch.from_numpy(observation.reshape((1, -1))).float()
        return int(torch.argmax(tree.get_output_pt(sample)).item())


def evaluate_episode(tree, env, seed: int, duration: int) -> dict:
    observation = env.reset(seed=int(seed))[0]
    total_reward = 0.0
    final_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    for _ in range(duration):
        action = select_action(tree, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        final_reward = float(reward)
        total_reward += final_reward
        steps += 1
        if terminated or truncated:
            break

    crashed = bool(terminated and final_reward <= -100.0)
    return {
        "score": float(total_reward),
        "steps": int(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "crashed": crashed,
        "survived": not crashed,
    }


def evaluate_same_episodes(
    baseline: RunData,
    improved: RunData,
    episode_count: int,
    seed: int,
    duration: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    episode_seeds = rng.integers(0, np.iinfo(np.int32).max, size=episode_count)
    trees = [(baseline, load_tree(baseline)), (improved, load_tree(improved))]
    rows = []
    env = gym.make("LunarLander-v2")

    try:
        for episode_idx, episode_seed in enumerate(episode_seeds, start=1):
            for run, tree in trees:
                result = evaluate_episode(tree, env, int(episode_seed), duration)
                rows.append(
                    {
                        "episode": episode_idx,
                        "seed": int(episode_seed),
                        "run": run.label,
                        **result,
                    }
                )
    finally:
        env.close()

    return pd.DataFrame(rows)


def plot_same_episodes(
    baseline: RunData,
    improved: RunData,
    out: Path,
    dpi: int,
    episode_count: int,
    seed: int,
    duration: int,
) -> None:
    df = evaluate_same_episodes(baseline, improved, episode_count, seed, duration)
    df.to_csv(out / "same_20_random_episode_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    x = np.arange(1, episode_count + 1)
    width = 0.38
    offsets = {baseline.label: -width / 2, improved.label: width / 2}
    color_map = {
        (baseline.label, True): BASELINE_SURVIVED,
        (baseline.label, False): BASELINE_CRASHED,
        (improved.label, True): IMPROVED_SURVIVED,
        (improved.label, False): IMPROVED_CRASHED,
    }

    for run in [baseline, improved]:
        run_df = df[df["run"] == run.label].sort_values("episode")
        colors = [color_map[(run.label, bool(row.survived))] for row in run_df.itertuples()]
        ax.bar(
            x + offsets[run.label],
            run_df["score"],
            width,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            label=run.label,
        )

    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    style_axes(
        ax,
        "Same 20 Random Episodes: Baseline vs Improved",
        xlabel="Episode",
        ylabel="Episode score",
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASELINE_SURVIVED, label=f"{baseline.label} survived"),
        plt.Rectangle((0, 0), 1, 1, color=BASELINE_CRASHED, label=f"{baseline.label} crashed"),
        plt.Rectangle((0, 0), 1, 1, color=IMPROVED_SURVIVED, label=f"{improved.label} survived"),
        plt.Rectangle((0, 0), 1, 1, color=IMPROVED_CRASHED, label=f"{improved.label} crashed"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper left")
    save(fig, out, "05_same_20_random_episode_scores.png", dpi)


def write_summary(baseline: RunData, improved: RunData, out: Path) -> None:
    rows = []
    for run in [baseline, improved]:
        test = run.metrics.get("test", {})
        rows.append(
            {
                "label": run.label,
                "run_path": str(run.path),
                "best_model_generation": run.metrics.get("best_model_generation"),
                "best_validation_score": run.metrics.get("best_validation_score"),
                "test_mean_score": test.get("mean_score"),
                "test_std_score": test.get("std_score"),
                "test_survival_rate": test.get("survival_rate"),
                "test_crash_rate": test.get("crash_rate"),
            }
        )
    pd.DataFrame(rows).to_csv(out / "comparison_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    out = prepare_out(args.out_dir, keep_existing=args.keep_existing)
    baseline = load_run(args.baseline, args.baseline_label)
    improved = load_run(args.improved, args.improved_label)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )

    write_summary(baseline, improved, out)
    plot_two_lines(
        baseline,
        improved,
        out,
        args.dpi,
        "best_fitness_so_far",
        "Best Fitness So Far",
        "Fitness",
        "01_best_fitness_so_far.png",
    )
    plot_two_lines(
        baseline,
        improved,
        out,
        args.dpi,
        "best_fitness",
        "Best Fitness in Each Generation",
        "Fitness",
        "02_generation_best_fitness.png",
    )
    plot_two_lines(
        baseline,
        improved,
        out,
        args.dpi,
        "episode_survival_rate",
        "Episode Survival Rate",
        "Survived episodes (%)",
        "03_episode_survival_rate.png",
        scale=100,
    )
    plot_validation_selection(improved, out, args.dpi)
    plot_same_episodes(
        baseline,
        improved,
        out,
        args.dpi,
        args.same_episode_count,
        args.same_episode_seed,
        args.same_episode_duration,
    )

    print(f"\nComparison plots written to {out}")
    print("Coefficient gating: remind me in the next question and I will explain it cleanly.")


if __name__ == "__main__":
    main()
