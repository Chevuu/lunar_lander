"""
Usage: python plot_run.py <run_name>

Generates a plots/ subdirectory inside the given experiment run with 7 plots.
"""

import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_run(run_name: str) -> tuple[Path, pd.DataFrame, dict]:
    base = Path(__file__).parent / "experiment_runs" / run_name
    if not base.exists():
        sys.exit(f"Run not found: {base}")

    df = pd.read_csv(base / "generation_history.csv")
    with open(base / "metrics.json") as f:
        metrics = json.load(f)

    return base, df, metrics


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ── Plot 1: Fitness per generation ────────────────────────────────────────────
def plot_fitness(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    gens = df["generation"]
    best_so_far_col = "best_fitness_so_far" if "best_fitness_so_far" in df.columns else "best_fitness"

    ax.fill_between(
        gens,
        df["mean_fitness"] - df["std_fitness"],
        df["mean_fitness"] + df["std_fitness"],
        alpha=0.2,
        label="mean ± 1 std",
    )
    ax.plot(gens, df["mean_fitness"], label="mean fitness")
    ax.plot(gens, df["best_fitness"], label="best of generation", linewidth=1.5)
    if best_so_far_col != "best_fitness":
        ax.plot(gens, df[best_so_far_col], label="best so far", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness per Generation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out / "01_fitness.png")


# ── Plot 2: Landing / survival rate per generation ────────────────────────────
def plot_survival(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    gens = df["generation"]

    ax.plot(gens, df["episode_survival_rate"] * 100, label="episode survival rate")
    ax.plot(gens, df["agent_survival_rate"] * 100, label="agent survival rate", linestyle="--")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Survival rate (%)")
    ax.set_title("Landing / Survival Rate per Generation")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out / "02_survival_rate.png")


# ── Plot 3: Std dev of agent performance per generation ───────────────────────
def plot_std(df: pd.DataFrame, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    gens = df["generation"]

    color1, color2 = "tab:blue", "tab:orange"
    ax1.plot(gens, df["std_fitness"], color=color1, label="std fitness (population)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Std fitness", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(gens, df["std_episode_score"], color=color2, linestyle="--", label="std episode score")
    ax2.set_ylabel("Std episode score", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Agent Performance Spread per Generation")
    ax1.grid(True, alpha=0.3)
    save(fig, out / "03_std_performance.png")


# ── Plot 4: Tree size evolution ───────────────────────────────────────────────
def plot_tree_size(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    gens = df["generation"]

    ax.plot(gens, df["best_tree_size"], label="best tree size")
    ax.plot(gens, df["mean_tree_size"], label="mean tree size", linestyle="--")

    if "max_tree_size" in df.columns:
        cap = df["max_tree_size"].iloc[0]
        ax.axhline(cap, color="red", linestyle=":", alpha=0.7, label=f"size cap ({cap})")
    else:
        cap = int(df["best_tree_size"].max())
        if df["best_tree_size"].iloc[-5:].std() < 1:
            ax.axhline(cap, color="red", linestyle=":", alpha=0.7, label=f"apparent cap ({cap})")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Tree size (nodes)")
    ax.set_title("Expression Tree Size Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out / "04_tree_size.png")


# ── Plot 5: Episode score distribution at artifact checkpoints ────────────────
def plot_artifact_scores(metrics: dict, out: Path) -> None:
    artifacts = metrics.get("generation_artifacts", [])
    if not artifacts:
        print("  skipping plot 5 — no generation artifacts in this run")
        return

    # Each artifact has a single test reward from the GIF run (not full episodes).
    # We show training_fitness at each checkpoint instead, which is the fitness
    # evaluated on num_episodes — more meaningful than the single GIF reward.
    gens = [a["generation"] for a in artifacts]
    train_fitness = [a["training_fitness"] for a in artifacts]

    raw_rewards = []
    opt_rewards = []
    for a in artifacts:
        raw_rewards.append(a.get("raw_gif", {}).get("reward", None))
        opt_rewards.append(a.get("optimized_gif", {}).get("reward", None))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: training fitness at each checkpoint
    axes[0].bar(gens, train_fitness, width=max(1, gens[1] - gens[0]) * 0.6, color="steelblue")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Training fitness")
    axes[0].set_title("Best Generation Fitness at Checkpoints")
    axes[0].set_xticks(gens)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: raw vs optimised GIF reward at each checkpoint
    if any(r is not None for r in raw_rewards):
        x = np.arange(len(gens))
        w = 0.35
        axes[1].bar(x - w / 2, raw_rewards, w, label="raw (before coeff opt)")
        axes[1].bar(x + w / 2, opt_rewards, w, label="optimized")
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Episode reward")
        axes[1].set_title("GIF Reward: Raw vs Optimized at Checkpoints")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(gens)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    save(fig, out / "05_checkpoint_scores.png")


# ── Plot 6: Final test episode scores ─────────────────────────────────────────
def plot_test_episodes(metrics: dict, out: Path) -> None:
    test = metrics.get("test", {})
    scores = test.get("episode_scores")
    if not scores:
        print("  skipping plot 8 — no test episode scores")
        return

    mean = test["mean_score"]
    std = test["std_score"]

    crashed = test.get("episode_crashed", [False] * len(scores))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scores))
    colors = ["tomato" if c else "steelblue" for c in crashed]
    ax.bar(x, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(mean, color="black", linestyle="--", linewidth=1.5, label=f"mean = {mean:.1f}")
    ax.axhspan(mean - std, mean + std, alpha=0.12, color="black", label=f"±1 std ({std:.1f})")
    ax.axhline(0, color="gray", linewidth=0.8)

    from matplotlib.patches import Patch
    stat_handles, _ = ax.get_legend_handles_labels()
    legend_handles = stat_handles + [Patch(color="steelblue", label="survived"), Patch(color="tomato", label="crashed")]

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Final Test Episode Scores")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ep {i+1}" for i in x], rotation=45)
    ax.legend(handles=legend_handles)
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, out / "06_test_episode_scores.png")


# ── Entry point ───────────────────────────────────────────────────────────────
def generate_plots(run_name: str) -> None:
    base, df, metrics = load_run(run_name)

    plots_dir = base / "plots"
    plots_dir.mkdir(exist_ok=True)
    print(f"Writing plots to {plots_dir}")

    plot_fitness(df, plots_dir)
    plot_survival(df, plots_dir)
    plot_std(df, plots_dir)
    plot_tree_size(df, plots_dir)
    plot_artifact_scores(metrics, plots_dir)
    plot_test_episodes(metrics, plots_dir)

    print("Done.")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python plot_run.py <run_name>")

    generate_plots(sys.argv[1])


if __name__ == "__main__":
    main()
