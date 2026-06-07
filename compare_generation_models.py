import argparse
import csv
import json
import pickle
import re
import secrets
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all generation checkpoint models on the same LunarLander episode."
    )
    parser.add_argument(
        "artifacts_dir",
        type=Path,
        help="Directory containing generation_XXXX_tree.pkl files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Episode seed. If omitted, one random seed is sampled and written to metadata.json.",
    )
    parser.add_argument("--duration", type=int, default=500, help="Maximum steps per model.")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="Save every n-th rendered frame to each GIF.",
    )
    parser.add_argument(
        "--resize",
        type=float,
        default=0.75,
        help="Scale GIF frames, e.g. 0.75 or 0.5 for faster/smaller GIFs.",
    )
    parser.add_argument(
        "--no-gifs",
        action="store_true",
        help="Only evaluate models and write CSV/metadata; do not render GIFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <artifacts_dir>/same_episode_comparison_seed_<seed>.",
    )
    return parser.parse_args()


def generation_number(path):
    match = re.search(r"generation_(\d+)_tree\.pkl$", path.name)
    return int(match.group(1)) if match else -1


def find_models(artifacts_dir):
    generation_models = sorted(artifacts_dir.glob("generation_*_tree.pkl"), key=generation_number)
    models = [
        {
            "label": f"generation_{generation_number(path):04d}",
            "generation": generation_number(path),
            "path": path,
        }
        for path in generation_models
    ]

    checkpoints_dir = artifacts_dir.parent / "checkpoints"
    for label, filename in [
        ("best_training", "best_training_tree.pkl"),
        ("best_validation", "best_validation_tree.pkl"),
    ]:
        checkpoint_path = checkpoints_dir / filename
        if checkpoint_path.exists():
            models.append(
                {
                    "label": label,
                    "generation": "",
                    "path": checkpoint_path,
                }
            )

    if not models:
        raise SystemExit(
            f"No generation_*_tree.pkl files found in {artifacts_dir} "
            "and no compatible ../checkpoints/*.pkl files found."
        )
    return models


def frame_to_image(frame, resize):
    image = Image.fromarray(frame)
    if resize != 1.0:
        width = max(1, int(image.width * resize))
        height = max(1, int(image.height * resize))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return image


def select_action(tree, observation):
    with torch.no_grad():
        input_sample = torch.from_numpy(observation.reshape((1, -1))).float()
        return int(torch.argmax(tree.get_output_pt(input_sample)).item())


def run_model(tree, env, episode_seed, duration, gif_path, frame_duration_ms, frame_stride, resize):
    observation = env.reset(seed=int(episode_seed))[0]
    frames = []
    if gif_path is not None:
        frames.append(frame_to_image(env.render(), resize))

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

        if gif_path is not None and (steps % frame_stride == 0 or terminated or truncated):
            frames.append(frame_to_image(env.render(), resize))

        if terminated or truncated:
            break

    if gif_path is not None:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )

    return {
        "reward": float(total_reward),
        "steps": int(steps),
        "gif_frames": int(len(frames)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "crashed": bool(terminated and final_reward <= -100.0),
    }


def main():
    args = parse_args()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts directory not found: {artifacts_dir}")
    if args.duration < 1:
        raise SystemExit("--duration must be at least 1")
    if args.fps < 1:
        raise SystemExit("--fps must be at least 1")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.resize <= 0:
        raise SystemExit("--resize must be positive")

    models = find_models(artifacts_dir)
    episode_seed = args.seed if args.seed is not None else secrets.randbelow(np.iinfo(np.int32).max)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = artifacts_dir / f"same_episode_comparison_seed_{episode_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_duration_ms = max(1, int(1000 / args.fps))
    rows = []
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    try:
        for idx, model in enumerate(models, start=1):
            model_path = model["path"]
            label = model["label"]
            gif_path = None
            if not args.no_gifs:
                gif_path = output_dir / f"{label}_seed_{episode_seed}.gif"

            print(f"{idx:02d}/{len(models)}: {label}", flush=True)
            with open(model_path, "rb") as f:
                tree = pickle.load(f)

            result = run_model(
                tree,
                env,
                episode_seed=episode_seed,
                duration=args.duration,
                gif_path=gif_path,
                frame_duration_ms=frame_duration_ms,
                frame_stride=args.frame_stride,
                resize=args.resize,
            )
            row = {
                "model_label": label,
                "generation": model["generation"],
                "model_path": str(model_path),
                "seed": int(episode_seed),
                "gif": str(gif_path) if gif_path else "",
                **result,
            }
            rows.append(row)
            status = "crash" if result["crashed"] else "ok"
            print(
                f"  reward={result['reward']:.2f}, steps={result['steps']}, {status}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nInterrupted; writing summary for completed models.", flush=True)
    finally:
        env.close()

    csv_path = output_dir / "same_episode_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "generation",
                "model_path",
                "seed",
                "gif",
                "reward",
                "steps",
                "gif_frames",
                "terminated",
                "truncated",
                "crashed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_row = max(rows, key=lambda row: row["reward"]) if rows else None
    metadata = {
        "artifacts_dir": str(artifacts_dir),
        "output_dir": str(output_dir.resolve()),
        "episode_seed": int(episode_seed),
        "models_found": int(len(models)),
        "models_completed": int(len(rows)),
        "duration": int(args.duration),
        "gifs_saved": not args.no_gifs,
        "frame_stride": int(args.frame_stride),
        "resize": float(args.resize),
        "best_model_label": best_row["model_label"] if best_row else None,
        "best_generation": best_row["generation"] if best_row else None,
        "best_reward": float(best_row["reward"]) if best_row else None,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"\nEpisode seed: {episode_seed}")
    print(f"Summary CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    if best_row:
        print(f"Best on this episode: {best_row['model_label']}, reward={best_row['reward']:.2f}")


if __name__ == "__main__":
    main()
