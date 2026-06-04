import argparse
import csv
import json
import pickle
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
        description="Render random LunarLander episodes for a saved GP tree."
    )
    parser.add_argument("model_path", type=Path, help="Path to a saved tree .pkl file.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of GIFs to render.")
    parser.add_argument("--duration", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for sampling random episode seeds.")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Save every n-th rendered frame to the GIF. Higher values are much faster/smaller.",
    )
    parser.add_argument(
        "--resize",
        type=float,
        default=1.0,
        help="Scale GIF frames, e.g. 0.75 or 0.5 for faster/smaller GIFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for GIFs. Defaults to <model_dir>/<model_name>_random_gifs.",
    )
    return parser.parse_args()


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


def render_episode(tree, env, episode_seed, gif_path, max_steps, frame_duration_ms, frame_stride, resize):
    observation = env.reset(seed=int(episode_seed))[0]
    frames = [frame_to_image(env.render(), resize)]
    total_reward = 0.0
    final_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    for _ in range(max_steps):
        action = select_action(tree, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        final_reward = float(reward)
        total_reward += final_reward
        steps += 1
        if steps % frame_stride == 0 or terminated or truncated:
            frames.append(frame_to_image(env.render(), resize))
        if terminated or truncated:
            break

    print(f"    saving {len(frames)} frames -> {gif_path.name}", flush=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )

    crashed = bool(terminated and final_reward <= -100.0)
    return {
        "seed": int(episode_seed),
        "gif": str(gif_path),
        "reward": float(total_reward),
        "steps": int(steps),
        "gif_frames": int(len(frames)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "crashed": crashed,
    }


def write_summary(output_dir, rows):
    csv_path = output_dir / "episode_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
    return csv_path


def main():
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = model_path.parent / f"{model_path.stem}_random_gifs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if args.duration < 1:
        raise SystemExit("--duration must be at least 1")
    if args.fps < 1:
        raise SystemExit("--fps must be at least 1")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.resize <= 0:
        raise SystemExit("--resize must be positive")

    sampler_seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(sampler_seed)
    episode_seeds = rng.integers(0, np.iinfo(np.int32).max, size=args.episodes)
    frame_duration_ms = max(1, int(1000 / args.fps))

    with open(model_path, "rb") as f:
        tree = pickle.load(f)

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    rows = []
    try:
        for idx, episode_seed in enumerate(episode_seeds, start=1):
            gif_path = output_dir / f"episode_{idx:02d}_seed_{int(episode_seed)}.gif"
            print(f"{idx:02d}/{args.episodes}: rendering seed={int(episode_seed)}", flush=True)
            result = render_episode(
                tree,
                env,
                episode_seed=episode_seed,
                gif_path=gif_path,
                max_steps=args.duration,
                frame_duration_ms=frame_duration_ms,
                frame_stride=args.frame_stride,
                resize=args.resize,
            )
            rows.append(result)
            status = "crash" if result["crashed"] else "ok"
            print(
                f"{idx:02d}/{args.episodes}: reward={result['reward']:.2f}, "
                f"steps={result['steps']}, gif_frames={result['gif_frames']}, "
                f"{status}, seed={result['seed']}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nInterrupted; writing summary for completed GIFs.", flush=True)
    finally:
        env.close()

    csv_path = write_summary(output_dir, rows)

    rewards = [row["reward"] for row in rows]
    metadata = {
        "model_path": str(model_path),
        "output_dir": str(output_dir.resolve()),
        "sampler_seed": int(sampler_seed),
        "episodes": int(args.episodes),
        "completed_episodes": int(len(rows)),
        "duration": int(args.duration),
        "frame_stride": int(args.frame_stride),
        "resize": float(args.resize),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "crashes": int(sum(row["crashed"] for row in rows)),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"\nSaved {len(rows)} GIFs to {output_dir}")
    print(f"Summary CSV: {csv_path}")
    print(f"Sampler seed: {sampler_seed}")


if __name__ == "__main__":
    main()
