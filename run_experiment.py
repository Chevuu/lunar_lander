import argparse
import copy
import csv
import json
import os
import pickle
import random
import subprocess
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

import numpy as np
import torch
import torch.optim as optim

import config
import sweep_config
from genepro.evo import Evolution
from genepro.node_impl import Constant, Feature
from train import Transition, env, fitness_function_pt, num_features, set_seed


def load_json_defaults(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def add_arg(parser, name, default, **kwargs):
    cli_name = "--" + name.replace("_", "-")
    parser.add_argument(cli_name, default=default, **kwargs)


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", default=None)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_json_defaults(pre_args.config_json)

    parser = argparse.ArgumentParser(
        description="Run only the necessary Lunar Lander GP experiment steps."
    )
    parser.add_argument("--config-json", default=pre_args.config_json)
    parser.add_argument(
        "--mode",
        choices=["train", "sweep", "sweep-train"],
        default=defaults.get("mode", "train"),
        help="train runs one experiment, sweep runs Optuna only, sweep-train tunes then trains with the best parameters.",
    )
    add_arg(parser, "experiment_name", defaults.get("experiment_name", "lunar_lander_gp"))
    add_arg(parser, "results_dir", defaults.get("results_dir", "experiment_runs"))
    add_arg(parser, "seed", defaults.get("seed", config.SEED), type=int)
    add_arg(parser, "n_jobs", defaults.get("n_jobs", config.N_JOBS), type=int)
    add_arg(parser, "verbose", defaults.get("verbose", config.VERBOSE), action=argparse.BooleanOptionalAction)

    add_arg(parser, "pop_size", defaults.get("pop_size", config.POP_SIZE), type=int)
    add_arg(parser, "max_gens", defaults.get("max_gens", config.MAX_GENS), type=int)
    add_arg(parser, "max_tree_size", defaults.get("max_tree_size", config.MAX_TREE_SIZE), type=int)
    add_arg(parser, "num_constants", defaults.get("num_constants", config.NUM_CONSTANTS), type=int)
    add_arg(parser, "num_episodes", defaults.get("num_episodes", config.NUM_EPISODES), type=int)
    add_arg(parser, "episode_duration", defaults.get("episode_duration", config.EPISODE_DURATION), type=int)

    add_arg(parser, "coeff_lr", defaults.get("coeff_lr", config.COEFF_LR), type=float)
    add_arg(parser, "coeff_opt_steps", defaults.get("coeff_opt_steps", config.COEFF_OPT_STEPS), type=int)
    add_arg(parser, "batch_size", defaults.get("batch_size", config.BATCH_SIZE), type=int)
    add_arg(parser, "gamma", defaults.get("gamma", config.GAMMA), type=float)
    add_arg(parser, "grad_clip", defaults.get("grad_clip", config.GRAD_CLIP), type=float)

    add_arg(parser, "test_episodes", defaults.get("test_episodes", config.TEST_EPISODES), type=int)
    add_arg(parser, "test_duration", defaults.get("test_duration", config.TEST_EPISODE_DURATION), type=int)
    add_arg(parser, "video", defaults.get("video", True), action=argparse.BooleanOptionalAction)
    add_arg(parser, "video_duration", defaults.get("video_duration", config.TEST_EPISODE_DURATION), type=int)
    add_arg(parser, "video_seed_offset", defaults.get("video_seed_offset", 20_000), type=int)

    add_arg(parser, "n_trials", defaults.get("n_trials", sweep_config.N_TRIALS), type=int)
    add_arg(parser, "sweep_gens", defaults.get("sweep_gens", sweep_config.SWEEP_GENS), type=int)
    add_arg(parser, "sweep_episodes", defaults.get("sweep_episodes", sweep_config.SWEEP_EPISODES), type=int)
    add_arg(parser, "resume_sweep", defaults.get("resume_sweep", False), action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def make_run_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.experiment_name.replace(" ", "_")
    run_dir = Path(args.results_dir) / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_fitness_fn(num_episodes, episode_duration, seed):
    def fn(multitree, render=False, ignore_done=False):
        return fitness_function_pt(
            multitree,
            num_episodes=num_episodes,
            episode_duration=episode_duration,
            render=render,
            ignore_done=ignore_done,
            seed=seed,
        )

    return fn


def update_runtime_config(args):
    config.SEED = args.seed
    config.N_JOBS = args.n_jobs
    config.POP_SIZE = args.pop_size
    config.MAX_GENS = args.max_gens
    config.MAX_TREE_SIZE = args.max_tree_size
    config.NUM_CONSTANTS = args.num_constants
    config.NUM_EPISODES = args.num_episodes
    config.EPISODE_DURATION = args.episode_duration
    config.COEFF_LR = args.coeff_lr
    config.COEFF_OPT_STEPS = args.coeff_opt_steps
    config.BATCH_SIZE = args.batch_size
    config.GAMMA = args.gamma
    config.GRAD_CLIP = args.grad_clip
    config.TEST_EPISODES = args.test_episodes
    config.TEST_EPISODE_DURATION = args.test_duration
    config.VERBOSE = args.verbose


def build_leaf_nodes(num_constants):
    leaf_nodes = [Feature(i) for i in range(num_features)]
    leaf_nodes += [Constant() for _ in range(num_constants)]
    return leaf_nodes


def optimize_constants(best, memory, args):
    constants = best.get_subtrees_consts()
    if len(constants) == 0 or len(memory) <= args.batch_size:
        return None

    optimizer = optim.AdamW(constants, lr=args.coeff_lr, amsgrad=True)
    last_loss = None
    for _ in range(args.coeff_opt_steps):
        target_tree = copy.deepcopy(best)
        transitions = memory.sample(args.batch_size)
        batch_data = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch_data.next_state)), dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch_data.next_state if s is not None])
        state_batch = torch.cat(batch_data.state)
        action_batch = torch.cat(batch_data.action)
        reward_batch = torch.cat(batch_data.reward)

        state_action_values = best.get_output_pt(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(args.batch_size, dtype=torch.float)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_tree.get_output_pt(non_final_next_states).max(1)[0].float()

        expected = (next_state_values * args.gamma) + reward_batch
        loss = torch.nn.SmoothL1Loss()(state_action_values, expected.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(constants, args.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())

    return last_loss


def evaluate_tree(tree, episodes, duration, seed):
    episode_rewards = []
    for episode_idx in range(episodes):
        observation = env.reset(seed=seed + episode_idx)[0]
        rewards = []
        for _ in range(duration):
            input_sample = torch.from_numpy(observation.reshape((1, -1))).float()
            action = torch.argmax(tree.get_output_pt(input_sample))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(float(reward))
            if terminated or truncated:
                break
        episode_rewards.append(sum(rewards))
    return {
        "total_score": float(sum(episode_rewards)),
        "mean_score": float(np.mean(episode_rewards)),
        "episode_scores": episode_rewards,
    }


def run_training(args):
    update_runtime_config(args)
    set_seed(args.seed)
    leaf_nodes = build_leaf_nodes(args.num_constants)
    fitness_fn = make_fitness_fn(args.num_episodes, args.episode_duration, args.seed)

    evo = Evolution(
        fitness_fn,
        config.INTERNAL_NODES,
        leaf_nodes,
        config.NUM_TREES,
        pop_size=args.pop_size,
        max_gens=args.max_gens,
        max_tree_size=args.max_tree_size,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    evo.evolve()

    best = max(evo.best_of_gens, key=lambda t: t.fitness)
    coeff_loss = optimize_constants(best, evo.memory, args)
    evaluation = evaluate_tree(
        best,
        episodes=args.test_episodes,
        duration=args.test_duration,
        seed=args.seed + 10_000,
    )
    return best, evo, coeff_loss, evaluation


def write_generation_history(evo, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "fitness", "tree_size"])
        writer.writeheader()
        for generation, tree in enumerate(evo.best_of_gens):
            writer.writerow(
                {
                    "generation": generation,
                    "fitness": float(tree.fitness),
                    "tree_size": int(len(tree)),
                }
            )


def render_gif(tree, path, seed, duration):
    from PIL import Image

    observation = env.reset(seed=seed)[0]
    frames = [env.render()]
    total_reward = 0.0
    for _ in range(duration):
        input_sample = torch.from_numpy(observation.reshape((1, -1))).float()
        action = torch.argmax(tree.get_output_pt(input_sample))
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += float(reward)
        frames.append(env.render())
        if terminated or truncated:
            break

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=50,
        loop=0,
    )
    return {"frames": len(frames), "reward": total_reward}


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def args_to_dict(args):
    data = vars(args).copy()
    data["git_commit"] = git_commit()
    return data


def run_sweep(args, run_dir):
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Optuna is required for --mode sweep or --mode sweep-train. Install it with: pip install optuna") from exc

    sweep_csv = run_dir / "sweep_trials.csv"
    storage_path = run_dir / "sweep.db"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name="lunar_lander_experiment_sweep",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=args.resume_sweep,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial):
        trial_args = copy.copy(args)
        trial_args.seed = args.seed + trial.number
        trial_args.verbose = False
        trial_args.pop_size = trial.suggest_categorical("pop_size", sweep_config.POP_SIZE_OPTIONS)
        trial_args.max_tree_size = trial.suggest_categorical("max_tree_size", sweep_config.MAX_TREE_SIZE_OPTIONS)
        trial_args.num_constants = trial.suggest_int(
            "num_constants",
            sweep_config.NUM_CONSTANTS_MIN,
            sweep_config.NUM_CONSTANTS_MAX,
        )
        trial_args.coeff_lr = trial.suggest_float(
            "coeff_lr",
            sweep_config.COEFF_LR_MIN,
            sweep_config.COEFF_LR_MAX,
            log=True,
        )
        trial_args.coeff_opt_steps = trial.suggest_categorical(
            "coeff_opt_steps",
            sweep_config.COEFF_OPT_STEPS_OPTIONS,
        )
        trial_args.gamma = trial.suggest_float("gamma", sweep_config.GAMMA_MIN, sweep_config.GAMMA_MAX)
        trial_args.max_gens = args.sweep_gens
        trial_args.num_episodes = args.sweep_episodes

        _, _, _, evaluation = run_training(trial_args)
        score = evaluation["total_score"]
        write_header = not sweep_csv.exists()
        with open(sweep_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "trial",
                    "score",
                    "pop_size",
                    "max_tree_size",
                    "num_constants",
                    "coeff_lr",
                    "coeff_opt_steps",
                    "gamma",
                    "seed",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "trial": trial.number,
                    "score": score,
                    "pop_size": trial_args.pop_size,
                    "max_tree_size": trial_args.max_tree_size,
                    "num_constants": trial_args.num_constants,
                    "coeff_lr": trial_args.coeff_lr,
                    "coeff_opt_steps": trial_args.coeff_opt_steps,
                    "gamma": trial_args.gamma,
                    "seed": trial_args.seed,
                }
            )
        print(f"Trial {trial.number + 1}/{args.n_trials}: score={score:.2f}")
        return score

    study.optimize(objective, n_trials=args.n_trials)
    best = {
        "value": float(study.best_value),
        "params": dict(study.best_params),
        "sweep_csv": str(sweep_csv),
        "sweep_db": str(storage_path),
    }
    write_json(run_dir / "best_sweep_params.json", best)
    return best


def apply_sweep_params(args, sweep_result):
    for key, value in sweep_result["params"].items():
        setattr(args, key, value)
    return args


def save_training_outputs(args, run_dir, best, evo, coeff_loss, evaluation, sweep_result=None):
    write_json(run_dir / "run_config.json", args_to_dict(args))
    write_generation_history(evo, run_dir / "generation_history.csv")

    with open(run_dir / "best_tree.txt", "w") as f:
        f.write(json.dumps(best.get_readable_repr(), indent=2))
        f.write("\n")

    with open(run_dir / "best_tree.pkl", "wb") as f:
        pickle.dump(best, f)

    video_info = None
    if args.video:
        gif_path = run_dir / "best_lander.gif"
        video_info = render_gif(
            best,
            gif_path,
            seed=args.seed + args.video_seed_offset,
            duration=args.video_duration,
        )
        video_info["path"] = str(gif_path)

    metrics = {
        "best_training_fitness": float(best.fitness),
        "coefficient_optimization_last_loss": coeff_loss,
        "test": evaluation,
        "video": video_info,
        "sweep": sweep_result,
    }
    write_json(run_dir / "metrics.json", metrics)

    with open(run_dir / "summary.txt", "w") as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Best training fitness: {best.fitness:.3f}\n")
        f.write(f"Test total score: {evaluation['total_score']:.3f}\n")
        f.write(f"Test mean score: {evaluation['mean_score']:.3f}\n")
        if video_info:
            f.write(f"GIF: {video_info['path']}\n")


def main():
    args = parse_args()
    run_dir = make_run_dir(args)
    write_json(run_dir / "requested_config.json", args_to_dict(args))

    sweep_result = None
    if args.mode in {"sweep", "sweep-train"}:
        sweep_result = run_sweep(args, run_dir)
        print("Best sweep params:", sweep_result["params"])

    if args.mode == "sweep":
        print(f"Sweep complete. Results written to {run_dir}")
        env.close()
        return

    if args.mode == "sweep-train":
        args = apply_sweep_params(args, sweep_result)
        args.mode = "sweep-train-final"
        args.verbose = True

    best, evo, coeff_loss, evaluation = run_training(args)
    save_training_outputs(args, run_dir, best, evo, coeff_loss, evaluation, sweep_result)
    print(f"Experiment complete. Results written to {run_dir}")
    print(f"Test total score: {evaluation['total_score']:.2f}")
    env.close()


if __name__ == "__main__":
    main()
