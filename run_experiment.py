import argparse
import copy
import csv
import json
import os
import pickle
import signal
import subprocess
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

import numpy as np
import torch
import torch.optim as optim
from joblib.parallel import Parallel, delayed

import config
import sweep_config
from genepro.evo import Evolution, generate_offspring, generate_random_multitree
from genepro.node_impl import Constant, Div, Feature, Minus, Plus, Times
from genepro.selection import tournament_selection
from train import ReplayMemory, Transition, env, num_features, set_seed


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
        choices=["train", "sweep", "sweep-train", "retest"],
        default=defaults.get("mode", "train"),
        help="train runs one experiment, sweep runs Optuna only, sweep-train tunes then trains with the best parameters, retest re-runs the final test on an existing run (requires --retest-dir).",
    )
    add_arg(parser, "retest_dir", defaults.get("retest_dir", None))
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
    add_arg(parser, "video", defaults.get("video", config.VIDEO), action=argparse.BooleanOptionalAction)
    add_arg(parser, "save_raw_gifs", defaults.get("save_raw_gifs", config.SAVE_RAW_GIFS), action=argparse.BooleanOptionalAction)
    add_arg(parser, "plots", defaults.get("plots", config.PLOTS), action=argparse.BooleanOptionalAction)
    add_arg(parser, "video_duration", defaults.get("video_duration", config.VIDEO_EPISODE_DURATION), type=int)
    add_arg(parser, "video_seed_offset", defaults.get("video_seed_offset", config.VIDEO_SEED_OFFSET), type=int)
    add_arg(parser, "artifact_interval", defaults.get("artifact_interval", config.ARTIFACT_INTERVAL), type=int)

    add_arg(parser, "baseline_original", defaults.get("baseline_original", False), action=argparse.BooleanOptionalAction)
    add_arg(parser, "elitism", defaults.get("elitism", True), action=argparse.BooleanOptionalAction)
    add_arg(parser, "random_seeds", defaults.get("random_seeds", config.RANDOM_SEEDS), action=argparse.BooleanOptionalAction)
    add_arg(parser, "seed_stride", defaults.get("seed_stride", config.SEED_STRIDE), type=int)
    add_arg(parser, "crash_penalty", defaults.get("crash_penalty", config.CRASH_PENALTY), action=argparse.BooleanOptionalAction)
    add_arg(parser, "crash_penalty_weight", defaults.get("crash_penalty_weight", config.CRASH_PENALTY_WEIGHT), type=float)
    add_arg(parser, "parsimony", defaults.get("parsimony", config.PARSIMONY), action=argparse.BooleanOptionalAction)
    add_arg(parser, "parsimony_weight", defaults.get("parsimony_weight", config.PARSIMONY_WEIGHT), type=float)
    add_arg(parser, "time_pressure", defaults.get("time_pressure", config.TIME_PRESSURE), action=argparse.BooleanOptionalAction)
    add_arg(parser, "time_pressure_weight", defaults.get("time_pressure_weight", config.TIME_PRESSURE_WEIGHT), type=float)

    add_arg(parser, "validation_selection", defaults.get("validation_selection", config.VALIDATION_SELECTION), action=argparse.BooleanOptionalAction)
    add_arg(parser, "validation_episodes", defaults.get("validation_episodes", config.VALIDATION_EPISODES), type=int)
    add_arg(parser, "validation_duration", defaults.get("validation_duration", config.VALIDATION_EPISODE_DURATION), type=int)
    add_arg(parser, "validation_interval", defaults.get("validation_interval", config.VALIDATION_INTERVAL), type=int)
    add_arg(parser, "validation_candidates", defaults.get("validation_candidates", config.VALIDATION_CANDIDATES), type=int)
    add_arg(parser, "validation_seed_offset", defaults.get("validation_seed_offset", config.VALIDATION_SEED_OFFSET), type=int)

    add_arg(parser, "tournament_size", defaults.get("tournament_size", config.TOURNAMENT_SIZE), type=int)
    add_arg(parser, "tournament_size_start", defaults.get("tournament_size_start", config.TOURNAMENT_SIZE_START), type=int)

    add_arg(parser, "save_best_each_gen", defaults.get("save_best_each_gen", config.SAVE_BEST_EACH_GEN), action=argparse.BooleanOptionalAction)
    add_arg(parser, "graceful_stop", defaults.get("graceful_stop", config.GRACEFUL_STOP), action=argparse.BooleanOptionalAction)
    add_arg(parser, "gate_coeff_optimization", defaults.get("gate_coeff_optimization", config.GATE_COEFF_OPTIMIZATION), action=argparse.BooleanOptionalAction)
    add_arg(parser, "coeff_gate_episodes", defaults.get("coeff_gate_episodes", config.COEFF_GATE_EPISODES), type=int)
    add_arg(parser, "coeff_gate_seed_offset", defaults.get("coeff_gate_seed_offset", config.COEFF_GATE_SEED_OFFSET), type=int)

    add_arg(parser, "n_trials", defaults.get("n_trials", sweep_config.N_TRIALS), type=int)
    add_arg(parser, "sweep_gens", defaults.get("sweep_gens", sweep_config.SWEEP_GENS), type=int)
    add_arg(parser, "sweep_episodes", defaults.get("sweep_episodes", sweep_config.SWEEP_EPISODES), type=int)
    add_arg(parser, "resume_sweep", defaults.get("resume_sweep", False), action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def apply_original_baseline(args):
    if not args.baseline_original:
        args.function_nodes = "configured"
        return args

    args.elitism = False
    args.validation_selection = False
    args.gate_coeff_optimization = False
    args.crash_penalty = False
    args.parsimony = False
    args.time_pressure = False
    args.random_seeds = True
    args.function_nodes = "original_plus_minus_times_div"
    return args


def make_run_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.experiment_name.replace(" ", "_")
    run_dir = Path(args.results_dir) / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def is_crash(terminated, final_reward):
    return bool(terminated and final_reward <= -100.0)


def validation_seed_for_generation(args, generation):
    return args.seed + args.validation_seed_offset + generation * args.seed_stride


def coeff_gate_seed(args):
    return args.seed + args.coeff_gate_seed_offset


def run_policy_episodes(tree, episodes, duration, seed, collect_memory=False, render=False, ignore_done=False):
    memory = ReplayMemory(config.REPLAY_MEMORY_SIZE) if collect_memory else None
    episode_rewards = []
    episode_lengths = []
    episode_crashed = []
    crashed_episodes = 0
    survived_episodes = 0
    terminated_episodes = 0
    truncated_episodes = 0

    for episode_idx in range(episodes):
        episode_seed = None if seed is None else seed + episode_idx
        observation = env.reset(seed=episode_seed)[0]
        total_reward = 0.0
        final_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        for _ in range(duration):
            input_sample = torch.from_numpy(observation.reshape((1, -1))).float()
            action = torch.argmax(tree.get_output_pt(input_sample))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            final_reward = float(reward)
            total_reward += final_reward
            steps += 1

            if collect_memory:
                output_sample = torch.from_numpy(observation.reshape((1, -1))).float()
                memory.push(
                    input_sample,
                    torch.tensor([[action.item()]]),
                    output_sample,
                    torch.tensor([reward]),
                )

            if render:
                env.render()

            if (terminated or truncated) and not ignore_done:
                break

        crashed = is_crash(terminated, final_reward)
        crashed_episodes += int(crashed)
        survived_episodes += int(not crashed)
        terminated_episodes += int(terminated)
        truncated_episodes += int(truncated)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_crashed.append(bool(crashed))

    total_episodes = len(episode_rewards)
    reward_std = float(np.std(episode_rewards)) if episode_rewards else 0.0
    stats = {
        "total_score": float(sum(episode_rewards)),
        "mean_score": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_score": reward_std,
        "episode_scores": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_crashed": episode_crashed,
        "total_episodes": total_episodes,
        "survived_episodes": survived_episodes,
        "crashed_episodes": crashed_episodes,
        "terminated_episodes": terminated_episodes,
        "truncated_episodes": truncated_episodes,
        "survival_rate": float(survived_episodes / total_episodes) if total_episodes else 0.0,
        "crash_rate": float(crashed_episodes / total_episodes) if total_episodes else 0.0,
    }
    return stats, memory


def make_fitness_fn(num_episodes, episode_duration, seed):
    def fn(multitree, render=False, ignore_done=False):
        eval_seed = None if config.RANDOM_SEEDS else seed
        stats, memory = run_policy_episodes(
            multitree,
            episodes=num_episodes,
            duration=episode_duration,
            seed=eval_seed,
            collect_memory=True,
            render=render,
            ignore_done=ignore_done,
        )
        fitness = stats["total_score"]
        if config.CRASH_PENALTY:
            fitness -= config.CRASH_PENALTY_WEIGHT * stats["crashed_episodes"]
        if config.PARSIMONY:
            fitness -= config.PARSIMONY_WEIGHT * len(multitree)
        if config.TIME_PRESSURE:
            fitness -= config.TIME_PRESSURE_WEIGHT * sum(stats["episode_lengths"])
        return fitness, memory, stats

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
    config.VIDEO = getattr(args, "video", config.VIDEO)
    config.SAVE_RAW_GIFS = getattr(args, "save_raw_gifs", config.SAVE_RAW_GIFS)
    config.PLOTS = getattr(args, "plots", config.PLOTS)
    config.VIDEO_EPISODE_DURATION = getattr(args, "video_duration", config.VIDEO_EPISODE_DURATION)
    config.VIDEO_SEED_OFFSET = getattr(args, "video_seed_offset", config.VIDEO_SEED_OFFSET)
    config.ARTIFACT_INTERVAL = getattr(args, "artifact_interval", config.ARTIFACT_INTERVAL)
    config.RANDOM_SEEDS = getattr(args, "random_seeds", config.RANDOM_SEEDS)
    config.SEED_STRIDE = getattr(args, "seed_stride", config.SEED_STRIDE)
    config.CRASH_PENALTY = getattr(args, "crash_penalty", config.CRASH_PENALTY)
    config.CRASH_PENALTY_WEIGHT = getattr(args, "crash_penalty_weight", config.CRASH_PENALTY_WEIGHT)
    config.PARSIMONY = getattr(args, "parsimony", config.PARSIMONY)
    config.PARSIMONY_WEIGHT = getattr(args, "parsimony_weight", config.PARSIMONY_WEIGHT)
    config.TIME_PRESSURE = getattr(args, "time_pressure", config.TIME_PRESSURE)
    config.TIME_PRESSURE_WEIGHT = getattr(args, "time_pressure_weight", config.TIME_PRESSURE_WEIGHT)
    config.VALIDATION_SELECTION = getattr(args, "validation_selection", config.VALIDATION_SELECTION)
    config.VALIDATION_EPISODES = getattr(args, "validation_episodes", config.VALIDATION_EPISODES)
    config.VALIDATION_EPISODE_DURATION = getattr(args, "validation_duration", config.VALIDATION_EPISODE_DURATION)
    config.VALIDATION_INTERVAL = getattr(args, "validation_interval", config.VALIDATION_INTERVAL)
    config.VALIDATION_CANDIDATES = getattr(args, "validation_candidates", config.VALIDATION_CANDIDATES)
    config.VALIDATION_SEED_OFFSET = getattr(args, "validation_seed_offset", config.VALIDATION_SEED_OFFSET)
    config.SAVE_BEST_EACH_GEN = getattr(args, "save_best_each_gen", config.SAVE_BEST_EACH_GEN)
    config.GRACEFUL_STOP = getattr(args, "graceful_stop", config.GRACEFUL_STOP)
    config.GATE_COEFF_OPTIMIZATION = getattr(args, "gate_coeff_optimization", config.GATE_COEFF_OPTIMIZATION)
    config.COEFF_GATE_EPISODES = getattr(args, "coeff_gate_episodes", config.COEFF_GATE_EPISODES)
    config.COEFF_GATE_SEED_OFFSET = getattr(args, "coeff_gate_seed_offset", config.COEFF_GATE_SEED_OFFSET)


def build_leaf_nodes(num_constants):
    leaf_nodes = [Feature(i) for i in range(num_features)]
    leaf_nodes += [Constant() for _ in range(num_constants)]
    return leaf_nodes


def build_internal_nodes(args):
    if getattr(args, "baseline_original", False):
        return [Plus(), Minus(), Times(), Div()]
    return config.INTERNAL_NODES


def merge_memories(memories):
    memory = memories[0]
    for next_memory in memories[1:]:
        memory += next_memory
    return memory


def summarize_population(generation, population, best_fitness_so_far=None):
    fitnesses = np.array([float(t.fitness) for t in population], dtype=float)
    sizes = np.array([int(len(t)) for t in population], dtype=int)
    best_idx = int(np.argmax(fitnesses))
    best_individual = population[best_idx]
    best_stats = getattr(best_individual, "_episode_stats", None)
    best_lander_fitness = (
        float(best_stats["mean_score"])
        if best_stats
        else float(fitnesses[best_idx])
    )
    episode_scores = []
    total_episodes = 0
    survived_episodes = 0
    crashed_episodes = 0
    survived_agents = 0
    crashed_agents = 0

    for individual in population:
        stats = getattr(individual, "_episode_stats", None)
        if not stats:
            continue
        episode_scores.extend(stats["episode_scores"])
        total_episodes += stats["total_episodes"]
        survived_episodes += stats["survived_episodes"]
        crashed_episodes += stats["crashed_episodes"]
        if stats["crashed_episodes"] == 0:
            survived_agents += 1
        else:
            crashed_agents += 1

    return {
        "generation": generation,
        "best_fitness": float(fitnesses[best_idx]),
        "best_fitness_so_far": float(best_fitness_so_far) if best_fitness_so_far is not None else float(fitnesses[best_idx]),
        "best_lander_fitness": best_lander_fitness,
        "mean_fitness": float(np.mean(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
        "best_tree_size": int(sizes[best_idx]),
        "mean_tree_size": float(np.mean(sizes)),
        "population_size": int(len(population)),
        "survived_agents": survived_agents,
        "crashed_agents": crashed_agents,
        "agent_survival_rate": float(survived_agents / len(population)) if population else 0.0,
        "agent_crash_rate": float(crashed_agents / len(population)) if population else 0.0,
        "total_episodes": total_episodes,
        "survived_episodes": survived_episodes,
        "crashed_episodes": crashed_episodes,
        "episode_survival_rate": float(survived_episodes / total_episodes) if total_episodes else 0.0,
        "episode_crash_rate": float(crashed_episodes / total_episodes) if total_episodes else 0.0,
        "mean_episode_score": float(np.mean(episode_scores)) if episode_scores else 0.0,
        "std_episode_score": float(np.std(episode_scores)) if episode_scores else 0.0,
    }


CSV_FIELDNAMES = [
    "generation", "best_fitness", "best_fitness_so_far", "best_lander_fitness",
    "mean_fitness", "std_fitness",
    "best_tree_size", "mean_tree_size", "population_size",
    "survived_agents", "crashed_agents", "agent_survival_rate", "agent_crash_rate",
    "total_episodes", "survived_episodes", "crashed_episodes",
    "episode_survival_rate", "episode_crash_rate", "mean_episode_score", "std_episode_score",
    "validation_best_score", "validation_best_mean", "validation_best_std",
    "validation_best_survival_rate", "best_validation_score_so_far",
]


class GracefulStopRequested(Exception):
    pass


def atomic_pickle(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, path)


def atomic_json(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


class ExperimentEvolution(Evolution):
    def __init__(self, *args, run_args=None, run_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_args = run_args
        self.run_dir = run_dir
        self.generation_records = []
        self.generation_artifacts = []
        self.elite = None
        self.best_training = None
        self.best_validation = None
        self.best_validation_score = None
        self.validation_records = []
        self.interrupted = False
        self.stop_reason = None

    def _evaluate_population(self, population):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fitness_function)(t) for t in population
        )
        fitnesses, memories, episode_stats = list(map(list, zip(*results)))
        for individual, fitness, stats in zip(population, fitnesses, episode_stats):
            individual.fitness = fitness
            individual._episode_stats = stats
        return merge_memories(memories)

    def _update_best_training(self, tree, generation):
        if self.best_training is None or tree.fitness > self.best_training.fitness:
            self.best_training = copy.deepcopy(tree)
            self.best_training._best_generation = generation

    def _validation_enabled(self, generation):
        args = self.run_args
        if args is None or not args.validation_selection:
            return False
        if args.validation_interval <= 0:
            return False
        return generation % args.validation_interval == 0

    def _validate_generation(self, generation, population, record):
        if not self._validation_enabled(generation):
            return

        args = self.run_args
        num_candidates = max(1, min(args.validation_candidates, len(population)))
        candidates = sorted(population, key=lambda t: t.fitness, reverse=True)[:num_candidates]
        eval_seed = validation_seed_for_generation(args, generation)

        generation_best = None
        generation_best_stats = None
        generation_best_score = None
        for candidate in candidates:
            stats = evaluate_tree(
                candidate,
                episodes=args.validation_episodes,
                duration=args.validation_duration,
                seed=eval_seed,
            )
            score = stats["total_score"]
            if generation_best_score is None or score > generation_best_score:
                generation_best = candidate
                generation_best_stats = stats
                generation_best_score = score

        record.update(
            {
                "validation_best_score": float(generation_best_score),
                "validation_best_mean": float(generation_best_stats["mean_score"]),
                "validation_best_std": float(generation_best_stats["std_score"]),
                "validation_best_survival_rate": float(generation_best_stats["survival_rate"]),
            }
        )
        self.validation_records.append(
            {
                "generation": generation,
                "score": float(generation_best_score),
                "stats": generation_best_stats,
                "training_fitness": float(generation_best.fitness),
                "tree_size": int(len(generation_best)),
            }
        )

        if self.best_validation_score is None or generation_best_score > self.best_validation_score:
            self.best_validation = copy.deepcopy(generation_best)
            self.best_validation._best_generation = generation
            self.best_validation._validation_stats = generation_best_stats
            self.best_validation_score = float(generation_best_score)
            self._write_best_files(generation, latest_tree=generation_best)

        record["best_validation_score_so_far"] = self.best_validation_score

    def _write_best_files(self, generation, latest_tree=None):
        args = self.run_args
        if self.run_dir is None or args is None:
            return
        if not (args.save_best_each_gen or args.validation_selection):
            return

        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        metadata = {
            "generation": generation,
            "best_training_fitness": float(self.best_training.fitness) if self.best_training else None,
            "best_training_generation": getattr(self.best_training, "_best_generation", None),
            "best_validation_score": self.best_validation_score,
            "best_validation_generation": getattr(self.best_validation, "_best_generation", None),
        }

        if latest_tree is not None:
            atomic_pickle(checkpoint_dir / "latest_tree.pkl", latest_tree)
            metadata["latest_training_fitness"] = float(latest_tree.fitness)

        if self.best_training is not None:
            atomic_pickle(checkpoint_dir / "best_training_tree.pkl", self.best_training)
        if self.best_validation is not None:
            atomic_pickle(checkpoint_dir / "best_validation_tree.pkl", self.best_validation)

        atomic_json(checkpoint_dir / "checkpoint_metadata.json", metadata)

    def _append_csv_row(self, record):
        if self.run_dir is None:
            return
        path = self.run_dir / "generation_history.csv"
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    def _write_checkpoint(self, generation):
        if self.run_dir is None or self.run_args is None:
            return
        args = self.run_args
        tree = self.best_of_gens[generation]
        artifact_dir = self.run_dir / "generation_artifacts"
        artifact_dir.mkdir(exist_ok=True)

        prefix = f"generation_{generation:04d}"
        raw_tree = copy.deepcopy(tree)
        optimized_tree, coeff_loss, coeff_gate = optimize_with_optional_gate(
            raw_tree,
            self.memory,
            args,
            gate_seed=args.seed + args.coeff_gate_seed_offset + generation * args.seed_stride,
        )

        pkl_path = artifact_dir / f"{prefix}_tree.pkl"
        atomic_pickle(pkl_path, optimized_tree)

        artifact = {
            "generation": generation,
            "training_fitness": float(raw_tree.fitness),
            "tree_size": int(len(optimized_tree)),
            "model_is_optimized": coeff_gate is None or bool(coeff_gate.get("kept_optimized", True)),
            "coefficient_optimization_last_loss": coeff_loss,
            "coefficient_gate": coeff_gate,
            "model_path": str(pkl_path),
        }
        if args.video:
            if args.save_raw_gifs:
                raw_gif_path = artifact_dir / f"{prefix}_before_optimization.gif"
                raw_gif_info = render_gif(raw_tree, raw_gif_path, seed=args.seed + args.video_seed_offset + generation, duration=args.video_duration)
                raw_gif_info["path"] = str(raw_gif_path)
                artifact["raw_gif"] = raw_gif_info
            optimized_gif_path = artifact_dir / f"{prefix}_after_optimization.gif"
            optimized_gif_info = render_gif(optimized_tree, optimized_gif_path, seed=args.seed + args.video_seed_offset + generation, duration=args.video_duration)
            optimized_gif_info["path"] = str(optimized_gif_path)
            artifact["optimized_gif"] = optimized_gif_info
            artifact["gif"] = optimized_gif_info

        self.generation_artifacts.append(artifact)
        print(f"  checkpoint written: {prefix}")

    def _initialize_population(self):
        self.population = Parallel(n_jobs=self.n_jobs)(
            delayed(generate_random_multitree)(
                self.n_trees,
                self.internal_nodes,
                self.leaf_nodes,
                max_depth=self.init_max_depth,
            )
            for _ in range(self.pop_size)
        )

        for individual in self.population:
            individual.get_readable_repr()

        self.memory = self._evaluate_population(self.population)
        self.num_evals += self.pop_size
        best = self.population[np.argmax([t.fitness for t in self.population])]
        self.best_of_gens.append(copy.deepcopy(best))
        if self.run_args is not None and self.run_args.elitism:
            self.elite = copy.deepcopy(best)
        self._update_best_training(best, generation=0)
        record = summarize_population(0, self.population, self.best_training.fitness)
        self._validate_generation(0, self.population, record)
        self.generation_records.append(record)
        self._append_csv_row(record)
        self._write_best_files(0, latest_tree=best)

    def _perform_generation(self):
        if self.run_args and self.run_args.tournament_size_start != self.run_args.tournament_size:
            progress = self.num_gens / max(1, self.max_gens - 1)
            t = self.run_args.tournament_size_start + progress * (self.run_args.tournament_size - self.run_args.tournament_size_start)
            divisors = [d for d in range(1, self.pop_size + 1) if self.pop_size % d == 0]
            self.selection["kwargs"]["tournament_size"] = min(divisors, key=lambda d: abs(d - t))

        sel_fun = self.selection["fun"]
        parents = sel_fun(self.population, self.pop_size, **self.selection["kwargs"])
        offspring_population = Parallel(n_jobs=self.n_jobs)(
            delayed(generate_offspring)(
                t,
                self.crossovers,
                self.mutations,
                self.coeff_opts,
                parents,
                self.internal_nodes,
                self.leaf_nodes,
                constraints={"max_tree_size": self.max_tree_size},
            )
            for t in parents
        )

        generation_memory = self._evaluate_population(offspring_population)
        self.memory = generation_memory + self.memory
        self.num_evals += self.pop_size

        generation_best = offspring_population[np.argmax([t.fitness for t in offspring_population])]
        self.num_gens += 1
        self.best_of_gens.append(copy.deepcopy(generation_best))
        self._update_best_training(generation_best, generation=self.num_gens)

        record = summarize_population(self.num_gens, offspring_population, self.best_training.fitness)
        self._validate_generation(self.num_gens, offspring_population, record)
        self.generation_records.append(record)
        self._append_csv_row(record)
        self._write_best_files(self.num_gens, latest_tree=generation_best)

        if self.run_args is not None and self.run_args.elitism and self.elite is not None:
            elite_candidate = copy.deepcopy(self.elite)
            if config.RANDOM_SEEDS:
                elite_candidate.fitness, _, elite_candidate._episode_stats = self.fitness_function(
                    elite_candidate
                )
            worst_idx = int(np.argmin([t.fitness for t in offspring_population]))
            offspring_population[worst_idx] = elite_candidate

        self.population = offspring_population
        best = self.population[np.argmax([t.fitness for t in self.population])]
        if self.run_args is not None and self.run_args.elitism:
            self.elite = copy.deepcopy(best)

        if self.run_args and self.run_args.artifact_interval > 0 and self.num_gens % self.run_args.artifact_interval == 0:
            self._write_checkpoint(self.num_gens)


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
    stats, _ = run_policy_episodes(
        tree,
        episodes=episodes,
        duration=duration,
        seed=seed,
        collect_memory=False,
    )
    return stats


def optimize_with_optional_gate(raw_tree, memory, args, gate_seed=None):
    optimized_tree = copy.deepcopy(raw_tree)
    coeff_loss = optimize_constants(optimized_tree, memory, args)

    if not args.gate_coeff_optimization:
        return optimized_tree, coeff_loss, None

    gate_info = {
        "enabled": True,
        "kept_optimized": False,
        "raw_score": None,
        "optimized_score": None,
        "reason": None,
    }
    if coeff_loss is None:
        gate_info["reason"] = "no_constants_or_not_enough_memory"
        return copy.deepcopy(raw_tree), coeff_loss, gate_info

    seed = gate_seed if gate_seed is not None else coeff_gate_seed(args)
    raw_eval = evaluate_tree(
        raw_tree,
        episodes=args.coeff_gate_episodes,
        duration=args.validation_duration,
        seed=seed,
    )
    optimized_eval = evaluate_tree(
        optimized_tree,
        episodes=args.coeff_gate_episodes,
        duration=args.validation_duration,
        seed=seed,
    )
    gate_info["raw_score"] = float(raw_eval["total_score"])
    gate_info["optimized_score"] = float(optimized_eval["total_score"])

    if optimized_eval["total_score"] >= raw_eval["total_score"]:
        gate_info["kept_optimized"] = True
        gate_info["reason"] = "optimized_validation_score_not_worse"
        return optimized_tree, coeff_loss, gate_info

    gate_info["reason"] = "raw_validation_score_better"
    return copy.deepcopy(raw_tree), coeff_loss, gate_info


def run_training(args, run_dir):
    update_runtime_config(args)
    set_seed(args.seed)
    internal_nodes = build_internal_nodes(args)
    leaf_nodes = build_leaf_nodes(args.num_constants)
    fitness_fn = make_fitness_fn(args.num_episodes, args.episode_duration, args.seed)

    evo = ExperimentEvolution(
        fitness_fn,
        internal_nodes,
        leaf_nodes,
        config.NUM_TREES,
        pop_size=args.pop_size,
        max_gens=args.max_gens,
        max_tree_size=args.max_tree_size,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        selection={"fun": tournament_selection, "kwargs": {"tournament_size": args.tournament_size_start}},
        run_args=args,
        run_dir=run_dir,
    )
    previous_sigterm_handler = None
    if args.graceful_stop:
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def handle_sigterm(signum, frame):
            raise GracefulStopRequested(f"signal_{signum}")

        signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        evo.evolve()
    except (KeyboardInterrupt, GracefulStopRequested) as exc:
        if not args.graceful_stop or not evo.best_of_gens:
            raise
        evo.interrupted = True
        evo.stop_reason = "keyboard_interrupt" if isinstance(exc, KeyboardInterrupt) else str(exc)
        print(f"\nGraceful stop requested after generation {evo.num_gens}; saving completed results.")
    finally:
        if previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

    if args.validation_selection and evo.best_validation is not None:
        raw_best = copy.deepcopy(evo.best_validation)
    elif evo.best_training is not None:
        raw_best = copy.deepcopy(evo.best_training)
    else:
        raw_best = copy.deepcopy(max(evo.best_of_gens, key=lambda t: t.fitness))

    best, coeff_loss, coeff_gate = optimize_with_optional_gate(raw_best, evo.memory, args)
    evaluation = evaluate_tree(
        best,
        episodes=args.test_episodes,
        duration=args.test_duration,
        seed=args.seed + 10_000,
    )
    return raw_best, best, evo, coeff_loss, coeff_gate, evaluation


def write_generation_history(evo, path):
    if getattr(evo, "generation_records", None):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(evo.generation_records)
        return

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


def save_generation_artifacts(args, run_dir, evo):
    interval = args.artifact_interval
    if interval <= 0:
        return []

    artifact_dir = run_dir / "generation_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifacts = []
    for generation, tree in enumerate(evo.best_of_gens):
        if generation == 0 or generation % interval != 0:
            continue

        prefix = f"generation_{generation:04d}"
        raw_tree = copy.deepcopy(tree)
        optimized_tree, coeff_loss, coeff_gate = optimize_with_optional_gate(
            raw_tree,
            evo.memory,
            args,
            gate_seed=args.seed + args.coeff_gate_seed_offset + generation * args.seed_stride,
        )
        pkl_path = artifact_dir / f"{prefix}_tree.pkl"
        atomic_pickle(pkl_path, optimized_tree)

        artifact = {
            "generation": generation,
            "training_fitness": float(raw_tree.fitness),
            "tree_size": int(len(optimized_tree)),
            "model_is_optimized": coeff_gate is None or bool(coeff_gate.get("kept_optimized", True)),
            "coefficient_optimization_last_loss": coeff_loss,
            "coefficient_gate": coeff_gate,
            "model_path": str(pkl_path),
        }
        if args.video:
            if args.save_raw_gifs:
                raw_gif_path = artifact_dir / f"{prefix}_before_optimization.gif"
                raw_gif_info = render_gif(
                    raw_tree,
                    raw_gif_path,
                    seed=args.seed + args.video_seed_offset + generation,
                    duration=args.video_duration,
                )
                raw_gif_info["path"] = str(raw_gif_path)
                artifact["raw_gif"] = raw_gif_info

            optimized_gif_path = artifact_dir / f"{prefix}_after_optimization.gif"
            optimized_gif_info = render_gif(
                optimized_tree,
                optimized_gif_path,
                seed=args.seed + args.video_seed_offset + generation,
                duration=args.video_duration,
            )
            optimized_gif_info["path"] = str(optimized_gif_path)
            artifact["optimized_gif"] = optimized_gif_info
            artifact["gif"] = optimized_gif_info
        artifacts.append(artifact)
    return artifacts


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
        trial_args.video = False
        trial_args.artifact_interval = 0

        _, _, _, _, _, evaluation = run_training(trial_args, run_dir=None)
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


def save_training_outputs(args, run_dir, raw_best, best, evo, coeff_loss, coeff_gate, evaluation, sweep_result=None):
    write_json(run_dir / "run_config.json", args_to_dict(args))
    # generation_history.csv and generation_artifacts are written incrementally during training
    csv_path = run_dir / "generation_history.csv"
    if not csv_path.exists():
        write_generation_history(evo, csv_path)
    generation_artifacts = evo.generation_artifacts if evo.generation_artifacts else save_generation_artifacts(args, run_dir, evo)

    with open(run_dir / "best_tree.txt", "w") as f:
        f.write(json.dumps(best.get_readable_repr(), indent=2))
        f.write("\n")

    atomic_pickle(run_dir / "best_tree.pkl", best)

    video_info = None
    if args.video:
        optimized_gif_path = run_dir / "best_lander.gif"
        optimized_video_info = render_gif(
            best,
            optimized_gif_path,
            seed=args.seed + args.video_seed_offset,
            duration=args.video_duration,
        )
        optimized_video_info["path"] = str(optimized_gif_path)
        video_info = {
            "path": str(optimized_gif_path),
            "optimized_gif": optimized_video_info,
        }
        if args.save_raw_gifs:
            raw_gif_path = run_dir / "best_lander_before_optimization.gif"
            raw_video_info = render_gif(
                raw_best,
                raw_gif_path,
                seed=args.seed + args.video_seed_offset,
                duration=args.video_duration,
            )
            raw_video_info["path"] = str(raw_gif_path)
            video_info["raw_gif"] = raw_video_info

    metrics = {
        "best_training_fitness": float(best.fitness),
        "best_training_fitness_before_optimization": float(raw_best.fitness),
        "best_model_source": "validation" if args.validation_selection and evo.best_validation is not None else "training",
        "best_model_generation": getattr(raw_best, "_best_generation", None),
        "best_validation_score": evo.best_validation_score,
        "best_model_is_optimized": coeff_gate is None or bool(coeff_gate.get("kept_optimized", True)),
        "coefficient_optimization_last_loss": coeff_loss,
        "coefficient_gate": coeff_gate,
        "test": evaluation,
        "video": video_info,
        "generation_artifacts": generation_artifacts,
        "validation_records": evo.validation_records,
        "interrupted": evo.interrupted,
        "stop_reason": evo.stop_reason,
        "sweep": sweep_result,
    }
    write_json(run_dir / "metrics.json", metrics)

    with open(run_dir / "summary.txt", "w") as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Interrupted: {evo.interrupted}\n")
        if evo.stop_reason:
            f.write(f"Stop reason: {evo.stop_reason}\n")
        f.write(f"Best model source: {metrics['best_model_source']}\n")
        if metrics["best_model_generation"] is not None:
            f.write(f"Best model generation: {metrics['best_model_generation']}\n")
        f.write(f"Best training fitness: {raw_best.fitness:.3f}\n")
        f.write(f"Best model saved after coefficient optimization: {metrics['best_model_is_optimized']}\n")
        if evo.best_validation_score is not None:
            f.write(f"Best validation score: {evo.best_validation_score:.3f}\n")
        f.write(f"Test total score: {evaluation['total_score']:.3f}\n")
        f.write(f"Test mean score: {evaluation['mean_score']:.3f}\n")
        f.write(f"Test score std: {evaluation['std_score']:.3f}\n")
        f.write(f"Test survival rate: {evaluation['survival_rate']:.3f}\n")
        f.write(f"Test crash rate: {evaluation['crash_rate']:.3f}\n")
        f.write(f"Generation artifacts: {len(generation_artifacts)} checkpoints\n")
        if video_info:
            if "raw_gif" in video_info:
                f.write(f"GIF before optimization: {video_info['raw_gif']['path']}\n")
            f.write(f"Best GIF: {video_info['optimized_gif']['path']}\n")


def run_retest(args):
    if not args.retest_dir:
        raise SystemExit("--retest-dir is required for --mode retest")
    run_dir = Path(args.retest_dir)
    pkl_path = run_dir / "best_tree.pkl"
    if not pkl_path.exists():
        raise SystemExit(f"No best_tree.pkl found in {run_dir}")
    with open(pkl_path, "rb") as f:
        best = pickle.load(f)
    print(f"Re-running final test on {run_dir.name} ({args.test_episodes} episodes)...")
    evaluation = evaluate_tree(best, episodes=args.test_episodes, duration=args.test_duration, seed=args.seed + 10_000)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics["test"] = evaluation
    write_json(metrics_path, metrics)
    print(f"Test total score: {evaluation['total_score']:.2f}  (survival rate: {evaluation['survival_rate']:.0%})")
    if args.plots:
        print("Regenerating plots...")
        from plot_run import generate_plots
        generate_plots(run_dir.name)


def main():
    args = parse_args()

    if args.mode == "retest":
        run_retest(args)
        env.close()
        return

    args = apply_original_baseline(args)
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

    raw_best, best, evo, coeff_loss, coeff_gate, evaluation = run_training(args, run_dir)
    save_training_outputs(args, run_dir, raw_best, best, evo, coeff_loss, coeff_gate, evaluation, sweep_result)
    print(f"Experiment complete. Results written to {run_dir}")
    print(f"Test total score: {evaluation['total_score']:.2f}")
    env.close()

    if args.plots:
        print("Generating plots...")
        from plot_run import generate_plots

        generate_plots(run_dir.name)


if __name__ == "__main__":
    main()
