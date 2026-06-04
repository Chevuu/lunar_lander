import argparse
import copy
import csv
import json
import os
import pickle
import random
import subprocess
import sys
import time
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
from genepro.node_impl import Constant, Feature
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
    add_arg(parser, "validation_episodes", defaults.get("validation_episodes", config.TEST_EPISODES), type=int)
    add_arg(parser, "validation_seed_offset", defaults.get("validation_seed_offset", 10_000), type=int)
    add_arg(parser, "test_seed_offset", defaults.get("test_seed_offset", 20_000), type=int)
    add_arg(parser, "video", defaults.get("video", True), action=argparse.BooleanOptionalAction)
    add_arg(parser, "video_duration", defaults.get("video_duration", config.TEST_EPISODE_DURATION), type=int)
    add_arg(parser, "video_seed_offset", defaults.get("video_seed_offset", 30_000), type=int)
    add_arg(parser, "artifact_interval", defaults.get("artifact_interval", 10), type=int)

    add_arg(parser, "random_seeds", defaults.get("random_seeds", config.RANDOM_SEEDS), action=argparse.BooleanOptionalAction)
    add_arg(parser, "crash_penalty", defaults.get("crash_penalty", config.CRASH_PENALTY), action=argparse.BooleanOptionalAction)
    add_arg(parser, "parsimony", defaults.get("parsimony", config.PARSIMONY), action=argparse.BooleanOptionalAction)
    add_arg(parser, "time_pressure", defaults.get("time_pressure", config.TIME_PRESSURE), action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--fitness-mode",
        choices=["total", "mean", "shaped"],
        default=defaults.get("fitness_mode", config.FITNESS_MODE),
        help="total keeps legacy summed reward; mean uses mean episode reward; shaped adds normalized crash/landing/std/time terms.",
    )
    add_arg(parser, "crash_penalty_weight", defaults.get("crash_penalty_weight", config.CRASH_PENALTY_WEIGHT), type=float)
    add_arg(parser, "landing_bonus_weight", defaults.get("landing_bonus_weight", config.LANDING_BONUS_WEIGHT), type=float)
    add_arg(parser, "std_penalty_weight", defaults.get("std_penalty_weight", config.STD_PENALTY_WEIGHT), type=float)
    add_arg(parser, "time_penalty_weight", defaults.get("time_penalty_weight", config.TIME_PENALTY_WEIGHT), type=float)
    add_arg(parser, "elite_count", defaults.get("elite_count", config.ELITE_COUNT), type=int)

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


def ensure_run_layout(run_dir):
    for name in ["evaluations", "models", "videos"]:
        (run_dir / name).mkdir(exist_ok=True)


def is_crash(terminated, final_reward):
    return bool(terminated and final_reward <= -100.0)


def is_landing(terminated, final_reward):
    return bool(terminated and final_reward >= 100.0)


def run_policy_episodes(tree, episodes, duration, seed, collect_memory=False, render=False, ignore_done=False):
    memory = ReplayMemory(config.REPLAY_MEMORY_SIZE) if collect_memory else None
    episode_rewards = []
    episode_lengths = []
    episode_outcomes = []
    episode_seeds = []
    crashed_episodes = 0
    landed_episodes = 0
    survived_episodes = 0
    terminated_episodes = 0
    truncated_episodes = 0

    for episode_idx in range(episodes):
        episode_seed = None if seed is None else seed + episode_idx
        episode_seeds.append(episode_seed)
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
        landed = is_landing(terminated, final_reward)
        timeout = bool(truncated and not terminated)
        outcome = "crash" if crashed else "landing" if landed else "timeout" if timeout else "other_termination" if terminated else "other"
        crashed_episodes += int(crashed)
        landed_episodes += int(landed)
        survived_episodes += int(not crashed)
        terminated_episodes += int(terminated)
        truncated_episodes += int(truncated)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_outcomes.append(outcome)

    total_episodes = len(episode_rewards)
    reward_std = float(np.std(episode_rewards)) if episode_rewards else 0.0
    stats = {
        "total_score": float(sum(episode_rewards)),
        "mean_score": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "median_score": float(np.median(episode_rewards)) if episode_rewards else 0.0,
        "std_score": reward_std,
        "min_score": float(np.min(episode_rewards)) if episode_rewards else 0.0,
        "max_score": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "episode_scores": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_outcomes": episode_outcomes,
        "episode_seeds": episode_seeds,
        "total_episodes": total_episodes,
        "survived_episodes": survived_episodes,
        "crashed_episodes": crashed_episodes,
        "landed_episodes": landed_episodes,
        "terminated_episodes": terminated_episodes,
        "truncated_episodes": truncated_episodes,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "median_episode_length": float(np.median(episode_lengths)) if episode_lengths else 0.0,
        "survival_rate": float(survived_episodes / total_episodes) if total_episodes else 0.0,
        "crash_rate": float(crashed_episodes / total_episodes) if total_episodes else 0.0,
        "landing_rate": float(landed_episodes / total_episodes) if total_episodes else 0.0,
        "timeout_rate": float(truncated_episodes / total_episodes) if total_episodes else 0.0,
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
        if config.FITNESS_MODE == "total":
            fitness = stats["total_score"]
        elif config.FITNESS_MODE == "mean":
            fitness = stats["mean_score"]
        elif config.FITNESS_MODE == "shaped":
            fitness = (
                stats["mean_score"]
                - config.CRASH_PENALTY_WEIGHT * stats["crash_rate"]
                + config.LANDING_BONUS_WEIGHT * stats["landing_rate"]
                - config.STD_PENALTY_WEIGHT * stats["std_score"]
                - config.TIME_PENALTY_WEIGHT * stats["mean_episode_length"]
            )
        else:
            raise ValueError(f"Unsupported fitness mode: {config.FITNESS_MODE}")

        # Legacy switches are kept for backwards-compatible experiment configs.
        if config.CRASH_PENALTY:
            fitness -= 100 * stats["crashed_episodes"]
        if config.PARSIMONY:
            fitness -= 1 * len(multitree)
        if config.TIME_PRESSURE:
            fitness -= 0.1 * sum(stats["episode_lengths"])
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
    config.RANDOM_SEEDS = args.random_seeds
    config.CRASH_PENALTY = args.crash_penalty
    config.PARSIMONY = args.parsimony
    config.TIME_PRESSURE = args.time_pressure
    config.FITNESS_MODE = args.fitness_mode
    config.CRASH_PENALTY_WEIGHT = args.crash_penalty_weight
    config.LANDING_BONUS_WEIGHT = args.landing_bonus_weight
    config.STD_PENALTY_WEIGHT = args.std_penalty_weight
    config.TIME_PENALTY_WEIGHT = args.time_penalty_weight
    config.ELITE_COUNT = args.elite_count


def build_leaf_nodes(num_constants):
    leaf_nodes = [Feature(i) for i in range(num_features)]
    leaf_nodes += [Constant() for _ in range(num_constants)]
    return leaf_nodes


def merge_memories(memories):
    memory = memories[0]
    for next_memory in memories[1:]:
        memory += next_memory
    return memory


def summarize_population(generation, population):
    fitnesses = np.array([float(t.fitness) for t in population], dtype=float)
    sizes = np.array([int(len(t)) for t in population], dtype=int)
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
        "best_fitness": float(np.max(fitnesses)),
        "mean_fitness": float(np.mean(fitnesses)),
        "median_fitness": float(np.median(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
        "best_tree_size": int(sizes[np.argmax(fitnesses)]),
        "mean_tree_size": float(np.mean(sizes)),
        "median_tree_size": float(np.median(sizes)),
        "min_tree_size": int(np.min(sizes)),
        "max_tree_size": int(np.max(sizes)),
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
        "median_episode_score": float(np.median(episode_scores)) if episode_scores else 0.0,
        "std_episode_score": float(np.std(episode_scores)) if episode_scores else 0.0,
    }


CSV_FIELDNAMES = [
    "generation", "best_fitness", "mean_fitness", "std_fitness",
    "median_fitness",
    "best_tree_size", "mean_tree_size", "median_tree_size", "min_tree_size", "max_tree_size", "population_size",
    "survived_agents", "crashed_agents", "agent_survival_rate", "agent_crash_rate",
    "total_episodes", "survived_episodes", "crashed_episodes",
    "episode_survival_rate", "episode_crash_rate", "mean_episode_score", "median_episode_score", "std_episode_score",
]


class ExperimentEvolution(Evolution):
    def __init__(self, *args, run_args=None, run_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_args = run_args
        self.run_dir = run_dir
        self.generation_records = []
        self.generation_artifacts = []
        self.elites = []
        self._progress_started = False

    def _progress_line(self, generation, record=None):
        if not self.max_gens:
            return f"gen {generation}"

        width = 28
        fraction = min(1.0, generation / self.max_gens)
        filled = int(round(width * fraction))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.time() - self.start_time
        pieces = [
            f"[{bar}]",
            f"{generation:>3}/{self.max_gens}",
            f"{fraction * 100:>5.1f}%",
            f"elapsed {elapsed:>6.1f}s",
        ]
        if record:
            pieces.extend([
                f"best {record['best_fitness']:.2f}",
                f"mean {record['mean_fitness']:.2f}",
                f"crash {record['episode_crash_rate'] * 100:.1f}%",
            ])
        return "  ".join(pieces)

    def _print_progress(self, generation):
        if not self.verbose:
            return
        record = self.generation_records[-1] if self.generation_records else None
        self._progress_started = True
        sys.stdout.write("\r" + self._progress_line(generation, record))
        sys.stdout.flush()

    def _finish_progress(self):
        if self.verbose and self._progress_started:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _evaluate_population(self, population):
        results = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in population)
        fitnesses, memories, episode_stats = list(map(list, zip(*results)))
        for individual, fitness, stats in zip(population, fitnesses, episode_stats):
            individual.fitness = fitness
            individual._episode_stats = stats
        return merge_memories(memories)

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

    def _elite_count(self):
        if self.run_args is None:
            return max(1, int(getattr(config, "ELITE_COUNT", 1)))
        return max(0, int(self.run_args.elite_count))

    def _update_elites(self, candidates):
        elite_count = self._elite_count()
        if elite_count <= 0:
            self.elites = []
            return

        pool = [copy.deepcopy(t) for t in self.elites]
        pool.extend(copy.deepcopy(t) for t in candidates)
        pool.sort(key=lambda t: t.fitness, reverse=True)
        self.elites = pool[:elite_count]

    def _fresh_elites_for_generation(self):
        elites = [copy.deepcopy(t) for t in self.elites]
        if config.RANDOM_SEEDS:
            for elite in elites:
                elite.fitness, _, elite._episode_stats = self.fitness_function(elite)
        return elites

    def _write_checkpoint(self, generation):
        if self.run_dir is None or self.run_args is None:
            return
        args = self.run_args
        tree = self.best_of_gens[generation]
        artifact_dir = self.run_dir / "generation_artifacts"
        artifact_dir.mkdir(exist_ok=True)

        prefix = f"generation_{generation:04d}"
        raw_tree = copy.deepcopy(tree)
        optimized_tree = copy.deepcopy(tree)
        coeff_loss = optimize_constants(optimized_tree, self.memory, args)

        pkl_path = artifact_dir / f"{prefix}_tree.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(optimized_tree, f)

        artifact = {
            "generation": generation,
            "training_fitness": float(raw_tree.fitness),
            "tree_size": int(len(optimized_tree)),
            "model_is_optimized": True,
            "coefficient_optimization_last_loss": coeff_loss,
            "model_path": str(pkl_path),
        }
        if args.video:
            raw_gif_path = artifact_dir / f"{prefix}_before_optimization.gif"
            raw_gif_info = render_gif(raw_tree, raw_gif_path, seed=args.seed + args.video_seed_offset + generation, duration=args.video_duration)
            raw_gif_info["path"] = str(raw_gif_path)
            optimized_gif_path = artifact_dir / f"{prefix}_after_optimization.gif"
            optimized_gif_info = render_gif(optimized_tree, optimized_gif_path, seed=args.seed + args.video_seed_offset + generation, duration=args.video_duration)
            optimized_gif_info["path"] = str(optimized_gif_path)
            artifact["raw_gif"] = raw_gif_info
            artifact["optimized_gif"] = optimized_gif_info
            artifact["gif"] = optimized_gif_info

        self.generation_artifacts.append(artifact)
        if self.verbose:
            sys.stdout.write("\n")
            print(f"  checkpoint written: {prefix}")
            self._print_progress(generation)

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
        self._update_elites(self.population)
        record = summarize_population(0, self.population)
        self.generation_records.append(record)
        self._append_csv_row(record)
        self._print_progress(0)

    def _perform_generation(self):
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

        elites = self._fresh_elites_for_generation()
        if elites:
            worst_indices = np.argsort([t.fitness for t in offspring_population])[:len(elites)]
            for worst_idx, elite_candidate in zip(worst_indices, elites):
                offspring_population[int(worst_idx)] = elite_candidate

        self.population = offspring_population
        self.num_gens += 1
        best = self.population[np.argmax([t.fitness for t in self.population])]
        self.best_of_gens.append(copy.deepcopy(best))

        self._update_elites(self.population)
        record = summarize_population(self.num_gens, self.population)
        self.generation_records.append(record)
        self._append_csv_row(record)

        if self.run_args and self.run_args.artifact_interval > 0 and self.num_gens % self.run_args.artifact_interval == 0:
            self._write_checkpoint(self.num_gens)

    def evolve(self):
        self.start_time = time.time()
        self._initialize_population()

        while not self._must_terminate():
            self._perform_generation()
            self._print_progress(self.num_gens)

        self._finish_progress()


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


def evaluate_tree(tree, episodes, duration, seed, label=None):
    stats, _ = run_policy_episodes(
        tree,
        episodes=episodes,
        duration=duration,
        seed=seed,
        collect_memory=False,
    )
    stats["label"] = label
    stats["seed_start"] = seed
    stats["seed_count"] = episodes
    stats["duration"] = duration
    return stats


def evaluate_selected_models(raw_best, optimized_best, args):
    validation_seed = args.seed + args.validation_seed_offset
    test_seed = args.seed + args.test_seed_offset
    return {
        "validation": {
            "raw": evaluate_tree(
                raw_best,
                episodes=args.validation_episodes,
                duration=args.test_duration,
                seed=validation_seed,
                label="validation_raw",
            ),
            "optimized": evaluate_tree(
                optimized_best,
                episodes=args.validation_episodes,
                duration=args.test_duration,
                seed=validation_seed,
                label="validation_optimized",
            ),
        },
        "test": {
            "raw": evaluate_tree(
                raw_best,
                episodes=args.test_episodes,
                duration=args.test_duration,
                seed=test_seed,
                label="test_raw",
            ),
            "optimized": evaluate_tree(
                optimized_best,
                episodes=args.test_episodes,
                duration=args.test_duration,
                seed=test_seed,
                label="test_optimized",
            ),
        },
    }


def run_training(args, run_dir=None):
    update_runtime_config(args)
    set_seed(args.seed)
    leaf_nodes = build_leaf_nodes(args.num_constants)
    fitness_fn = make_fitness_fn(args.num_episodes, args.episode_duration, args.seed)

    evo = ExperimentEvolution(
        fitness_fn,
        config.INTERNAL_NODES,
        leaf_nodes,
        config.NUM_TREES,
        pop_size=args.pop_size,
        max_gens=args.max_gens,
        max_tree_size=args.max_tree_size,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        run_args=args,
        run_dir=run_dir,
    )
    evo.evolve()

    raw_best = copy.deepcopy(max(evo.best_of_gens, key=lambda t: t.fitness))
    best = copy.deepcopy(raw_best)
    coeff_loss = optimize_constants(best, evo.memory, args)
    evaluations = evaluate_selected_models(raw_best, best, args)
    return raw_best, best, evo, coeff_loss, evaluations


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
        optimized_tree = copy.deepcopy(tree)
        coeff_loss = optimize_constants(optimized_tree, evo.memory, args)
        pkl_path = artifact_dir / f"{prefix}_tree.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(optimized_tree, f)

        artifact = {
            "generation": generation,
            "training_fitness": float(raw_tree.fitness),
            "tree_size": int(len(optimized_tree)),
            "model_is_optimized": True,
            "coefficient_optimization_last_loss": coeff_loss,
            "model_path": str(pkl_path),
        }
        if args.video:
            raw_gif_path = artifact_dir / f"{prefix}_before_optimization.gif"
            raw_gif_info = render_gif(
                raw_tree,
                raw_gif_path,
                seed=args.seed + args.video_seed_offset + generation,
                duration=args.video_duration,
            )
            raw_gif_info["path"] = str(raw_gif_path)

            optimized_gif_path = artifact_dir / f"{prefix}_after_optimization.gif"
            optimized_gif_info = render_gif(
                optimized_tree,
                optimized_gif_path,
                seed=args.seed + args.video_seed_offset + generation,
                duration=args.video_duration,
            )
            optimized_gif_info["path"] = str(optimized_gif_path)
            artifact["raw_gif"] = raw_gif_info
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

        _, _, _, _, evaluations = run_training(trial_args)
        score = evaluations["validation"]["optimized"]["mean_score"]
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


def write_evaluation_files(run_dir, evaluations):
    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir(exist_ok=True)
    for split, models in evaluations.items():
        for model_name, stats in models.items():
            write_json(eval_dir / f"{split}_{model_name}.json", stats)


def comparison_metrics(evaluations):
    raw_test = evaluations["test"]["raw"]
    opt_test = evaluations["test"]["optimized"]
    raw_val = evaluations["validation"]["raw"]
    opt_val = evaluations["validation"]["optimized"]
    return {
        "raw_vs_optimized_test_mean_delta": opt_test["mean_score"] - raw_test["mean_score"],
        "raw_vs_optimized_test_crash_rate_delta": opt_test["crash_rate"] - raw_test["crash_rate"],
        "raw_vs_optimized_validation_mean_delta": opt_val["mean_score"] - raw_val["mean_score"],
        "raw_vs_optimized_validation_crash_rate_delta": opt_val["crash_rate"] - raw_val["crash_rate"],
        "optimized_validation_vs_test_mean_delta": opt_test["mean_score"] - opt_val["mean_score"],
        "raw_validation_vs_test_mean_delta": raw_test["mean_score"] - raw_val["mean_score"],
    }


def write_summary_markdown(run_dir, args, raw_best, best, coeff_loss, evaluations):
    selection = {
        "criterion": "best_training_fitness",
        "raw_training_fitness": float(raw_best.fitness),
        "raw_tree_size": int(len(raw_best)),
        "optimized_tree_size": int(len(best)),
    }
    rows = []
    for split in ["validation", "test"]:
        for model_name in ["raw", "optimized"]:
            stats = evaluations[split][model_name]
            rows.append(
                "| {split} | {model} | {mean:.3f} | {median:.3f} | {std:.3f} | {crash:.3f} | {landing:.3f} | {timeout:.3f} |".format(
                    split=split,
                    model=model_name,
                    mean=stats["mean_score"],
                    median=stats["median_score"],
                    std=stats["std_score"],
                    crash=stats["crash_rate"],
                    landing=stats["landing_rate"],
                    timeout=stats["timeout_rate"],
                )
            )

    with open(run_dir / "summary.md", "w") as f:
        f.write(f"# {args.experiment_name}\n\n")
        f.write("## Selection\n")
        f.write(f"- Seed: {args.seed}\n")
        f.write(f"- Criterion: {selection['criterion']}\n")
        f.write(f"- Raw training fitness: {selection['raw_training_fitness']:.3f}\n")
        f.write(f"- Raw tree size: {selection['raw_tree_size']}\n")
        f.write(f"- Optimized tree size: {selection['optimized_tree_size']}\n")
        f.write(f"- Coefficient optimization last loss: {coeff_loss}\n\n")
        f.write("## Algorithm Settings\n")
        f.write(f"- Fitness mode: {args.fitness_mode}\n")
        f.write(f"- Crash penalty weight: {args.crash_penalty_weight}\n")
        f.write(f"- Landing bonus weight: {args.landing_bonus_weight}\n")
        f.write(f"- Std penalty weight: {args.std_penalty_weight}\n")
        f.write(f"- Time penalty weight: {args.time_penalty_weight}\n")
        f.write(f"- Elite count: {args.elite_count}\n")
        f.write(f"- Random training seeds: {args.random_seeds}\n\n")
        f.write("## Evaluation\n")
        f.write("| Split | Model | Mean | Median | Std | Crash rate | Landing rate | Timeout rate |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        f.write("\n".join(rows))
        f.write("\n")


def save_training_outputs(args, run_dir, raw_best, best, evo, coeff_loss, evaluations, sweep_result=None):
    ensure_run_layout(run_dir)
    write_json(run_dir / "run_config.json", args_to_dict(args))
    write_json(run_dir / "resolved_config.json", args_to_dict(args))
    # generation_history.csv and generation_artifacts are written incrementally during training
    csv_path = run_dir / "generation_history.csv"
    if not csv_path.exists():
        write_generation_history(evo, csv_path)
    generation_artifacts = evo.generation_artifacts if evo.generation_artifacts else save_generation_artifacts(args, run_dir, evo)
    write_evaluation_files(run_dir, evaluations)

    with open(run_dir / "best_tree.txt", "w") as f:
        f.write(json.dumps(best.get_readable_repr(), indent=2))
        f.write("\n")

    with open(run_dir / "best_tree.pkl", "wb") as f:
        pickle.dump(best, f)
    with open(run_dir / "models" / "selected_raw.pkl", "wb") as f:
        pickle.dump(raw_best, f)
    with open(run_dir / "models" / "selected_optimized.pkl", "wb") as f:
        pickle.dump(best, f)
    with open(run_dir / "models" / "selected_optimized.txt", "w") as f:
        f.write(json.dumps(best.get_readable_repr(), indent=2))
        f.write("\n")

    video_info = None
    if args.video:
        raw_gif_path = run_dir / "videos" / "best_lander_before_optimization.gif"
        raw_video_info = render_gif(
            raw_best,
            raw_gif_path,
            seed=args.seed + args.video_seed_offset,
            duration=args.video_duration,
        )
        raw_video_info["path"] = str(raw_gif_path)

        optimized_gif_path = run_dir / "videos" / "best_lander.gif"
        optimized_video_info = render_gif(
            best,
            optimized_gif_path,
            seed=args.seed + args.video_seed_offset,
            duration=args.video_duration,
        )
        optimized_video_info["path"] = str(optimized_gif_path)
        video_info = {
            "path": str(optimized_gif_path),
            "raw_gif": raw_video_info,
            "optimized_gif": optimized_video_info,
        }
        # Keep old top-level paths for existing notebooks and plots.
        legacy_raw = run_dir / "best_lander_before_optimization.gif"
        legacy_opt = run_dir / "best_lander.gif"
        if not legacy_raw.exists():
            legacy_raw.write_bytes(raw_gif_path.read_bytes())
        if not legacy_opt.exists():
            legacy_opt.write_bytes(optimized_gif_path.read_bytes())

    selection = {
        "criterion": "best_training_fitness",
        "raw_training_fitness": float(raw_best.fitness),
        "raw_tree_size": int(len(raw_best)),
        "optimized_tree_size": int(len(best)),
    }
    post_processing = {
        "coefficient_optimization": {
            "enabled": True,
            "steps": args.coeff_opt_steps,
            "learning_rate": args.coeff_lr,
            "last_loss": coeff_loss,
        }
    }
    comparisons = comparison_metrics(evaluations)
    metrics = {
        "best_training_fitness": float(best.fitness),
        "best_training_fitness_before_optimization": float(raw_best.fitness),
        "best_model_is_optimized": True,
        "coefficient_optimization_last_loss": coeff_loss,
        "selection": selection,
        "post_processing": post_processing,
        "evaluations": evaluations,
        "comparisons": comparisons,
        "algorithm": {
            "fitness_mode": args.fitness_mode,
            "crash_penalty_weight": args.crash_penalty_weight,
            "landing_bonus_weight": args.landing_bonus_weight,
            "std_penalty_weight": args.std_penalty_weight,
            "time_penalty_weight": args.time_penalty_weight,
            "elite_count": args.elite_count,
            "random_seeds": args.random_seeds,
        },
        "test": evaluations["test"]["optimized"],
        "video": video_info,
        "generation_artifacts": generation_artifacts,
        "sweep": sweep_result,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", {
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "selection": selection,
        "post_processing": post_processing,
        "evaluation_files": {
            "validation_raw": "evaluations/validation_raw.json",
            "validation_optimized": "evaluations/validation_optimized.json",
            "test_raw": "evaluations/test_raw.json",
            "test_optimized": "evaluations/test_optimized.json",
        },
        "model_files": {
            "raw": "models/selected_raw.pkl",
            "optimized": "models/selected_optimized.pkl",
        },
        "algorithm": {
            "fitness_mode": args.fitness_mode,
            "crash_penalty_weight": args.crash_penalty_weight,
            "landing_bonus_weight": args.landing_bonus_weight,
            "std_penalty_weight": args.std_penalty_weight,
            "time_penalty_weight": args.time_penalty_weight,
            "elite_count": args.elite_count,
            "random_seeds": args.random_seeds,
        },
    })
    write_summary_markdown(run_dir, args, raw_best, best, coeff_loss, evaluations)

    with open(run_dir / "summary.txt", "w") as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Best training fitness: {raw_best.fitness:.3f}\n")
        f.write(f"Fitness mode: {args.fitness_mode}\n")
        f.write(f"Elite count: {args.elite_count}\n")
        f.write("Best model saved after coefficient optimization: yes\n")
        for model_name in ["raw", "optimized"]:
            test = evaluations["test"][model_name]
            f.write(f"Test {model_name} mean score: {test['mean_score']:.3f}\n")
            f.write(f"Test {model_name} median score: {test['median_score']:.3f}\n")
            f.write(f"Test {model_name} score std: {test['std_score']:.3f}\n")
            f.write(f"Test {model_name} crash rate: {test['crash_rate']:.3f}\n")
            f.write(f"Test {model_name} landing rate: {test['landing_rate']:.3f}\n")
        f.write(f"Generation artifacts: {len(generation_artifacts)} checkpoints\n")
        if video_info:
            f.write(f"GIF before optimization: {video_info['raw_gif']['path']}\n")
            f.write(f"GIF after optimization: {video_info['optimized_gif']['path']}\n")


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

    raw_best, best, evo, coeff_loss, evaluations = run_training(args, run_dir)
    save_training_outputs(args, run_dir, raw_best, best, evo, coeff_loss, evaluations, sweep_result)
    print(f"Experiment complete. Results written to {run_dir}")
    print(f"Optimized test mean score: {evaluations['test']['optimized']['mean_score']:.2f}")
    print(f"Optimized test crash rate: {evaluations['test']['optimized']['crash_rate']:.1%}")
    env.close()

    print("Generating plots...")
    import sys
    subprocess.run([sys.executable, str(Path(__file__).parent / "plot_run.py"), run_dir.name], check=False)


if __name__ == "__main__":
    main()
