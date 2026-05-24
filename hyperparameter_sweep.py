# pip install optuna
import argparse
import copy
import csv
import os
from datetime import datetime

import numpy as np
import optuna
import torch
import torch.optim as optim

from genepro.node_impl import Feature, Constant
from genepro.evo import Evolution

import config
import sweep_config
from train import env, num_features, Transition, ReplayMemory, fitness_function_pt, get_test_score


def make_fitness_fn(num_episodes, episode_duration):
    def fn(multitree, render=False, ignore_done=False):
        return fitness_function_pt(multitree, num_episodes=num_episodes,
                                   episode_duration=episode_duration,
                                   render=render, ignore_done=ignore_done)
    return fn


def run_trial(pop_size, max_tree_size, num_constants, coeff_lr, coeff_opt_steps, gamma,
              max_gens, num_episodes):
    leaf_nodes = [Feature(i) for i in range(num_features)]
    leaf_nodes += [Constant() for _ in range(num_constants)]

    fitness_fn = make_fitness_fn(num_episodes, config.EPISODE_DURATION)

    evo = Evolution(
        fitness_fn, config.INTERNAL_NODES, leaf_nodes,
        config.NUM_TREES,
        pop_size=pop_size,
        max_gens=max_gens,
        max_tree_size=max_tree_size,
        n_jobs=config.N_JOBS,
        verbose=False,
    )
    evo.evolve()

    best = max(evo.best_of_gens, key=lambda t: t.fitness)
    constants = best.get_subtrees_consts()

    if len(constants) > 0 and len(evo.memory) > config.BATCH_SIZE:
        optimizer = optim.AdamW(constants, lr=coeff_lr, amsgrad=True)
        for _ in range(coeff_opt_steps):
            if len(evo.memory) > config.BATCH_SIZE:
                target_tree = copy.deepcopy(best)
                transitions = evo.memory.sample(config.BATCH_SIZE)
                batch_data = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch_data.next_state)), dtype=torch.bool
                )
                non_final_next_states = torch.cat([s for s in batch_data.next_state if s is not None])
                state_batch = torch.cat(batch_data.state)
                action_batch = torch.cat(batch_data.action)
                reward_batch = torch.cat(batch_data.reward)

                state_action_values = best.get_output_pt(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(config.BATCH_SIZE, dtype=torch.float)
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_tree.get_output_pt(non_final_next_states).max(1)[0].float()

                expected = (next_state_values * gamma) + reward_batch
                loss = torch.nn.SmoothL1Loss()(state_action_values, expected.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(constants, config.GRAD_CLIP)
                optimizer.step()

    return get_test_score(best), best


def objective(trial, results_file=sweep_config.RESULTS_FILE):
    print(f"Trial {trial.number + 1}/{sweep_config.N_TRIALS} starting...")
    pop_size        = trial.suggest_categorical("pop_size", sweep_config.POP_SIZE_OPTIONS)
    max_tree_size   = trial.suggest_categorical("max_tree_size", sweep_config.MAX_TREE_SIZE_OPTIONS)
    num_constants   = trial.suggest_int("num_constants", sweep_config.NUM_CONSTANTS_MIN, sweep_config.NUM_CONSTANTS_MAX)
    coeff_lr        = trial.suggest_float("coeff_lr", sweep_config.COEFF_LR_MIN, sweep_config.COEFF_LR_MAX, log=True)
    coeff_opt_steps = trial.suggest_categorical("coeff_opt_steps", sweep_config.COEFF_OPT_STEPS_OPTIONS)
    gamma           = trial.suggest_float("gamma", sweep_config.GAMMA_MIN, sweep_config.GAMMA_MAX)

    score, _ = run_trial(
        pop_size, max_tree_size, num_constants, coeff_lr, coeff_opt_steps, gamma,
        max_gens=sweep_config.SWEEP_GENS,
        num_episodes=sweep_config.SWEEP_EPISODES,
    )

    completed = [t.value for t in trial.study.trials if t.value is not None]
    best_so_far = max(completed) if completed else score
    print(f"Trial {trial.number + 1}/{sweep_config.N_TRIALS} | score: {score:.1f} | best so far: {best_so_far:.1f}")
    print(f"  pop={pop_size}, tree_size={max_tree_size}, constants={num_constants}, lr={coeff_lr:.5f}, opt_steps={coeff_opt_steps}, gamma={gamma:.4f}")

    write_header = not os.path.exists(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["trial", "score", "pop_size", "max_tree_size",
                             "num_constants", "coeff_lr", "coeff_opt_steps", "gamma"])
        writer.writerow([trial.number, score, pop_size, max_tree_size,
                         num_constants, coeff_lr, coeff_opt_steps, gamma])

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for GP lunar lander")
    parser.add_argument("--n-trials",       type=int,  default=sweep_config.N_TRIALS)
    parser.add_argument("--sweep-gens",     type=int,  default=sweep_config.SWEEP_GENS)
    parser.add_argument("--sweep-episodes", type=int,  default=sweep_config.SWEEP_EPISODES)
    parser.add_argument("--n-jobs",         type=int,  default=config.N_JOBS)
    parser.add_argument("--resume",         action="store_true",  default=sweep_config.RESUME)
    parser.add_argument("--no-resume",      action="store_false", dest="resume")
    args = parser.parse_args()

    sweep_config.N_TRIALS       = args.n_trials
    sweep_config.SWEEP_GENS     = args.sweep_gens
    sweep_config.SWEEP_EPISODES = args.sweep_episodes
    sweep_config.RESUME         = args.resume
    config.N_JOBS               = args.n_jobs

    if not sweep_config.RESUME and os.path.exists(sweep_config.DB_FILE):
        os.remove(sweep_config.DB_FILE)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"sweep_results_{timestamp}.csv" if not sweep_config.RESUME else sweep_config.RESULTS_FILE

    study = optuna.create_study(
        study_name=sweep_config.STUDY_NAME,
        storage=f"sqlite:///{sweep_config.DB_FILE}",
        load_if_exists=sweep_config.RESUME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    completed = len(study.trials)
    remaining = max(0, sweep_config.N_TRIALS - completed)
    print(f"Resuming from trial {completed + 1}, {remaining} trials remaining.")
    study.optimize(lambda trial: objective(trial, results_file), n_trials=remaining)

    best = study.best_params
    print("\nBest params found:", best)
    print(f"Best sweep score: {study.best_value:.1f}")

    print("\nRe-running best config at full budget...")
    score, best_tree = run_trial(
        pop_size=best["pop_size"],
        max_tree_size=best["max_tree_size"],
        num_constants=best["num_constants"],
        coeff_lr=best["coeff_lr"],
        coeff_opt_steps=best["coeff_opt_steps"],
        gamma=best["gamma"],
        max_gens=config.MAX_GENS,
        num_episodes=config.NUM_EPISODES,
    )
    print(f"Full-budget score: {score:.1f}")
    print("Best tree:", best_tree.get_readable_repr())

    env.close()
