# pip install optuna
import optuna
import copy
import csv
import os
import random
from collections import namedtuple, deque

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from genepro.node_impl import Feature, Constant, Plus, Minus, Times, Div, Sin, Cos, Exp, Log, Sqrt
from genepro.evo import Evolution

import config
import sweep_config

env = gym.make("LunarLander-v2", render_mode="rgb_array")
num_features = env.observation_space.shape[0]

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        return self

    def __add__(self, other):
        self.memory = self.memory + other.memory
        return self


def make_fitness_fn(num_episodes, episode_duration):
    def fitness_function_pt(multitree, num_episodes=num_episodes, episode_duration=episode_duration, render=False, ignore_done=False):
        memory = ReplayMemory(config.REPLAY_MEMORY_SIZE)
        rewards = []
        for _ in range(num_episodes):
            observation = env.reset()[0]
            for _ in range(episode_duration):
                input_sample = torch.from_numpy(observation.reshape((1, -1))).float()
                action = torch.argmax(multitree.get_output_pt(input_sample))
                observation, reward, terminated, truncated, _ = env.step(action.item())
                rewards.append(reward)
                output_sample = torch.from_numpy(observation.reshape((1, -1))).float()
                memory.push(input_sample, torch.tensor([[action.item()]]), output_sample, torch.tensor([reward]))
                if (terminated or truncated) and not ignore_done:
                    break
        return np.sum(rewards), memory
    return fitness_function_pt


def run_trial(pop_size, max_tree_size, num_constants, coeff_lr, coeff_opt_steps, gamma,
              max_gens, num_episodes):
    leaf_nodes = [Feature(i) for i in range(num_features)]
    leaf_nodes += [Constant() for _ in range(num_constants)]
    internal_nodes = [Plus(), Minus(), Times(), Div(), Sin(), Cos(), Exp(), Log(), Sqrt()]

    fitness_fn = make_fitness_fn(num_episodes, config.EPISODE_DURATION)

    evo = Evolution(
        fitness_fn, internal_nodes, leaf_nodes,
        config.NUM_TREES,
        pop_size=pop_size,
        max_gens=max_gens,
        max_tree_size=max_tree_size,
        n_jobs=config.N_JOBS,
        verbose=False,
    )
    evo.evolve()

    best = evo.best_of_gens[-1]
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

    # fixed seeds so every trial is scored on the same episodes
    rewards = []
    for i in range(config.TEST_EPISODES):
        observation = env.reset(seed=i)[0]
        for _ in range(config.TEST_EPISODE_DURATION):
            input_sample = torch.from_numpy(observation.reshape((1, -1))).float()
            action = torch.argmax(best.get_output_pt(input_sample))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            if terminated or truncated:
                break

    return np.sum(rewards), best


def objective(trial):
    print(f"Trial {trial.number + 1}/{sweep_config.N_TRIALS} starting...")
    pop_size = trial.suggest_categorical("pop_size", sweep_config.POP_SIZE_OPTIONS)
    max_tree_size = trial.suggest_categorical("max_tree_size", sweep_config.MAX_TREE_SIZE_OPTIONS)
    num_constants = trial.suggest_int("num_constants", sweep_config.NUM_CONSTANTS_MIN, sweep_config.NUM_CONSTANTS_MAX)
    coeff_lr = trial.suggest_float("coeff_lr", sweep_config.COEFF_LR_MIN, sweep_config.COEFF_LR_MAX, log=True)
    coeff_opt_steps = trial.suggest_categorical("coeff_opt_steps", sweep_config.COEFF_OPT_STEPS_OPTIONS)
    gamma = trial.suggest_float("gamma", sweep_config.GAMMA_MIN, sweep_config.GAMMA_MAX)

    score, _ = run_trial(
        pop_size, max_tree_size, num_constants, coeff_lr, coeff_opt_steps, gamma,
        max_gens=sweep_config.SWEEP_GENS,
        num_episodes=sweep_config.SWEEP_EPISODES,
    )

    completed = [t.value for t in trial.study.trials if t.value is not None]
    best_so_far = max(completed) if completed else score
    print(f"Trial {trial.number + 1}/{sweep_config.N_TRIALS} | score: {score:.1f} | best so far: {best_so_far:.1f}")
    print(f"  pop={pop_size}, tree_size={max_tree_size}, constants={num_constants}, lr={coeff_lr:.5f}, opt_steps={coeff_opt_steps}, gamma={gamma:.4f}")

    write_header = not os.path.exists(sweep_config.RESULTS_FILE)
    with open(sweep_config.RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["trial", "score", "pop_size", "max_tree_size",
                             "num_constants", "coeff_lr", "coeff_opt_steps", "gamma"])
        writer.writerow([trial.number, score, pop_size, max_tree_size,
                         num_constants, coeff_lr, coeff_opt_steps, gamma])

    return score


if __name__ == "__main__":
    if not sweep_config.RESUME and os.path.exists(sweep_config.DB_FILE):
        os.remove(sweep_config.DB_FILE)

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
    study.optimize(objective, n_trials=remaining)

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
