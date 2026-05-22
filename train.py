import gymnasium as gym

from genepro.node_impl import Feature, Constant
from genepro.evo import Evolution

import torch
import torch.optim as optim

import random
import copy
from collections import namedtuple, deque

import numpy as np

import config

env = gym.make("LunarLander-v2", render_mode="rgb_array")

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


def fitness_function_pt(multitree, num_episodes=config.NUM_EPISODES, episode_duration=config.EPISODE_DURATION, render=False, ignore_done=False):
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


num_features = env.observation_space.shape[0]
leaf_nodes = [Feature(i) for i in range(num_features)]
leaf_nodes += [Constant() for _ in range(config.NUM_CONSTANTS)]

evo = Evolution(
    fitness_function_pt, config.INTERNAL_NODES, leaf_nodes,
    config.NUM_TREES,
    pop_size=config.POP_SIZE,
    max_gens=config.MAX_GENS,
    max_tree_size=config.MAX_TREE_SIZE,
    n_jobs=config.N_JOBS,
    verbose=config.VERBOSE,
)

evo.evolve()

best = evo.best_of_gens[-1]
constants = best.get_subtrees_consts()

if len(constants) > 0 and len(evo.memory) > config.BATCH_SIZE:
    optimizer = optim.AdamW(constants, lr=config.COEFF_LR, amsgrad=True)
    for _ in range(config.COEFF_OPT_STEPS):
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

            expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch
            loss = torch.nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(constants, config.GRAD_CLIP)
            optimizer.step()


def get_test_score(tree):
    rewards = []
    for i in range(config.TEST_EPISODES):
        observation = env.reset(seed=i)[0]
        for _ in range(config.TEST_EPISODE_DURATION):
            input_sample = torch.from_numpy(observation.reshape((1, -1))).float()
            action = torch.argmax(tree.get_output_pt(input_sample))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            if terminated or truncated:
                break
    return np.sum(rewards)


print(best.get_readable_repr())
print(f"Test score: {get_test_score(best)}")

env.close()
