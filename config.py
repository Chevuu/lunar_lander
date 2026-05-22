from genepro.node_impl import Plus, Minus, Times, Div, Sin, Cos, Exp, Log, Sqrt

NUM_EPISODES = 15          # episodes per fitness evaluation; more = less noisy signal
EPISODE_DURATION = 500     # max steps per episode before forced termination
REPLAY_MEMORY_SIZE = 10000 # max transitions stored; oldest are dropped when full

NUM_TREES = 4              # one tree per action (LunarLander has 4 discrete actions)
POP_SIZE = 64              # number of individuals in the population
MAX_GENS = 50              # number of generations to run
MAX_TREE_SIZE = 63         # max nodes per tree; larger = more complex expressions
N_JOBS = 8                 # parallel workers for fitness evaluation
VERBOSE = True             # print progress each generation
NUM_CONSTANTS = 3          # constant leaf nodes added alongside features (~27% sample rate)

INTERNAL_NODES = [Plus(), Minus(), Times(), Div(), Sin(), Cos(), Exp(), Log(), Sqrt()]  # allowed operators

COEFF_OPT_STEPS = 1000     # gradient update steps on best tree after evolution
COEFF_LR = 1e-3            # AdamW learning rate for constant optimisation
BATCH_SIZE = 128           # transitions sampled per gradient step
GAMMA = 0.99               # discount factor for future rewards
GRAD_CLIP = 100            # gradient clipping value to prevent exploding gradients

TEST_EPISODES = 10         # episodes used to compute final test score
TEST_EPISODE_DURATION = 500  # max steps per test episode
