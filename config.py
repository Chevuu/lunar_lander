from genepro.node_impl import Plus, Minus, Times, Div, Sin, Cos, Exp, Log, Sqrt, Max, Min

SEED = 42                  # fixed seed for reproducible experiments
NUM_EPISODES = 15          # episodes per fitness evaluation; more = less noisy signal
EPISODE_DURATION = 500     # max steps per episode before forced termination
REPLAY_MEMORY_SIZE = 10000 # max transitions stored; oldest are dropped when full

NUM_TREES = 4              # one tree per action (LunarLander has 4 discrete actions)
POP_SIZE = 128             # number of individuals in the population; best from sweep
MAX_GENS = 35              # number of generations to run
MAX_TREE_SIZE = 63         # max nodes per tree; best from sweep
N_JOBS = 8                 # parallel workers for fitness evaluation
VERBOSE = True             # print progress each generation
NUM_CONSTANTS = 2          # constant leaf nodes added alongside features; best from sweep

INTERNAL_NODES = [Plus(), Minus(), Times(), Div(), Sin(), Cos(), Exp(), Log(), Sqrt(), Max(), Min()]  # allowed operators

COEFF_OPT_STEPS = 250      # gradient update steps on best tree after evolution; best from sweep
COEFF_LR = 1.69e-4         # AdamW learning rate for constant optimisation; best from sweep
BATCH_SIZE = 128           # transitions sampled per gradient step
GAMMA = 0.9796             # discount factor for future rewards; best from sweep
GRAD_CLIP = 100            # gradient clipping value to prevent exploding gradients

TEST_EPISODES = 10         # episodes used to compute final test score
TEST_EPISODE_DURATION = 500  # max steps per test episode
