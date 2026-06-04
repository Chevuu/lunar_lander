from genepro.node_impl import Plus, Minus, Times, Div, Sin, Cos, Exp, Log, Sqrt, Max, Min

SEED = 142                 # fixed seed used for the selected long run
NUM_EPISODES = 20          # episodes per fitness evaluation; more = less noisy signal
EPISODE_DURATION = 500     # max steps per episode before forced termination
REPLAY_MEMORY_SIZE = 10000 # max transitions stored; oldest are dropped when full

NUM_TREES = 4              # one tree per action (LunarLander has 4 discrete actions)
POP_SIZE = 128             # number of individuals in the population
MAX_GENS = 200             # number of generations to run
MAX_TREE_SIZE = 63         # max nodes per tree
N_JOBS = 8                 # parallel workers for fitness evaluation
VERBOSE = True             # print progress each generation
NUM_CONSTANTS = 2          # constant leaf nodes added alongside features

INTERNAL_NODES = [Plus(), Minus(), Times(), Div(), Sin(), Cos(), Exp(), Log(), Sqrt(), Max(), Min()]  # allowed operators

COEFF_OPT_STEPS = 250      # gradient update steps on best tree after evolution
COEFF_LR = 1.69e-4         # AdamW learning rate for constant optimisation
BATCH_SIZE = 128           # transitions sampled per gradient step
GAMMA = 0.9796             # discount factor for future rewards
GRAD_CLIP = 100            # gradient clipping value to prevent exploding gradients

TEST_EPISODES = 30         # episodes used to compute final test score
TEST_EPISODE_DURATION = 500  # max steps per test episode
VIDEO = True               # render final and checkpoint GIFs
SAVE_RAW_GIFS = False      # skip before-optimization GIFs by default
PLOTS = True               # generate plots after training
VIDEO_EPISODE_DURATION = 500
VIDEO_SEED_OFFSET = 20_000
ARTIFACT_INTERVAL = 10     # save generation artifacts every n generations

RANDOM_SEEDS = True        # randomise episode seeds per evaluation to prevent fixed-scenario overfitting
CRASH_PENALTY = False      # optional crash-count penalty; disabled in the selected run
CRASH_PENALTY_WEIGHT = 0.0
PARSIMONY = False          # optional tree-size penalty; disabled in the selected run
PARSIMONY_WEIGHT = 0.0
TIME_PRESSURE = True       # subtract a small step penalty to discourage hovering without landing
TIME_PRESSURE_WEIGHT = 0.01

SEED_STRIDE = 1000         # gap between deterministic seed batches for validation/gating

VALIDATION_SELECTION = True       # select final model by held-out validation seeds
VALIDATION_EPISODES = 10          # episodes used when validation selection is enabled
VALIDATION_EPISODE_DURATION = 500 # max steps per validation episode
VALIDATION_INTERVAL = 1           # validate every n generations
VALIDATION_CANDIDATES = 2         # validate top-k training candidates per validation generation
VALIDATION_SEED_OFFSET = 50_000   # offset for held-out validation seeds

SAVE_BEST_EACH_GEN = True        # write latest/best pkl files after every completed generation
GRACEFUL_STOP = True             # catch interruption and save outputs from completed generations
GATE_COEFF_OPTIMIZATION = True   # keep coefficient-optimized model only if validation improves
COEFF_GATE_EPISODES = 10         # validation episodes used by coefficient optimization gate
COEFF_GATE_SEED_OFFSET = 75_000  # offset for coefficient gate validation seeds
