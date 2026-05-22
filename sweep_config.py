N_TRIALS = 60
SWEEP_GENS = 35        # reduced from config.MAX_GENS to keep each trial fast
SWEEP_EPISODES = 8     # reduced from config.NUM_EPISODES, same reason
RESULTS_FILE = "sweep_results.csv"
DB_FILE = "sweep.db"
STUDY_NAME = "lunar_lander_sweep"
RESUME = True         # set to True to continue from a previous run

# search spaces
POP_SIZE_OPTIONS = [16, 32, 64, 128]
MAX_TREE_SIZE_OPTIONS = [15, 31, 63]
NUM_CONSTANTS_MIN = 1
NUM_CONSTANTS_MAX = 5
COEFF_LR_MIN = 1e-4
COEFF_LR_MAX = 1e-2
COEFF_OPT_STEPS_OPTIONS = [250, 500, 1000]
GAMMA_MIN = 0.95
GAMMA_MAX = 0.999
