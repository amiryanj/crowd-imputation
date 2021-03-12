import os
import sys

# Config
BIPED_MODE = True
LIDAR_ENABLED = True
RECORD_MAIN_SCENE = False
TRACKING_ENABLED = True
OVERRIDE_TRACKS = True  # Use ground-truth loc-vel of the agents
PROJECTION_ENABLED = True
READ_LIDAR_FROM_CARLA = False
NUM_HYPOTHESES = 1
MAP_RESOLUTION = 8  # per meter
FRAME_RANGE = range(0, sys.maxsize)
# FRAME_RANGE = range(443, sys.maxsize)  # => filter the frame range
# ****************************

# Debug/Visualization Settings
VISUALIZER_ENABLED = True
DEBUG_VISDOM = False
VIDEO_ENABLED = False

LOG_SIM_TRAJS = False  # Deprecated?
LOG_PRED_OUTPUTS = True
LOG_PRED_EVALS = True
SAVE_PRED_PLOTS = False
SAVE_SIM_DATA_DIR = os.path.abspath(__file__ + "../../../../data/prior-social-ties")
EVAL_RESULT_PATH = '/home/cyrus/Music/num-results/num_results-1.csv'
PRED_RESULTS_OUTPUT_DIR = '/home/cyrus/Music/crowdrep_bot-outputs-1'
SCENARIO_CONFIG_FILE = os.path.abspath(os.path.join(__file__, "../../..",
                                                    "config/repbot_sim/real_scenario_config.yaml"))
# ****************************

# Setup Random Seed
RAND_SEED = 4
# ****************************
