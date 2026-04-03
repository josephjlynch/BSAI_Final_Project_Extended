"""
Analysis Constants
==================

Centralized parameters for the Visual Behavior change detection project.
All values sourced from Bennett et al. (2025) unless noted otherwise.
"""

# =============================================================================
# STIMULUS TIMING (Bennett Methods: Visual Behavior task)
# =============================================================================

STIMULUS_DURATION_MS = 250
GRAY_SCREEN_DURATION_MS = 500
OMISSION_DURATION_MS = 1250
RESPONSE_WINDOW_MS = (150, 750)

# =============================================================================
# ANALYSIS WINDOWS (milliseconds relative to change onset)
# =============================================================================

BASELINE_WINDOW_MS = (-50, 0)
DECISION_WINDOW_MS = (20, 100)
FEEDFORWARD_WINDOW_MS = (20, 60)
RECURRENT_WINDOW_MS = (60, 100)

# =============================================================================
# BINNING
# =============================================================================

PSTH_BIN_SIZE_MS = 1.0
GLM_BIN_SIZE_MS = 25.0
RATE_BIN_SIZE_MS = 50.0

# =============================================================================
# CROSS-CORRELOGRAM (CCG) PARAMETERS
# Sourced from: Tang, Disheng et al. (2024) Nature Communications ccg_library.py
# https://github.com/HChoiLab/functional-network
# =============================================================================

CCG_BIN_SIZE_MS = 1.0          # 1ms bins (Disheng's implementation)
CCG_WINDOW_BINS = 100          # lag window in bins (= 100ms at 1ms bins, one-sided 0→+100ms)
CCG_WINDOW_MS = CCG_WINDOW_BINS * CCG_BIN_SIZE_MS  # 100ms

# Jitter: pattern jitter (Harrison & Geman 2009), history-preserving
CCG_N_SURROGATES = 100         # num_jitter in ccg_library.py default
CCG_JITTER_WINDOW_BINS = 25    # L parameter (jitter window in bins = 25ms)
CCG_JITTER_MEMORY = False      # memory=False: simple spike jitter (not history-preserving)
CCG_N_SIGMA = 5.0              # significance threshold in z-score units

# CCG epoch definitions (Disheng feedback: separate stimulus from gray screen)
CCG_STIMULUS_EPOCH_MS = 250    # stimulus window only (primary analysis)
CCG_GRAY_EPOCH_MS = 500        # gray screen window (secondary, optional)

# CCG unit inclusion (Disheng feedback + Tang et al. 2024 Methods):
# "only neurons with a firing rate of at least 2 Hz during all stimuli"
CCG_MIN_FIRING_RATE_HZ = 2.0

# Peak classification (retained for connection typing, consistent with Bennett)
MONOSYNAPTIC_PEAK_WIDTH_MS = 2.0
COMMON_INPUT_PEAK_WIDTH_MS = 5.0

# =============================================================================
# UNIT CLASSIFICATION
# =============================================================================

RS_FS_THRESHOLD_MS = 0.4  # trough-to-peak; >0.4 = RS (excitatory), <0.4 = FS (PV inhibitory)

# Laminar depth assignment (Week 5)
# Normalized cortical depth boundaries: 0 = pia, 1 = white matter.
CORTICAL_LAYER_BOUNDARIES_NORM = {
    'L2/3': (0.0, 0.35),
    'L4': (0.35, 0.50),
    'L5': (0.50, 0.75),
    'L6': (0.75, 1.0),
}
SUBCORTICAL_LAYER_LABEL = 'subcortical'
UNKNOWN_LAYER_LABEL = 'unknown'

# L6 consolidation: Allen CCF ontology distinguishes L6a and L6b.
# Both are mapped to 'L6' here because:
#   (a) L6b contains too few units for reliable per-area statistics
#       (~10% of L6 units across the 6 visual areas in this dataset)
#   (b) Jia et al. 2022 and Tang et al. 2024 do not distinguish L6 sublaminae
# If L6a/L6b separation is required for a specific analysis, change
# layer_suffixes in _build_ccf_layer_lookup() to:
#   '6a': 'L6a', '6b': 'L6b'
# and update CORTICAL_LAYER_BOUNDARIES_NORM accordingly.
CORTICAL_LAYER_L6_SUBLAYERS_MERGED = True

# CCF annotation volume for atlas-registered laminar assignment
CCF_ANNOTATION_URL = (
    'http://download.alleninstitute.org/informatics-archive/'
    'current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd'
)
CCF_ANNOTATION_PATH = 'data/allen_cache/ccf_annotation_25.nrrd'
CCF_ONTOLOGY_URL = (
    'http://api.brain-map.org/api/v2/data/Structure/query.json'
    '?criteria=[graph_id$eq1]&num_rows=all'
)
CCF_ONTOLOGY_PATH = 'data/allen_cache/ccf_structure_tree.json'
CCF_RESOLUTION_UM = 25

# =============================================================================
# UNIT QUALITY FILTERS (Bennett Methods, §Data processing)
# =============================================================================

PRESENCE_RATIO_MIN = 0.9
ISI_VIOLATIONS_MAX = 0.5
AMPLITUDE_CUTOFF_MAX = 0.1
EXPECTED_UNIT_COUNT = 76091

# =============================================================================
# BEHAVIORAL ENGAGEMENT (Bennett Methods)
# =============================================================================

ENGAGEMENT_REWARD_RATE_MIN = 2.0  # rewards per minute threshold

# =============================================================================
# TRIAL-TYPE ALIGNMENT
# =============================================================================

MAX_LABELED_REPEAT_POSITION = 4   # positions 1-4 labeled; 5+ lumped as "5plus"
GLM_HISTORY_LAGS = 10             # 10-lag outcome history (Molano-Mazón Eq. 3)
RESET_INDEX_MIN_TRIALS = 30       # min after-hit/after-miss trials for RI

# =============================================================================
# BEHAVIORAL STRATEGY (Piet et al. 2023)
# Classification: visual comparison vs. timing estimation strategy
# =============================================================================

STRATEGY_INDEX_THRESHOLD = 0.5  # visual > 0.5, timing <= 0.5
STRATEGY_MIN_TRIALS = 20        # minimum engaged trials for reliable logistic fit
ANTICIPATORY_WINDOW_S = 0.5     # gray screen duration before each stimulus (500ms)

# =============================================================================
# Piet et al. (2023) dynamic logistic regression / licking segmentation
# =============================================================================
LICK_BOUT_ILI_S = 0.700  # inter-lick interval threshold for bout segmentation

# PsyTrack random-walk prior hyperparameter search bounds (guardrails).
# (psytrack.hyperOpt performs unconstrained optimization in log2 space.)
PSYTRACK_SIGMA_BOUNDS = (1e-4, 1e4)

# Timing strategy sigmoid parameters:
# images since the end of the last licking bout, mapped through a sigmoid.
TIMING_SIGMOID_MIDPOINT = 4
TIMING_SIGMOID_SLOPE = 1.0

# =============================================================================
# SESSION / MOUSE COUNTS (Bennett dataset)
# =============================================================================

N_SESSIONS_EXPECTED = 103
N_MICE_EXPECTED = 54

# =============================================================================
# GRAPH ANALYSIS
# =============================================================================

LOUVAIN_RANDOM_STATE = 42
NULL_MODEL_N_REWIRINGS = 1000
MOTIF_NULL_N_PERMUTATIONS = 10000

# =============================================================================
# RNN PARAMETERS (behavioral prediction model)
# =============================================================================

RNN_HIDDEN_SIZE = 64
RNN_N_LAYERS = 2
RNN_DROPOUT = 0.3
RNN_LEARNING_RATE = 0.001

# =============================================================================
# STATISTICAL
# =============================================================================

ALPHA = 0.05
BOOTSTRAP_N = 10000
RANDOM_SEED = 42
