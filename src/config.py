"""
Configuration and constants for Chess Neural Network game.
"""
import os

# Window and board settings
WIDTH, HEIGHT = 600, 640
BOARD_SIZE = 8
SQUARE_SIZE = WIDTH // BOARD_SIZE
CAPTURED_AREA_HEIGHT = 40
GAME_AREA_HEIGHT = HEIGHT - CAPTURED_AREA_HEIGHT
FPS = 30

# AI settings
AI_THINK_TIME = 5.0
ANIMATION_DURATION = 6

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (238, 238, 210)
DARK_SQUARE = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0, 150)
POSSIBLE_MOVE_COLOR = (135, 206, 250, 150)
CHECK_HIGHLIGHT_COLOR = (255, 0, 0, 100)
MENU_BG_COLOR = (40, 40, 40)
MENU_TEXT_COLOR = (220, 220, 220)
CAPTURED_BG_COLOR = (60, 60, 60)
CAPTURED_TEXT_COLOR = (200, 200, 200)
FORFEIT_BUTTON_COLOR = (180, 0, 0)
FORFEIT_TEXT_COLOR = (255, 255, 255)
PROMOTION_BG_COLOR = (50, 50, 50, 230)
PROMOTION_BORDER_COLOR = (150, 150, 150)
PROMOTION_CHOICE_COLOR = (80, 80, 80)
PROMOTION_TEXT_COLOR = (220, 220, 220)

# Piece settings
PIECES = {
    'wP', 'wR', 'wN', 'wB', 'wQ', 'wK',
    'bP', 'bR', 'bN', 'bB', 'bQ', 'bK'
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, 'assets', 'images')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, 'chess_model.weights.h5')
TRAINING_PROGRESS_FILE = os.path.join(MODEL_DIR, 'training_progress.json')

# Chess evaluation values (in centipawns)
PIECE_VALUES = {
    'P': 100, 'N': 300, 'B': 300, 'R': 500, 'Q': 900, 'K': 10000
}

# Piece-square tables for positional evaluation
POSITION_SCORES = {
    'P': [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [90, 90, 90, 90, 90, 90, 90, 90],
        [30, 30, 40, 60, 60, 40, 30, 30],
        [10, 10, 20, 40, 40, 20, 10, 10],
        [5,  5, 10, 20, 20, 10,  5,  5],
        [0,  0,  0,-10,-10,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
    ],
    'N': [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ],
    'B': [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 15, 15, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ],
    'R': [
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 5, 20, 20, 20, 20, 20, 20,  5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [ 0,  0,  0,  5,  5,  0,  0,  0]
    ],
    'Q': [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [ -5,  0,  5, 10, 10,  5,  0, -5],
        [ -5,  0,  5, 10, 10,  5,  0, -5],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20]
    ],
    'K': [
        [-50,-30,-30,-30,-30,-30,-30,-50],
        [-30,-30,  0,  0,  0,  0,-30,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-20,-10,  0,  0,-10,-20,-30],
        [-50,-40,-30,-20,-20,-30,-40,-50]
    ]
}

# Neural network settings
N_PIECE_TYPES = 6
N_COLORS = 2
# Enhanced board representation: 19 planes
# - 12 piece planes (6 types x 2 colors)
# - 1 turn indicator
# - 2 castling rights (kingside, queenside)
# - 1 en passant
# - 1 halfmove clock (normalized)
# - 1 repetition indicator
# - 1 check indicator
N_PLANES = 19
INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, N_PLANES)

# Supervised training settings
import os as _os
PGN_DATABASE_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'assets')
TRAIN_MODEL_ON_START = False
TRAINING_GAME_LIMIT = 200  # Process games per training session
TRAINING_BATCH_SIZE = 128
TRAINING_EPOCHS = 10
VALIDATION_SPLIT = 0.2  # Increased from 0.1
EARLY_STOPPING_PATIENCE = 3

# =============================================================================
# Reinforcement Learning Settings (TD-Lambda)
# =============================================================================

# TD-Lambda hyperparameters
RL_LAMBDA = 0.7              # TD-Lambda trace decay (0=TD(0), 1=Monte Carlo)
RL_GAMMA = 1.0               # Discount factor (1.0 for episodic chess games)
RL_LEARNING_RATE = 0.0001    # Lower than supervised for stable RL updates

# Training loop settings
RL_BATCH_SIZE = 64           # Batch size for RL training updates
RL_GAMES_PER_ITERATION = 50  # Self-play games per training iteration
RL_TRAINING_ITERATIONS = 100 # Total training iterations
RL_UPDATES_PER_ITERATION = 100  # Gradient updates per iteration

# Self-play settings
RL_MINIMAX_DEPTH = 3         # Fixed depth for training games (faster)
RL_MINIMAX_TIME = 1.0        # Time limit per move during self-play (seconds)
RL_TEMPERATURE = 1.0         # Move selection temperature (higher = more exploration)
RL_TEMPERATURE_THRESHOLD = 30  # Moves before switching to greedy (temperature=0)

# Experience replay buffer
RL_REPLAY_BUFFER_SIZE = 100000  # Maximum positions in replay buffer
RL_REPLAY_MIN_SIZE = 5000       # Minimum positions before training starts

# Checkpointing and evaluation
RL_CHECKPOINT_INTERVAL = 10     # Save checkpoint every N iterations
RL_EVAL_INTERVAL = 10           # Evaluate against Stockfish every N iterations
RL_EVAL_GAMES = 20              # Games per evaluation

# Paths for RL training
RL_CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'rl_checkpoints')
RL_BEST_MODEL_FILE = os.path.join(RL_CHECKPOINT_DIR, 'best_model.weights.h5')
RL_TRAINING_LOG_FILE = os.path.join(RL_CHECKPOINT_DIR, 'training_log.json')

# Stockfish settings for evaluation
STOCKFISH_SKILL_LEVELS = [1, 5, 10, 15, 20]  # Skill levels for Elo estimation
STOCKFISH_TIME_PER_MOVE = 0.1  # Seconds per move during evaluation
