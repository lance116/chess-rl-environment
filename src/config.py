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

# Training settings
import os as _os
PGN_DATABASE_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'assets')
TRAIN_MODEL_ON_START = False
TRAINING_GAME_LIMIT = 200  # Process games per training session
TRAINING_BATCH_SIZE = 128
TRAINING_EPOCHS = 10
VALIDATION_SPLIT = 0.2  # Increased from 0.1
EARLY_STOPPING_PATIENCE = 3
