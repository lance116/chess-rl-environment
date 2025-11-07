# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chess Neural Network is a Python-based chess game featuring a TensorFlow-powered AI opponent. The entire application is contained in a single monolithic file (`src/chess_game.py` - 1349 lines) that handles game logic, neural network training, AI move generation, and Pygame-based GUI rendering.

## Running and Development

### Running the game
```bash
# Primary method
python src/chess_game.py

# Alternative (uses wrapper)
python main.py
```

### Installing dependencies
```bash
pip install -r requirements.txt
```

Dependencies: TensorFlow 2.12.0, Pygame 2.5.0+, python-chess 1.999+, NumPy 1.21.0+

## Architecture

### Single-File Design
All functionality resides in `src/chess_game.py`. The file is organized as:
- Lines 1-161: Imports, constants, Pygame initialization, image loading, model configuration
- Lines 165-218: Neural network model definition (`build_model`) and weight loading
- Lines 220-447: PGN training data parsing and model training (`load_parse_and_train`, `board_to_bitboard`)
- Lines 449-685: Board evaluation and move scoring (classic heuristics + optional NN evaluation)
- Lines 687-809: Minimax AI with iterative deepening and alpha-beta pruning (`minimax`, `get_ai_move`)
- Lines 813-904: Board representation conversion and coordinate transforms (pygame ↔ python-chess)
- Lines 906-1029: Rendering functions (board, pieces, menus, game over screens, promotion UI)
- Lines 1030-1067: Game state management (`reset_game`, `start_animation`)
- Lines 1069-1349: Main game loop with event handling for MENU, PLAYING, PROMOTION, GAME_OVER states

### Board Representation Dual System
The codebase uses **python-chess** as the source of truth for game state:
- `board_pychess` (chess.Board): Authoritative state, legal move generation, rule validation
- Board lists (8×8 arrays): Only used for rendering via `pychess_to_board_list()`
- Coordinate systems:
  - **Internal**: Row 0 = rank 8 (top), Col 0 = a-file (left)
  - **python-chess squares**: 0 = a1, 63 = h8
  - **Display**: Flipped for black player via `to_display_coords()` and `to_internal_coords()`

### Neural Network Architecture
- **Input**: 8×8×13 bitboard tensor
  - 12 planes for piece positions (6 piece types × 2 colors)
  - 1 plane encoding current turn (1.0 for white, 0.0 for black)
- **Processing**: 2 Conv2D layers (64, 128 filters) → BatchNorm → Dense(192) → Dropout(0.4) → Dense(96) → Dropout(0.4)
- **Output**: Single sigmoid value (0-1) representing white's win probability
- **Evaluation conversion**: `(predicted_value - 0.5) * 1200` maps to centipawn score for minimax

Model weights: `models/chess_model.weights.h5` (20MB)

### AI Move Selection
Uses iterative deepening minimax with:
- Time limit: 5 seconds (`AI_THINK_TIME`)
- Alpha-beta pruning
- Move ordering by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
- Two evaluation modes:
  - `use_nn_eval = False` (default): Classical piece-square tables + material (see `PIECE_VALUES`, `POSITION_SCORES`)
  - `use_nn_eval = True`: Neural network position evaluation via `evaluate_board_nn()`

### Training System
- **Flag**: Set `TRAIN_MODEL_ON_START = True` in chess_game.py to trigger training on startup
- **Data source**: PGN files in `PGN_DATABASE_PATH` (currently hardcoded Windows path: `A:\\Chess Neural Network\\Lichess Elite Database\\Lichess Elite Database`)
- **Progress tracking**: `models/training_progress.json` stores processed files and positions to allow incremental training
- **Training config**:
  - `TRAINING_GAME_LIMIT = 1000` games per run
  - Label noise (8%) to reduce overfitting
  - Early stopping on validation loss (patience=2)
  - Model checkpoint saves best weights
- **IMPORTANT**: After training completes, set `TRAIN_MODEL_ON_START = False` to prevent retraining on every launch

### Game State Machine
Four states managed by `game_state` variable:
1. **MENU**: Player selects color (W/B keys)
2. **PLAYING**: Active gameplay, handles player + AI turns
3. **PROMOTION**: Pawn promotion piece selection UI
4. **GAME_OVER**: Checkmate/stalemate/forfeit screen

Animation system (`is_animating` flag) blocks input during move animations (6 frames @ 30 FPS).

## Important Configuration Constants

Located at top of `src/chess_game.py`:
- `WIDTH, HEIGHT = 600, 640` (game window size)
- `AI_THINK_TIME = 5.0` (seconds for AI move search)
- `ANIMATION_DURATION = 6` (frames)
- `MODEL_WEIGHTS_FILE`: Path to model weights
- `PGN_DATABASE_PATH`: Training data location (needs updating for cross-platform use)
- `TRAIN_MODEL_ON_START = False`: Training toggle
- `TRAINING_GAME_LIMIT = 1000`: Games to process per training session

## Development Notes

### Modifying the Neural Network
If you change the model architecture in `build_model()`:
1. Delete or rename `models/chess_model.weights.h5` (backup is auto-created)
2. The code will automatically start with fresh weights
3. Re-train with `TRAIN_MODEL_ON_START = True`

### Adding Features
Key areas to modify:
- **New game rules**: Update via python-chess board methods (validation is automatic)
- **AI behavior**: Modify `evaluate_board_ai()`, `score_move()`, or minimax search depth
- **UI changes**: Update rendering functions (lines 906-1029) and event handling in main loop
- **Training data**: Change `PGN_DATABASE_PATH` to point to new PGN files

### Performance Considerations
- GPU acceleration: TensorFlow auto-detects GPU (lines 18-28)
- Alpha-beta pruning reduces search tree significantly
- Move ordering (line 703) improves pruning effectiveness
- Iterative deepening allows time-bounded searches with best move fallback

### Common Pitfalls
- **Coordinate confusion**: Always use conversion functions (`to_internal_coords`, `to_display_coords`) when translating between screen clicks and board positions
- **Animation blocking**: Moves only apply after animation completes (see `pending_move_uci` mechanism)
- **Promotion handling**: Requires special UCI move construction with promotion piece type (lines 1103-1110)
- **Training path**: `PGN_DATABASE_PATH` is Windows-specific; update for your environment before training
