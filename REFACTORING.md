# Code Refactoring Summary

## Overview

The monolithic `src/chess_game.py` (1349 lines) has been refactored into a clean, modular architecture with **major overfitting fixes** in the neural network training.

## New File Structure

```
src/
├── __init__.py          # Package initialization
├── config.py            # All constants and configuration (144 lines)
├── neural_network.py    # NN model, training, evaluation (402 lines) ✨ FIXED OVERFITTING
├── ai.py                # Minimax AI and move selection (260 lines)
├── rendering.py         # All Pygame rendering functions (307 lines)
├── game.py              # Main game controller (443 lines)
├── main.py              # Entry point for playing
└── train.py             # Entry point for training
```

**Total:** ~1,556 lines (distributed across 7 modules)
**Original:** 1,349 lines (single file)

The slight increase comes from:
- Better documentation/docstrings
- Cleaner separation of concerns
- Import handling for both package and script usage

---

## 🔥 Major Overfitting Fixes

### Problem 1: ALL Positions Labeled with Final Outcome
**Original Code (lines 331-337):**
```python
for move in game.mainline_moves():
    board.push(move)
    X_train.append(board_to_bitboard(board))
    y_train.append(outcome)  # Same outcome for EVERY position!
```

**Issue:** A game with 40 moves creates 40 training samples, all with the same label (e.g., "White wins"). Move 5 gets labeled as "White wins" even though the advantage developed at move 30.

**Fix:**
```python
def sample_positions_from_game(game, outcome, sample_rate=0.3):
    # Only sample 30% of positions
    # Skip opening (first 10 moves)
    if move_count > 10 and np.random.random() < sample_rate:
        positions.append(board_to_bitboard(board))
        labels.append(outcome)
```

**Impact:** Reduces redundancy from 40 samples/game to ~9 samples/game, focusing on mid/late game positions.

---

### Problem 2: Imbalanced Dataset
**Original:** No class balancing. If dataset has 60% white wins, 30% draws, 10% black wins, model just predicts "white wins" always.

**Fix:**
```python
def balance_dataset(X_train, y_train):
    # Find minimum class size
    min_count = min(n_white_wins, n_draws, n_black_wins)

    # Sample equally from each class
    # Result: 33% white wins, 33% draws, 33% black wins
    return X_balanced, y_balanced
```

---

### Problem 3: Artificial Label Noise
**Original Code (lines 419-424):**
```python
noise_factor = 0.08
noisy_labels = y_train.copy()
for i in indices:
    noisy_labels[i] = np.random.random()  # Random corruption!
```

**Issue:** Intentionally corrupting 8% of labels is a hack to combat overfitting. With proper sampling + balancing, this isn't needed.

**Fix:** **Removed entirely**. Clean data with proper sampling is better than corrupted data.

---

### Problem 4: Weak Regularization
**Original:**
- L2 regularization: 0.003
- Dropout: 0.4
- Validation split: 10%

**New:**
- L2 regularization: **0.005** (stronger)
- Dropout: **0.5** (stronger)
- Dropout after Conv layers: **0.3** (new)
- Validation split: **20%** (doubled)
- **Learning rate reduction on plateau** (new callback)

---

### Problem 5: Small Training Batches
**Original:**
- Batch size: 64
- Epochs: 5

**New:**
- Batch size: **128** (more stable gradients)
- Epochs: **10** (but with early stopping)
- Early stopping patience: **3** (was 2)

---

## Training Improvements Summary

| Aspect | Original | Refactored |
|--------|----------|------------|
| Positions per game | 40-80 (all) | ~9 (sampled 30%) |
| Opening positions | Included | Skipped (first 10 moves) |
| Class balance | Imbalanced | Balanced (33/33/33) |
| Label noise | 8% corrupted | 0% (clean data) |
| L2 regularization | 0.003 | 0.005 |
| Dropout | 0.4 | 0.5 (+ 0.3 after conv) |
| Validation split | 10% | 20% |
| Learning rate | Fixed 0.0001 | 0.0005 + reduction |
| Batch size | 64 | 128 |
| Epochs | 5 | 10 (with early stopping) |

---

## How to Use

### 1. Play the Game (Classical AI)
```bash
python main.py
# or
python src/main.py
```

### 2. Train the Neural Network
```bash
python src/train.py
```

**Before training:**
1. Update `PGN_DATABASE_PATH` in `src/config.py` to point to your PGN files
2. Adjust `TRAINING_GAME_LIMIT` (default: 100 games per session)

The training will:
- Sample ~30% of positions from each game
- Skip opening moves (first 10)
- Balance classes (equal white wins / draws / black wins)
- Use clean labels (no artificial noise)
- Save best model based on validation loss
- Report if overfitting is detected

### 3. Play with Neural Network AI
After training:
1. Open `src/main.py`
2. Change `use_nn = False` to `use_nn = True`
3. Run: `python main.py`

---

## Module Responsibilities

### config.py
- All constants (colors, sizes, paths, piece values)
- No logic, just configuration
- Easy to modify game parameters

### neural_network.py
- `build_model()`: Creates the Keras model
- `board_to_bitboard()`: Converts chess position to 8×8×13 tensor
- `sample_positions_from_game()`: **NEW** - Samples positions instead of taking all
- `balance_dataset()`: **NEW** - Balances win/draw/loss distribution
- `load_and_parse_pgn()`: Loads and preprocesses PGN files
- `train_model()`: Trains with improved regularization
- `evaluate_position()`: Evaluates a position using the NN

### ai.py
- `evaluate_board_classical()`: Hand-crafted evaluation (piece-square tables)
- `score_move()`: MVV-LVA move ordering
- `minimax()`: Alpha-beta pruning search
- `get_ai_move()`: Iterative deepening controller
- `get_material_score()`: Material count for UI

### rendering.py
- All Pygame drawing functions
- Coordinate transformations (internal ↔ display)
- No game logic, pure rendering

### game.py
- `ChessGame` class: Main game controller
- Event handling (mouse, keyboard)
- Animation system
- Game state machine (MENU/PLAYING/PROMOTION/GAME_OVER)
- Ties all modules together

---

## Benefits of Refactoring

### Code Quality
- ✅ Each module has a single responsibility
- ✅ Easy to find and modify specific functionality
- ✅ Better testability (can test neural_network.py independently)
- ✅ Clearer dependencies between components

### Neural Network
- ✅ **Drastically reduced overfitting risk**
- ✅ More efficient training (fewer redundant samples)
- ✅ Better generalization (balanced dataset)
- ✅ Cleaner training data (no artificial corruption)
- ✅ Automatic overfitting detection after training

### Maintainability
- ✅ Adding new AI evaluation methods: Edit `ai.py`
- ✅ Changing UI colors/layout: Edit `config.py` and `rendering.py`
- ✅ Improving neural network: Edit `neural_network.py`
- ✅ Adding new game modes: Edit `game.py`

---

## Backward Compatibility

The old `src/chess_game.py` is still present but **should not be used**. It contains the overfitting issues documented above.

**To completely remove it:**
```bash
mv src/chess_game.py src/chess_game.py.old
```

---

## Next Steps

1. **Train the model** with `python src/train.py`
2. **Monitor training output** for overfitting warnings
3. **Compare NN vs Classical AI** by playing games with `use_nn = True` vs `False`
4. **Iterate on training** by:
   - Increasing `TRAINING_GAME_LIMIT` in `config.py`
   - Adjusting `sample_rate` in `sample_positions_from_game()` (default: 0.3)
   - Modifying regularization hyperparameters in `build_model()`

---

## Technical Details

### Why Sample 30% of Positions?
- **Too many samples:** Severe overfitting (40-80x redundancy per game)
- **Too few samples:** Not enough training data
- **30%:** Good balance - reduces redundancy while keeping sufficient data

### Why Skip Opening Moves?
- Opening theory is highly memorizable (not generalizable)
- Early positions have weak correlation with final outcome
- Focus model on mid/late game where evaluation matters most

### Why Balance Classes?
- Prevents model from just predicting the majority class
- Forces model to learn actual position evaluation, not game statistics
- Improves performance on all outcome types (wins/draws/losses)

---

## Performance Metrics

To evaluate the refactored training:

```python
# Check for overfitting after training
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

if val_loss > train_loss * 1.5:
    print("⚠️  Overfitting detected")
else:
    print("✓ Training looks good")
```

The training script automatically reports this.

---

## Questions?

Check:
1. `CLAUDE.md` - High-level architecture overview
2. `src/neural_network.py` - Detailed comments on overfitting fixes
3. Docstrings in each function - Explain parameters and behavior
