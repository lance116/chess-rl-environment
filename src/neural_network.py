"""
Neural network model for chess position evaluation.
Fixes major overfitting issues from the original implementation.
"""
import os
import json
import numpy as np
import chess
import chess.pgn
import traceback
from typing import Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    # Configure GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"TensorFlow using GPU: {physical_devices[0]}")
    else:
        print("TensorFlow using CPU.")
except ImportError as e:
    print(f"TensorFlow not found: {e}")
    raise

try:
    from .config import (
        INPUT_SHAPE, BOARD_SIZE, MODEL_WEIGHTS_FILE, TRAINING_PROGRESS_FILE,
        TRAINING_BATCH_SIZE, TRAINING_EPOCHS, VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE
    )
except ImportError:
    from config import (
        INPUT_SHAPE, BOARD_SIZE, MODEL_WEIGHTS_FILE, TRAINING_PROGRESS_FILE,
        TRAINING_BATCH_SIZE, TRAINING_EPOCHS, VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE
    )


def build_model(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Build the chess evaluation neural network.

    IMPROVEMENTS FROM ORIGINAL:
    - Stronger regularization (L2=0.005 from 0.003)
    - More dropout (0.5 from 0.4)
    - Batch normalization after each layer
    - Reduced model capacity to prevent memorization

    Args:
        input_shape: Shape of input tensor (8, 8, 13)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Convolutional layers for spatial pattern recognition
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),  # Add dropout after conv layers

        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Dense layers for evaluation
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output: win probability for white [0, 1]
        layers.Dense(1, activation="sigmoid")
    ])

    # Use lower learning rate for more stable training
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def board_to_bitboard(board: chess.Board) -> np.ndarray:
    """
    Convert python-chess Board to neural network input tensor.

    Creates 8×8×19 tensor (enhanced representation):
    - Planes 0-5: White pieces (P, N, B, R, Q, K)
    - Planes 6-11: Black pieces (P, N, B, R, Q, K)
    - Plane 12: Turn indicator (1.0 = White's turn, 0.0 = Black's turn)
    - Plane 13: Kingside castling rights (white OR black has rights)
    - Plane 14: Queenside castling rights (white OR black has rights)
    - Plane 15: En passant square (1.0 at en passant target)
    - Plane 16: Halfmove clock (normalized 0-1, capped at 100)
    - Plane 17: Repetition indicator (1.0 if position repeated)
    - Plane 18: Check indicator (1.0 if current side is in check)

    Args:
        board: python-chess Board object

    Returns:
        NumPy array of shape (1, 8, 8, 19) ready for model.predict()
    """
    bitboard = np.zeros(INPUT_SHAPE, dtype=np.float32)
    plane_index = 0

    # Planes 0-11: Piece positions
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                          chess.ROOK, chess.QUEEN, chess.KING]:
            for square in board.pieces(piece_type, color):
                row = 7 - (square // 8)
                col = square % 8
                bitboard[row, col, plane_index] = 1.0
            plane_index += 1

    # Plane 12: Turn indicator
    bitboard[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Plane 13: Kingside castling rights
    has_kingside = (board.has_kingside_castling_rights(chess.WHITE) or
                    board.has_kingside_castling_rights(chess.BLACK))
    bitboard[:, :, 13] = 1.0 if has_kingside else 0.0

    # Plane 14: Queenside castling rights
    has_queenside = (board.has_queenside_castling_rights(chess.WHITE) or
                     board.has_queenside_castling_rights(chess.BLACK))
    bitboard[:, :, 14] = 1.0 if has_queenside else 0.0

    # Plane 15: En passant square
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        bitboard[row, col, 15] = 1.0

    # Plane 16: Halfmove clock (normalized, important for 50-move rule)
    halfmove_normalized = min(board.halfmove_clock / 100.0, 1.0)
    bitboard[:, :, 16] = halfmove_normalized

    # Plane 17: Repetition indicator
    is_repetition = board.is_repetition(2)  # True if position occurred 2+ times
    bitboard[:, :, 17] = 1.0 if is_repetition else 0.0

    # Plane 18: Check indicator
    bitboard[:, :, 18] = 1.0 if board.is_check() else 0.0

    return np.expand_dims(bitboard, axis=0)


def sample_positions_from_game(game: chess.pgn.Game, outcome: float,
                               sample_rate: float = 0.3) -> Tuple[list, list]:
    """
    Sample positions from a game instead of taking ALL positions.

    FIX #1 FOR OVERFITTING: Don't use every position from a game.
    Taking all positions means the same outcome gets repeated 40-80 times,
    causing severe overfitting.

    Args:
        game: chess.pgn.Game object
        outcome: Final outcome (1.0 = White wins, 0.0 = Black wins, 0.5 = Draw)
        sample_rate: Fraction of positions to sample (0.3 = 30%)

    Returns:
        Tuple of (bitboards, labels)
    """
    positions = []
    labels = []

    board = game.board()
    move_count = 0

    for move in game.mainline_moves():
        board.push(move)
        move_count += 1

        # Sample positions probabilistically
        # Skip opening (first 10 moves) - too early to have outcome signal
        if move_count > 10 and np.random.random() < sample_rate:
            bitboard_tensor = board_to_bitboard(board)[0]
            positions.append(bitboard_tensor)
            labels.append(outcome)

    return positions, labels


def balance_dataset(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance the dataset so wins/draws/losses are evenly represented.

    FIX #2 FOR OVERFITTING: Balance class distribution.
    If dataset has 60% white wins, 30% draws, 10% black wins,
    the model will just predict "white wins" most of the time.

    Args:
        X_train: Training positions
        y_train: Training labels

    Returns:
        Balanced X_train and y_train
    """
    # Separate by outcome
    white_wins = y_train >= 0.9  # White wins (1.0)
    draws = (y_train > 0.4) & (y_train < 0.6)  # Draws (0.5)
    black_wins = y_train <= 0.1  # Black wins (0.0)

    # Count each class
    n_white_wins = np.sum(white_wins)
    n_draws = np.sum(draws)
    n_black_wins = np.sum(black_wins)

    print(f"Dataset distribution: White wins: {n_white_wins}, "
          f"Draws: {n_draws}, Black wins: {n_black_wins}")

    # Find minimum class size
    min_count = min(n_white_wins, n_draws, n_black_wins)

    if min_count == 0:
        print("Warning: One class has no examples. Skipping balancing.")
        return X_train, y_train

    # Sample equally from each class
    white_win_indices = np.where(white_wins)[0]
    draw_indices = np.where(draws)[0]
    black_win_indices = np.where(black_wins)[0]

    # Randomly sample min_count from each
    selected_white = np.random.choice(white_win_indices, size=min_count, replace=False)
    selected_draw = np.random.choice(draw_indices, size=min_count, replace=False)
    selected_black = np.random.choice(black_win_indices, size=min_count, replace=False)

    # Combine
    selected_indices = np.concatenate([selected_white, selected_draw, selected_black])
    np.random.shuffle(selected_indices)

    X_balanced = X_train[selected_indices]
    y_balanced = y_train[selected_indices]

    print(f"Balanced dataset: {len(X_balanced)} positions "
          f"({min_count} per class)")

    return X_balanced, y_balanced


def load_and_parse_pgn(pgn_dir_path: str, game_limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and parse PGN files for training.

    IMPROVEMENTS FROM ORIGINAL:
    - Samples positions instead of taking all
    - Balances dataset
    - No artificial label noise (not needed with proper sampling)

    Args:
        pgn_dir_path: Directory containing PGN files
        game_limit: Maximum number of games to process

    Returns:
        Tuple of (X_train, y_train) NumPy arrays
    """
    print(f"Starting PGN parsing from: {pgn_dir_path}")
    X_train = []
    y_train = []
    games_parsed = 0
    positions_parsed = 0

    # Load training progress
    processed_files = set()
    total_games_processed = 0
    current_file_position = 0
    current_file_name = None

    if os.path.exists(TRAINING_PROGRESS_FILE):
        try:
            with open(TRAINING_PROGRESS_FILE, 'r') as f:
                training_progress = json.load(f)
                processed_files = set(training_progress.get('processed_files', []))
                total_games_processed = training_progress.get('total_games_processed', 0)
                current_file_name = training_progress.get('current_file_name', None)
                current_file_position = training_progress.get('current_file_position', 0)
                print(f"Loaded training progress: {len(processed_files)} files processed, "
                      f"{total_games_processed} games total")
        except Exception as e:
            print(f"Error loading training progress: {e}. Starting fresh.")

    # Get PGN files
    if not os.path.exists(pgn_dir_path):
        print(f"Error: PGN directory not found: {pgn_dir_path}")
        print("Please update PGN_DATABASE_PATH in config.py or disable training.")
        return np.array([]), np.array([])

    pgn_files = [os.path.join(pgn_dir_path, f) for f in os.listdir(pgn_dir_path)
                 if f.lower().endswith('.pgn')]

    if not pgn_files:
        print(f"Error: No .pgn files found in directory: {pgn_dir_path}")
        return np.array([]), np.array([])

    pgn_files.sort()
    pgn_files = [f for f in pgn_files if os.path.basename(f) not in processed_files]

    # Resume from partial file if needed
    if current_file_name and current_file_position > 0:
        current_file_path = os.path.join(pgn_dir_path, current_file_name)
        if current_file_path in pgn_files:
            pgn_files.remove(current_file_path)
            pgn_files.insert(0, current_file_path)

    print(f"Found {len(pgn_files)} PGN files to process.")

    # Parse games
    for pgn_file_path in pgn_files:
        if game_limit is not None and games_parsed >= game_limit:
            break

        file_basename = os.path.basename(pgn_file_path)
        is_resuming = (file_basename == current_file_name)
        start_pos = current_file_position if is_resuming else 0

        print(f"Processing file: {file_basename}...")

        try:
            with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                if start_pos > 0:
                    pgn_file.seek(start_pos)

                while True:
                    current_position = pgn_file.tell()

                    if game_limit is not None and games_parsed >= game_limit:
                        current_file_position = current_position
                        current_file_name = file_basename
                        break

                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        current_file_position = 0
                        current_file_name = None
                        break

                    games_parsed += 1

                    try:
                        result = game.headers["Result"]
                        if result == '1-0':
                            outcome = 1.0
                        elif result == '0-1':
                            outcome = 0.0
                        elif result == '1/2-1/2':
                            outcome = 0.5
                        else:
                            continue

                        # FIX: Sample positions instead of taking all
                        positions, labels = sample_positions_from_game(game, outcome)
                        X_train.extend(positions)
                        y_train.extend(labels)
                        positions_parsed += len(positions)

                        if games_parsed % 50 == 0:
                            print(f"  Parsed {games_parsed} games, {positions_parsed} positions...")

                    except KeyError:
                        pass  # Skip games without result
                    except Exception as e_inner:
                        print(f"Error processing game {games_parsed}: {e_inner}")

        except FileNotFoundError:
            print(f"Error: PGN file not found at {pgn_file_path}")
        except Exception as e_outer:
            print(f"Error reading PGN file {pgn_file_path}: {e_outer}")
            traceback.print_exc()

    if not X_train:
        print("No valid positions parsed from PGN files.")
        return np.array([]), np.array([])

    print(f"\nTotal games parsed: {games_parsed}")
    print(f"Total positions extracted: {len(X_train)}")

    # Convert to numpy arrays
    X_train_np = np.array(X_train, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.float32)

    # FIX: Balance the dataset
    X_train_np, y_train_np = balance_dataset(X_train_np, y_train_np)

    # Save training progress
    files_processed_this_run = [os.path.basename(f) for f in pgn_files
                                if os.path.basename(f) not in processed_files]

    if files_processed_this_run and game_limit is not None and games_parsed >= game_limit:
        last_processed_file = files_processed_this_run[-1] if current_file_name else None
        files_fully_processed = files_processed_this_run[:-1] if last_processed_file else files_processed_this_run
        current_file_name = last_processed_file
    else:
        files_fully_processed = files_processed_this_run
        current_file_name = None

    processed_files.update(files_fully_processed)

    training_progress = {
        'processed_files': list(processed_files),
        'total_games_processed': total_games_processed + games_parsed,
        'current_file_name': current_file_name,
        'current_file_position': current_file_position
    }

    try:
        os.makedirs(os.path.dirname(TRAINING_PROGRESS_FILE), exist_ok=True)
        with open(TRAINING_PROGRESS_FILE, 'w') as f:
            json.dump(training_progress, f)
        print(f"Saved training progress: {len(processed_files)} files processed")
    except Exception as e:
        print(f"Warning: Could not save training progress: {e}")

    return X_train_np, y_train_np


def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
               weights_save_path: str) -> keras.callbacks.History:
    """
    Train the chess evaluation model.

    IMPROVEMENTS FROM ORIGINAL:
    - No artificial label noise (clean data with proper sampling)
    - Larger validation split (20% vs 10%)
    - Learning rate reduction on plateau
    - More patient early stopping

    Args:
        model: Keras model to train
        X_train: Training positions
        y_train: Training labels
        weights_save_path: Path to save best weights

    Returns:
        Training history object
    """
    print("\nStarting model training...")
    print(f"Training on {len(X_train)} positions")

    # Callbacks for better training
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        weights_save_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    try:
        # NO LABEL NOISE - we're using proper sampling now
        history = model.fit(
            X_train,
            y_train,
            epochs=TRAINING_EPOCHS,
            batch_size=TRAINING_BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            shuffle=True,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint, reduce_lr]
        )

        print("\nTraining complete!")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

        # Check for overfitting
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        if val_loss > train_loss * 1.5:
            print("\nWARNING: Model may still be overfitting!")
            print(f"Validation loss is {val_loss/train_loss:.2f}x higher than training loss")
        else:
            print("\n✓ Model training looks good - no severe overfitting detected")

        return history

    except Exception as e:
        print(f"Error during model training: {e}")
        traceback.print_exc()
        return None


def load_model(weights_path: str) -> Optional[keras.Model]:
    """
    Load a trained model from weights file.

    Args:
        weights_path: Path to .h5 weights file

    Returns:
        Loaded model or None if loading fails
    """
    model = build_model(INPUT_SHAPE)

    if os.path.exists(weights_path):
        try:
            # Create backup before loading
            if os.path.getsize(weights_path) > 0:
                backup_file = f"{weights_path}.backup"
                import shutil
                shutil.copy2(weights_path, backup_file)
                print(f"Created backup at {backup_file}")

            model.load_weights(weights_path)
            print(f"Loaded model weights from {weights_path}")
            return model

        except ValueError as ve:
            print(f"Error: Model architecture changed, weights incompatible: {ve}")
            print("Starting with fresh weights.")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print(f"Weights file {weights_path} not found. Starting with fresh weights.")

    return model


def evaluate_position(model: keras.Model, board: chess.Board) -> float:
    """
    Evaluate a chess position using the neural network.

    Args:
        model: Trained Keras model
        board: python-chess Board object

    Returns:
        Evaluation score in centipawns (positive = White advantage)
    """
    # Handle terminal positions
    if board.is_insufficient_material() or board.is_stalemate():
        return 0.0

    if board.is_checkmate():
        return -10000.0 if board.turn == chess.WHITE else 10000.0

    # Get neural network prediction
    board_tensor = board_to_bitboard(board)
    predicted_value = model.predict(board_tensor, verbose=0)[0][0]

    # Convert probability [0, 1] to centipawn score
    # 0.5 -> 0 (equal), 1.0 -> +600, 0.0 -> -600
    eval_score = (predicted_value - 0.5) * 1200

    return eval_score
