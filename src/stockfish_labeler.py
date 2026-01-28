"""
Stockfish-based position labeling for training data generation.

Uses Stockfish engine to evaluate positions and generate high-quality
training labels instead of relying on game outcomes.
"""
import os
import math
import numpy as np
import chess
import chess.engine
import chess.pgn
from typing import Optional, Tuple, List
from tqdm import tqdm

try:
    from .config import MODEL_DIR
    from .neural_network import board_to_bitboard
except ImportError:
    from config import MODEL_DIR
    from neural_network import board_to_bitboard


def find_stockfish() -> Optional[str]:
    """
    Find Stockfish executable on the system.

    Returns:
        Path to Stockfish executable, or None if not found
    """
    # Common locations
    possible_paths = [
        '/usr/local/bin/stockfish',
        '/usr/bin/stockfish',
        '/opt/homebrew/bin/stockfish',  # Homebrew on Apple Silicon
        'stockfish',  # In PATH
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
        # Try which command
        import shutil
        if shutil.which(path):
            return shutil.which(path)

    return None


def centipawn_to_value(cp: int) -> float:
    """
    Convert centipawn score to neural network value [-1, 1].

    Uses a sigmoid-like transformation to map centipawn scores:
    - cp=0 -> 0.0 (equal position)
    - cp=+400 -> ~0.76 (significant white advantage)
    - cp=-400 -> ~-0.76 (significant black advantage)
    - cp=+infinity -> 1.0 (white wins)
    - cp=-infinity -> -1.0 (black wins)

    Args:
        cp: Centipawn score (positive = white advantage)

    Returns:
        Value in [-1, 1] range
    """
    # Sigmoid-like transformation with scaling factor
    # 400 centipawns (4 pawns) maps to ~0.76
    return 2.0 / (1.0 + math.exp(-cp / 400.0)) - 1.0


def evaluate_position_stockfish(board: chess.Board, engine: chess.engine.SimpleEngine,
                                 depth: int = 12, time_limit: float = 0.1) -> float:
    """
    Evaluate a single position using Stockfish.

    Args:
        board: Chess position to evaluate
        engine: Stockfish engine instance
        depth: Search depth
        time_limit: Time limit in seconds

    Returns:
        Evaluation in [-1, 1] range (from white's perspective)
    """
    try:
        # Use depth limit for consistency
        info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
        score = info['score'].white()  # Always from white's perspective

        if score.is_mate():
            mate_in = score.mate()
            # Mate scores: closer to +-1 for imminent mates
            if mate_in > 0:
                return 1.0 - (0.01 * min(mate_in, 10))  # White wins
            else:
                return -1.0 + (0.01 * min(abs(mate_in), 10))  # Black wins
        else:
            cp = score.score()
            return centipawn_to_value(cp)

    except Exception as e:
        print(f"Error evaluating position: {e}")
        return 0.0


def label_positions_from_pgn(pgn_path: str, stockfish_path: str,
                              output_path: str, game_limit: int = 100,
                              sample_rate: float = 0.3, depth: int = 12) -> Tuple[int, int]:
    """
    Generate training data by labeling PGN positions with Stockfish evaluations.

    Args:
        pgn_path: Path to PGN file
        stockfish_path: Path to Stockfish executable
        output_path: Path to save output .npz file
        game_limit: Maximum number of games to process
        sample_rate: Fraction of positions to sample per game
        depth: Stockfish search depth

    Returns:
        Tuple of (games_processed, positions_generated)
    """
    positions = []
    labels = []
    games_processed = 0

    print(f"Opening Stockfish engine at: {stockfish_path}")

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        print(f"Processing PGN file: {pgn_path}")

        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            pbar = tqdm(total=game_limit, desc="Processing games")

            while games_processed < game_limit:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                board = game.board()
                move_count = 0

                for move in game.mainline_moves():
                    board.push(move)
                    move_count += 1

                    # Skip opening moves (first 8 moves = 16 half-moves)
                    if move_count < 16:
                        continue

                    # Skip endgame positions with few pieces
                    if len(board.piece_map()) < 6:
                        continue

                    # Sample probabilistically
                    if np.random.random() < sample_rate:
                        # Get Stockfish evaluation
                        value = evaluate_position_stockfish(board, engine, depth=depth)

                        # Get board tensor
                        tensor = board_to_bitboard(board)[0]

                        positions.append(tensor)
                        labels.append(value)

                games_processed += 1
                pbar.update(1)

            pbar.close()

    if positions:
        # Convert to numpy arrays
        X = np.array(positions, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, positions=X, labels=y)

        print(f"\nSaved {len(positions)} positions to {output_path}")
        print(f"Label statistics: min={y.min():.3f}, max={y.max():.3f}, "
              f"mean={y.mean():.3f}, std={y.std():.3f}")

    return games_processed, len(positions)


def load_stockfish_labeled_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-generated Stockfish-labeled training data.

    Args:
        data_path: Path to .npz file

    Returns:
        Tuple of (positions, labels) numpy arrays
    """
    data = np.load(data_path)
    return data['positions'], data['labels']


def augment_positions(positions: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data with horizontal flip.

    Chess has vertical symmetry (a-file <-> h-file), so we can
    flip positions horizontally to double the training data.

    Args:
        positions: Position tensors (N, 8, 8, 19)
        labels: Position labels (N,)

    Returns:
        Augmented (positions, labels) with 2N samples
    """
    # Flip positions horizontally (along axis 1 = columns)
    flipped = np.flip(positions, axis=2)

    # Combine original and flipped
    X_aug = np.concatenate([positions, flipped], axis=0)
    y_aug = np.concatenate([labels, labels], axis=0)  # Labels unchanged by flip

    # Shuffle
    indices = np.random.permutation(len(X_aug))

    return X_aug[indices], y_aug[indices]


if __name__ == '__main__':
    # Example usage
    stockfish = find_stockfish()
    if stockfish:
        print(f"Found Stockfish at: {stockfish}")
    else:
        print("Stockfish not found! Please install it:")
        print("  brew install stockfish  (macOS)")
        print("  apt install stockfish   (Ubuntu)")
