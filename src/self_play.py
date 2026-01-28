"""
Self-Play Game Engine for TD-Lambda reinforcement learning.

Generates training games by playing the neural network against itself
using minimax search. Records positions and values for training.
"""
import time
import random
import numpy as np
import chess
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

try:
    from .config import (
        RL_MINIMAX_DEPTH, RL_MINIMAX_TIME, RL_TEMPERATURE,
        RL_TEMPERATURE_THRESHOLD, INPUT_SHAPE
    )
    from .neural_network import board_to_bitboard, evaluate_position
    from .ai import minimax, score_move, evaluate_board_classical
except ImportError:
    from config import (
        RL_MINIMAX_DEPTH, RL_MINIMAX_TIME, RL_TEMPERATURE,
        RL_TEMPERATURE_THRESHOLD, INPUT_SHAPE
    )
    from neural_network import board_to_bitboard, evaluate_position
    from ai import minimax, score_move, evaluate_board_classical


@dataclass
class GameRecord:
    """Record of a self-play game."""
    positions: List[np.ndarray]  # Board states
    values: List[float]          # Position values (assigned after game ends)
    moves: List[str]             # UCI move strings
    outcome: float               # Final outcome: 1.0 (white), 0.0 (black), 0.5 (draw)
    num_moves: int               # Total moves
    termination: str             # How game ended


class SelfPlayEngine:
    """
    Engine for generating self-play games.

    Uses minimax with neural network evaluation to play games against
    itself and record training data.
    """

    def __init__(self, model=None, use_nn: bool = True,
                 depth: int = RL_MINIMAX_DEPTH,
                 time_limit: float = RL_MINIMAX_TIME,
                 temperature: float = RL_TEMPERATURE,
                 temp_threshold: int = RL_TEMPERATURE_THRESHOLD):
        """
        Initialize self-play engine.

        Args:
            model: Neural network model for evaluation
            use_nn: Whether to use NN evaluation (False = classical)
            depth: Minimax search depth
            time_limit: Time limit per move
            temperature: Move selection temperature
            temp_threshold: Moves before switching to greedy
        """
        self.model = model
        self.use_nn = use_nn and model is not None
        self.depth = depth
        self.time_limit = time_limit
        self.temperature = temperature
        self.temp_threshold = temp_threshold

        # Statistics
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0

    def get_eval_function(self):
        """Get the evaluation function based on settings."""
        if self.use_nn and self.model is not None:
            return lambda board: evaluate_position(self.model, board)
        else:
            return evaluate_board_classical

    def select_move_with_temperature(self, board: chess.Board,
                                      move_scores: List[Tuple[chess.Move, float]],
                                      move_number: int) -> chess.Move:
        """
        Select move using temperature-based sampling.

        Higher temperature = more exploration.
        After temp_threshold moves, switch to greedy (best move).

        Args:
            board: Current board state
            move_scores: List of (move, score) tuples
            move_number: Current move number

        Returns:
            Selected move
        """
        if not move_scores:
            return random.choice(list(board.legal_moves))

        # After threshold, be greedy
        if move_number >= self.temp_threshold or self.temperature == 0:
            return max(move_scores, key=lambda x: x[1])[0]

        # Temperature-based selection
        scores = np.array([s for _, s in move_scores])

        # Normalize scores to prevent overflow
        scores = scores - scores.max()

        # Apply temperature
        if self.temperature > 0:
            probs = np.exp(scores / self.temperature)
            probs = probs / probs.sum()
        else:
            probs = np.zeros_like(scores)
            probs[np.argmax(scores)] = 1.0

        # Sample from distribution
        idx = np.random.choice(len(move_scores), p=probs)
        return move_scores[idx][0]

    def get_move_with_eval(self, board: chess.Board, move_number: int) -> Tuple[chess.Move, float]:
        """
        Get best move using minimax search.

        Args:
            board: Current board position
            move_number: Current move number (for temperature)

        Returns:
            Tuple of (selected_move, evaluation)
        """
        eval_func = self.get_eval_function()
        is_maximizing = board.turn == chess.WHITE
        start_time = time.time()

        # Get all legal moves with their scores
        moves = list(board.legal_moves)
        if not moves:
            return None, 0.0

        # Sort moves for better alpha-beta pruning
        moves.sort(key=lambda m: score_move(board, m), reverse=True)

        # Run minimax
        try:
            best_score, best_move = minimax(
                board.copy(), self.depth,
                -float('inf'), float('inf'),
                is_maximizing, start_time, self.time_limit,
                eval_func
            )

            if best_move is None:
                best_move = random.choice(moves)
                best_score = eval_func(board)

        except TimeoutError:
            # If timeout, use first legal move
            best_move = moves[0]
            best_score = eval_func(board)

        # Collect move scores for temperature selection
        move_scores = [(best_move, best_score)]

        # For temperature selection, we could evaluate more moves
        # But for efficiency, just use the minimax result
        selected_move = self.select_move_with_temperature(
            board, move_scores, move_number
        )

        return selected_move, best_score

    def play_game(self, max_moves: int = 300) -> GameRecord:
        """
        Play a complete self-play game.

        Args:
            max_moves: Maximum moves before declaring draw

        Returns:
            GameRecord with positions, values, and outcome
        """
        board = chess.Board()
        positions = []
        pre_move_evals = []  # Evaluation before each move
        moves = []
        move_number = 0

        while not board.is_game_over() and move_number < max_moves:
            # Record position before move
            position_tensor = board_to_bitboard(board)[0]
            positions.append(position_tensor)

            # Get move and evaluation
            move, eval_score = self.get_move_with_eval(board, move_number)

            if move is None:
                break

            # Store evaluation from white's perspective
            if board.turn == chess.BLACK:
                eval_score = -eval_score
            pre_move_evals.append(eval_score)

            # Make move
            moves.append(move.uci())
            board.push(move)
            move_number += 1

        # Determine outcome
        if board.is_checkmate():
            # Winner is the side that delivered checkmate
            outcome = 0.0 if board.turn == chess.WHITE else 1.0
            termination = "checkmate"
        elif board.is_stalemate():
            outcome = 0.5
            termination = "stalemate"
        elif board.is_insufficient_material():
            outcome = 0.5
            termination = "insufficient_material"
        elif board.is_seventyfive_moves():
            outcome = 0.5
            termination = "75_move_rule"
        elif board.is_fivefold_repetition():
            outcome = 0.5
            termination = "fivefold_repetition"
        elif move_number >= max_moves:
            outcome = 0.5
            termination = "max_moves"
        else:
            outcome = 0.5
            termination = "unknown"

        # Assign values to positions based on outcome
        # Use TD-Lambda style: blend evaluation with final outcome
        values = self.assign_values(positions, pre_move_evals, outcome)

        # Update statistics
        self.games_played += 1
        if outcome == 1.0:
            self.white_wins += 1
        elif outcome == 0.0:
            self.black_wins += 1
        else:
            self.draws += 1

        return GameRecord(
            positions=positions,
            values=values,
            moves=moves,
            outcome=outcome,
            num_moves=move_number,
            termination=termination
        )

    def assign_values(self, positions: List[np.ndarray],
                      evals: List[float], outcome: float,
                      lambda_decay: float = 0.7) -> List[float]:
        """
        Assign values to positions using TD-Lambda style blending.

        Blends immediate evaluation with final outcome:
        - Early positions: more weight on evaluation
        - Late positions: more weight on outcome

        Args:
            positions: List of position tensors
            evals: List of pre-move evaluations
            outcome: Final game outcome
            lambda_decay: Decay factor for temporal difference

        Returns:
            List of position values
        """
        n = len(positions)
        if n == 0:
            return []

        values = []
        # Convert outcome to [-1, 1]: 1.0->1, 0.5->0, 0.0->-1
        final_value = outcome * 2 - 1

        for i in range(n):
            # Distance from end of game (0 = last move, 1 = second to last, etc.)
            distance_from_end = n - 1 - i

            # Weight on final outcome increases as game progresses
            # lambda_decay^distance gives weight on outcome
            outcome_weight = lambda_decay ** distance_from_end

            # Blend evaluation with outcome
            eval_score = evals[i] if i < len(evals) else 0.0

            # Normalize evaluation to [-1, 1] range (assuming centipawn scale)
            eval_normalized = np.clip(eval_score / 600.0, -1.0, 1.0)

            # Blend
            value = (1 - outcome_weight) * eval_normalized + outcome_weight * final_value

            # Adjust for side to move (position i is before move i)
            # Even indices = white to move, odd = black to move
            if i % 2 == 1:  # Black's perspective
                value = -value

            values.append(float(value))

        return values

    def play_games(self, num_games: int, verbose: bool = True) -> List[GameRecord]:
        """
        Play multiple self-play games.

        Args:
            num_games: Number of games to play
            verbose: Whether to print progress

        Returns:
            List of GameRecords
        """
        games = []
        start_time = time.time()

        for i in range(num_games):
            game = self.play_game()
            games.append(game)

            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                games_per_sec = (i + 1) / elapsed
                print(f"  Played {i+1}/{num_games} games "
                      f"({games_per_sec:.2f} games/sec)")

        if verbose:
            total_time = time.time() - start_time
            print(f"\nCompleted {num_games} games in {total_time:.1f}s")
            print(f"Results: W={self.white_wins}, B={self.black_wins}, D={self.draws}")

        return games

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        total = self.games_played or 1
        return {
            'games_played': self.games_played,
            'white_wins': self.white_wins,
            'black_wins': self.black_wins,
            'draws': self.draws,
            'white_win_rate': self.white_wins / total,
            'black_win_rate': self.black_wins / total,
            'draw_rate': self.draws / total,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0


def games_to_training_data(games: List[GameRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert game records to training data.

    Args:
        games: List of GameRecord objects

    Returns:
        Tuple of (positions, values) numpy arrays
    """
    all_positions = []
    all_values = []

    for game in games:
        all_positions.extend(game.positions)
        all_values.extend(game.values)

    return np.array(all_positions), np.array(all_values)


if __name__ == '__main__':
    # Test self-play engine
    print("Testing Self-Play Engine...")

    engine = SelfPlayEngine(model=None, use_nn=False, depth=2, time_limit=0.5)

    print("\nPlaying 5 test games with classical evaluation...")
    games = engine.play_games(num_games=5, verbose=True)

    print(f"\nGame lengths: {[g.num_moves for g in games]}")
    print(f"Terminations: {[g.termination for g in games]}")

    # Convert to training data
    X, y = games_to_training_data(games)
    print(f"\nTraining data: {X.shape} positions, values in [{y.min():.2f}, {y.max():.2f}]")
