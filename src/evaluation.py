"""
Elo Estimation and Model Evaluation System.

Estimates the playing strength of trained models by playing
matches against Stockfish at various skill levels.
"""
import os
import math
import time
import chess
import chess.engine
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    from .config import (
        STOCKFISH_SKILL_LEVELS, STOCKFISH_TIME_PER_MOVE,
        RL_MINIMAX_DEPTH, RL_MINIMAX_TIME, RL_EVAL_GAMES
    )
    from .neural_network import evaluate_position
    from .ai import minimax, evaluate_board_classical
    from .stockfish_labeler import find_stockfish
except ImportError:
    from config import (
        STOCKFISH_SKILL_LEVELS, STOCKFISH_TIME_PER_MOVE,
        RL_MINIMAX_DEPTH, RL_MINIMAX_TIME, RL_EVAL_GAMES
    )
    from neural_network import evaluate_position
    from ai import minimax, evaluate_board_classical
    from stockfish_labeler import find_stockfish


# Stockfish skill level to approximate Elo mapping
# Based on community estimates and Stockfish documentation
STOCKFISH_SKILL_ELO = {
    0: 800,
    1: 1000,
    2: 1100,
    3: 1200,
    4: 1300,
    5: 1400,
    6: 1500,
    7: 1600,
    8: 1700,
    9: 1800,
    10: 1900,
    11: 2000,
    12: 2100,
    13: 2200,
    14: 2350,
    15: 2500,
    16: 2600,
    17: 2700,
    18: 2850,
    19: 3000,
    20: 3200,
}


@dataclass
class MatchResult:
    """Result of a single match."""
    wins: int
    losses: int
    draws: int
    games: int

    @property
    def score(self) -> float:
        """Score from model's perspective (0-1)."""
        if self.games == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def win_rate(self) -> float:
        """Win rate (0-1)."""
        if self.games == 0:
            return 0.0
        return self.wins / self.games


def expected_score(elo_diff: float) -> float:
    """
    Calculate expected score given Elo difference.

    Args:
        elo_diff: Rating difference (positive = higher rated)

    Returns:
        Expected score (0-1)
    """
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400))


def elo_from_score(score: float, opponent_elo: float) -> float:
    """
    Calculate Elo rating from match score.

    Args:
        score: Match score (0-1)
        opponent_elo: Opponent's Elo rating

    Returns:
        Estimated Elo rating
    """
    if score <= 0.01:
        return opponent_elo - 400
    if score >= 0.99:
        return opponent_elo + 400

    # Inverse of expected_score formula
    elo_diff = -400 * math.log10((1.0 / score) - 1)
    return opponent_elo + elo_diff


class EloEstimator:
    """
    Estimates Elo rating by playing against Stockfish.

    Uses matches against Stockfish at various skill levels
    to estimate the model's playing strength.
    """

    def __init__(self, model=None, stockfish_path: Optional[str] = None):
        """
        Initialize Elo estimator.

        Args:
            model: Neural network model for evaluation
            stockfish_path: Path to Stockfish executable
        """
        self.model = model
        self.stockfish_path = stockfish_path or find_stockfish()

        if self.stockfish_path is None:
            print("Warning: Stockfish not found. Elo estimation will not work.")

    def get_model_move(self, board: chess.Board, depth: int = RL_MINIMAX_DEPTH,
                       time_limit: float = RL_MINIMAX_TIME) -> chess.Move:
        """
        Get move from model using minimax search.

        Args:
            board: Current position
            depth: Search depth
            time_limit: Time limit

        Returns:
            Best move
        """
        if self.model is not None:
            eval_func = lambda b: evaluate_position(self.model, b)
        else:
            eval_func = evaluate_board_classical

        is_maximizing = board.turn == chess.WHITE
        start_time = time.time()

        try:
            _, move = minimax(
                board.copy(), depth,
                -float('inf'), float('inf'),
                is_maximizing, start_time, time_limit,
                eval_func
            )

            if move is None:
                moves = list(board.legal_moves)
                move = moves[0] if moves else None

        except Exception as e:
            print(f"Error in minimax: {e}")
            moves = list(board.legal_moves)
            move = moves[0] if moves else None

        return move

    def play_game(self, engine: chess.engine.SimpleEngine,
                  skill_level: int, model_is_white: bool,
                  max_moves: int = 200) -> str:
        """
        Play a single game against Stockfish.

        Args:
            engine: Stockfish engine instance
            skill_level: Stockfish skill level (0-20)
            model_is_white: Whether model plays white
            max_moves: Maximum moves before draw

        Returns:
            Result string: "1-0", "0-1", or "1/2-1/2"
        """
        board = chess.Board()

        # Configure Stockfish skill level
        engine.configure({"Skill Level": skill_level})

        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            is_model_turn = (board.turn == chess.WHITE) == model_is_white

            if is_model_turn:
                # Model's turn
                move = self.get_model_move(board)
            else:
                # Stockfish's turn
                result = engine.play(
                    board,
                    chess.engine.Limit(time=STOCKFISH_TIME_PER_MOVE)
                )
                move = result.move

            if move is None:
                break

            board.push(move)
            move_count += 1

        if board.is_checkmate():
            return "1-0" if board.turn == chess.BLACK else "0-1"
        else:
            return "1/2-1/2"

    def play_match(self, skill_level: int, num_games: int = RL_EVAL_GAMES,
                   verbose: bool = True) -> MatchResult:
        """
        Play a match against Stockfish at given skill level.

        Args:
            skill_level: Stockfish skill level (0-20)
            num_games: Number of games to play
            verbose: Print progress

        Returns:
            MatchResult with wins/losses/draws
        """
        if self.stockfish_path is None:
            print("Error: Stockfish not found")
            return MatchResult(0, 0, 0, 0)

        wins = 0
        losses = 0
        draws = 0

        if verbose:
            sf_elo = STOCKFISH_SKILL_ELO.get(skill_level, 1500)
            print(f"\nPlaying {num_games} games vs Stockfish (skill {skill_level}, ~{sf_elo} Elo)...")

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            for game_num in range(num_games):
                # Alternate colors
                model_is_white = (game_num % 2 == 0)

                result = self.play_game(engine, skill_level, model_is_white)

                # Interpret result from model's perspective
                if result == "1-0":
                    if model_is_white:
                        wins += 1
                    else:
                        losses += 1
                elif result == "0-1":
                    if model_is_white:
                        losses += 1
                    else:
                        wins += 1
                else:
                    draws += 1

                if verbose and (game_num + 1) % 5 == 0:
                    print(f"  {game_num + 1}/{num_games}: W={wins} L={losses} D={draws}")

        if verbose:
            match_result = MatchResult(wins, losses, draws, num_games)
            print(f"Match result: W={wins} L={losses} D={draws} (Score: {match_result.score:.2%})")

        return MatchResult(wins, losses, draws, num_games)

    def estimate_elo(self, num_games_per_level: int = 10,
                     skill_levels: List[int] = None,
                     verbose: bool = True) -> Tuple[float, Dict]:
        """
        Estimate Elo by playing against multiple Stockfish levels.

        Args:
            num_games_per_level: Games per skill level
            skill_levels: Skill levels to test (default: config values)
            verbose: Print progress

        Returns:
            Tuple of (estimated_elo, detailed_results)
        """
        if skill_levels is None:
            skill_levels = STOCKFISH_SKILL_LEVELS

        if verbose:
            print("\n" + "=" * 60)
            print("Elo Estimation via Stockfish Matches")
            print("=" * 60)

        results = {}
        elo_estimates = []
        weights = []

        for level in skill_levels:
            match_result = self.play_match(level, num_games_per_level, verbose)
            results[level] = match_result

            # Estimate Elo from this match
            if match_result.games > 0:
                sf_elo = STOCKFISH_SKILL_ELO.get(level, 1500)
                estimated = elo_from_score(match_result.score, sf_elo)
                elo_estimates.append(estimated)
                # Weight by number of games and confidence
                weights.append(match_result.games)

        # Weighted average of estimates
        if elo_estimates:
            total_weight = sum(weights)
            final_elo = sum(e * w for e, w in zip(elo_estimates, weights)) / total_weight
        else:
            final_elo = 1200  # Default

        if verbose:
            print("\n" + "=" * 60)
            print(f"Estimated Elo: {final_elo:.0f}")
            print("=" * 60)

            for level, result in results.items():
                sf_elo = STOCKFISH_SKILL_ELO.get(level, 1500)
                level_elo = elo_from_score(result.score, sf_elo)
                print(f"  Skill {level:2d} (~{sf_elo}): "
                      f"W={result.wins} L={result.losses} D={result.draws} "
                      f"-> {level_elo:.0f} Elo")

        return final_elo, results

    def quick_estimate(self, verbose: bool = True) -> float:
        """
        Quick Elo estimate using fewer games.

        Args:
            verbose: Print progress

        Returns:
            Estimated Elo
        """
        # Test against skill levels 5, 10, 15 (1400, 1900, 2500)
        quick_levels = [5, 10, 15]
        elo, _ = self.estimate_elo(
            num_games_per_level=5,
            skill_levels=quick_levels,
            verbose=verbose
        )
        return elo


def evaluate_model(model, num_games: int = 20, verbose: bool = True) -> float:
    """
    Convenience function to evaluate a model's Elo.

    Args:
        model: Neural network model
        num_games: Games per skill level
        verbose: Print progress

    Returns:
        Estimated Elo
    """
    estimator = EloEstimator(model)
    elo, _ = estimator.estimate_elo(num_games_per_level=num_games, verbose=verbose)
    return elo


if __name__ == '__main__':
    # Test Elo estimation with classical evaluation (no model)
    print("Testing Elo Estimation with Classical Evaluation...")

    estimator = EloEstimator(model=None)

    if estimator.stockfish_path:
        print(f"Found Stockfish at: {estimator.stockfish_path}")

        # Quick estimate
        elo = estimator.quick_estimate()
        print(f"\nEstimated Elo (classical eval): {elo:.0f}")
    else:
        print("Stockfish not found. Please install:")
        print("  brew install stockfish  (macOS)")
        print("  apt install stockfish   (Ubuntu)")
