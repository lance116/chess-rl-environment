"""
AI move selection using minimax with alpha-beta pruning.
"""
import math
import time
import random
import chess
from typing import Optional, Tuple
from tensorflow import keras

try:
    from .config import PIECE_VALUES, POSITION_SCORES, BOARD_SIZE
except ImportError:
    from config import PIECE_VALUES, POSITION_SCORES, BOARD_SIZE


def evaluate_board_classical(board: chess.Board) -> float:
    """
    Classical chess evaluation using material and piece-square tables.

    Args:
        board: python-chess Board object

    Returns:
        Evaluation score in centipawns (positive = White advantage)
    """
    if board.is_insufficient_material() or board.is_stalemate():
        return 0.0

    if board.is_checkmate():
        return -100000.0 if board.turn == chess.WHITE else 100000.0

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type_char = piece.symbol().upper()
            base_value = PIECE_VALUES.get(piece_type_char, 0)

            # Positional bonus
            pos_score = 0
            if piece_type_char in POSITION_SCORES:
                r = 7 - (square // 8)
                c = square % 8
                # Flip board for black pieces
                pos_r = r if piece.color == chess.WHITE else BOARD_SIZE - 1 - r
                pos_c = c
                try:
                    pos_score = POSITION_SCORES[piece_type_char][pos_r][pos_c]
                except IndexError:
                    pos_score = 0

            current_piece_score = base_value + pos_score
            score += current_piece_score if piece.color == chess.WHITE else -current_piece_score

    return score


def score_move(board: chess.Board, move: chess.Move) -> int:
    """
    Score a move for move ordering (MVV-LVA).

    Args:
        board: python-chess Board object
        move: chess.Move to score

    Returns:
        Score (higher = should be searched first)
    """
    score = 0

    # Promotion bonus
    if move.promotion is not None:
        score += PIECE_VALUES.get(chess.piece_symbol(move.promotion).upper(), 900)

    # Capture bonus (Most Valuable Victim - Least Valuable Attacker)
    if board.is_capture(move):
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)

        if victim is None:  # En passant
            if board.is_en_passant(move):
                victim_value = PIECE_VALUES['P']
            else:
                victim_value = 0
        else:
            victim_value = PIECE_VALUES.get(victim.symbol().upper(), 0)

        attacker_value = PIECE_VALUES.get(attacker.symbol().upper(), 0)
        score += 1000 + (victim_value * 10) - attacker_value

    return score


def minimax(board: chess.Board, depth: int, alpha: float, beta: float,
           maximizing_player: bool, start_time: float, time_limit: float,
           eval_func) -> Tuple[float, Optional[chess.Move]]:
    """
    Minimax algorithm with alpha-beta pruning.

    Args:
        board: python-chess Board object
        depth: Search depth remaining
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        maximizing_player: True if maximizing, False if minimizing
        start_time: Time when search started
        time_limit: Maximum time for search
        eval_func: Function to evaluate leaf positions

    Returns:
        Tuple of (evaluation, best_move)
    """
    # Time check
    if time.time() - start_time > time_limit:
        raise TimeoutError

    # Terminal node checks
    if board.is_game_over():
        if board.is_checkmate():
            return (-math.inf if maximizing_player else math.inf), None
        else:
            return 0.0, None

    # Leaf node - evaluate
    if depth == 0:
        return eval_func(board), None

    # Move ordering - search promising moves first
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: score_move(board, m), reverse=True)

    best_move = None

    if maximizing_player:
        max_eval = -math.inf
        for move in moves:
            board.push(move)
            try:
                evaluation, _ = minimax(board, depth - 1, alpha, beta, False,
                                       start_time, time_limit, eval_func)
            except TimeoutError:
                board.pop()
                raise TimeoutError
            board.pop()

            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move

            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break  # Beta cutoff

        return max_eval, best_move

    else:
        min_eval = math.inf
        for move in moves:
            board.push(move)
            try:
                evaluation, _ = minimax(board, depth - 1, alpha, beta, True,
                                       start_time, time_limit, eval_func)
            except TimeoutError:
                board.pop()
                raise TimeoutError
            board.pop()

            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move

            beta = min(beta, evaluation)
            if beta <= alpha:
                break  # Alpha cutoff

        return min_eval, best_move


def get_ai_move(board: chess.Board, time_limit: float, use_neural_network: bool = False,
               nn_model: Optional[keras.Model] = None) -> Optional[chess.Move]:
    """
    Get the best move for the AI using iterative deepening minimax.

    Args:
        board: python-chess Board object
        time_limit: Maximum time in seconds
        use_neural_network: Whether to use neural network evaluation
        nn_model: Neural network model (required if use_neural_network=True)

    Returns:
        Best chess.Move found, or None if no move available
    """
    start_time = time.time()
    depth = 0
    best_move_overall = None
    best_score_overall = -math.inf if board.turn == chess.WHITE else math.inf
    is_maximizing = board.turn == chess.WHITE

    # Select evaluation function
    if use_neural_network and nn_model is not None:
        try:
            from .neural_network import evaluate_position
        except ImportError:
            from neural_network import evaluate_position
        eval_func = lambda b: evaluate_position(nn_model, b)
        eval_type = "NN"
    else:
        eval_func = evaluate_board_classical
        eval_type = "Classical"

    print(f"\nAI ({'White' if is_maximizing else 'Black'}) thinking... "
          f"(Eval: {eval_type})")

    try:
        # Iterative deepening - search increasing depths
        while True:
            depth += 1
            elapsed_time = time.time() - start_time
            remaining_time = time_limit - elapsed_time

            # Stop if running out of time
            if remaining_time <= 0.1:
                print(f"Stopping at depth {depth-1}: Low time remaining")
                break

            if elapsed_time > time_limit * 0.8 and depth > 2:
                print(f"Stopping at depth {depth-1}: >80% time used")
                break

            print(f"  Searching depth {depth}...")
            depth_start_time = time.time()

            try:
                score, move = minimax(board.copy(), depth, -math.inf, math.inf,
                                     is_maximizing, start_time, time_limit, eval_func)

                if move is not None:
                    best_move_overall = move
                    best_score_overall = score
                    depth_time = time.time() - depth_start_time
                    print(f"  Depth {depth} complete: {move.uci()} "
                          f"(score: {score:.1f}, time: {depth_time:.2f}s)")
                else:
                    print(f"  No move at depth {depth} (game over)")
                    break

            except TimeoutError:
                print(f"  Timeout at depth {depth}, using depth {depth-1} result")
                break

    except Exception as e:
        print(f"Error during AI search: {e}")
        if best_move_overall:
            print("Falling back to last completed depth move")
        else:
            print("No move found due to error")

    final_time = time.time() - start_time
    if best_move_overall:
        print(f"AI decision: {best_move_overall.uci()} "
              f"(score: {best_score_overall:.1f}, time: {final_time:.2f}s)\n")
    else:
        print("AI could not find a move - choosing randomly")
        legal_moves = list(board.legal_moves)
        if legal_moves:
            best_move_overall = random.choice(legal_moves)

    return best_move_overall


def get_material_score(board: chess.Board, color: chess.Color) -> int:
    """
    Calculate material score for one side (for display).

    Args:
        board: python-chess Board object
        color: chess.WHITE or chess.BLACK

    Returns:
        Material score in pawns (not centipawns)
    """
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            piece_type = piece.piece_type
            if piece_type != chess.KING:
                score += PIECE_VALUES.get(piece.symbol().upper(), 0)
    return score // 100
