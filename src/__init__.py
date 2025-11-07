"""
Chess Neural Network Game
A modular chess game with neural network AI.
"""

__version__ = "2.0.0"
__author__ = "Lance"

from .game import ChessGame
from .neural_network import load_model, train_model, load_and_parse_pgn
from . import config

__all__ = ['ChessGame', 'load_model', 'train_model', 'load_and_parse_pgn', 'config']
