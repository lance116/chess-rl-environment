#!/usr/bin/env python3
"""
Main entry point for Chess Neural Network game.
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from game import ChessGame


def main():
    """Run the chess game."""
    # Set to True to use neural network evaluation (if trained)
    use_nn = True

    print("=" * 50)
    print("Chess Neural Network Game v2.0")
    print("=" * 50)
    print(f"AI Evaluation: {'Neural Network' if use_nn else 'Classical'}")
    print("=" * 50)
    print()

    game = ChessGame(use_neural_network=use_nn)
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
