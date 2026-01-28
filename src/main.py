#!/usr/bin/env python3
"""
Main entry point for Chess Neural Network game.

Usage:
    python main.py                  # Default: try RL model, fallback to classical
    python main.py --classical      # Use classical evaluation (no neural network)
    python main.py --model PATH     # Use specific model weights file
"""
import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from game import ChessGame
from config import MODEL_WEIGHTS_FILE, RL_BEST_MODEL_FILE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Chess Neural Network Game with RL AI'
    )
    parser.add_argument('--classical', action='store_true',
                        help='Use classical evaluation (no neural network)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights file')
    parser.add_argument('--supervised', action='store_true',
                        help='Use supervised learning model instead of RL')
    return parser.parse_args()


def main():
    """Run the chess game."""
    args = parse_args()

    # Determine model to use
    if args.classical:
        use_nn = False
        model_path = None
        eval_type = "Classical"
    elif args.model:
        use_nn = True
        model_path = args.model
        eval_type = f"Custom ({os.path.basename(args.model)})"
    elif args.supervised:
        use_nn = True
        model_path = MODEL_WEIGHTS_FILE
        eval_type = "Supervised NN"
    else:
        # Default: try RL model first, then supervised, then classical
        use_nn = True
        if os.path.exists(RL_BEST_MODEL_FILE):
            model_path = RL_BEST_MODEL_FILE
            eval_type = "RL Model"
        elif os.path.exists(MODEL_WEIGHTS_FILE):
            model_path = MODEL_WEIGHTS_FILE
            eval_type = "Supervised NN"
        else:
            use_nn = False
            model_path = None
            eval_type = "Classical (no model found)"

    print("=" * 50)
    print("Chess Neural Network Game v2.0")
    print("=" * 50)
    print(f"AI Evaluation: {eval_type}")
    if model_path:
        print(f"Model: {model_path}")
    print("=" * 50)
    print()

    game = ChessGame(use_neural_network=use_nn, model_path=model_path)
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
