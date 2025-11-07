#!/usr/bin/env python3
"""
Training script for the chess neural network.
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from neural_network import build_model, load_and_parse_pgn, train_model
from config import INPUT_SHAPE, MODEL_WEIGHTS_FILE, PGN_DATABASE_PATH, TRAINING_GAME_LIMIT


def main():
    """Train the chess evaluation model."""
    print("=" * 60)
    print("Chess Neural Network Training")
    print("=" * 60)
    print()

    # Build model
    print("Building model...")
    model = build_model(INPUT_SHAPE)
    model.summary()
    print()

    # Load training data
    print(f"Loading PGN data from: {PGN_DATABASE_PATH}")
    print(f"Game limit: {TRAINING_GAME_LIMIT}")
    print()

    X_train, y_train = load_and_parse_pgn(PGN_DATABASE_PATH, game_limit=TRAINING_GAME_LIMIT)

    if len(X_train) == 0:
        print("\nERROR: No training data loaded!")
        print("Please check:")
        print(f"  1. PGN_DATABASE_PATH in config.py: {PGN_DATABASE_PATH}")
        print("  2. Directory exists and contains .pgn files")
        print("  3. PGN files are valid")
        return 1

    print(f"\nLoaded {len(X_train)} training positions")
    print()

    # Train model
    history = train_model(model, X_train, y_train, MODEL_WEIGHTS_FILE)

    if history is None:
        print("\nTraining failed!")
        return 1

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model weights saved to: {MODEL_WEIGHTS_FILE}")
    print("=" * 60)
    print()
    print("To use the neural network in gameplay:")
    print("  1. Open src/main.py")
    print("  2. Set use_nn = True")
    print("  3. Run: python src/main.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
