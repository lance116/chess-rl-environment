#!/usr/bin/env python3
"""
TD-Lambda Reinforcement Learning Training Script.

Usage:
    python scripts/train_rl.py                    # Full training
    python scripts/train_rl.py --quick            # Quick test (10 iterations)
    python scripts/train_rl.py --resume checkpoint.h5  # Resume from checkpoint
    python scripts/train_rl.py --supervised data.npz   # Pre-train on supervised data
"""
import os
import sys
import argparse
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    RL_TRAINING_ITERATIONS, RL_GAMES_PER_ITERATION,
    RL_UPDATES_PER_ITERATION, RL_CHECKPOINT_INTERVAL,
    RL_CHECKPOINT_DIR, INPUT_SHAPE
)
from neural_network import build_model, load_model
from td_training import TDLambdaTrainer
from stockfish_labeler import load_stockfish_labeled_data, augment_positions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='TD-Lambda Reinforcement Learning Training for Chess'
    )

    parser.add_argument('--iterations', type=int, default=RL_TRAINING_ITERATIONS,
                        help=f'Number of training iterations (default: {RL_TRAINING_ITERATIONS})')
    parser.add_argument('--games', type=int, default=RL_GAMES_PER_ITERATION,
                        help=f'Self-play games per iteration (default: {RL_GAMES_PER_ITERATION})')
    parser.add_argument('--updates', type=int, default=RL_UPDATES_PER_ITERATION,
                        help=f'Gradient updates per iteration (default: {RL_UPDATES_PER_ITERATION})')
    parser.add_argument('--checkpoint-interval', type=int, default=RL_CHECKPOINT_INTERVAL,
                        help=f'Iterations between checkpoints (default: {RL_CHECKPOINT_INTERVAL})')

    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (10 iterations, 20 games each)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')
    parser.add_argument('--supervised', type=str, default=None,
                        help='Pre-train on supervised data (npz file)')

    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--lambda', type=float, default=0.7, dest='lambda_param',
                        help='TD-Lambda parameter (default: 0.7)')

    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')

    return parser.parse_args()


def configure_tensorflow(use_gpu: bool = True):
    """Configure TensorFlow settings."""
    import tensorflow as tf

    if not use_gpu:
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled, using CPU")
    else:
        # Check for Apple Silicon Metal
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found GPU: {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("No GPU found, using CPU")


def supervised_pretrain(model, data_path: str, epochs: int = 10):
    """
    Pre-train model on supervised data before RL training.

    Args:
        model: Neural network model
        data_path: Path to npz file with positions and labels
        epochs: Training epochs
    """
    print("\n" + "=" * 60)
    print("Supervised Pre-Training")
    print("=" * 60)

    # Load data
    print(f"Loading data from {data_path}...")
    positions, labels = load_stockfish_labeled_data(data_path)

    print(f"Loaded {len(positions)} positions")
    print(f"Label range: [{labels.min():.3f}, {labels.max():.3f}]")

    # Augment data
    print("Augmenting with horizontal flip...")
    positions, labels = augment_positions(positions, labels)
    print(f"Augmented to {len(positions)} positions")

    # Train
    print(f"\nTraining for {epochs} epochs...")
    from tensorflow import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
        )
    ]

    history = model.fit(
        positions, labels.reshape(-1, 1),
        epochs=epochs,
        batch_size=256,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nPre-training complete!")
    print(f"Final loss: {history.history['val_loss'][-1]:.4f}")

    return model


def main():
    """Main training entry point."""
    args = parse_args()

    # Configure TensorFlow
    configure_tensorflow(use_gpu=not args.no_gpu)

    print("\n" + "=" * 60)
    print("Chess RL Environment - TD-Lambda Training")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Quick mode overrides
    if args.quick:
        args.iterations = 10
        args.games = 20
        args.updates = 50
        args.checkpoint_interval = 5
        print("\nQUICK MODE: Running abbreviated training")

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Games per iteration: {args.games}")
    print(f"  Updates per iteration: {args.updates}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Lambda: {args.lambda_param}")

    # Build or load model
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model = load_model(args.resume)
        if model is None:
            print("Failed to load checkpoint, starting fresh")
            model = build_model(INPUT_SHAPE)
    else:
        print("\nBuilding new model...")
        model = build_model(INPUT_SHAPE)

    model.summary()

    # Supervised pre-training if requested
    if args.supervised:
        if os.path.exists(args.supervised):
            model = supervised_pretrain(model, args.supervised)
        else:
            print(f"Warning: Supervised data file not found: {args.supervised}")

    # Create trainer
    trainer = TDLambdaTrainer(
        model=model,
        learning_rate=args.learning_rate,
        lambda_param=args.lambda_param
    )

    # Load existing replay buffer if resuming
    if args.resume:
        trainer.replay_buffer.load()

    # Run training
    try:
        metrics = trainer.run_training(
            num_iterations=args.iterations,
            games_per_iter=args.games,
            updates_per_iter=args.updates,
            checkpoint_interval=args.checkpoint_interval
        )

        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Total iterations: {len(metrics)}")
        print(f"Total games: {trainer.total_games}")
        print(f"Total positions: {trainer.total_positions}")

        if metrics:
            final_loss = metrics[-1]['loss']
            print(f"Final loss: {final_loss:.4f}")

            # Loss trend
            losses = [m['loss'] for m in metrics if m.get('loss', 0) > 0]
            if len(losses) > 1:
                improvement = losses[0] - losses[-1]
                print(f"Loss improvement: {improvement:.4f}")

        print("=" * 60)
        print(f"Checkpoints saved to: {RL_CHECKPOINT_DIR}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(suffix="interrupted")
        trainer.save_training_log()

    return 0


if __name__ == "__main__":
    sys.exit(main())
