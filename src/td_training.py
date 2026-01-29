"""
TD-Lambda Reinforcement Learning Training Loop.

Implements temporal difference learning with eligibility traces
to train the chess evaluation neural network through self-play.
"""
import os
import json
import time
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise ImportError("TensorFlow is required for TD-Lambda training")

try:
    from .config import (
        RL_LAMBDA, RL_GAMMA, RL_LEARNING_RATE, RL_BATCH_SIZE,
        RL_GAMES_PER_ITERATION, RL_TRAINING_ITERATIONS, RL_UPDATES_PER_ITERATION,
        RL_REPLAY_MIN_SIZE, RL_CHECKPOINT_INTERVAL, RL_CHECKPOINT_DIR,
        RL_BEST_MODEL_FILE, RL_TRAINING_LOG_FILE, INPUT_SHAPE
    )
    from .neural_network import build_model, board_to_bitboard
    from .replay_buffer import ReplayBuffer
    from .self_play import SelfPlayEngine, games_to_training_data
    from .evaluation import EloEstimator
except ImportError:
    from config import (
        RL_LAMBDA, RL_GAMMA, RL_LEARNING_RATE, RL_BATCH_SIZE,
        RL_GAMES_PER_ITERATION, RL_TRAINING_ITERATIONS, RL_UPDATES_PER_ITERATION,
        RL_REPLAY_MIN_SIZE, RL_CHECKPOINT_INTERVAL, RL_CHECKPOINT_DIR,
        RL_BEST_MODEL_FILE, RL_TRAINING_LOG_FILE, INPUT_SHAPE
    )
    from neural_network import build_model, board_to_bitboard
    from replay_buffer import ReplayBuffer
    from self_play import SelfPlayEngine, games_to_training_data
    from evaluation import EloEstimator


class LearningRateScheduler:
    """
    Learning rate scheduler with warmup and decay.

    Schedule:
    1. Warmup phase: Linear increase from 0 to initial LR
    2. Stable phase: Constant learning rate
    3. Decay phase: Exponential decay
    """

    def __init__(self, initial_lr: float = RL_LEARNING_RATE,
                 warmup_iterations: int = 5,
                 decay_start: int = 50,
                 decay_factor: float = 0.95,
                 min_lr: float = 1e-6):
        self.initial_lr = initial_lr
        self.warmup_iterations = warmup_iterations
        self.decay_start = decay_start
        self.decay_factor = decay_factor
        self.min_lr = min_lr

    def get_lr(self, iteration: int) -> float:
        """Get learning rate for given iteration."""
        if iteration < self.warmup_iterations:
            # Warmup: linear increase
            return self.initial_lr * (iteration + 1) / self.warmup_iterations
        elif iteration < self.decay_start:
            # Stable phase
            return self.initial_lr
        else:
            # Decay phase
            decay_steps = iteration - self.decay_start
            lr = self.initial_lr * (self.decay_factor ** decay_steps)
            return max(lr, self.min_lr)


class OpponentPool:
    """
    Pool of previous model checkpoints for diverse self-play.

    Playing against previous versions prevents overfitting to
    the current model's weaknesses.
    """

    def __init__(self, max_opponents: int = 5):
        self.max_opponents = max_opponents
        self.opponents: List[str] = []  # Checkpoint paths

    def add(self, checkpoint_path: str):
        """Add a checkpoint to the pool."""
        if checkpoint_path not in self.opponents:
            self.opponents.append(checkpoint_path)
            # Keep only recent opponents
            if len(self.opponents) > self.max_opponents:
                self.opponents.pop(0)

    def sample(self) -> Optional[str]:
        """Sample a random opponent from the pool."""
        if not self.opponents:
            return None
        return np.random.choice(self.opponents)

    def __len__(self) -> int:
        return len(self.opponents)


class TDLambdaTrainer:
    """
    TD-Lambda trainer for chess neural network.

    Training loop:
    1. Generate self-play games
    2. Add positions to replay buffer
    3. Sample batches and perform gradient updates
    4. Periodically evaluate and checkpoint

    Enhancements:
    - Learning rate scheduling with warmup and decay
    - Opponent pool for diverse self-play
    - Automatic Elo evaluation
    """

    def __init__(self, model: Optional[keras.Model] = None,
                 learning_rate: float = RL_LEARNING_RATE,
                 lambda_param: float = RL_LAMBDA,
                 gamma: float = RL_GAMMA,
                 use_lr_scheduler: bool = True,
                 use_opponent_pool: bool = True):
        """
        Initialize TD-Lambda trainer.

        Args:
            model: Neural network model (builds new if None)
            learning_rate: Learning rate for gradient updates
            lambda_param: TD-Lambda trace decay
            gamma: Discount factor
            use_lr_scheduler: Enable learning rate scheduling
            use_opponent_pool: Enable opponent pool for self-play
        """
        self.model = model if model is not None else build_model(INPUT_SHAPE)
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.gamma = gamma

        # Optimizer with configurable learning rate
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile model for training
        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])

        # Training components
        self.replay_buffer = ReplayBuffer()
        self.self_play_engine = SelfPlayEngine(
            model=self.model, use_nn=True
        )

        # Training enhancements
        self.lr_scheduler = LearningRateScheduler(learning_rate) if use_lr_scheduler else None
        self.opponent_pool = OpponentPool() if use_opponent_pool else None

        # Training state
        self.iteration = 0
        self.total_games = 0
        self.total_positions = 0
        self.best_loss = float('inf')
        self.best_elo = 0

        # Training log
        self.training_log: List[Dict] = []

        # Ensure checkpoint directory exists
        os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)

    def generate_self_play_data(self, num_games: int = RL_GAMES_PER_ITERATION,
                                 verbose: bool = True) -> int:
        """
        Generate self-play games and add to replay buffer.

        Uses opponent pool when available: 20% of games are played against
        a previous version for training diversity.

        Args:
            num_games: Number of games to generate
            verbose: Whether to print progress

        Returns:
            Number of positions generated
        """
        if verbose:
            print(f"\nGenerating {num_games} self-play games...")

        # Update self-play engine with current model
        self.self_play_engine.model = self.model

        # Determine how many games to play against opponents from pool
        opponent_games = 0
        if self.opponent_pool is not None and len(self.opponent_pool) > 0:
            opponent_games = max(1, num_games // 5)  # 20% against old versions
            if verbose:
                print(f"  ({opponent_games} games vs opponent pool, "
                      f"{num_games - opponent_games} self-play)")

        # Generate self-play games (current model vs itself)
        self_play_count = num_games - opponent_games
        games = self.self_play_engine.play_games(self_play_count, verbose=verbose)

        # Generate opponent pool games
        if opponent_games > 0:
            opponent_path = self.opponent_pool.sample()
            if opponent_path and os.path.exists(opponent_path):
                try:
                    # Build a separate model for the opponent
                    opponent_model = build_model(INPUT_SHAPE)
                    opponent_model.load_weights(opponent_path)

                    # Create temporary engine with opponent model
                    opponent_engine = SelfPlayEngine(
                        model=opponent_model, use_nn=True
                    )
                    opponent_games_data = opponent_engine.play_games(
                        opponent_games, verbose=False
                    )
                    games.extend(opponent_games_data)

                    if verbose:
                        print(f"  Played {opponent_games} games vs "
                              f"{os.path.basename(opponent_path)}")
                except Exception as e:
                    if verbose:
                        print(f"  Opponent pool game failed: {e}")
                    # Fall back to self-play for these games
                    extra = self.self_play_engine.play_games(
                        opponent_games, verbose=False
                    )
                    games.extend(extra)

        # Add to replay buffer
        positions_added = 0
        for game in games:
            self.replay_buffer.add_game(game.positions, game.values)
            positions_added += len(game.positions)

        self.total_games += num_games
        self.total_positions += positions_added

        if verbose:
            print(f"Added {positions_added} positions to replay buffer")
            print(f"Buffer size: {len(self.replay_buffer)}")

        return positions_added

    def train_step(self, batch_size: int = RL_BATCH_SIZE) -> float:
        """
        Perform single training step on sampled batch.

        Args:
            batch_size: Batch size for training

        Returns:
            Training loss
        """
        # Sample from replay buffer
        states, values = self.replay_buffer.sample(batch_size)

        # Reshape values for Keras
        values = values.reshape(-1, 1)

        # Train on batch
        loss = self.model.train_on_batch(states, values)

        # train_on_batch returns loss (and metrics if any)
        if isinstance(loss, list):
            return loss[0]
        return loss

    def update_learning_rate(self):
        """Update learning rate based on scheduler."""
        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.get_lr(self.iteration)
            self.optimizer.learning_rate.assign(new_lr)
            return new_lr
        return self.learning_rate

    def train_iteration(self, num_updates: int = RL_UPDATES_PER_ITERATION,
                        batch_size: int = RL_BATCH_SIZE,
                        verbose: bool = True) -> Dict:
        """
        Perform one training iteration.

        Args:
            num_updates: Number of gradient updates
            batch_size: Batch size per update
            verbose: Whether to print progress

        Returns:
            Dictionary of training metrics
        """
        if not self.replay_buffer.is_ready(RL_REPLAY_MIN_SIZE):
            if verbose:
                print(f"Buffer not ready ({len(self.replay_buffer)}/{RL_REPLAY_MIN_SIZE})")
            return {'loss': 0.0, 'skipped': True, 'learning_rate': self.learning_rate}

        # Update learning rate (only when actually training)
        current_lr = self.update_learning_rate()

        losses = []
        start_time = time.time()

        for i in range(num_updates):
            loss = self.train_step(batch_size)
            losses.append(loss)

            if verbose and (i + 1) % 25 == 0:
                print(f"  Update {i+1}/{num_updates}: loss={np.mean(losses[-25:]):.4f}")

        train_time = time.time() - start_time
        avg_loss = np.mean(losses)

        metrics = {
            'iteration': self.iteration,
            'loss': float(avg_loss),
            'loss_std': float(np.std(losses)),
            'updates': num_updates,
            'train_time': train_time,
            'buffer_size': len(self.replay_buffer),
            'learning_rate': current_lr,
        }

        if verbose:
            print(f"  Average loss: {avg_loss:.4f} ({train_time:.1f}s)")

        return metrics

    def run_training(self, num_iterations: int = RL_TRAINING_ITERATIONS,
                     games_per_iter: int = RL_GAMES_PER_ITERATION,
                     updates_per_iter: int = RL_UPDATES_PER_ITERATION,
                     checkpoint_interval: int = RL_CHECKPOINT_INTERVAL,
                     eval_interval: int = 0,
                     verbose: bool = True) -> List[Dict]:
        """
        Run full TD-Lambda training loop.

        Args:
            num_iterations: Total training iterations
            games_per_iter: Self-play games per iteration
            updates_per_iter: Gradient updates per iteration
            checkpoint_interval: Iterations between checkpoints
            eval_interval: Iterations between Elo evaluations (0 = disabled)
            verbose: Whether to print progress

        Returns:
            List of training metrics per iteration
        """
        print("=" * 60)
        print("TD-Lambda Reinforcement Learning Training")
        print("=" * 60)
        print(f"Iterations: {num_iterations}")
        print(f"Games per iteration: {games_per_iter}")
        print(f"Updates per iteration: {updates_per_iter}")
        print(f"Lambda: {self.lambda_param}, Gamma: {self.gamma}")
        print(f"Learning rate: {self.learning_rate}")
        print("=" * 60)

        all_metrics = []
        training_start = time.time()

        for i in range(num_iterations):
            self.iteration = i + 1
            iter_start = time.time()

            print(f"\n{'='*60}")
            print(f"Iteration {self.iteration}/{num_iterations}")
            print(f"{'='*60}")

            # Generate self-play data
            positions = self.generate_self_play_data(games_per_iter, verbose)

            # Train on replay buffer
            metrics = self.train_iteration(updates_per_iter, verbose=verbose)
            metrics['positions_generated'] = positions
            metrics['total_games'] = self.total_games
            metrics['total_positions'] = self.total_positions

            # Add self-play stats
            sp_stats = self.self_play_engine.get_stats()
            metrics.update({f'selfplay_{k}': v for k, v in sp_stats.items()})

            all_metrics.append(metrics)
            self.training_log.append(metrics)

            # Checkpoint
            if self.iteration % checkpoint_interval == 0:
                checkpoint_path = self.save_checkpoint()

                # Add to opponent pool for diverse self-play
                if self.opponent_pool is not None and checkpoint_path:
                    self.opponent_pool.add(checkpoint_path)
                    if verbose:
                        print(f"Opponent pool size: {len(self.opponent_pool)}")

                # Save if best loss
                if metrics['loss'] < self.best_loss:
                    self.best_loss = metrics['loss']
                    self.save_best_model()

            # Periodic Elo evaluation
            if eval_interval > 0 and self.iteration % eval_interval == 0:
                elo = self.evaluate_elo(verbose=verbose)
                metrics['elo'] = elo
                if elo > self.best_elo:
                    self.best_elo = elo
                    if verbose:
                        print(f"New best Elo: {elo:.0f}")

            iter_time = time.time() - iter_start
            print(f"\nIteration time: {iter_time:.1f}s")

            # Reset self-play stats for next iteration
            self.self_play_engine.reset_stats()

        total_time = time.time() - training_start
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Total games: {self.total_games}")
        print(f"Total positions: {self.total_positions}")
        print(f"Final loss: {all_metrics[-1]['loss']:.4f}")
        print(f"{'='*60}")

        # Save final checkpoint
        self.save_checkpoint()
        self.save_training_log()

        return all_metrics

    def evaluate_elo(self, num_games: int = 5, verbose: bool = True) -> float:
        """
        Evaluate current model's Elo rating.

        Args:
            num_games: Games per skill level
            verbose: Print progress

        Returns:
            Estimated Elo rating
        """
        if verbose:
            print("\nEvaluating Elo rating...")

        try:
            estimator = EloEstimator(model=self.model)
            elo = estimator.quick_estimate(verbose=verbose)
            return elo
        except Exception as e:
            print(f"Elo evaluation failed: {e}")
            return 0.0

    def save_checkpoint(self, suffix: str = None) -> str:
        """Save model checkpoint.

        Returns:
            Path to saved checkpoint
        """
        if suffix is None:
            suffix = f"iter_{self.iteration:04d}"

        path = os.path.join(RL_CHECKPOINT_DIR, f"model_{suffix}.weights.h5")
        self.model.save_weights(path)
        print(f"Saved checkpoint: {path}")

        # Also save replay buffer
        self.replay_buffer.save()

        return path

    def save_best_model(self):
        """Save best model."""
        self.model.save_weights(RL_BEST_MODEL_FILE)
        print(f"Saved best model: {RL_BEST_MODEL_FILE}")

    def save_training_log(self):
        """Save training log to JSON."""
        with open(RL_TRAINING_LOG_FILE, 'w') as f:
            json.dump({
                'config': {
                    'lambda': self.lambda_param,
                    'gamma': self.gamma,
                    'learning_rate': self.learning_rate,
                },
                'log': self.training_log,
                'final_stats': {
                    'total_games': self.total_games,
                    'total_positions': self.total_positions,
                    'best_loss': self.best_loss,
                }
            }, f, indent=2)
        print(f"Saved training log: {RL_TRAINING_LOG_FILE}")

    def load_checkpoint(self, path: str) -> bool:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint weights

        Returns:
            True if loaded successfully
        """
        try:
            self.model.load_weights(path)
            print(f"Loaded checkpoint: {path}")

            # Update self-play engine
            self.self_play_engine.model = self.model

            # Try to load replay buffer
            self.replay_buffer.load()

            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False


def quick_train(iterations: int = 10, games_per_iter: int = 20,
                updates_per_iter: int = 50) -> TDLambdaTrainer:
    """
    Quick training for testing.

    Args:
        iterations: Number of iterations
        games_per_iter: Games per iteration
        updates_per_iter: Updates per iteration

    Returns:
        Trained TDLambdaTrainer instance
    """
    trainer = TDLambdaTrainer()
    trainer.run_training(
        num_iterations=iterations,
        games_per_iter=games_per_iter,
        updates_per_iter=updates_per_iter,
        checkpoint_interval=5
    )
    return trainer


if __name__ == '__main__':
    # Test TD-Lambda training
    print("Testing TD-Lambda Training...")

    # Quick test with classical evaluation (no NN)
    trainer = TDLambdaTrainer()

    # Generate some data first
    print("\nGenerating initial self-play data...")
    trainer.generate_self_play_data(num_games=10)

    print(f"\nBuffer stats: {trainer.replay_buffer.get_stats()}")

    # Run a few training updates
    print("\nRunning training iteration...")
    metrics = trainer.train_iteration(num_updates=20)
    print(f"Training metrics: {metrics}")
