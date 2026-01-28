"""
Experience Replay Buffer for TD-Lambda reinforcement learning.

Stores game positions and their values for batch training updates.
Implements a FIFO queue with random sampling for training.
"""
import os
import json
import random
import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, asdict

try:
    from .config import RL_REPLAY_BUFFER_SIZE, RL_CHECKPOINT_DIR
except ImportError:
    from config import RL_REPLAY_BUFFER_SIZE, RL_CHECKPOINT_DIR


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray      # Board position tensor (8, 8, 19)
    value: float           # Position value [-1, 1]
    game_id: int           # Which game this came from
    move_number: int       # Move number in the game


class ReplayBuffer:
    """
    Experience replay buffer for TD-Lambda training.

    Features:
    - FIFO queue with configurable max size
    - Random sampling for training batches
    - Save/load functionality for persistence
    - Statistics tracking
    """

    def __init__(self, max_size: int = RL_REPLAY_BUFFER_SIZE):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.game_counter = 0
        self.total_experiences_added = 0

        # Statistics
        self.stats = {
            'games_added': 0,
            'positions_added': 0,
            'samples_drawn': 0,
        }

    def add(self, state: np.ndarray, value: float, game_id: int = -1,
            move_number: int = 0):
        """
        Add a single experience to the buffer.

        Args:
            state: Position tensor (8, 8, 19)
            value: Position value [-1, 1]
            game_id: Game identifier
            move_number: Move number in game
        """
        exp = Experience(
            state=state.astype(np.float32),
            value=float(value),
            game_id=game_id if game_id >= 0 else self.game_counter,
            move_number=move_number
        )
        self.buffer.append(exp)
        self.total_experiences_added += 1
        self.stats['positions_added'] += 1

    def add_game(self, positions: List[np.ndarray], values: List[float]):
        """
        Add all positions from a game to the buffer.

        Args:
            positions: List of position tensors
            values: List of position values
        """
        self.game_counter += 1
        self.stats['games_added'] += 1

        for move_num, (pos, val) in enumerate(zip(positions, values)):
            self.add(pos, val, self.game_counter, move_num)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, values) numpy arrays
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)

        states = np.array([exp.state for exp in batch])
        values = np.array([exp.value for exp in batch])

        self.stats['samples_drawn'] += batch_size

        return states, values

    def sample_with_info(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Sample batch with additional information.

        Returns:
            Tuple of (states, values, info_list)
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)

        states = np.array([exp.state for exp in batch])
        values = np.array([exp.value for exp in batch])
        info = [{'game_id': exp.game_id, 'move_number': exp.move_number}
                for exp in batch]

        self.stats['samples_drawn'] += batch_size

        return states, values, info

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return len(self.buffer) >= min_size

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        values = [exp.value for exp in self.buffer]
        return {
            **self.stats,
            'current_size': len(self.buffer),
            'max_size': self.max_size,
            'fill_percentage': len(self.buffer) / self.max_size * 100,
            'value_mean': np.mean(values) if values else 0.0,
            'value_std': np.std(values) if values else 0.0,
            'value_min': np.min(values) if values else 0.0,
            'value_max': np.max(values) if values else 0.0,
        }

    def save(self, path: Optional[str] = None):
        """
        Save buffer to disk.

        Args:
            path: Save path (defaults to checkpoint directory)
        """
        if path is None:
            os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)
            path = os.path.join(RL_CHECKPOINT_DIR, 'replay_buffer.npz')

        # Convert to numpy arrays for efficient storage
        states = np.array([exp.state for exp in self.buffer])
        values = np.array([exp.value for exp in self.buffer])
        game_ids = np.array([exp.game_id for exp in self.buffer])
        move_numbers = np.array([exp.move_number for exp in self.buffer])

        np.savez_compressed(
            path,
            states=states,
            values=values,
            game_ids=game_ids,
            move_numbers=move_numbers,
            game_counter=self.game_counter,
            total_experiences_added=self.total_experiences_added
        )

        # Save stats separately
        stats_path = path.replace('.npz', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)

        print(f"Saved replay buffer to {path} ({len(self.buffer)} experiences)")

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load buffer from disk.

        Args:
            path: Load path (defaults to checkpoint directory)

        Returns:
            True if loaded successfully
        """
        if path is None:
            path = os.path.join(RL_CHECKPOINT_DIR, 'replay_buffer.npz')

        if not os.path.exists(path):
            print(f"No replay buffer found at {path}")
            return False

        try:
            data = np.load(path)

            self.buffer.clear()
            for state, value, game_id, move_num in zip(
                data['states'], data['values'],
                data['game_ids'], data['move_numbers']
            ):
                exp = Experience(state, float(value), int(game_id), int(move_num))
                self.buffer.append(exp)

            self.game_counter = int(data['game_counter'])
            self.total_experiences_added = int(data['total_experiences_added'])

            print(f"Loaded replay buffer from {path} ({len(self.buffer)} experiences)")
            return True

        except Exception as e:
            print(f"Error loading replay buffer: {e}")
            return False

    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.stats = {
            'games_added': 0,
            'positions_added': 0,
            'samples_drawn': 0,
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer.

    Prioritizes experiences with higher TD-error for more efficient learning.
    (Optional enhancement for future use)
    """

    def __init__(self, max_size: int = RL_REPLAY_BUFFER_SIZE, alpha: float = 0.6):
        """
        Initialize prioritized buffer.

        Args:
            max_size: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        """
        super().__init__(max_size)
        self.priorities: deque = deque(maxlen=max_size)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state: np.ndarray, value: float, game_id: int = -1,
            move_number: int = 0, priority: float = None):
        """Add experience with priority."""
        super().add(state, value, game_id, move_number)
        # New experiences get max priority
        self.priorities.append(priority if priority else self.max_priority)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample based on priorities."""
        if len(self.buffer) == 0:
            return np.array([]), np.array([])

        batch_size = min(batch_size, len(self.buffer))

        # Convert priorities to probabilities
        priorities = np.array(list(self.priorities))
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(
            len(self.buffer), size=batch_size, replace=False, p=probabilities
        )

        buffer_list = list(self.buffer)
        states = np.array([buffer_list[i].state for i in indices])
        values = np.array([buffer_list[i].value for i in indices])

        self.stats['samples_drawn'] += batch_size

        return states, values

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        priorities_list = list(self.priorities)
        for idx, td_error in zip(indices, td_errors):
            priorities_list[idx] = abs(td_error) + 1e-6  # Small epsilon for stability
            self.max_priority = max(self.max_priority, priorities_list[idx])
        self.priorities = deque(priorities_list, maxlen=self.max_size)


if __name__ == '__main__':
    # Test the replay buffer
    buffer = ReplayBuffer(max_size=1000)

    # Add some dummy experiences
    for game in range(10):
        positions = [np.random.randn(8, 8, 19).astype(np.float32) for _ in range(50)]
        values = [np.random.uniform(-1, 1) for _ in range(50)]
        buffer.add_game(positions, values)

    print(f"Buffer size: {len(buffer)}")
    print(f"Stats: {buffer.get_stats()}")

    # Sample a batch
    states, values = buffer.sample(64)
    print(f"Sampled batch: states={states.shape}, values={values.shape}")
