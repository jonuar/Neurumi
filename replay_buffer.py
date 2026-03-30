import random
from collections import deque
from dataclasses import dataclass
import torch


@dataclass
class Experience:
    """
    A single transition tuple: (state, action, reward, next_state, done).

    This is the standard unit of experience in DQN.
    - state      : drives tensor before the action [5]
    - action_idx : integer index of the action taken (0-3)
    - reward     : scalar reward received after the action
    - next_state : drives tensor after the action [5]
    - done       : whether the episode ended (always False here — Neurumi lives on)
    """
    state:      torch.Tensor
    action_idx: int
    reward:     float
    next_state: torch.Tensor
    done:       bool = False


class ReplayBuffer:
    """
    Fixed-size circular buffer that stores past experiences.

    Implemented with collections.deque(maxlen=N) — when full,
    the oldest experience is automatically discarded. O(1) append.

    Why random sampling matters:
    Sequential training creates temporal correlations — the network
    overfits to recent events and catastrophically forgets older ones.
    Random mini-batches break that correlation and stabilize training.
    This is the key insight from DeepMind's DQN paper (2013).
    """

    def __init__(self, capacity: int = 2000):
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Adds one experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Returns a random mini-batch of experiences.
        Requires buffer to have at least batch_size entries.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """
        Training only starts once the buffer has enough experiences.
        Training on 1-2 samples produces noisy, unstable gradients.
        We wait for at least 64 experiences before starting.
        """
        return len(self.buffer) >= 64