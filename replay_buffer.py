import random
from collections import deque
from dataclasses import dataclass
import torch


@dataclass
class Experience:
    """
    One transition tuple stored in the replay buffer: (s, a, r, s', done).

    state      : drive tensor before the action        shape [5]
    action_idx : integer index of the action taken     0-3
    reward     : scalar reward received after acting
    next_state : drive tensor after the action          shape [5]
    done       : True if the episode ended (always False — Neurumi lives on)
    """
    state:      torch.Tensor
    action_idx: int
    reward:     float
    next_state: torch.Tensor
    done:       bool = False


class ReplayBuffer:
    """
    Fixed-size circular buffer that stores past experiences.

    Uses collections.deque(maxlen=N): when full, the oldest experience
    is automatically dropped. Append is O(1).

    Why random sampling matters:
    Training sequentially creates temporal correlations — the network
    overfits to recent events and forgets older ones (catastrophic forgetting).
    Random mini-batches break that correlation and stabilize training.
    This is the key insight from DeepMind's DQN paper (2013).
    """

    def __init__(self, capacity: int = 2000):
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Adds one experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Returns a random mini-batch. Caller must ensure buffer is ready."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """
        Training only starts once the buffer holds at least 64 experiences.
        Fewer samples produce noisy, unstable gradient updates.
        """
        return len(self.buffer) >= 64