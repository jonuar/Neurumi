import torch
import torch.nn as nn


# Fixed action space, the 4 things Neurumi can experience
ACTIONS = ["feed", "play", "pet", "ignore"]
N_ACTIONS = len(ACTIONS)   # 4


class QBrain(nn.Module):
    """
    The Q-network. Maps (state) → Q-values for each action.

    Output shape: [4]  — one Q-value per possible action.
    The agent selects the action with the highest Q-value.

    Architecture is intentionally simple. For a 5-dimensional state space
    and 4 actions, a deep network would overfit immediately.

    Key difference from AnimaBrain (Phase 1):
    - Phase 1 output: 5 deltas (what should change)
    - Q-network output: 4 Q-values (how good is each action)
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(5, 64),    # state → hidden
            nn.ReLU(),
            nn.Linear(64, 64),   # hidden → hidden
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),  # hidden → Q-values (one per action)
            # No activation on output, Q-values can be any real number,
            # not bounded like Tanh. We need the full range.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [5] for single state, or [batch, 5] for batch training.
        Returns Q-values shape: [4] or [batch, 4].
        """
        return self.net(x)


def save_q_brain(brain: "QBrain", path: str = "neurumi_qbrain.pt"):
    torch.save(brain.state_dict(), path)


def load_q_brain(path: str = "neurumi_qbrain.pt") -> "QBrain":
    brain = QBrain()
    brain.load_state_dict(torch.load(path, map_location="cpu"))
    brain.eval()
    return brain