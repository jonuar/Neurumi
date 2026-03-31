import torch
import torch.nn as nn

# Fixed action space — the 4 things Neurumi can experience
ACTIONS   = ["feed", "play", "pet", "ignore"]
N_ACTIONS = len(ACTIONS)  # 4


class QBrain(nn.Module):
    """
    The Q-network. Maps a state tensor → Q-values for each possible action.

    Output shape: [4] — one Q-value per action.
    The agent picks the action with the highest Q-value (argmax).

    Key difference from Phase 1 AnimaBrain:
      Phase 1 output → 5 deltas  (what should the state change to?)
      Q-network output → 4 Q-values  (how good is each action right now?)

    No output activation: Q-values are unbounded real numbers.
    Tanh would artificially cap them and distort the Bellman targets.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),      # 5 drives → 64 hidden units
            nn.ReLU(),
            nn.Linear(64, 64),     # 64 → 64 (second hidden layer)
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),  # 64 → 4 Q-values (no activation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [5]          for a single state (inference)
                 [batch, 5]   for a batch         (training)
        Returns: [4] or [batch, 4]
        """
        return self.net(x)


def save_q_brain(brain: "QBrain", path: str = "neurumi_qbrain.pt"):
    torch.save(brain.state_dict(), path)


def load_q_brain(path: str = "neurumi_qbrain.pt") -> "QBrain":
    brain = QBrain()
    brain.load_state_dict(torch.load(path, map_location="cpu"))
    brain.eval()
    return brain


if __name__ == "__main__":
    brain = QBrain()
    print(brain)