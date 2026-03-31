import torch
import torch.nn as nn


class AnimaBrain(nn.Module):
    """
    Phase 1 supervised network.

    Maps 5 drives → 5 deltas representing how the state should change.
    Trained by imitating ACTION_EFFECTS targets — not by reward.

    Architecture: 5 → 16 → 8 → 5
    Output activation: Tanh (bounds deltas to [-1, 1])
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 5)
        self.relu   = nn.ReLU()
        self.tanh   = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))   # [5]  → [16]
        x = self.relu(self.layer2(x))   # [16] → [8]
        x = self.tanh(self.layer3(x))   # [8]  → [5], bounded to [-1, 1]
        return x


def save_brain(brain: AnimaBrain, path: str = "neurumi_brain.pt"):
    torch.save(brain.state_dict(), path)


def load_brain(path: str = "neurumi_brain.pt") -> AnimaBrain:
    brain = AnimaBrain()
    brain.load_state_dict(torch.load(path, map_location="cpu"))
    brain.eval()
    return brain