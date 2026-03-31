import torch
import torch.nn as nn
import torch.optim as optim

from brain import AnimaBrain
from state import NeurumiState, ACTION_EFFECTS


class NeurumiTrainer:
    """
    Phase 1 supervised trainer.

    Trains AnimaBrain to imitate ACTION_EFFECTS targets.
    The network learns "what should change when the player does X"
    rather than "what maximizes Neurumi's wellbeing".
    """

    def __init__(self, brain: AnimaBrain, lr: float = 0.001):
        self.brain     = brain
        self.loss_fn   = nn.MSELoss()
        self.optimizer = optim.Adam(brain.parameters(), lr=lr)
        self.loss_history: list[float] = []

    def train_step(
        self,
        state_tensor:  torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> float:
        """Single forward + backward + optimizer step."""
        self.optimizer.zero_grad()
        predicted = self.brain(state_tensor)
        loss      = self.loss_fn(predicted, target_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_on_action(
        self,
        action:  str,
        neurumi: NeurumiState,
        steps:   int = 8,
    ) -> float:
        """
        Trains the network for `steps` iterations on a single (state, target) pair.
        Repeating the same example multiple times strengthens the learned association
        without needing a larger dataset.
        """
        state_tensor  = neurumi.to_tensor()
        target_tensor = ACTION_EFFECTS[action]

        total = sum(self.train_step(state_tensor, target_tensor) for _ in range(steps))
        avg   = round(total / steps, 5)

        self.loss_history.append(avg)
        if len(self.loss_history) > 50:
            self.loss_history.pop(0)

        return avg

    def infer(self, neurumi: NeurumiState) -> torch.Tensor:
        """
        Forward pass only — no gradient computation.
        Used by 'Pass time' to let the network predict state drift autonomously.
        """
        self.brain.eval()
        with torch.no_grad():
            deltas = self.brain(neurumi.to_tensor())
        self.brain.train()
        return deltas

    def get_last_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else 0.0