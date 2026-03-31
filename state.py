import torch
import json
from dataclasses import dataclass, asdict

# Fixed drive order — must never change, the network depends on this index mapping
DRIVE_KEYS = ["hunger", "curiosity", "affection", "energy", "fear"]


@dataclass
class NeurumiState:
    """
    Neurumi's internal state. Five drives clamped to [0.0, 1.0].

    Uses dataclass for free __repr__ and easy JSON serialization.
    Persisted to disk as neurumi_state.json between sessions.
    """
    hunger:    float = 0.3
    curiosity: float = 0.7
    affection: float = 0.5
    energy:    float = 0.8
    fear:      float = 0.1
    age:       int   = 0
    name:      str   = "Neurumi"

    def to_tensor(self) -> torch.Tensor:
        """Converts the 5 numeric drives to a float32 tensor. Age and name excluded."""
        values = [getattr(self, k) for k in DRIVE_KEYS]
        return torch.tensor(values, dtype=torch.float32)

    def apply_deltas(self, deltas_tensor: torch.Tensor, scale: float = 0.1):
        """
        Applies predicted deltas to drives.
        scale controls the magnitude of change per tick.
        Drives are clamped to [0, 1] after every update.
        """
        for i, key in enumerate(DRIVE_KEYS):
            delta = deltas_tensor[i].item()
            new_val = getattr(self, key) + delta * scale
            setattr(self, key, max(0.0, min(1.0, new_val)))

    def apply_action_effect(self, effects: torch.Tensor, scale: float = 0.15):
        """
        Applies a player or agent action effect to drives.
        Uses a larger scale than apply_deltas — direct actions
        should feel more impactful than passive time-based drift.
        """
        self.apply_deltas(effects, scale=scale)

    def tick(self):
        """
        Simulates time passing. Called after every interaction.
        Drives drift toward their natural resting tension:
        hunger always creeps up, affection always decays without attention.
        """
        self.age       += 1
        self.hunger     = min(1.0, self.hunger    + 0.015)
        self.curiosity  = min(1.0, self.curiosity + 0.008)
        self.affection  = max(0.0, self.affection - 0.010)
        self.energy     = max(0.0, self.energy    - 0.005)
        self.fear       = max(0.0, self.fear      - 0.003)  # fear decays on its own

    def get_emotion(self) -> str:
        """Translates the drive vector into a single readable emotion label."""
        if self.fear      > 0.65: return "scared"
        if self.hunger    > 0.78: return "hungry"
        if self.energy    < 0.18: return "sleepy"
        if self.affection > 0.78: return "happy"
        if self.affection < 0.22: return "lonely"
        if self.curiosity > 0.82: return "curious"
        return "calm"

    def get_wellness(self) -> float:
        """
        Overall wellness score in [0, 1].
        Hunger and fear are inverted — high values are bad for those drives.
        """
        return round((
            (1.0 - self.hunger) +
            self.affection      +
            self.energy         +
            (1.0 - self.fear)
        ) / 4.0, 3)

    def save(self, path: str = "neurumi_state.json"):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = "neurumi_state.json") -> "NeurumiState":
        with open(path) as f:
            return cls(**json.load(f))


# ── Player / agent action effects ─────────────────────────────────────────────
# Tensor order matches DRIVE_KEYS: [hunger, curiosity, affection, energy, fear]
# Values in [-1, 1] to match the Tanh output range of Phase 1's AnimaBrain.
# Used by both the supervised trainer (Phase 1) and the DQN agent (Phase 2).

ACTION_EFFECTS = {
    "feed": torch.tensor(
        [-0.8,  0.1,  0.2,  0.5, -0.1], dtype=torch.float32
    ),
    "play": torch.tensor(
        [-0.1,  0.8,  0.3, -0.4, -0.3], dtype=torch.float32
    ),
    "pet": torch.tensor(
        [ 0.0,  0.1,  0.9,  0.0, -0.6], dtype=torch.float32
    ),
    "ignore": torch.tensor(
        [-0.1, -0.2, -0.5,  0.1,  0.3], dtype=torch.float32
    ),
}

# ── Emotion display metadata ──────────────────────────────────────────────────

EMOTION_META = {
    "happy":   {"emoji": "✦", "label": "Happy",   "color": "#5DCAA5"},
    "calm":    {"emoji": "◉", "label": "Calm",    "color": "#378ADD"},
    "curious": {"emoji": "◈", "label": "Curious", "color": "#7F77DD"},
    "hungry":  {"emoji": "◌", "label": "Hungry",  "color": "#D85A30"},
    "scared":  {"emoji": "◍", "label": "Scared",  "color": "#BA7517"},
    "sleepy":  {"emoji": "◎", "label": "Sleepy",  "color": "#888780"},
    "lonely":  {"emoji": "◯", "label": "Lonely",  "color": "#D4537E"},
}