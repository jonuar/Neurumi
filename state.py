import torch
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict


# Orden fijo de los drives. El orden importa porque el tensor
# debe construirse siempre de la misma forma para que la red lo entienda.
DRIVE_KEYS = ["hunger", "curiosity", "affection", "energy", "fear"]


@dataclass
class NeurumiState:
    """
    El estado interno de NEURUMI. Cinco drives entre 0.0 y 1.0.

    Usamos dataclass para tener __repr__ gratis y facilitar
    la serialización a JSON (para persistir entre sesiones).
    """

    hunger: float = 0.3
    curiosity: float = 0.7
    affection: float = 0.5
    energy: float = 0.8
    fear: float = 0.1
    age: int = 0  # ticks de vida acumulados
    name: str = "NEURUMI-01"

    def to_tensor(self) -> torch.Tensor:
        """
        Convierte los drives a un tensor de PyTorch.
        Solo incluye los 5 drives numéricos, no la edad ni el nombre.
        """
        values = [getattr(self, k) for k in DRIVE_KEYS]
        return torch.tensor(values, dtype=torch.float32)

    def apply_deltas(self, deltas_tensor: torch.Tensor, scale: float = 0.1):
        """
        Aplica los deltas predichos por la red al estado actual.

        scale=0.1 significa que el cambio máximo por tick es ±10%,
        lo que genera cambios graduales y naturales.
        """
        for i, key in enumerate(DRIVE_KEYS):
            delta = deltas_tensor[i].item()  # tensor → float Python
            current = getattr(self, key)
            new_value = current + delta * scale
            # Clamping: los drives no pueden salir del rango [0, 1]
            setattr(self, key, max(0.0, min(1.0, new_value)))

    def apply_action_effect(self, effects: torch.Tensor, scale: float = 0.15):
        """
        Aplica el efecto directo de una acción del jugador.
        Usa un scale mayor que apply_deltas porque el jugador
        interactuando debe tener un impacto más notable.
        """
        self.apply_deltas(effects, scale=scale)

    def tick(self):
        """
        Simula el paso del tiempo. Cada tick:
        - El hambre aumenta lentamente (la criatura siempre tiene hambre)
        - La curiosidad sube (quiere explorar)
        - El afecto decae (necesita atención)
        - La energía baja (se cansa)
        """
        self.age += 1
        self.hunger = min(1.0, self.hunger + 0.015)
        self.curiosity = min(1.0, self.curiosity + 0.008)
        self.affection = max(0.0, self.affection - 0.010)
        self.energy = max(0.0, self.energy - 0.005)
        # El miedo decae naturalmente con el tiempo (se calma solo)
        self.fear = max(0.0, self.fear - 0.003)

    def get_emotion(self) -> str:
        """Traduce el estado interno a una emoción legible."""
        if self.fear > 0.65:
            return "scared"
        if self.hunger > 0.78:
            return "hungry"
        if self.energy < 0.18:
            return "sleepy"
        if self.affection > 0.78:
            return "happy"
        if self.affection < 0.22:
            return "lonely"
        if self.curiosity > 0.82:
            return "curious"
        return "calm"

    def get_wellness(self) -> float:
        """
        Score de bienestar entre 0 y 1.
        Usado para visualizar qué tan bien está NEURUMI.

        hunger alta = malo → invertimos con (1 - hunger)
        fear alta = malo → invertimos con (1 - fear)
        """
        return round(
            ((1.0 - self.hunger) + self.affection + self.energy + (1.0 - self.fear))
            / 4.0,
            3,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NeurumiState":
        return cls(**data)

    def save(self, path: str = "neurumi_state.json"):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str = "neurumi_state.json") -> "NeurumiState":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


# ─── Efectos de las acciones del jugador ────────────────────────────────────
# Cada tensor tiene 5 valores en orden: [hunger, curiosity, affection, energy, fear]
# Rango [-1, 1] para ser compatibles con la salida Tanh de la red.

ACTION_EFFECTS = {
    "feed": torch.tensor(
        [-0.8, 0.1, 0.2, 0.5, -0.1],
        dtype=torch.float32,
        # hunger baja mucho, energía sube, afecto sube un poco
    ),
    "play": torch.tensor(
        [-0.1, 0.8, 0.3, -0.4, -0.3],
        dtype=torch.float32,
        # curiosidad sube mucho, afecto sube, energía baja (se cansa)
    ),
    "pet": torch.tensor(
        [0.0, 0.1, 0.9, 0.0, -0.6],
        dtype=torch.float32,
        # afecto sube muchísimo, miedo cae mucho
    ),
    "ignore": torch.tensor(
        [-0.1, -0.2, -0.5, 0.1, 0.3],
        dtype=torch.float32,
        # afecto cae, miedo sube, energía sube un poco (descansa)
    ),
}

EMOTION_META = {
    "happy": {"emoji": "(≧◡≦)", "label": "Happy", "color": "#5DCAA5"},
    "calm": {"emoji": "( ˘ᵕ˘)", "label": "Calm", "color": "#378ADD"},
    "curious": {"emoji": "(⊙ᗜ⊙)", "label": "Curious", "color": "#7F77DD"},
    "hungry": {"emoji": "(◑﹏◐)", "label": "Hungry", "color": "#D85A30"},
    "scared": {"emoji": "(⓪△⓪)", "label": "Scared", "color": "#BA7517"},
    "sleepy": {"emoji": "(-_-)zzz", "label": "Sleepy", "color": "#888780"},
    "lonely": {"emoji": "(ಥ﹏ಥ)", "label": "Lonely", "color": "#D4537E"},
}
