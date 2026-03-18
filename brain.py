import torch
import torch.nn as nn


class NeurumiBrain(nn.Module):
    """
    La mente de NEURUMI. Una red neuronal feed-forward pequeña.

    Recibe 5 drives (estado actual) y produce 5 deltas
    que representan cómo debería cambiar el estado interno.

    Arquitectura: 5 → 16 → 8 → 5
    """

    def __init__(self):
        super().__init__()

        # Capa 1: de 5 drives a 16 "pensamientos" internos
        self.layer1 = nn.Linear(5, 16)

        # Capa 2: refina los 16 valores a 8
        self.layer2 = nn.Linear(16, 8)

        # Capa de salida: produce 5 deltas (uno por drive)
        self.layer3 = nn.Linear(8, 5)

        # ReLU para capas ocultas: elimina valores negativos
        self.relu = nn.ReLU()

        # Tanh para la salida: comprime a [-1, 1]
        # Los deltas pueden ser positivos (drive sube) o negativos (drive baja)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.layer1(x))   # [5] → [16]
        x = self.relu(self.layer2(x))   # [16] → [8]
        x = self.tanh(self.layer3(x))   # [8] → [5]
        return x


def save_brain(brain: NeurumiBrain, path: str = "neurumi_brain.pt"):
    """Serializa los pesos del modelo a disco."""
    torch.save(brain.state_dict(), path)


def load_brain(path: str = "neurumi_brain.pt") -> NeurumiBrain:
    """Carga un modelo previamente guardado."""
    brain = NeurumiBrain()
    brain.load_state_dict(torch.load(path, map_location="cpu"))
    brain.eval()
    return brain

if __name__ == "__main__":
    brain = NeurumiBrain()
    save_brain(brain)