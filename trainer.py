import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from brain import NeurumiBrain
from state import NeurumiState, ACTION_EFFECTS


class NeurumiTrainer:
    """
    Encapsula la lógica de entrenamiento e inferencia.

    Separar el entrenamiento del modelo (NeurumiBrain) y del estado (NeurumiState)
    sigue el principio de responsabilidad única — cada clase hace una sola cosa.
    """

    def __init__(self, brain: NeurumiBrain, lr: float = 0.001):
        self.brain = brain

        # MSELoss: mide el error cuadrático medio entre predicción y target
        # Ideal para regresión (predecir números continuos como deltas)
        self.loss_fn = nn.MSELoss()

        # Adam: optimizador estándar, robusto y sin mucho tuning necesario
        # lr (learning rate) controla el tamaño de cada paso de ajuste
        self.optimizer = optim.Adam(brain.parameters(), lr=lr)

        # Historial de pérdidas para visualizar el aprendizaje en la UI
        self.loss_history: list[float] = []

    def train_step(
        self,
        state_tensor: torch.Tensor,
        target_tensor: torch.Tensor
    ) -> float:
        """
        Un paso de entrenamiento.

        1. Limpia gradientes anteriores
        2. Forward pass: la red predice
        3. Calcula pérdida: qué tan equivocado estuvo
        4. Backward pass: calcula gradientes
        5. Optimizer step: ajusta los pesos

        Retorna el valor de la pérdida como float (para logging).
        """

        # 1. Zero gradients — obligatorio antes de cada paso
        self.optimizer.zero_grad()

        # 2. Forward pass
        predicted = self.brain(state_tensor)

        # 3. Loss
        loss = self.loss_fn(predicted, target_tensor)

        # 4. Backpropagation — PyTorch calcula automáticamente los gradientes
        loss.backward()

        # 5. Actualizar pesos
        self.optimizer.step()

        return loss.item()

    def train_on_action(
        self,
        action: str,
        neurumi: NeurumiState,
        steps: int = 8
    ) -> float:
        """
        Entrena la red varias veces con el mismo ejemplo.
        `steps` = cuántas veces repetimos el entrenamiento con este dato.

        Repetir el mismo ejemplo varias veces es útil cuando el dataset
        es pequeño (una sola interacción). En ML se llama "overfitting local"
        pero aquí es intencional — queremos que la red aprenda bien esta acción.
        """
        state_tensor = neurumi.to_tensor()
        target_tensor = ACTION_EFFECTS[action]

        total_loss = 0.0
        for _ in range(steps):
            loss = self.train_step(state_tensor, target_tensor)
            total_loss += loss

        avg_loss = total_loss / steps
        self.loss_history.append(round(avg_loss, 5))

        # Mantener solo los últimos 50 valores para la visualización
        if len(self.loss_history) > 50:
            self.loss_history.pop(0)

        return avg_loss

    def infer(self, neurumi: NeurumiState) -> torch.Tensor:
        """
        Inferencia: la red predice los deltas para el estado actual.
        No entrena — solo predice.

        torch.no_grad() es importante aquí: desactiva el cálculo
        de gradientes porque no los necesitamos. Es más rápido y
        consume menos memoria.
        """
        self.brain.eval()  # modo inferencia
        with torch.no_grad():
            state_tensor = neurumi.to_tensor()
            deltas = self.brain(state_tensor)
        self.brain.train()  # volver a modo entrenamiento
        return deltas

    def get_last_loss(self) -> float:
        if self.loss_history:
            return self.loss_history[-1]
        return 0.0