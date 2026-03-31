import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from q_brain import QBrain, ACTIONS, N_ACTIONS
from replay_buffer import ReplayBuffer, Experience
from state import NeurumiState, ACTION_EFFECTS
from reward import compute_reward


class DQNTrainer:
    """
    Deep Q-Network trainer implementing the core DQN algorithm:

    1. Epsilon-greedy action selection
       Balances exploring new actions vs exploiting learned Q-values.

    2. Experience replay
       Random mini-batches from the ReplayBuffer break temporal
       correlations and prevent catastrophic forgetting.

    3. Target network
       A frozen copy of the Q-network that provides stable Bellman
       targets. Without it, training chases a moving goalpost — the
       network updates toward targets that themselves change every step.
       The target network syncs with the live network every N steps.

    4. Gradient clipping
       Prevents exploding gradients, common in RL where Q-value
       errors can cascade across many Bellman updates.
    """

    def __init__(
        self,
        brain: QBrain,
        lr: float = 0.001,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 300,
        batch_size: int = 32,
        target_update_freq: int = 20,
    ):
        self.brain = brain

        # Frozen copy of the live network used only for computing
        # stable Bellman targets. Never trained directly — only synced.
        self.target_brain = copy.deepcopy(brain)
        self.target_brain.eval()

        self.optimizer = optim.Adam(brain.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.buffer    = ReplayBuffer(capacity=2000)

        self.gamma             = gamma          # future reward discount factor
        self.epsilon           = epsilon_start  # current exploration rate
        self.epsilon_end       = epsilon_end    # minimum epsilon floor
        self.epsilon_decay     = epsilon_decay  # steps to reach epsilon_end
        self.batch_size        = batch_size
        self.target_update_freq = target_update_freq

        self.steps          = 0
        self.loss_history:   list[float] = []
        self.reward_history: list[float] = []

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state_tensor: torch.Tensor) -> tuple[int, str]:
        """
        Epsilon-greedy selection.
        epsilon → 1.0: mostly random exploration
        epsilon → 0.0: mostly greedy exploitation
        """
        if random.random() < self.epsilon:
            idx = random.randint(0, N_ACTIONS - 1)  # explore
        else:
            self.brain.eval()
            with torch.no_grad():
                idx = self.brain(state_tensor).argmax().item()  # exploit
            self.brain.train()

        return idx, ACTIONS[idx]

    def decay_epsilon(self):
        """
        Linear epsilon decay from epsilon_start to epsilon_end
        over epsilon_decay steps.
        """
        progress = min(1.0, self.steps / self.epsilon_decay)
        self.epsilon = max(
            self.epsilon_end,
            1.0 - progress * (1.0 - self.epsilon_end)
        )

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self) -> float | None:
        """
        One DQN training step on a random mini-batch.

        Bellman update:
            Q_target(s, a) = r + γ × max_a'[ Q_target_network(s', a') ]

        The live brain learns to predict Q_target.
        The target_brain provides the stable right-hand side.
        """
        if not self.buffer.is_ready:
            return None

        batch = self.buffer.sample(self.batch_size)

        # Stack individual experience tensors into batch matrices
        states      = torch.stack([e.state      for e in batch])  # [B, 5]
        next_states = torch.stack([e.next_state for e in batch])  # [B, 5]
        actions     = torch.tensor([e.action_idx for e in batch], dtype=torch.long)
        rewards     = torch.tensor([e.reward     for e in batch], dtype=torch.float32)
        dones       = torch.tensor([e.done       for e in batch], dtype=torch.float32)

        # Current Q-values for the actions that were actually taken
        # brain(states) → [B, 4]; gather selects the column = action taken
        current_q = self.brain(states).gather(
            1, actions.unsqueeze(1)  # [B, 1]
        ).squeeze(1)                 # [B]

        # Bellman targets using the frozen target network
        with torch.no_grad():
            next_q_max = self.target_brain(next_states).max(dim=1).values  # [B]
            # (1 - done) zeroes out future reward for terminal states
            target_q = rewards + self.gamma * next_q_max * (1 - dones)    # [B]

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent large Bellman errors from destabilizing training
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def sync_target_network(self):
        """Copies live brain weights into the frozen target network."""
        self.target_brain.load_state_dict(self.brain.state_dict())

    # ── Full step ─────────────────────────────────────────────────────────────

    def step(self, neurumi: NeurumiState) -> dict:
        """
        One complete RL step:
        1. Select action via epsilon-greedy
        2. Apply action to Neurumi's drives
        3. Compute reward from resulting state
        4. Store (s, a, r, s') in replay buffer
        5. Train on a random mini-batch
        6. Decay epsilon
        7. Sync target network if scheduled

        Returns step metadata for the UI.
        """
        state_tensor = neurumi.to_tensor()

        # 1. Select action
        action_idx, action_name = self.select_action(state_tensor)

        # 2. Apply action — modifies neurumi in place
        neurumi.apply_action_effect(ACTION_EFFECTS[action_name])
        neurumi.tick()

        # 3. Reward from the resulting state
        reward = compute_reward(neurumi)
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        # 4. Store experience
        self.buffer.push(Experience(
            state      = state_tensor,
            action_idx = action_idx,
            reward     = reward,
            next_state = neurumi.to_tensor(),
            done       = False,
        ))

        # 5. Train
        loss = self.train_step()
        if loss is not None:
            self.loss_history.append(round(loss, 5))
            if len(self.loss_history) > 50:
                self.loss_history.pop(0)

        # 6. Decay epsilon
        self.steps += 1
        self.decay_epsilon()

        # 7. Sync target network on schedule
        if self.steps % self.target_update_freq == 0:
            self.sync_target_network()

        return {
            "action":  action_name,
            "reward":  reward,
            "loss":    loss,
            "epsilon": round(self.epsilon, 3),
            "steps":   self.steps,
            "buffer":  len(self.buffer),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_q_values(self, neurumi: NeurumiState) -> dict[str, float]:
        """
        Returns current Q-values for all actions given Neurumi's state.
        Used in the UI to visualize the agent's preferences.
        """
        self.brain.eval()
        with torch.no_grad():
            q = self.brain(neurumi.to_tensor())
        self.brain.train()
        return {action: round(q[i].item(), 3) for i, action in enumerate(ACTIONS)}

    def get_last_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else 0.0

    def get_avg_reward(self) -> float:
        if not self.reward_history:
            return 0.0
        return round(sum(self.reward_history) / len(self.reward_history), 3)