import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from q_brain import QBrain, ACTIONS, N_ACTIONS
from replay_buffer import ReplayBuffer, Experience
from state import NeurumiState, ACTION_EFFECTS
from reward import compute_reward


class DQNTrainer:
    """
    Deep Q-Network trainer.

    Implements the core DQN algorithm:
    1. Epsilon-greedy action selection
    2. Experience replay (random mini-batches from ReplayBuffer)
    3. Target network (frozen copy of Q-network, updated periodically)

    The target network is the key stabilization trick in DQN.
    Without it, training is circular: the network updates toward a target
    that itself changes every step, like chasing a moving goalpost.
    The frozen target network provides stable Q-value estimates for
    a fixed number of steps before syncing with the live network.
    """

    def __init__(
        self,
        brain: QBrain,
        lr: float = 0.001,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 300,    # steps until epsilon reaches minimum
        batch_size: int = 32,
        target_update_freq: int = 20 # sync target network every N steps
    ):
        self.brain = brain

        # Target network: a frozen copy of the Q-network.
        # Updated every `target_update_freq` steps by copying live weights.
        # Used only for computing stable Q-value targets during training.
        self.target_brain = copy.deepcopy(brain)
        self.target_brain.eval()  # target network never trains directly

        self.optimizer = optim.Adam(brain.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.buffer    = ReplayBuffer(capacity=2000)

        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update_freq = target_update_freq

        self.steps       = 0
        self.loss_history: list[float] = []
        self.reward_history: list[float] = []

    # ── Action selection 

    def select_action(self, state_tensor: torch.Tensor) -> tuple[int, str]:
        """
        Epsilon-greedy action selection.

        With probability epsilon  → random action  (explore)
        With probability 1-epsilon → best Q-action (exploit)

        Returns both the action index and its string name.
        """
        if random.random() < self.epsilon:
            # Exploration: pick a random action
            idx = random.randint(0, N_ACTIONS - 1)
        else:
            # Exploitation: pick the action with the highest Q-value
            self.brain.eval()
            with torch.no_grad():
                q_values = self.brain(state_tensor)   # shape: [4]
            self.brain.train()
            idx = q_values.argmax().item()   # index of the max Q-value

        return idx, ACTIONS[idx]

    def decay_epsilon(self):
        """
        Exponential epsilon decay.
        Epsilon decreases from epsilon_start to epsilon_end
        over epsilon_decay steps.

        This schedule means:
        - Early steps: mostly random (exploring the state space)
        - Later steps: mostly greedy (exploiting learned Q-values)
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_end + (1.0 - self.epsilon_end)
            * (1 - self.steps / self.epsilon_decay)
        )

    # ── Training

    def train_step(self) -> float | None:
        """
        One DQN training step using a random mini-batch from the buffer.

        Returns the loss value, or None if the buffer isn't ready yet.

        The Bellman update:
            Q_target(s, a) = r + γ × max_a'[ Q_target(s', a') ]

        We compute this target using the frozen target_brain,
        then update the live brain to predict closer to that target.
        """
        if not self.buffer.is_ready:
            return None

        # Sample a random mini-batch of experiences
        batch = self.buffer.sample(self.batch_size)

        # Stack individual tensors into batch tensors for vectorized computation
        # shape: [batch_size, 5]
        states      = torch.stack([e.state      for e in batch])
        next_states = torch.stack([e.next_state for e in batch])

        # shape: [batch_size]
        actions = torch.tensor([e.action_idx for e in batch], dtype=torch.long)
        rewards = torch.tensor([e.reward      for e in batch], dtype=torch.float32)
        dones   = torch.tensor([e.done        for e in batch], dtype=torch.float32)

        # ── Current Q-values
        # brain(states) shape: [batch_size, 4]
        # We select only the Q-value of the action that was actually taken
        # gather(1, actions) picks one value per row using actions as column index
        current_q = self.brain(states).gather(
            1, actions.unsqueeze(1)  # [batch_size, 1]
        ).squeeze(1)                 # back to [batch_size]

        # ── Target Q-values (Bellman equation) 
        with torch.no_grad():
            # max Q-value of next state, from the frozen target network
            next_q_max = self.target_brain(next_states).max(dim=1).values
            # Bellman: r + γ * max Q(s') * (1 - done)
            # (1 - done) masks out future rewards for terminal states
            target_q = rewards + self.gamma * next_q_max * (1 - dones)

        # ── Loss and backprop 
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients
        # Clips gradient norm to max 1.0, common practice in RL
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Copies live brain weights into the frozen target network.
        Called every `target_update_freq` steps.
        """
        self.target_brain.load_state_dict(self.brain.state_dict())

    # ── Full step 

    def step(self, neurumi: NeurumiState) -> dict:
        """
        One complete RL step:
        1. Select action (epsilon-greedy)
        2. Apply action to Neurumi's state
        3. Compute reward
        4. Store experience in buffer
        5. Train on a random mini-batch
        6. Decay epsilon
        7. Sync target network if needed

        Returns a dict with step metadata for logging and UI display.
        """
        state_tensor = neurumi.to_tensor()

        # 1. Select action
        action_idx, action_name = self.select_action(state_tensor)

        # 2. Apply action — modifies neurumi in place
        neurumi.apply_action_effect(ACTION_EFFECTS[action_name])
        neurumi.tick()

        # 3. Compute reward from the resulting state
        reward = compute_reward(neurumi)
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        # 4. Store experience
        next_state_tensor = neurumi.to_tensor()
        self.buffer.push(Experience(
            state      = state_tensor,
            action_idx = action_idx,
            reward     = reward,
            next_state = next_state_tensor,
            done       = False
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

        # 7. Sync target network periodically
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()

        return {
            "action":   action_name,
            "reward":   reward,
            "loss":     loss,
            "epsilon":  round(self.epsilon, 3),
            "step":     self.steps,
            "explored": action_idx != self.brain(state_tensor).argmax().item()
                        if not self.buffer.is_ready else False,
        }

    def get_q_values(self, neurumi: NeurumiState) -> dict[str, float]:
        """
        Returns the current Q-values for all actions given Neurumi's state.
        Used in the UI to visualize what the agent thinks about each action.
        """
        self.brain.eval()
        with torch.no_grad():
            q = self.brain(neurumi.to_tensor())
        self.brain.train()
        return {action: round(q[i].item(), 3) for i, action in enumerate(ACTIONS)}