"""DQN (Deep Q-Network) — Deep RL approach for strategy selection.

Uses a 3-layer MLP to approximate Q(state, action).
Includes experience replay buffer and target network for stable training.

This tests whether non-linear feature interactions help beyond LinUCB's
linear model, at the cost of more compute and less interpretability.

Reference: Mnih, V., et al. (2015). "Human-level control through deep
           reinforcement learning." Nature, 518(7540), 529–533.
"""
import numpy as np
import random
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ReplayBuffer:
    """Fixed-size experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        return states, actions, rewards

    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """3-layer MLP: input_dim → 128 → 64 → n_actions."""

        def __init__(self, input_dim: int, n_actions: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, n_actions),
            )

        def forward(self, x):
            return self.net(x)


class DQNAgent:
    """Deep Q-Network agent for contextual strategy selection.

    Note: Since our problem is a single-step (contextual bandit, no next-state),
    the Q-target simplifies to just the immediate reward: Q_target = r
    (no discount factor needed). This makes DQN essentially a neural network
    reward predictor with experience replay for stability.
    """

    def __init__(self, n_actions: int = 4, feature_dim: int = None,
                 lr: float = 1e-3, epsilon: float = 0.3,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.05,
                 batch_size: int = 32, buffer_capacity: int = 10000,
                 tau: float = 0.01):
        """
        Args:
            n_actions: Number of arms (strategies).
            feature_dim: Input feature dimension. Lazily initialized if None.
            lr: Learning rate for Adam optimizer.
            epsilon: Initial exploration rate.
            epsilon_decay: Multiplicative decay per update.
            epsilon_min: Minimum exploration rate.
            batch_size: Mini-batch size for replay training.
            buffer_capacity: Maximum replay buffer size.
            tau: Soft update coefficient for target network.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")

        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.tau = tau
        self._initialized = False

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.device = torch.device("cpu")  # Keep on CPU for this project scale
        self.update_count = 0

    def _initialize(self, dim: int):
        """Lazily initialize networks when feature dimension is known."""
        self.feature_dim = dim
        self.q_network = QNetwork(dim, self.n_actions).to(self.device)
        self.target_network = QNetwork(dim, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self._initialized = True

    def get_name(self) -> str:
        return "DQN"

    def select_action(self, state: np.ndarray) -> int:
        """Select action using ε-greedy over Q-network."""
        if not self._initialized:
            self._initialize(len(state))

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def update(self, state: np.ndarray, action: int, reward: float):
        """Store transition and train on a mini-batch from replay buffer."""
        if not self._initialized:
            self._initialize(len(state))

        # Store experience
        self.replay_buffer.push(state, action, reward)

        # Only train when we have enough samples (train every 4 steps for efficiency)
        self.update_count += 1
        if len(self.replay_buffer) < self.batch_size or self.update_count % 4 != 0:
            return

        # Sample mini-batch
        states, actions, rewards = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)

        # Q-values for chosen actions
        q_values = self.q_network(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target: just the immediate reward (single-step bandit, no next state)
        q_targets = rewards_t

        # Compute loss and backprop
        loss = self.loss_fn(q_selected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.update_count += 1

    def _soft_update(self):
        """Polyak averaging: θ_target ← τ·θ_online + (1−τ)·θ_target"""
        for target_param, online_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
