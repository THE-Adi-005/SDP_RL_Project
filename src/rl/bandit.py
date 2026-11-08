from typing import Optional
import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_actions=4, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.05, lr=0.1, n_arms=None, **kwargs):
        if n_arms is not None:
            n_actions = n_arms
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.q_values = {}  # action -> running average reward (context-free fallback)
        self.weights = None  # linear model per action, set on first update

    def get_name(self) -> str:
        return "EpsilonGreedy"

    def _init_weights(self, dim):
        self.weights = np.zeros((self.n_actions, dim))

    def select_action(self, state: Optional[np.ndarray] = None) -> int:
        if state is None:
            if np.random.random() < self.epsilon or not self.q_values:
                return np.random.randint(self.n_actions)
            return max(self.q_values, key=self.q_values.get)

        if self.weights is None:
            self._init_weights(len(state))
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        scores = self.weights @ state
        return int(np.argmax(scores))

    def update(self, *args, **kwargs):
        if len(args) == 2:
            action, reward = args
            state = kwargs.get('state', None)
        elif len(args) >= 3:
            state, action, reward = args[0], args[1], args[2]
        else:
            return

        action = int(action)
        reward = float(reward)

        # Context-free running average
        if action not in self.q_values:
            self.q_values[action] = reward
        else:
            self.q_values[action] = 0.9 * self.q_values[action] + 0.1 * reward

        # Contextual weights
        if state is not None:
            if self.weights is None:
                self._init_weights(len(state))
            prediction = self.weights[action] @ state
            error = reward - prediction
            self.weights[action] += self.lr * error * state / (np.linalg.norm(state) + 1e-8)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
