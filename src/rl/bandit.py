import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_actions=4, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.05, lr=0.1):
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

    def select_action(self, state: np.ndarray) -> int:
        if self.weights is None:
            self._init_weights(len(state))
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        scores = self.weights @ state
        return int(np.argmax(scores))

    def update(self, state: np.ndarray, action: int, reward: float):
        if self.weights is None:
            self._init_weights(len(state))
        prediction = self.weights[action] @ state
        error = reward - prediction
        self.weights[action] += self.lr * error * state / (np.linalg.norm(state) + 1e-8)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
