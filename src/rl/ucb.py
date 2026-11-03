"""UCB1 (Upper Confidence Bound) — Non-contextual multi-armed bandit.

Selects arm = argmax( Q̄ₐ + c * sqrt(ln(t) / Nₐ) )
where Q̄ₐ is the average reward for arm a, Nₐ is the count, t is total pulls.

Reference: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
           "Finite-time analysis of the multiarmed bandit problem."
           Machine Learning, 47(2-3), 235–256.
"""
import numpy as np
import math


class UCB1:
    """Non-contextual UCB1 bandit — ignores features, uses per-arm statistics."""

    def __init__(self, n_actions: int = 4, c: float = 1.414):
        """
        Args:
            n_actions: Number of arms (strategies).
            c: Exploration parameter (default sqrt(2), theoretically optimal).
        """
        self.n_actions = n_actions
        self.c = c
        self.counts = np.zeros(n_actions, dtype=int)     # Nₐ
        self.sum_rewards = np.zeros(n_actions)            # Σ rewards per arm
        self.total_pulls = 0

    def get_name(self) -> str:
        return "UCB1"

    def select_action(self, state: np.ndarray) -> int:
        """Select action using UCB1 (state is ignored — non-contextual)."""
        # If any arm hasn't been tried, try it first (initialization)
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return int(untried[0])

        # UCB1 formula: Q̄ₐ + c * sqrt(ln(t) / Nₐ)
        avg_rewards = self.sum_rewards / self.counts
        ucb_bonus = self.c * np.sqrt(math.log(self.total_pulls) / self.counts)
        ucb_values = avg_rewards + ucb_bonus
        return int(np.argmax(ucb_values))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Update arm statistics after observing reward."""
        self.counts[action] += 1
        self.sum_rewards[action] += reward
        self.total_pulls += 1
