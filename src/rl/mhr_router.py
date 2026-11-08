"""MHR-Router (Multimodal Heterogeneous Retrieval-Enhanced Adaptive Router).

Reference:
- "MHR-Router: Multimodal Heterogeneous Retrieval-Enhanced Adaptive Routing" (Oct 2026).
- Maintains success memory and failure memory buffers for each VLM arm.
- For incoming multimodal state s_t, retrieves k-nearest neighbors in historical memory
  and computes empirical success expectation per arm via fast vectorized cosine similarity.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np


class MHRRouter:
    """Retrieval-enhanced multimodal adaptive router baseline with vectorized k-NN."""

    def __init__(
        self,
        n_actions: int = 8,
        k: int = 5,
        max_memory_per_arm: int = 1000,
        n_arms: Optional[int] = None,
        state_dim: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            n_actions: Number of available VLM routing arms.
            k: Number of nearest neighbors to retrieve per arm.
            max_memory_per_arm: Maximum memory buffer capacity per arm.
        """
        if n_arms is not None:
            n_actions = n_arms
        self.n_actions = n_actions
        self.k = k
        self.max_memory_per_arm = max_memory_per_arm
        self.state_dim = state_dim

        # Memory buffers: arm -> list of (state_vector, reward)
        self.success_memory: Dict[int, List[Tuple[np.ndarray, float]]] = {
            a: [] for a in range(n_actions)
        }
        self.failure_memory: Dict[int, List[Tuple[np.ndarray, float]]] = {
            a: [] for a in range(n_actions)
        }
        # Pre-cached unit feature matrices and rewards for O(1) select_action speed
        self._cached_feats_unit: Dict[int, Optional[np.ndarray]] = {a: None for a in range(n_actions)}
        self._cached_rewards: Dict[int, Optional[np.ndarray]] = {a: None for a in range(n_actions)}
        self.total_steps = 0

    def get_name(self) -> str:
        return f"MHRRouter(k={self.k})"

    def _rebuild_cache_for_arm(self, arm: int):
        succ = self.success_memory[arm]
        fail = self.failure_memory[arm]
        combined = succ + fail
        if len(combined) < self.k:
            self._cached_feats_unit[arm] = None
            self._cached_rewards[arm] = None
            return

        feats = np.array([item[0] for item in combined], dtype=np.float32)
        rewards = np.array([item[1] for item in combined], dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self._cached_feats_unit[arm] = feats / norms
        self._cached_rewards[arm] = rewards

    def select_action(self, state: np.ndarray, budget_fraction: Optional[float] = None) -> int:
        """Select action by scoring expected reward from k-NN historical memory."""
        state = state.astype(np.float32).flatten()
        state_norm = np.linalg.norm(state)
        state_unit = state / state_norm if state_norm > 1e-12 else state

        scores = np.zeros(self.n_actions, dtype=np.float32)

        for arm in range(self.n_actions):
            feats_unit = self._cached_feats_unit[arm]
            rewards = self._cached_rewards[arm]

            if feats_unit is None or len(rewards) < self.k:
                scores[arm] = 0.5  # Prior before sufficient memory
            else:
                sims = feats_unit @ state_unit
                k_actual = min(self.k, len(rewards))
                top_k_idxs = np.argpartition(sims, -k_actual)[-k_actual:]
                scores[arm] = float(np.mean(rewards[top_k_idxs]))

        return int(np.argmax(scores))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Store interaction experience in arm's success/failure memory buffer."""
        state = state.astype(np.float32).flatten()
        act = int(action)
        rew = float(reward)

        target_buf = self.success_memory[act] if rew > 0.5 else self.failure_memory[act]
        target_buf.append((state, rew))

        if len(target_buf) > self.max_memory_per_arm:
            target_buf.pop(0)

        self._rebuild_cache_for_arm(act)
        self.total_steps += 1
