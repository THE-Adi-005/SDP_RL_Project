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
        # Pre-allocated flat contiguous numpy buffers for O(1) incremental appends
        self._buf_feats: Dict[int, Optional[np.ndarray]] = {a: None for a in range(n_actions)}
        self._buf_rews: Dict[int, Optional[np.ndarray]] = {a: None for a in range(n_actions)}
        self._buf_len: Dict[int, int] = {a: 0 for a in range(n_actions)}
        self._dirty_arms = set()
        self.total_steps = 0

    def get_name(self) -> str:
        return f"MHRRouter(k={self.k})"

    def _rebuild_cache_for_arm(self, arm: int):
        combined = self.success_memory[arm] + self.failure_memory[arm]
        n = len(combined)
        if n == 0:
            self._buf_len[arm] = 0
            return
        cap = 2 * self.max_memory_per_arm + 8
        dim = len(combined[0][0])
        if self._buf_feats[arm] is None or self._buf_feats[arm].shape != (cap, dim):
            self._buf_feats[arm] = np.empty((cap, dim), dtype=np.float32)
            self._buf_rews[arm] = np.empty(cap, dtype=np.float32)
        feats = np.array([item[0] for item in combined], dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self._buf_feats[arm][:n] = feats / norms
        self._buf_rews[arm][:n] = np.array([item[1] for item in combined], dtype=np.float32)
        self._buf_len[arm] = n

    def select_action(self, state: np.ndarray, budget_fraction: Optional[float] = None) -> int:
        """Select action by scoring expected reward from k-NN historical memory."""
        if self._dirty_arms:
            for dirty_arm in list(self._dirty_arms):
                self._rebuild_cache_for_arm(dirty_arm)
            self._dirty_arms.clear()

        state = state.astype(np.float32).flatten()
        state_norm = np.linalg.norm(state)
        state_unit = state / state_norm if state_norm > 1e-12 else state

        scores = np.zeros(self.n_actions, dtype=np.float32)

        for arm in range(self.n_actions):
            n = self._buf_len[arm]
            if n < self.k or self._buf_feats[arm] is None:
                scores[arm] = 0.5  # Prior before sufficient memory
            else:
                sims = self._buf_feats[arm][:n] @ state_unit
                k_actual = min(self.k, n)
                top_k_idxs = np.argpartition(sims, -k_actual)[-k_actual:]
                scores[arm] = float(np.mean(self._buf_rews[arm][:n][top_k_idxs]))

        return int(np.argmax(scores))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Store interaction experience in arm's success/failure memory buffer."""
        state = state.astype(np.float32).flatten()
        act = int(action)
        rew = float(reward)

        target_buf = self.success_memory[act] if rew > 0.5 else self.failure_memory[act]
        target_buf.append((state, rew))

        evicted = False
        if len(target_buf) > self.max_memory_per_arm:
            target_buf.pop(0)
            evicted = True

        dim = len(state)
        cap = 2 * self.max_memory_per_arm + 8
        if self._buf_feats[act] is None:
            self._buf_feats[act] = np.empty((cap, dim), dtype=np.float32)
            self._buf_rews[act] = np.empty(cap, dtype=np.float32)

        if evicted or self._buf_len[act] >= cap:
            self._dirty_arms.add(act)
        else:
            pos = self._buf_len[act]
            sn = float(np.linalg.norm(state))
            self._buf_feats[act][pos] = state / sn if sn > 1e-12 else state
            self._buf_rews[act][pos] = np.float32(rew)
            self._buf_len[act] = pos + 1
        self.total_steps += 1
