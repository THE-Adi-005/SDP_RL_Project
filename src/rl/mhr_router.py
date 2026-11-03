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

    def __init__(self, n_actions: int = 8, k: int = 5, max_memory_per_arm: int = 1000):
        """
        Args:
            n_actions: Number of available VLM routing arms.
            k: Number of nearest neighbors to retrieve per arm.
            max_memory_per_arm: Maximum memory buffer capacity per arm.
        """
        self.n_actions = n_actions
        self.k = k
        self.max_memory_per_arm = max_memory_per_arm

        # Memory buffers: arm -> list of (state_vector, reward)
        self.success_memory: Dict[int, List[Tuple[np.ndarray, float]]] = {
            a: [] for a in range(n_actions)
        }
        self.failure_memory: Dict[int, List[Tuple[np.ndarray, float]]] = {
            a: [] for a in range(n_actions)
        }
        self.total_steps = 0

    def get_name(self) -> str:
        return f"MHRRouter(k={self.k})"

    def select_action(self, state: np.ndarray, budget_fraction: Optional[float] = None) -> int:
        """Select action by scoring expected reward from k-NN historical memory."""
        state = state.astype(np.float64).flatten()
        state_norm = np.linalg.norm(state)
        if state_norm > 1e-12:
            state_unit = state / state_norm
        else:
            state_unit = state

        scores = np.zeros(self.n_actions, dtype=np.float64)

        for arm in range(self.n_actions):
            succ = self.success_memory[arm]
            fail = self.failure_memory[arm]
            total_examples = len(succ) + len(fail)

            if total_examples < self.k:
                scores[arm] = 0.5  # Prior before sufficient memory
            else:
                combined = succ + fail
                feats = np.array([item[0] for item in combined], dtype=np.float64)
                rewards = np.array([item[1] for item in combined], dtype=np.float64)

                # Vectorized cosine similarity
                norms = np.linalg.norm(feats, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                feats_unit = feats / norms
                sims = feats_unit @ state_unit

                # Top-k nearest neighbors
                k_actual = min(self.k, len(combined))
                top_k_idxs = np.argpartition(sims, -k_actual)[-k_actual:]
                scores[arm] = float(np.mean(rewards[top_k_idxs]))

        return int(np.argmax(scores))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Store interaction experience in arm's success/failure memory buffer."""
        state = state.astype(np.float64).flatten()
        act = int(action)
        rew = float(reward)

        target_buf = self.success_memory[act] if rew > 0.5 else self.failure_memory[act]
        target_buf.append((state, rew))

        if len(target_buf) > self.max_memory_per_arm:
            target_buf.pop(0)

        self.total_steps += 1
