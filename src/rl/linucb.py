"""LinUCB (Disjoint) — Contextual bandit with linear payoffs and UCB exploration.

For each arm a, maintains:
  Aₐ = d×d matrix (initialized to identity)
  bₐ = d-dim vector (initialized to zero)

Prediction: θ̂ₐ = Aₐ⁻¹ bₐ  (ridge regression solution)
Selection:  arm = argmax( θ̂ₐᵀx + α_t * sqrt(xᵀ Aₐ⁻¹ x) )

The second term is the UCB bonus — wider for less-explored (state, arm) pairs.

CascadeRoute-VL Enhancements:
1. Warm-Start: Initializes Aₐ and bₐ from offline logged VLM interaction pairs,
   eliminating cold-start regret while maintaining online adaptation.
2. Budget-Aware Annealing: α_t = α_0 * b_t where b_t is the remaining budget fraction,
   providing formal budget feasibility guarantees.
3. Sherman-Morrison O(d²) rank-1 inverse updates.

Reference:
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010).
  "A contextual-bandit approach to personalized news article recommendation." WWW 2010.
- CascadeRoute-VL (CVPR 2027): "Taming the Reasoning Tax in Vision-Language Models
  via Online Contextual Bandit Routing"
"""
from typing import Optional, Union, List, Dict
import numpy as np


class LinUCB:
    """Disjoint LinUCB — contextual bandit algorithm with warm-start & budget annealing."""

    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 1.0,
        feature_dim: Optional[int] = None,
        alpha_anneal: bool = False
    ):
        """
        Args:
            n_actions: Number of arms (strategies).
            alpha: Exploration parameter controlling UCB width (α_0).
                   Higher α → more exploration.
            feature_dim: Dimension of the context feature vector.
                         If None, initialized on first call.
            alpha_anneal: If True, scales alpha by remaining budget fraction b_t.
        """
        self.n_actions = n_actions
        self.alpha_0 = float(alpha)
        self.alpha = float(alpha)
        self.feature_dim = feature_dim
        self.alpha_anneal = alpha_anneal
        self._initialized = False
        self.warm_started = False
        self.total_steps = 0

        # Per-arm matrices — initialized lazily when feature_dim is known
        self.A: Optional[List[np.ndarray]] = None      # List of d×d matrices
        self.b: Optional[List[np.ndarray]] = None      # List of d-dim vectors
        self.A_inv: Optional[List[np.ndarray]] = None  # Cached inverses for efficiency

        if feature_dim is not None:
            self._initialize(feature_dim)

    def _initialize(self, dim: int):
        """Lazily initialize per-arm matrices when feature dimension is known."""
        self.feature_dim = dim
        self.A = [np.eye(dim, dtype=np.float64) for _ in range(self.n_actions)]
        self.b = [np.zeros(dim, dtype=np.float64) for _ in range(self.n_actions)]
        self.A_inv = [np.eye(dim, dtype=np.float64) for _ in range(self.n_actions)]
        self._initialized = True

    def get_name(self) -> str:
        tag = f"LinUCB(α={self.alpha:.2f}"
        if self.warm_started:
            tag += ", warm-start"
        if self.alpha_anneal:
            tag += ", annealed"
        tag += ")"
        return tag

    def warm_start(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        regularization: float = 1.0
    ):
        """Warm-start bandit matrices from logged offline outcomes.

        Aₐ = λI + ∑_{i: a_i=a} x_i x_iᵀ
        bₐ = ∑_{i: a_i=a} r_i x_i
        Aₐ⁻¹ = inv(Aₐ)

        Args:
            states: Array of context vectors (N, d).
            actions: Array of selected action indices (N,).
            rewards: Array of scalar rewards (N,).
            regularization: Ridge regularization λ (default 1.0).
        """
        N, d = states.shape
        self._initialize(d)

        for a in range(self.n_actions):
            self.A[a] = np.eye(d, dtype=np.float64) * regularization
            self.b[a] = np.zeros(d, dtype=np.float64)

            mask = (actions == a)
            if np.any(mask):
                X_a = states[mask].astype(np.float64)
                r_a = rewards[mask].astype(np.float64)
                self.A[a] += X_a.T @ X_a
                self.b[a] += X_a.T @ r_a

            # Compute precise initial inverse
            self.A_inv[a] = np.linalg.pinv(self.A[a])

        self.warm_started = True

    def select_action(
        self,
        state: np.ndarray,
        budget_fraction: Optional[float] = None
    ) -> int:
        """Select action using LinUCB policy with optional budget-aware exploration.

        For each arm a:
          θ̂ₐ = Aₐ⁻¹ bₐ
          α_t = α_0 * b_t if alpha_anneal else α_0
          pₐ = θ̂ₐᵀx + α_t * sqrt(xᵀ Aₐ⁻¹ x)
        Return argmax(pₐ)
        """
        if not self._initialized:
            self._initialize(len(state))

        x = state.astype(np.float64)
        effective_alpha = self.alpha_0
        if self.alpha_anneal and budget_fraction is not None:
            effective_alpha = self.alpha_0 * max(0.0, min(1.0, float(budget_fraction)))

        ucb_values = np.zeros(self.n_actions, dtype=np.float64)

        for a in range(self.n_actions):
            theta_hat = self.A_inv[a] @ self.b[a]             # Ridge regression estimate
            pred = float(theta_hat @ x)                        # Expected reward
            uncertainty = float(np.sqrt(max(0.0, x @ self.A_inv[a] @ x))) # Confidence width
            ucb_values[a] = pred + effective_alpha * uncertainty

        return int(np.argmax(ucb_values))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Update arm's model after observing reward.

        Aₐ ← Aₐ + xxᵀ
        bₐ ← bₐ + r·x
        Aₐ⁻¹ updated via Sherman-Morrison for O(d²) computational efficiency.
        """
        if not self._initialized:
            self._initialize(len(state))

        x = state.astype(np.float64)
        a = int(action)
        r = float(reward)

        # Update A and b
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x

        # Sherman-Morrison inverse update: (A + xxᵀ)⁻¹ = A⁻¹ - (A⁻¹ x xᵀ A⁻¹) / (1 + xᵀ A⁻¹ x)
        A_inv_x = self.A_inv[a] @ x
        denom = 1.0 + float(x @ A_inv_x)
        if abs(denom) > 1e-12:
            self.A_inv[a] -= np.outer(A_inv_x, A_inv_x) / denom
        else:
            # Recompute directly in case of numerical instability
            self.A_inv[a] = np.linalg.pinv(self.A[a])

        self.total_steps += 1
