"""Thompson Sampling with Linear Payoffs — Bayesian contextual bandit.

For each arm a, maintains a posterior distribution over the weight vector:
  θₐ ~ N(μₐ, v² Bₐ⁻¹)

At each step:
  1. Sample θ̃ₐ ~ N(μₐ, v² Bₐ⁻¹) for each arm
  2. Select arm = argmax(θ̃ₐᵀ x)
  3. Update posterior with observed (x, reward)

Reference: Agrawal, S. & Goyal, N. (2013).
           "Thompson Sampling for Contextual Bandits with Linear Payoffs."
           Proceedings of the 30th International Conference on Machine Learning (ICML).

Regret bound: O(d * sqrt(T) * polylog(T))
"""
import numpy as np


class ThompsonSamplingLinear:
    """Thompson Sampling with linear payoffs — Bayesian exploration."""

    def __init__(self, n_actions: int = 4, v_squared: float = 1.0,
                 lambda_reg: float = 1.0, feature_dim: int = None,
                 n_arms: int = None, d: int = None, v: float = None, **kwargs):
        """
        Args:
            n_actions: Number of arms (strategies).
            v_squared: Variance scaling for posterior sampling.
                       Controls exploration intensity. Higher → more exploration.
            lambda_reg: Regularization parameter for ridge regression prior.
            feature_dim: Dimension of context features. Lazily initialized if None.
        """
        if n_arms is not None:
            n_actions = n_arms
        if d is not None:
            feature_dim = d
        if v is not None:
            v_squared = v ** 2

        self.n_actions = n_actions
        self.v_squared = v_squared
        self.lambda_reg = lambda_reg
        self.feature_dim = feature_dim
        self._initialized = False

        self.B = None      # Per-arm precision matrices (B = λI + ΣxxᵀΤ)
        self.mu = None     # Per-arm posterior mean vectors
        self.f = None      # Per-arm cumulative reward-weighted features (Σ r·x)
        self.B_inv = None  # Cached inverses

    def _initialize(self, dim: int):
        """Lazily initialize when feature dimension is known."""
        self.feature_dim = dim
        self.B = [self.lambda_reg * np.eye(dim) for _ in range(self.n_actions)]
        self.B_inv = [(1.0 / self.lambda_reg) * np.eye(dim) for _ in range(self.n_actions)]
        self.f = [np.zeros(dim) for _ in range(self.n_actions)]
        self.mu = [np.zeros(dim) for _ in range(self.n_actions)]
        self._initialized = True

    def get_name(self) -> str:
        return f"ThompsonSampling(v²={self.v_squared})"

    def select_action(self, state: np.ndarray) -> int:
        """Sample from posterior and select arm with highest sampled reward."""
        if not self._initialized:
            self._initialize(len(state))

        x = state.astype(np.float64)
        sampled_rewards = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            # Fast exact scalar Gaussian sampling: θ̃ₐᵀx ~ N(μₐᵀx, v² xᵀ Bₐ⁻¹ x)
            # Mathematically identical to drawing multivariate θ̃ₐ and taking dot product with x,
            # but avoids O(d³) Cholesky decomposition of d×d covariance at every step.
            mean_val = float(self.mu[a] @ x)
            var_val = float(max(1e-12, self.v_squared * (x @ self.B_inv[a] @ x)))
            sampled_rewards[a] = np.random.normal(mean_val, np.sqrt(var_val))

        return int(np.argmax(sampled_rewards))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Update posterior for the chosen arm.

        Bₐ ← Bₐ + xxᵀ
        fₐ ← fₐ + r·x
        μₐ ← Bₐ⁻¹ fₐ
        """
        if not self._initialized:
            self._initialize(len(state))

        x = state.astype(np.float64)
        a = action

        # Update precision matrix and reward-weighted features
        self.B[a] += np.outer(x, x)
        self.f[a] += reward * x

        # Sherman-Morrison update of B_inv
        B_inv_x = self.B_inv[a] @ x
        denom = 1.0 + x @ B_inv_x
        self.B_inv[a] -= np.outer(B_inv_x, B_inv_x) / denom

        # Update posterior mean
        self.mu[a] = self.B_inv[a] @ self.f[a]
