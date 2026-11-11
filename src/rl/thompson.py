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

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


class ThompsonSamplingLinear:
    """Thompson Sampling with linear payoffs — Bayesian exploration."""

    def __init__(self, n_actions: int = 4, v_squared: float = 1.0,
                 lambda_reg: float = 1.0, feature_dim: int = None,
                 n_arms: int = None, d: int = None, v: float = None,
                 use_gpu: bool = False, **kwargs):
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
        self.use_gpu = bool(use_gpu and _TORCH_OK and torch.cuda.is_available())
        self._device = torch.device("cuda") if self.use_gpu else None
        self._initialized = False

        self.B = None      # Per-arm precision matrices (B = λI + ΣxxᵀΤ)
        self.mu = None     # Per-arm posterior mean vectors
        self.f = None      # Per-arm cumulative reward-weighted features (Σ r·x)
        self.B_inv = None  # Cached inverses
        self._B_inv_g = None
        self._f_g = None
        self._mu_g = None

    def _initialize(self, dim: int):
        """Lazily initialize when feature dimension is known."""
        self.feature_dim = dim
        self.B = [self.lambda_reg * np.eye(dim) for _ in range(self.n_actions)]
        self.B_inv = [(1.0 / self.lambda_reg) * np.eye(dim) for _ in range(self.n_actions)]
        self._B_inv_stack = np.stack(self.B_inv, axis=0)
        self.f = [np.zeros(dim) for _ in range(self.n_actions)]
        self.mu = [np.zeros(dim) for _ in range(self.n_actions)]
        self._mu_stack = np.zeros((self.n_actions, dim), dtype=np.float64)
        if self.use_gpu:
            self._B_inv_g = torch.from_numpy(self._B_inv_stack.astype(np.float32)).to(self._device)
            self._f_g = torch.zeros((self.n_actions, dim), dtype=torch.float32, device=self._device)
            self._mu_g = torch.zeros((self.n_actions, dim), dtype=torch.float32, device=self._device)
        self._initialized = True

    def get_name(self) -> str:
        return f"ThompsonSampling(v²={self.v_squared})"

    def select_action(self, state: np.ndarray) -> int:
        """Sample from posterior and select arm with highest sampled reward."""
        if not self._initialized:
            self._initialize(len(state))

        if self.use_gpu:
            xg = state if isinstance(state, torch.Tensor) else torch.from_numpy(state.astype(np.float32)).to(self._device)
            means = torch.mv(self._mu_g, xg)
            Bx = torch.mv(self._B_inv_g.view(self.n_actions * self.feature_dim, self.feature_dim), xg).view(
                self.n_actions, self.feature_dim
            )
            vars_a = torch.clamp(self.v_squared * torch.mv(Bx, xg), min=1e-12)
            sampled = means + torch.sqrt(vars_a) * torch.randn_like(means)
            return int(torch.argmax(sampled).item())

        x = state.astype(np.float64)
        means = self._mu_stack @ x
        Bx = (self._B_inv_stack.reshape(self.n_actions * self.feature_dim, self.feature_dim) @ x).reshape(
            self.n_actions, self.feature_dim
        )
        vars_a = np.maximum(1e-12, self.v_squared * (Bx @ x))
        sampled_rewards = np.random.normal(means, np.sqrt(vars_a))
        return int(np.argmax(sampled_rewards))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Update posterior for the chosen arm.

        Bₐ ← Bₐ + xxᵀ
        fₐ ← fₐ + r·x
        μₐ ← Bₐ⁻¹ fₐ
        """
        if not self._initialized:
            self._initialize(len(state))

        a = int(action)
        r = float(reward)

        if self.use_gpu:
            xg = state if isinstance(state, torch.Tensor) else torch.from_numpy(state.astype(np.float32)).to(self._device)
            Bx_a = torch.mv(self._B_inv_g[a], xg)
            denom = 1.0 + float(torch.dot(xg, Bx_a).item())
            if abs(denom) > 1e-12:
                self._B_inv_g[a].addr_(Bx_a, Bx_a, alpha=-1.0 / denom)
            self._f_g[a].add_(xg, alpha=r)
            self._mu_g[a] = torch.mv(self._B_inv_g[a], self._f_g[a])
            return

        x = state.astype(np.float64)
        # Update precision matrix and reward-weighted features
        self.B[a] += np.outer(x, x)
        self.f[a] += r * x

        # Sherman-Morrison update of B_inv
        B_inv_x = self._B_inv_stack[a] @ x
        denom = 1.0 + float(x @ B_inv_x)
        self.B_inv[a] -= np.outer(B_inv_x, B_inv_x) / denom
        self._B_inv_stack[a] = self.B_inv[a]

        # Update posterior mean
        self.mu[a] = self._B_inv_stack[a] @ self.f[a]
        self._mu_stack[a] = self.mu[a]
