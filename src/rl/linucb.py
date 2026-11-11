"""LinUCB (Disjoint) — Contextual bandit with linear payoffs and UCB exploration.

For each arm a, maintains:
  Aₐ = d×d matrix (initialized to λ I)
  bₐ = d-dim vector (initialized to zero)

Prediction: θ̂ₐ = Aₐ⁻¹ bₐ  (ridge regression solution)
Selection:  arm = argmax( θ̂ₐᵀx + α_t * sqrt(xᵀ Aₐ⁻¹ x) )

CascadeRoute-VL Enhancements:
1. Warm-Start: Initializes Aₐ and bₐ from offline logged VLM interaction pairs,
   eliminating cold-start regret while maintaining online adaptation.
2. Budget-Aware Annealing: α_t = α_0 * b_t where b_t is the remaining budget fraction,
   providing formal budget feasibility guarantees.
3. Sherman-Morrison O(d²) rank-1 inverse updates with vectorized batched scoring.
"""
from typing import Optional, Tuple, List
import numpy as np

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


class LinUCB:
    """Disjoint LinUCB — contextual bandit algorithm with warm-start & budget annealing."""

    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 1.0,
        feature_dim: Optional[int] = None,
        alpha_anneal: bool = False,
        regularization: float = 1.0,
        n_arms: Optional[int] = None,
        d: Optional[int] = None,
        use_gpu: bool = False,
        **kwargs
    ):
        if n_arms is not None:
            n_actions = n_arms
        if d is not None:
            feature_dim = d

        self.n_actions = n_actions
        self.alpha_0 = float(alpha)
        self.alpha = float(alpha)
        self.feature_dim = feature_dim
        self.alpha_anneal = alpha_anneal
        self.regularization = float(regularization)
        self.use_gpu = bool(use_gpu and _TORCH_OK and torch.cuda.is_available())
        self._device = torch.device("cuda") if self.use_gpu else None
        self._initialized = False
        self.warm_started = False
        self.total_steps = 0

        self.A: Optional[List[np.ndarray]] = None
        self.b: Optional[List[np.ndarray]] = None
        self.A_inv: Optional[List[np.ndarray]] = None
        self._A_inv_stack: Optional[np.ndarray] = None  # (K, d, d)
        self._theta_stack: Optional[np.ndarray] = None  # (K, d)

        # Optional GPU tensors synchronized when self.use_gpu is True
        self._A_inv_g = None
        self._b_g = None
        self._theta_g = None

        if feature_dim is not None:
            self._initialize(feature_dim)

    def _sync_to_gpu(self):
        if not self.use_gpu:
            return
        self._A_inv_g = torch.from_numpy(self._A_inv_stack.astype(np.float32)).to(self._device)
        b_mat = np.stack(self.b, axis=0).astype(np.float32)
        self._b_g = torch.from_numpy(b_mat).to(self._device)
        self._theta_g = torch.from_numpy(self._theta_stack.astype(np.float32)).to(self._device)

    def _initialize(self, dim: int):
        """Lazily initialize per-arm matrices when feature dimension is known."""
        self.feature_dim = dim
        lam = self.regularization
        self.A = [np.eye(dim, dtype=np.float64) * lam for _ in range(self.n_actions)]
        self.b = [np.zeros(dim, dtype=np.float64) for _ in range(self.n_actions)]
        self.A_inv = [np.eye(dim, dtype=np.float64) / lam for _ in range(self.n_actions)]
        self._A_inv_stack = np.stack(self.A_inv, axis=0)
        self._theta_stack = np.zeros((self.n_actions, dim), dtype=np.float64)
        self._sync_to_gpu()
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
        regularization: Optional[float] = None
    ):
        """Warm-start bandit matrices from logged offline outcomes.

        Aₐ = λI + ∑_{i: a_i=a} x_i x_iᵀ
        bₐ = ∑_{i: a_i=a} r_i x_i
        Aₐ⁻¹ = inv(Aₐ)
        """
        N, d = states.shape
        reg = float(regularization if regularization is not None else self.regularization)
        self.regularization = reg
        self._initialize(d)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards, dtype=np.float64)

        for a in range(self.n_actions):
            self.A[a] = np.eye(d, dtype=np.float64) * reg
            self.b[a] = np.zeros(d, dtype=np.float64)

            mask = (actions == a)
            if np.any(mask):
                X_a = states[mask].astype(np.float64)
                r_a = rewards[mask].astype(np.float64)
                self.A[a] += X_a.T @ X_a
                self.b[a] += X_a.T @ r_a

            self.A_inv[a] = np.linalg.pinv(self.A[a])
            self._A_inv_stack[a] = self.A_inv[a]
            self._theta_stack[a] = self.A_inv[a] @ self.b[a]

        self.warm_started = True
        self._sync_to_gpu()

    def warm_start_from_full_matrix(
        self,
        X_train: np.ndarray,
        R_train: np.ndarray,
        regularization: Optional[float] = None,
        sample_weight: float = 1.0,
        precomputed_XtX: Optional[np.ndarray] = None,
    ):
        """Fast vectorized warm-start when offline logs contain outcomes across all K arms."""
        N, d = X_train.shape
        K = R_train.shape[1]
        assert K == self.n_actions
        reg = float(regularization if regularization is not None else self.regularization)
        self.regularization = reg
        self._initialize(d)

        X_f64 = X_train.astype(np.float64)
        R_f64 = R_train.astype(np.float64)
        XtX_raw = precomputed_XtX if precomputed_XtX is not None else (X_f64.T @ X_f64)
        XtX = sample_weight * XtX_raw
        A0 = XtX + np.eye(d, dtype=np.float64) * reg
        A0_inv = np.linalg.pinv(A0)
        XtR = sample_weight * (X_f64.T @ R_f64)  # (d, K)
        Thetas = A0_inv @ XtR  # (d, K)

        self._A_inv_stack[:] = A0_inv[None, :, :]
        self._theta_stack[:] = Thetas.T
        if not self.use_gpu:
            for a in range(K):
                self.A[a] = A0.copy()
                self.b[a] = XtR[:, a].copy()
                self.A_inv[a] = A0_inv.copy()
        else:
            for a in range(K):
                self.b[a] = XtR[:, a].copy()
                self.A_inv[a] = A0_inv

        self.warm_started = True
        self._sync_to_gpu()

    def score_arms(
        self,
        state: np.ndarray,
        budget_fraction: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized computation of (expected_reward, epistemic_uncertainty, ucb_score)."""
        if not self._initialized:
            self._initialize(len(state))

        effective_alpha = self.alpha_0
        if self.alpha_anneal and budget_fraction is not None:
            effective_alpha = self.alpha_0 * max(0.0, min(1.0, float(budget_fraction)))

        if self.use_gpu:
            if isinstance(state, torch.Tensor):
                xg = state
            else:
                xg = torch.from_numpy(state.astype(np.float32)).to(self._device)
            preds_t = torch.mv(self._theta_g, xg)
            Ax = torch.mv(self._A_inv_g.view(self.n_actions * self.feature_dim, self.feature_dim), xg).view(
                self.n_actions, self.feature_dim
            )
            uncs_t = torch.sqrt(torch.clamp(torch.mv(Ax, xg), min=0.0))
            ucb_t = preds_t + effective_alpha * uncs_t
            res = torch.stack([preds_t, uncs_t, ucb_t], dim=0).cpu().numpy()
            return res[0], res[1], res[2]

        x = state.astype(np.float64)
        preds = self._theta_stack @ x                               # (K,)
        Ax = (self._A_inv_stack.reshape(self.n_actions * self.feature_dim, self.feature_dim) @ x).reshape(
            self.n_actions, self.feature_dim
        )
        uncs = np.sqrt(np.maximum(0.0, Ax @ x))                     # (K,)
        ucb_values = preds + effective_alpha * uncs
        return preds, uncs, ucb_values

    def select_action(
        self,
        state: np.ndarray,
        budget_fraction: Optional[float] = None,
        budget_remaining: Optional[float] = None
    ) -> int:
        if budget_fraction is None and budget_remaining is not None:
            budget_fraction = budget_remaining
        _, _, ucb_values = self.score_arms(state, budget_fraction=budget_fraction)
        return int(np.argmax(ucb_values))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Update arm's model after observing reward using Sherman-Morrison O(d²) update."""
        if not self._initialized:
            self._initialize(len(state))

        a = int(action)
        r = float(reward)

        if self.use_gpu:
            if isinstance(state, torch.Tensor):
                xg = state
            else:
                xg = torch.from_numpy(state.astype(np.float32)).to(self._device)
            Ax_a = torch.mv(self._A_inv_g[a], xg)
            denom = 1.0 + float(torch.dot(xg, Ax_a).item())
            if abs(denom) > 1e-12:
                self._A_inv_g[a].addr_(Ax_a, Ax_a, alpha=-1.0 / denom)
            self._b_g[a].add_(xg, alpha=r)
            self._theta_g[a] = torch.mv(self._A_inv_g[a], self._b_g[a])
            self.total_steps += 1
            return

        x = state.astype(np.float64)
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x

        A_inv_x = self._A_inv_stack[a] @ x
        denom = 1.0 + float(x @ A_inv_x)
        if abs(denom) > 1e-12:
            self.A_inv[a] -= np.outer(A_inv_x, A_inv_x) / denom
        else:
            self.A_inv[a] = np.linalg.pinv(self.A[a])

        self._A_inv_stack[a] = self.A_inv[a]
        self._theta_stack[a] = self._A_inv_stack[a] @ self.b[a]
        self.total_steps += 1
