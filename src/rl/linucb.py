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
        alpha_min: float = 0.0,
        regularization: float = 1.0,
        discount_factor: float = 1.0,
        sliding_window: Optional[int] = None,
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
        self.alpha_min = float(alpha_min)
        self.feature_dim = feature_dim
        self.alpha_anneal = alpha_anneal
        self.regularization = float(regularization)
        self.discount_factor = float(discount_factor)
        self.sliding_window = int(sliding_window) if sliding_window is not None else None
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
        self._arm_pulls: Optional[List[int]] = None
        self._sw_buffers: Optional[List[List[Tuple[np.ndarray, float]]]] = None
        self._A_base_prior: Optional[List[np.ndarray]] = None
        self._b_base_prior: Optional[List[np.ndarray]] = None

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
        self._arm_pulls = [0 for _ in range(self.n_actions)]
        self._sw_buffers = [[] for _ in range(self.n_actions)]
        self._A_base_prior = [np.eye(dim, dtype=np.float64) * lam for _ in range(self.n_actions)]
        self._b_base_prior = [np.zeros(dim, dtype=np.float64) for _ in range(self.n_actions)]
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
        self._A_g = None
        if self.discount_factor < 1.0 and self.use_gpu:
            self._A_g = torch.from_numpy(A0.astype(np.float32)).to(self._device).unsqueeze(0).repeat(K, 1, 1)
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
            effective_alpha = max(self.alpha_min, self.alpha_0 * max(0.0, min(1.0, float(budget_fraction))))

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

    def select_action_budgeted(
        self,
        state: np.ndarray,
        budget_remaining_usd: float,
        total_budget_usd: float,
        arm_cost_est: np.ndarray,
        cheapest_arm: Optional[int] = None,
        reserve_multiplier: float = 1.0,
        remaining_steps: Optional[int] = None,
        cheapest_arm_cost_bound: Optional[float] = None,
    ) -> Tuple[int, bool]:
        """Deterministic budget-guarded action selection (Theorem 2).

        Reserves the cheapest-arm horizon cost ((remaining_steps - 1) * c_min_safe when
        horizon is known, or reserve_multiplier * c_min otherwise). Once remaining USD
        drops below the cheapest-arm horizon reserve (or no non-cheapest arm fits inside
        remaining headroom), forces the cheapest arm so realized total spend satisfies
        C_T <= B_max + c_max deterministically whenever B_max >= T * c_min_safe.
        """
        c_est = np.asarray(arm_cost_est, dtype=np.float64)
        c_arm = int(np.argmin(c_est)) if cheapest_arm is None else int(cheapest_arm)
        c_min = float(c_est[c_arm])
        c_safe = float(cheapest_arm_cost_bound) if cheapest_arm_cost_bound is not None else c_min
        b_frac = max(0.0, min(1.0, float(budget_remaining_usd) / max(float(total_budget_usd), 1e-12)))

        if remaining_steps is not None and remaining_steps > 1:
            future_reserve = (remaining_steps - 1) * c_safe
        else:
            future_reserve = reserve_multiplier * c_safe

        if budget_remaining_usd <= future_reserve + reserve_multiplier * c_safe:
            return c_arm, True

        _, _, ucb = self.score_arms(state, budget_fraction=b_frac)
        headroom = budget_remaining_usd - future_reserve
        feasible = c_est <= headroom
        feasible[c_arm] = True  # cheapest fallback arm is always admissible
        if not np.any(feasible):
            return c_arm, True
        masked = np.where(feasible, ucb, -1e12)
        return int(np.argmax(masked)), False

    def hot_onboard_arm(
        self,
        arm: int,
        mean_prior_theta: Optional[np.ndarray] = None,
        prior_strength: float = 1.0,
        init_reg: Optional[float] = None,
    ):
        """Online onboarding of a newly registered VLM arm at test time without full retraining."""
        if not self._initialized:
            raise RuntimeError("LinUCB must be initialized before onboarding an arm.")
        a = int(arm)
        d = self.feature_dim
        lam = float(init_reg if init_reg is not None else self.regularization)
        A0 = np.eye(d, dtype=np.float64) * lam
        A0_inv = np.eye(d, dtype=np.float64) / lam
        b0 = (lam * prior_strength * mean_prior_theta.astype(np.float64)) if mean_prior_theta is not None else np.zeros(d, dtype=np.float64)
        theta0 = A0_inv @ b0

        self.A[a] = A0
        self.A_inv[a] = A0_inv
        self.b[a] = b0
        self._A_inv_stack[a] = A0_inv
        self._theta_stack[a] = theta0
        self._arm_pulls[a] = 0
        self._sw_buffers[a] = []
        if self.use_gpu:
            self._A_inv_g[a] = torch.from_numpy(A0_inv.astype(np.float32)).to(self._device)
            self._b_g[a] = torch.from_numpy(b0.astype(np.float32)).to(self._device)
            self._theta_g[a] = torch.from_numpy(theta0.astype(np.float32)).to(self._device)
            if getattr(self, "_A_g", None) is not None:
                self._A_g[a] = torch.from_numpy(A0.astype(np.float32)).to(self._device)

    def _refresh_inverse_gpu(self, a: int):
        A_inv_new = torch.linalg.inv(self._A_g[a] + 1e-6 * torch.eye(self.feature_dim, device=self._device))
        self._A_inv_g[a] = 0.5 * (A_inv_new + A_inv_new.T)
        self._theta_g[a] = torch.mv(self._A_inv_g[a], self._b_g[a])

    def update(self, state: np.ndarray, action: int, reward: float):
        """Update arm's model after observing reward using Sherman-Morrison O(d²) update,
        Discounted LinUCB (discount_factor < 1), or Sliding-Window LinUCB (sliding_window > 0)."""
        if not self._initialized:
            self._initialize(len(state))

        a = int(action)
        r = float(reward)
        self._arm_pulls[a] += 1

        # Optional Sliding-Window non-stationary LinUCB
        if self.sliding_window is not None and self.sliding_window > 0:
            if self.use_gpu:
                xg = state if isinstance(state, torch.Tensor) else torch.from_numpy(state.astype(np.float32)).to(self._device)
                buf = self._sw_buffers[a]
                buf.append((xg, r))
                if len(buf) > self.sliding_window:
                    buf.pop(0)
                    if (self._arm_pulls[a] % 25 == 0) or (len(buf) == self.sliding_window and self._arm_pulls[a] == self.sliding_window + 1):
                        Xs_g = torch.stack([item[0] for item in buf], dim=0)
                        rs_g = torch.tensor([item[1] for item in buf], device=self._device, dtype=torch.float32)
                        if getattr(self, "_eye_g", None) is None or self._eye_g.shape[0] != self.feature_dim:
                            self._eye_g = torch.eye(self.feature_dim, device=self._device, dtype=torch.float32)
                        A_mat = self._eye_g * float(self.regularization) + Xs_g.T @ Xs_g
                        self._A_inv_g[a] = torch.linalg.inv(A_mat + 1e-6 * self._eye_g)
                        self._b_g[a] = Xs_g.T @ rs_g
                        self._theta_g[a] = torch.mv(self._A_inv_g[a], self._b_g[a])
                        self.total_steps += 1
                        return
            else:
                x = np.asarray(state, dtype=np.float64)
                buf = self._sw_buffers[a]
                buf.append((x.copy(), r))
                if len(buf) > self.sliding_window:
                    buf.pop(0)
                    if (self._arm_pulls[a] % 25 == 0) or (len(buf) == self.sliding_window and self._arm_pulls[a] == self.sliding_window + 1):
                        Xs = np.stack([item[0] for item in buf], axis=0)
                        rs = np.array([item[1] for item in buf], dtype=np.float64)
                        self.A[a] = np.eye(self.feature_dim, dtype=np.float64) * self.regularization + Xs.T @ Xs
                        self.b[a] = Xs.T @ rs
                        self.A_inv[a] = np.linalg.inv(self.A[a])
                        self._A_inv_stack[a] = self.A_inv[a]
                        self._theta_stack[a] = self._A_inv_stack[a] @ self.b[a]
                        self.total_steps += 1
                        return

        # Optional Exponentially Discounted LinUCB (D-LinUCB)
        if self.discount_factor < 1.0:
            beta = self.discount_factor
            lam = self.regularization
            if self.use_gpu:
                if getattr(self, "_A_g", None) is None:
                    self._A_g = torch.stack([torch.from_numpy(m.astype(np.float32)).to(self._device) for m in self.A], dim=0)
                if getattr(self, "_eye_g", None) is None or self._eye_g.shape[0] != self.feature_dim:
                    self._eye_g = torch.eye(self.feature_dim, device=self._device, dtype=torch.float32)
                xg = state if isinstance(state, torch.Tensor) else torch.from_numpy(state.astype(np.float32)).to(self._device)
                self._A_g[a].mul_(beta).add_(self._eye_g, alpha=(1.0 - beta) * lam).addr_(xg, xg)
                self._b_g[a].mul_(beta).add_(xg, alpha=r)
                # Keep A_inv scaled and periodically refresh exact inverse every 30 updates per arm
                if self._arm_pulls[a] % 30 == 0:
                    self._refresh_inverse_gpu(a)
                else:
                    self._A_inv_g[a].div_(beta)
                    Ax_a = torch.mv(self._A_inv_g[a], xg)
                    denom = 1.0 + float(torch.dot(xg, Ax_a).item())
                    if abs(denom) > 1e-12:
                        self._A_inv_g[a].addr_(Ax_a, Ax_a, alpha=-1.0 / denom)
                    self._theta_g[a] = torch.mv(self._A_inv_g[a], self._b_g[a])
                self.total_steps += 1
                return
            x = state.astype(np.float64)
            self.A[a] = beta * self.A[a] + (1.0 - beta) * lam * np.eye(self.feature_dim) + np.outer(x, x)
            self.b[a] = beta * self.b[a] + r * x
            if self._arm_pulls[a] % 20 == 0:
                self.A_inv[a] = np.linalg.pinv(self.A[a])
            else:
                self.A_inv[a] /= beta
                A_inv_x = self.A_inv[a] @ x
                denom = 1.0 + float(x @ A_inv_x)
                if abs(denom) > 1e-12:
                    self.A_inv[a] -= np.outer(A_inv_x, A_inv_x) / denom
            self._A_inv_stack[a] = self.A_inv[a]
            self._theta_stack[a] = self._A_inv_stack[a] @ self.b[a]
            self.total_steps += 1
            return

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
