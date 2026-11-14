"""Neural Contextual Bandits (NeuralUCB, NeuralTS) and Bandits-with-Knapsacks (BwK) Primal-Dual Pacing.

References:
- Zhou, D., Li, L., & Gu, Q. (2020). "Neural Contextual Bandits with UCB-based Exploration." ICML.
- Zhang, W., Zhou, D., Li, L., & Gu, Q. (2021). "Neural Thompson Sampling." ICLR.
- Agrawal, S. & Devanur, N. R. (2016). "Linear Contextual Bandits with Knapsacks." NeurIPS.
"""
from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from src.rl.linucb import LinUCB


if _TORCH_OK:
    class _NeuralTrunk(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int = 128, n_actions: int = 17):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.head = nn.Linear(hidden_dim, n_actions)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            h = self.trunk(x)
            return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.encode(x))


class _BaseLastLayerNeuralBandit:
    """Shared neural trunk + per-arm ridge/precision last-layer Sherman-Morrison UCB/TS."""

    def __init__(
        self,
        n_actions: int = 17,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 128,
        alpha: float = 0.25,
        reg_lambda: float = 2.0,
        lr: float = 1e-3,
        train_every: int = 16,
        batch_size: int = 64,
        thompson: bool = False,
        use_gpu: bool = True,
        seed: int = 42,
    ):
        if not _TORCH_OK:
            raise ImportError("PyTorch is required for NeuralUCB/NeuralTS.")
        self.n_actions = int(n_actions)
        self.feature_dim = feature_dim
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.reg_lambda = float(reg_lambda)
        self.lr = float(lr)
        self.train_every = int(train_every)
        self.batch_size = int(batch_size)
        self.thompson = bool(thompson)
        self.use_gpu = bool(use_gpu and torch.cuda.is_available())
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        self._initialized = False
        self.model: Optional[_NeuralTrunk] = None
        self.opt = None
        self.lin_head: Optional[LinUCB] = None
        self._buf_x = []
        self._buf_a = []
        self._buf_r = []
        self.steps = 0

        if feature_dim is not None:
            self._initialize(feature_dim)

    def _initialize(self, dim: int):
        self.feature_dim = int(dim)
        self.model = _NeuralTrunk(dim, self.hidden_dim, self.n_actions).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.lin_head = LinUCB(
            n_actions=self.n_actions,
            alpha=self.alpha,
            feature_dim=self.hidden_dim,
            regularization=self.reg_lambda,
            use_gpu=self.use_gpu,
        )
        self._initialized = True

    def pretrain_warm_start(
        self,
        X_train: np.ndarray,
        R_train: np.ndarray,
        epochs: int = 12,
        warm_weight: float = 0.20,
    ):
        """Optional offline warm-start pretraining of neural trunk + last-layer ridge matrices."""
        if not self._initialized:
            self._initialize(X_train.shape[1])
        x_t = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        r_t = torch.from_numpy(R_train.astype(np.float32)).to(self.device)
        loss_fn = nn.MSELoss()
        bs = 512
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(x_t), device=self.device)
            for s in range(0, len(x_t), bs):
                idx = perm[s : s + bs]
                self.opt.zero_grad()
                pred = self.model(x_t[idx])
                loss = loss_fn(pred, r_t[idx])
                loss.backward()
                self.opt.step()
        self.model.eval()
        with torch.no_grad():
            phi = self.model.encode(x_t).cpu().numpy()
        self.lin_head.warm_start_from_full_matrix(
            phi, R_train, regularization=self.reg_lambda, sample_weight=warm_weight
        )

    @torch.no_grad()
    def _encode_one(self, state: np.ndarray) -> np.ndarray:
        if isinstance(state, torch.Tensor):
            xg = state.to(self.device).unsqueeze(0)
        else:
            xg = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device).unsqueeze(0)
        self.model.eval()
        return self.model.encode(xg).squeeze(0).cpu().numpy()

    def select_action(self, state: np.ndarray, budget_fraction: Optional[float] = None) -> int:
        if not self._initialized:
            dim = int(state.shape[0]) if hasattr(state, "shape") else len(state)
            self._initialize(dim)
        phi = self._encode_one(state)
        preds, uncs, ucb = self.lin_head.score_arms(phi, budget_fraction=budget_fraction)
        if self.thompson:
            sampled = preds + self.alpha * uncs * self.rng.randn(self.n_actions)
            return int(np.argmax(sampled))
        return int(np.argmax(ucb))

    def update(self, state: np.ndarray, action: int, reward: float):
        if not self._initialized:
            dim = int(state.shape[0]) if hasattr(state, "shape") else len(state)
            self._initialize(dim)
        x_np = np.asarray(state.cpu().numpy() if isinstance(state, torch.Tensor) else state, dtype=np.float32)
        self._buf_x.append(x_np)
        self._buf_a.append(int(action))
        self._buf_r.append(float(reward))
        if len(self._buf_x) > 4096:
            self._buf_x.pop(0)
            self._buf_a.pop(0)
            self._buf_r.pop(0)

        phi = self._encode_one(x_np)
        self.lin_head.update(phi, int(action), float(reward))
        self.steps += 1

        if len(self._buf_x) >= self.batch_size and (self.steps % self.train_every == 0):
            idx = self.rng.choice(len(self._buf_x), size=self.batch_size, replace=False)
            xb = torch.from_numpy(np.stack([self._buf_x[i] for i in idx])).to(self.device)
            ab = torch.tensor([self._buf_a[i] for i in idx], dtype=torch.long, device=self.device)
            rb = torch.tensor([self._buf_r[i] for i in idx], dtype=torch.float32, device=self.device)
            self.model.train()
            self.opt.zero_grad()
            q = self.model(xb).gather(1, ab.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(q, rb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.model.eval()


class NeuralUCB(_BaseLastLayerNeuralBandit):
    """NeuralUCB contextual bandit (Zhou et al., ICML 2020)."""

    def __init__(self, **kwargs):
        kwargs["thompson"] = False
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return f"NeuralUCB(alpha={self.alpha:.2f})"


class NeuralTS(_BaseLastLayerNeuralBandit):
    """Neural Thompson Sampling contextual bandit (Zhang et al., ICLR 2021)."""

    def __init__(self, **kwargs):
        kwargs["thompson"] = True
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return f"NeuralTS(v={self.alpha:.2f})"


class BwKPrimalDualLinUCB:
    """Bandits-with-Knapsacks (BwK) Primal-Dual contextual pacing baseline.

    Maintains separate Disjoint LinUCB accuracy estimators and an online Lagrangian
    cost dual multiplier mu_t updated via projected gradient ascent on per-step
    budget excess:
        mu_{t+1} = clip(mu_t + eta * (c_{t, a_t} - B_max / T_horizon), 0, mu_max).
    """

    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        total_budget_usd: float,
        horizon: int,
        arm_mean_cost_usd: np.ndarray,
        alpha: float = 0.10,
        regularization: float = 4.0,
        dual_lr: float = 50.0,
        mu_init: float = 0.0,
        mu_max: float = 5000.0,
        use_gpu: bool = True,
    ):
        self.acc_bandit = LinUCB(
            n_actions=n_actions,
            feature_dim=feature_dim,
            alpha=alpha,
            alpha_anneal=False,
            regularization=regularization,
            use_gpu=use_gpu,
        )
        self.total_budget_usd = float(total_budget_usd)
        self.horizon = max(1, int(horizon))
        self.target_per_step = self.total_budget_usd / self.horizon
        self.arm_mean_cost_usd = np.asarray(arm_mean_cost_usd, dtype=np.float64)
        self.dual_lr = float(dual_lr)
        self.mu = float(mu_init)
        self.mu_max = float(mu_max)
        self.spent_usd = 0.0
        self.cheapest_arm = int(np.argmin(self.arm_mean_cost_usd))

    def warm_start(self, X_train: np.ndarray, Y_train: np.ndarray, sample_weight: float = 0.20, precomputed_XtX=None):
        self.acc_bandit.warm_start_from_full_matrix(
            X_train, Y_train, regularization=self.acc_bandit.regularization,
            sample_weight=sample_weight, precomputed_XtX=precomputed_XtX,
        )

    def select_action(self, state: np.ndarray, enforce_hard_cap: bool = True) -> int:
        rem = self.total_budget_usd - self.spent_usd
        if enforce_hard_cap and rem <= self.arm_mean_cost_usd[self.cheapest_arm]:
            return self.cheapest_arm
        _, _, acc_ucb = self.acc_bandit.score_arms(state)
        lagrangian_score = acc_ucb - self.mu * self.arm_mean_cost_usd
        if enforce_hard_cap:
            feas = self.arm_mean_cost_usd <= rem
            if np.any(feas):
                lagrangian_score = np.where(feas, lagrangian_score, -1e12)
            else:
                return self.cheapest_arm
        return int(np.argmax(lagrangian_score))

    def update(self, state: np.ndarray, action: int, y_correct: float, observed_cost_usd: float):
        self.acc_bandit.update(state, int(action), float(y_correct))
        c = float(observed_cost_usd)
        self.spent_usd += c
        self.mu = float(np.clip(self.mu + self.dual_lr * (c - self.target_per_step), 0.0, self.mu_max))

    def get_name(self) -> str:
        return f"BwK-PrimalDual(mu={self.mu:.1f})"

