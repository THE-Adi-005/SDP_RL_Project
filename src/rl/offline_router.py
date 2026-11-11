"""Offline Cost-Aware Router Baseline (Per-Arm Expected Utility Regression).

Reference:
- Huang et al., "VL-RouterBench: A Benchmark for Vision-Language Model Routing" (arXiv:2512.23562, 2025).
- Predicts per-arm correctness probability y_hat_a(x) via L2-regularized ridge/logistic regression
  trained on offline logs, then selects argmax_a (y_hat_a(x) - lambda * normalized_cost_a).
"""
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler


class OfflineCostAwareRouter:
    """Supervised offline per-arm utility router parameterized by cost trade-off lambda."""

    def __init__(self, n_actions: int = 8, lambda_weight: float = 0.5, reg_lambda: float = 25.0):
        self.n_actions = n_actions
        self.lambda_weight = float(lambda_weight)
        self.reg_lambda = float(reg_lambda)
        self.scaler = StandardScaler()
        self.W: Optional[np.ndarray] = None  # (d, K)
        self.b: Optional[np.ndarray] = None  # (K,)
        self.mean_norm_cost: Optional[np.ndarray] = None
        self.is_fitted = False

    def get_name(self) -> str:
        return f"OfflineRouter(λ={self.lambda_weight})"

    def fit_from_matrices(self, X: np.ndarray, Y: np.ndarray, C: np.ndarray):
        """Fit per-arm linear quality predictors W, b from full offline interaction matrices.

        Avoids multiclass argmax tie-collapse by learning an independent regression head per VLM arm:
            y_hat_a(x) = w_a^T x + b_a ≈ E[Y_a | x]
            action(x) = argmax_a [ clip(y_hat_a(x), 0, 1) - lambda * mean_norm_cost_a ]
        """
        N, d = X.shape
        _, K = Y.shape
        self.n_actions = K

        X_scaled = self.scaler.fit_transform(X).astype(np.float64)
        self.b = Y.mean(axis=0).astype(np.float64)
        Y_centered = Y.astype(np.float64) - self.b[None, :]

        # Closed-form multi-output ridge regression (identical matrix solve for all K arms simultaneously)
        XtX = X_scaled.T @ X_scaled + self.reg_lambda * np.eye(d, dtype=np.float64)
        self.W = np.linalg.solve(XtX, X_scaled.T @ Y_centered)  # (d, K)

        max_cost = float(np.max(C)) if np.max(C) > 0 else 1.0
        self.mean_norm_cost = (C.mean(axis=0).astype(np.float64)) / max_cost
        self.is_fitted = True

    def predict_utilities_batch(self, states: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(states).astype(np.float64)
        y_hat = np.clip((X_scaled @ self.W) + self.b[None, :], 0.0, 1.0)
        return y_hat - self.lambda_weight * self.mean_norm_cost[None, :]

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(states), dtype=np.int64)
        return np.argmax(self.predict_utilities_batch(states), axis=1)

    def select_action(self, state: np.ndarray, budget_fraction: Optional[float] = None) -> int:
        if not self.is_fitted:
            return 0
        utils = self.predict_utilities_batch(state.reshape(1, -1))[0]
        return int(np.argmax(utils))

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.ones(self.n_actions) / self.n_actions
        utils = self.predict_utilities_batch(state.reshape(1, -1))[0]
        e = np.exp(utils - np.max(utils))
        return e / np.sum(e)

    def update(self, state: np.ndarray, action: int, reward: float):
        """Offline router has frozen weights during test-time deployment."""
        pass
