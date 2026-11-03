"""Offline Cost-Aware Soft-Target Router Baseline.

Reference:
- "VL-RouterBench: A Benchmark for Vision-Language Model Routing" (2026).
- Maps multimodal query features to target VLM arms using offline supervised learning
  with cost-performance trade-off parameter λ.
- Used to rigorously benchmark static offline routers against online LinUCB.
"""
from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class OfflineCostAwareRouter:
    """Supervised offline router parameterized by trade-off lambda."""

    def __init__(self, n_actions: int = 8, lambda_weight: float = 0.5, C_reg: float = 1.0):
        """
        Args:
            n_actions: Number of candidate VLM routing arms.
            lambda_weight: Trade-off parameter.
                           0.0 = pure quality optimization.
                           1.0 = heavy cost penalization.
                           0.5 = balanced Pareto trade-off.
            C_reg: Regularization strength for Logistic Regression.
        """
        self.n_actions = n_actions
        self.lambda_weight = float(lambda_weight)
        self.C_reg = float(C_reg)
        self.clf = LogisticRegression(
            C=self.C_reg,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def get_name(self) -> str:
        return f"OfflineRouter(λ={self.lambda_weight})"

    def fit_from_matrices(self, X: np.ndarray, Y: np.ndarray, C: np.ndarray):
        """Derive soft labels and fit supervised classifier.

        Args:
            X: Multimodal feature vectors (N, d).
            Y: Correctness matrix (N, K) with values in {0, 1}.
            C: Cost/token matrix (N, K).
        """
        N, K = Y.shape
        # Normalize costs per sample or globally
        max_cost = np.max(C) if np.max(C) > 0 else 1.0
        normalized_C = C / max_cost

        # Utility score for each arm
        # score = Y - lambda * norm_cost
        utility = Y.astype(np.float64) - self.lambda_weight * normalized_C

        # Best arm for each sample
        labels = np.argmax(utility, axis=1)

        # Scale features and train classifier
        X_scaled = self.scaler.fit_transform(X)

        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            self.single_class_fallback_ = int(unique_classes[0])
            self.is_fitted = True
            return

        self.single_class_fallback_ = None
        self.clf.fit(X_scaled, labels)
        self.is_fitted = True

    def select_action(self, state: np.ndarray, budget_fraction: Optional[float] = None) -> int:
        """Predict optimal arm statically."""
        if not self.is_fitted:
            return 0
        if getattr(self, "single_class_fallback_", None) is not None:
            return self.single_class_fallback_
        state_2d = state.reshape(1, -1)
        state_scaled = self.scaler.transform(state_2d)
        return int(self.clf.predict(state_scaled)[0])

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        """Return probability distribution over arms."""
        if not self.is_fitted:
            return np.ones(self.n_actions) / self.n_actions
        if getattr(self, "single_class_fallback_", None) is not None:
            p = np.zeros(self.n_actions, dtype=np.float64)
            p[self.single_class_fallback_] = 1.0
            return p
        state_2d = state.reshape(1, -1)
        state_scaled = self.scaler.transform(state_2d)
        probs = self.clf.predict_proba(state_scaled)[0]

        full_probs = np.zeros(self.n_actions, dtype=np.float64)
        for idx, cls in enumerate(self.clf.classes_):
            if cls < self.n_actions:
                full_probs[cls] = probs[idx]
        return full_probs

    def update(self, state: np.ndarray, action: int, reward: float):
        """Offline router has frozen weights during test-time deployment."""
        pass
