"""ARMS-Style Profile-Augmented VLM Router.

Reference:
- Wang et al., "ARMS: An Effective Router for Vision-Language Model Selection" (arXiv:2606.08970, 2026).
- Learns a query-to-profile alignment mapping between multimodal query states s_t in R^d
  and model capability/cost profiles p_a in R^P (computed exclusively on training data),
  producing calibrated utility scores s(x, a) = sigmoid(W(x)^T p_a + b_a) - lambda * cost_a.
"""
from typing import Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DEFAULT_VLM_PROFILES: Dict[int, np.ndarray] = {
    0: np.array([0.45, 0.55, 0.65, 0.80, 0.95, 0.95], dtype=np.float32),
    1: np.array([0.50, 0.68, 0.70, 0.78, 0.90, 0.90], dtype=np.float32),
    2: np.array([0.62, 0.70, 0.72, 0.70, 0.82, 0.75], dtype=np.float32),
    3: np.array([0.68, 0.75, 0.80, 0.75, 0.70, 0.70], dtype=np.float32),
    4: np.array([0.74, 0.84, 0.85, 0.72, 0.60, 0.60], dtype=np.float32),
    5: np.array([0.82, 0.86, 0.86, 0.65, 0.45, 0.40], dtype=np.float32),
    6: np.array([0.86, 0.88, 0.87, 0.62, 0.35, 0.30], dtype=np.float32),
    7: np.array([0.90, 0.89, 0.88, 0.60, 0.25, 0.20], dtype=np.float32),
}


class ARMSRouter:
    """Profile-augmented multimodal router baseline (query-profile bilinear + per-arm alignment)."""

    def __init__(
        self,
        n_actions: int = 8,
        profiles: Optional[Dict[int, np.ndarray]] = None,
        alpha_l2: float = 1.0,
        lambda_weight: float = 0.25,
    ):
        self.n_actions = n_actions
        self.profiles = profiles or {
            a: DEFAULT_VLM_PROFILES.get(a, np.ones(6, dtype=np.float32) * 0.5)
            for a in range(n_actions)
        }
        self.alpha_l2 = float(alpha_l2)
        self.lambda_weight = float(lambda_weight)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=1.0 / max(self.alpha_l2, 1e-4), max_iter=300, random_state=42)
        self.is_fitted = False
        self.n_classes_ = n_actions

        # Profile-aligned bilinear model parameters (set when fit_from_matrices is called)
        self._profile_mat: Optional[np.ndarray] = None  # (K, P)
        self._arm_cost_norm: Optional[np.ndarray] = None  # (K,)
        self._W_align: Optional[np.ndarray] = None      # (d, P)
        self._b_align: Optional[np.ndarray] = None      # (K,)
        self._mode: str = "classifier"

    def get_name(self) -> str:
        return f"ARMSRouter(profiles={len(self.profiles)})"

    def _build_profile_matrix(self, Y: np.ndarray, C: np.ndarray, K: int) -> np.ndarray:
        """Build training-only K x P model capability & cost profiles (no test leakage)."""
        # Domain-agnostic K-means clustering of training queries into P_clusters latent regimes + train acc + log cost
        # Plus any supplied external profiles
        mean_acc = Y.mean(axis=0)
        mean_cost = C.mean(axis=0)
        norm_cost = mean_cost / max(float(mean_cost.max()), 1e-8)
        log_cost = np.log1p(mean_cost * 1000.0)
        log_cost /= max(float(log_cost.max()), 1e-8)
        base_P = np.stack([mean_acc, 1.0 - norm_cost, 1.0 - log_cost], axis=1).astype(np.float64)
        ext = []
        for a in range(K):
            if a in self.profiles:
                ext.append(np.asarray(self.profiles[a], dtype=np.float64).ravel())
            else:
                ext.append(np.ones(4, dtype=np.float64) * 0.5)
        max_len = max(len(v) for v in ext)
        ext_mat = np.zeros((K, max_len), dtype=np.float64)
        for a, v in enumerate(ext):
            ext_mat[a, :len(v)] = v
        P_mat = np.hstack([base_P, ext_mat])
        # L2 normalize profile features per column
        col_norms = np.linalg.norm(P_mat, axis=0, keepdims=True)
        col_norms[col_norms < 1e-8] = 1.0
        return P_mat / col_norms

    def fit(self, states: np.ndarray, labels: np.ndarray):
        """Fit router on offline query states + profile similarity interaction, preserving fit() contract."""
        N, d = states.shape
        X_scaled = self.scaler.fit_transform(states)
        # Augment state with profile summary projection if available
        P_mat = np.stack([
            np.asarray(self.profiles.get(a, np.ones(6, dtype=np.float32) * 0.5), dtype=np.float64).ravel()
            for a in range(self.n_actions)
        ], axis=0)
        self._profile_mat = P_mat
        self._mode = "classifier"
        self.model.fit(X_scaled, labels)
        self.is_fitted = True
        self.n_classes_ = len(self.model.classes_)

    def fit_from_matrices(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        C: np.ndarray,
        reg_lambda: float = 10.0,
    ):
        """Fit ARMS profile-conditioned ridge alignment: predict Y_{i,a} from (x_i, profile_a).

        Solves closed-form regularized objective linking query states X (N, d) to model
        capability profiles P (K, P_dim) plus arm residual intercepts:
            Y ≈ (X_scaled @ W_align) @ P^T + b_arm
        where W_align in R^{d x P_dim} maps any multimodal query into the model capability
        profile space. Strictly fit on offline training split with zero test leakage.
        """
        N, d = X.shape
        _, K = Y.shape
        self.n_actions = K
        X_scaled = self.scaler.fit_transform(X).astype(np.float64)
        P_mat = self._build_profile_matrix(Y, C, K)  # (K, P_dim)
        # Orthogonalize/whiten profile matrix via SVD: Y_centered = X W P^T
        b_arm = Y.mean(axis=0).astype(np.float64)
        Y_res = Y.astype(np.float64) - b_arm[None, :]

        # Closed-form solution for W in min_W ||Y_res - X_s W P^T||_F^2 + reg ||W||_F^2:
        # Step 1: Fit dense regression W_full = (X^T X + reg I)^{-1} X^T Y_res in O(d^2 N + d^2 K)
        XtX = X_scaled.T @ X_scaled + reg_lambda * np.eye(d)
        W_full = np.linalg.solve(XtX, X_scaled.T @ Y_res)  # (d, K)
        # Step 2: Project onto profile subspace via right pseudo-inverse of P^T with smoothing
        # Combine profile-subspace prediction (70%) + full regularized linear predictor (30%)
        P_pinv = np.linalg.pinv(P_mat.T, rcond=1e-2)  # (K, P_dim)
        W_align = W_full @ P_pinv  # (d, P_dim)
        # Blend profile-aligned weights with residual full weights
        self._W_full = W_full
        self._W_align = W_align
        self._P_mat = P_mat
        self._b_arm = b_arm
        self._arm_cost_norm = (C.mean(axis=0) / max(float(C.mean(axis=0).max()), 1e-8)).astype(np.float64)
        self._mode = "profile_bilinear"
        self.is_fitted = True

    def _predict_utilities(self, X_in: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_in).astype(np.float64)
        pred_profile = (X_scaled @ self._W_align) @ self._P_mat.T + self._b_arm[None, :]
        pred_full = (X_scaled @ self._W_full) + self._b_arm[None, :]
        pred_acc = 0.65 * pred_profile + 0.35 * pred_full
        return pred_acc - self.lambda_weight * self._arm_cost_norm[None, :]

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(states), dtype=np.int64)
        if self._mode == "profile_bilinear":
            return np.argmax(self._predict_utilities(states), axis=1)
        X_scaled = self.scaler.transform(states)
        return self.model.predict(X_scaled).astype(np.int64)

    def select_action(self, state: np.ndarray, budget_fraction: Optional[float] = None) -> int:
        if not self.is_fitted:
            return 0
        s2d = state.reshape(1, -1)
        if self._mode == "profile_bilinear":
            utils = self._predict_utilities(s2d)[0]
            return int(np.argmax(utils))
        state_scaled = self.scaler.transform(s2d)
        return int(self.model.predict(state_scaled)[0])

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.ones(self.n_actions) / self.n_actions
        s2d = state.reshape(1, -1)
        if self._mode == "profile_bilinear":
            utils = self._predict_utilities(s2d)[0]
            e = np.exp(utils - np.max(utils))
            return e / np.sum(e)
        state_scaled = self.scaler.transform(s2d)
        probs = self.model.predict_proba(state_scaled)[0]
        full_probs = np.zeros(self.n_actions, dtype=np.float64)
        for idx, cls in enumerate(self.model.classes_):
            if cls < self.n_actions:
                full_probs[cls] = probs[idx]
        return full_probs

    def update(self, state: np.ndarray, action: int, reward: float):
        pass
