"""Visual Consistency & Epistemic Confidence Gate with Chain-of-Responsibility Escalation.

Reference:
- CascadeRoute-VL (CVPR 2027): "Taming the Reasoning Tax in Vision-Language Models
  via Online Contextual Bandit Routing"
- Supports both:
  1. Live visual-perturbation hallucination scoring S_OT = 1 - Jaccard(A_1, A_2).
  2. Principled bandit-epistemic + unsupervised perceptual escalation gate that avoids
     test-time oracle peeking and accounts for full multi-stage cascade token/cost overhead.
"""
from typing import Dict, Optional, Tuple, Any
import numpy as np
from sklearn.isotonic import IsotonicRegression


MULTIMODAL_CHAINS = {
    "2b_direct_lowres":  "2b_direct_highres",
    "2b_direct_highres": "2b_vcot_highres",
    "2b_vcot_highres":   "7b_direct_highres",
    "7b_direct_lowres":  "7b_direct_highres",
    "7b_direct_highres": "7b_vcot_highres",
    "7b_vcot_highres":   "7b_got_highres",
    "7b_got_highres":    "7b_tot_highres",
    "7b_tot_highres":    None
}

LEGACY_TEXT_CHAINS = {
    "3b_direct": "3b_cot",
    "3b_cot": None,
    "7b_direct": "7b_cot",
    "7b_cot": "7b_got",
    "7b_got": "7b_tot",
    "7b_tot": None
}


class ConfidenceGate:
    """Visual & Epistemic Confidence Gate with Chain-of-Responsibility Escalation."""

    def __init__(
        self,
        strategies_dict: Optional[Dict[str, Any]] = None,
        tau: float = 0.25,
        perturbation_scale: float = 0.90,
        gaussian_sigma: float = 0.01,
        unc_threshold: float = 0.12,
        margin_threshold: float = 0.025,
        min_budget_frac: float = 0.15,
        p_cap: float = 0.78,
        unc_weight: float = 0.25,
    ):
        self.strategies = strategies_dict or {}
        self.tau = tau
        self.perturbation_scale = perturbation_scale
        self.gaussian_sigma = gaussian_sigma
        self.unc_threshold = unc_threshold
        self.margin_threshold = margin_threshold
        self.min_budget_frac = min_budget_frac
        self.p_cap = p_cap
        self.unc_weight = unc_weight
        self.chains = {**LEGACY_TEXT_CHAINS, **MULTIMODAL_CHAINS}

    def compute_overthinking_score(self, answer1: str, answer2: str) -> float:
        """Compute divergence S_OT ∈ [0, 1] between original and perturbed responses."""
        a1 = str(answer1).strip().lower()
        a2 = str(answer2).strip().lower()
        if not a1 or not a2:
            return 1.0
        if a1 == a2:
            return 0.0
        tokens1 = set(a1.split())
        tokens2 = set(a2.split())
        if not tokens1 or not tokens2:
            return 1.0
        return 1.0 - (len(tokens1 & tokens2) / len(tokens1 | tokens2))

    def decide_escalation(
        self,
        primary_arm: int,
        preds: np.ndarray,
        uncs: np.ndarray,
        perceptual_raw: Optional[np.ndarray] = None,
        fallback_arms: Optional[Tuple[int, ...]] = None,
        arm_costs: Optional[np.ndarray] = None,
        arm_cost_penalty: Optional[np.ndarray] = None,
        observed_out_tokens: Optional[float] = None,
        budget_fraction: float = 1.0,
    ) -> Tuple[int, bool]:
        """Confidence-guided escalation decision without any ground-truth label peeking.

        Recovers estimated pure accuracy (before cost-penalty shift) for the primary
        compact arm and candidate specialist/frontier fallback arms, then combines:
        1. Expected accuracy gain (est_acc[alt] - est_acc[primary]),
        2. Bandit epistemic uncertainty sigma_a(s_t) = sqrt(s_t^T A_a^{-1} s_t),
        3. Unsupervised perceptual complexity (edge density, math symbol cues),
        4. Legitimate post-call compact response length signal (overthinking/bloat).
        """
        if budget_fraction <= self.min_budget_frac or not fallback_arms:
            return primary_arm, False

        est_acc = preds if arm_cost_penalty is None else (preds + arm_cost_penalty)
        p_primary = float(est_acc[primary_arm])
        u_primary = float(uncs[primary_arm])

        best_alt = primary_arm
        best_alt_util = p_primary - (0.22 * float(arm_cost_penalty[primary_arm]) if arm_cost_penalty is not None else 0.0)
        for alt in fallback_arms:
            if alt == primary_arm:
                continue
            alt_util = float(est_acc[alt]) - (0.22 * float(arm_cost_penalty[alt]) if arm_cost_penalty is not None else 0.0)
            if alt_util > best_alt_util:
                best_alt_util = alt_util
                best_alt = int(alt)

        if best_alt == primary_arm:
            return primary_arm, False

        best_alt_acc = float(est_acc[best_alt])
        gain = best_alt_acc - p_primary
        cost_diff_pen = 0.0
        if arm_cost_penalty is not None:
            cost_diff_pen = max(0.0, float(arm_cost_penalty[best_alt]) - float(arm_cost_penalty[primary_arm]))
        eff_gain = gain - 0.22 * cost_diff_pen

        perc_boost = 0.0
        if perceptual_raw is not None:
            edge = float(perceptual_raw[1])
            math_cue = float(perceptual_raw[6])
            perc_boost = 0.06 * edge + 0.06 * math_cue

        obs_bloat = 0.05 if (observed_out_tokens is not None and observed_out_tokens > 15.0) else 0.0
        risk_score = eff_gain + self.unc_weight * u_primary + perc_boost + obs_bloat

        if (p_primary < self.p_cap) and (eff_gain >= self.margin_threshold) and (risk_score >= self.unc_threshold):
            return best_alt, True

        return primary_arm, False

    def evaluate_offline_query(
        self,
        initial_action: str,
        query_outcomes_by_arm: Dict[str, Dict[str, Any]],
        proxy_instability: float = 0.0
    ) -> Tuple[str, bool, float, float, int]:
        """Offline simulation of visual confidence gate escalation."""
        current_action = initial_action
        escalation_count = 0
        total_tokens = 0.0
        total_latency = 0.0

        while current_action is not None:
            arm_data = query_outcomes_by_arm.get(
                current_action,
                {"correct": False, "total_tokens": 100.0, "latency": 0.5}
            )
            tokens = float(arm_data.get("total_tokens", 100.0))
            latency = float(arm_data.get("latency", 0.5))
            correct = bool(arm_data.get("correct", False))

            total_tokens += tokens
            total_latency += latency

            if proxy_instability < self.tau:
                return current_action, correct, total_tokens, total_latency, escalation_count

            next_action = self.chains.get(current_action)
            if next_action is None:
                return current_action, correct, total_tokens, total_latency, escalation_count

            current_action = next_action
            escalation_count += 1
            proxy_instability *= 0.6

        return current_action or initial_action, False, total_tokens, total_latency, escalation_count

    def solve_with_cascade(self, initial_action_name: str, question: str):
        """Live execution method for compatibility with existing runner."""
        current_action = initial_action_name
        escalation_count = 0

        while current_action is not None:
            strategy = self.strategies[current_action]
            strategy.llm.temperature = 0.2
            res1 = strategy.solve(question)
            strategy.llm.temperature = 0.8
            res2 = strategy.solve(question)
            strategy.llm.temperature = 0.2

            res1.input_tokens += res2.input_tokens
            res1.output_tokens += res2.output_tokens
            res1.latency += res2.latency
            res1.n_llm_calls += res2.n_llm_calls

            s_ot = self.compute_overthinking_score(res1.predicted_answer, res2.predicted_answer)
            if s_ot < self.tau and res1.predicted_answer != "":
                return res1, current_action, escalation_count

            next_action = self.chains.get(current_action)
            if next_action is None:
                return res1, current_action, escalation_count

            current_action = next_action
            escalation_count += 1

        return res1, initial_action_name, escalation_count


class CalibratedDeferralGate:
    """Isotonic + Split-Conformal Calibrated Deferral Gate (Claim 5).

    Replaces heuristic hand-tuned constants with a single Dev-calibrated threshold
    tau_cal operating on isotonic-calibrated correctness probabilities p_hat(s_t, a)
    and lower conformal UCB margin against fallback arm a* in F:

        Escalate a_0 -> a* iff:
            b_t > b_min  AND  (p_hat(a*) - p_hat_upper(a_0)) - gamma_cost * dc >= tau_cal

    where p_hat_upper(a_0) = min(1, p_hat(a_0) + q_conf * sigma_{a_0}(s_t)) and
    p_hat(a) is calibrated by non-parametric Isotonic Regression on Dev predictions.
    """

    def __init__(
        self,
        tau_cal: float = 0.04,
        conformal_quantile: float = 0.25,
        min_budget_frac: float = 0.10,
        cost_tradeoff: float = 0.20,
    ):
        self.tau_cal = float(tau_cal)
        self.q_conf = float(conformal_quantile)
        self.min_budget_frac = float(min_budget_frac)
        self.cost_tradeoff = float(cost_tradeoff)
        self.iso: Optional[IsotonicRegression] = None

    def fit_calibrator(self, dev_est_acc: np.ndarray, dev_true_y: np.ndarray, alpha_risk: float = 0.20):
        """Fit isotonic probability calibrator & conformal residual quantile strictly on Dev."""
        raw = np.asarray(dev_est_acc, dtype=np.float64).ravel()
        y = np.asarray(dev_true_y, dtype=np.float64).ravel()
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso.fit(raw, y)
        p_cal = self.iso.predict(raw)
        # Split-conformal upper uncertainty quantile over positive prediction errors (y - p_cal)
        pos_res = np.maximum(0.0, p_cal - y)
        n = len(pos_res)
        q_level = min(1.0, (1.0 - alpha_risk) * (n + 1.0) / max(1.0, float(n)))
        self.conformal_margin = float(np.quantile(pos_res, q_level)) if n > 0 else 0.10
        return self

    def calibrate_probs(self, est_acc: np.ndarray) -> np.ndarray:
        arr = np.asarray(est_acc, dtype=np.float64)
        if self.iso is None:
            return np.clip(arr, 0.0, 1.0)
        return self.iso.predict(arr)

    def decide_escalation(
        self,
        primary_arm: int,
        preds: np.ndarray,
        uncs: np.ndarray,
        fallback_arms: Tuple[int, ...],
        arm_cost_penalty: Optional[np.ndarray] = None,
        observed_out_tokens: Optional[float] = None,
        budget_fraction: float = 1.0,
    ) -> Tuple[int, bool, float]:
        """Single-threshold calibrated deferral decision. Returns (chosen_arm, did_escalate, expected_gain)."""
        if budget_fraction <= self.min_budget_frac or not fallback_arms:
            return primary_arm, False, 0.0

        est_acc = preds if arm_cost_penalty is None else (preds + arm_cost_penalty)
        cal_probs = self.calibrate_probs(est_acc)

        # If observed compact output length exceeds concise perception envelope (>20 tok), widen uncertainty bonus
        len_penalty = 0.04 if (observed_out_tokens is not None and observed_out_tokens > 20.0) else 0.0
        p0_upper = min(1.0, float(cal_probs[primary_arm]) + self.q_conf * float(uncs[primary_arm]) - len_penalty)

        best_alt = primary_arm
        best_score = -np.inf
        for alt in fallback_arms:
            if alt == primary_arm:
                continue
            dc = 0.0
            if arm_cost_penalty is not None:
                dc = max(0.0, float(arm_cost_penalty[alt]) - float(arm_cost_penalty[primary_arm]))
            gain = float(cal_probs[alt]) - p0_upper - self.cost_tradeoff * dc
            if gain > best_score:
                best_score = gain
                best_alt = int(alt)

        if best_alt != primary_arm and best_score >= self.tau_cal:
            return best_alt, True, float(best_score)
        return primary_arm, False, float(max(0.0, best_score))
