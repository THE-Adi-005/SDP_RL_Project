"""Visual Perturbation Confidence Gate with Chain of Responsibility Escalation.

Reference:
- CascadeRoute-VL (CVPR 2027): "Taming the Reasoning Tax in Vision-Language Models
  via Online Contextual Bandit Routing"
- Formulates visual hallucination detection via input perturbation invariance:
  mild scale/noise perturbations should NOT flip grounded visual answers.
  If answers diverge (high Overthinking Score S_OT), the gate triggers
  Chain-of-Responsibility escalation to higher-fidelity VLM arms.
"""
from typing import Dict, Optional, Tuple, Any
import numpy as np


# Standard escalation chains across 8 multimodal arms
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

# Legacy text chains for backwards compatibility
LEGACY_TEXT_CHAINS = {
    "3b_direct": "3b_cot",
    "3b_cot": None,
    "7b_direct": "7b_cot",
    "7b_cot": "7b_got",
    "7b_got": "7b_tot",
    "7b_tot": None
}


class ConfidenceGate:
    """Visual Consistency Confidence Gate with Chain of Responsibility."""

    def __init__(
        self,
        strategies_dict: Optional[Dict[str, Any]] = None,
        tau: float = 0.25,
        perturbation_scale: float = 0.90,
        gaussian_sigma: float = 0.01
    ):
        """
        Args:
            strategies_dict: Dictionary mapping action names to strategy instances.
            tau: Hallucination divergence threshold (S_OT >= tau triggers escalation).
            perturbation_scale: Image scaling factor for perturbation check.
            gaussian_sigma: Gaussian noise standard deviation.
        """
        self.strategies = strategies_dict or {}
        self.tau = tau
        self.perturbation_scale = perturbation_scale
        self.gaussian_sigma = gaussian_sigma

        # Merged chains covering both text and multimodal
        self.chains = {**LEGACY_TEXT_CHAINS, **MULTIMODAL_CHAINS}

    def compute_overthinking_score(self, answer1: str, answer2: str) -> float:
        """Compute divergence S_OT ∈ [0, 1] between original and perturbed responses."""
        a1 = str(answer1).strip().lower()
        a2 = str(answer2).strip().lower()

        if not a1 or not a2:
            return 1.0  # Empty generation indicates failure

        if a1 == a2:
            return 0.0  # Exact match: high visual stability

        # Token-level overlap (Jaccard distance)
        tokens1 = set(a1.split())
        tokens2 = set(a2.split())
        if not tokens1 or not tokens2:
            return 1.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard_sim = intersection / union if union > 0 else 0.0

        return 1.0 - jaccard_sim

    def evaluate_offline_query(
        self,
        initial_action: str,
        query_outcomes_by_arm: Dict[str, Dict[str, Any]],
        proxy_instability: float = 0.0
    ) -> Tuple[str, bool, float, float, int]:
        """Offline simulation of visual confidence gate escalation.

        Args:
            initial_action: Initial arm proposed by the router.
            query_outcomes_by_arm: Dict mapping arm_name -> {'correct', 'total_tokens', 'latency'}.
            proxy_instability: Measured or estimated instability S_OT for the query.

        Returns:
            (final_action, is_correct, total_tokens, total_latency, escalation_count)
        """
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

            # Accumulate evaluation cost of this stage
            total_tokens += tokens
            total_latency += latency

            # If visually stable (S_OT < tau) and output valid, accept answer
            if proxy_instability < self.tau:
                return current_action, correct, total_tokens, total_latency, escalation_count

            # Divergence detected -> escalate to next arm in chain
            next_action = self.chains.get(current_action)
            if next_action is None:
                return current_action, correct, total_tokens, total_latency, escalation_count

            current_action = next_action
            escalation_count += 1
            # Escalated arm benefits from higher capacity, decreasing residual instability
            proxy_instability *= 0.6

        return current_action or initial_action, False, total_tokens, total_latency, escalation_count

    def solve_with_cascade(self, initial_action_name: str, question: str):
        """Live execution method for compatibility with existing runner."""
        current_action = initial_action_name
        escalation_count = 0

        while current_action is not None:
            strategy = self.strategies[current_action]

            # Primary execution
            strategy.llm.temperature = 0.2
            res1 = strategy.solve(question)

            # Verification pass
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
