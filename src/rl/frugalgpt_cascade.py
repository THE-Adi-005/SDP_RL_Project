"""FrugalGPT Sequential Cascade Baseline.

Reference:
- Chen, L., Zaharia, M., & Zou, J. (2023). "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance."
  arXiv:2305.05176.
- Sequential cascade: starts at cheapest tier (Arm 0), uses a scoring judge to accept or escalate
  to subsequent tiers. Note: incurs latency and token overhead of all traversed intermediate models.
"""
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class FrugalGPTCascade:
    """Sequential escalation cascade baseline."""

    def __init__(self, n_actions: int = 8, acceptance_threshold: float = 0.70):
        """
        Args:
            n_actions: Number of arms ordered by increasing cost/complexity.
            acceptance_threshold: Confidence threshold to accept output without escalating.
        """
        self.n_actions = n_actions
        self.threshold = acceptance_threshold
        self.escalation_counts = 0
        self.total_steps = 0

    def get_name(self) -> str:
        return f"FrugalGPT(τ={self.threshold})"

    def select_action(self, state: np.ndarray, budget_fraction: Optional[float] = None) -> int:
        """In upfront routing evaluation, FrugalGPT initiates at arm 0 (cheapest)."""
        return 0

    def execute_cascade(
        self,
        query_outcomes: Dict[int, Dict[str, Any]]
    ) -> Tuple[int, bool, float, float, int]:
        """Simulate sequential escalation execution on a query.

        Args:
            query_outcomes: Dict mapping arm_index -> {
                'correct': bool, 'total_tokens': float, 'latency': float, 'confidence': float
            }

        Returns:
            (final_arm, is_correct, total_tokens_accumulated, latency_accumulated, n_escalations)
        """
        cum_tokens = 0.0
        cum_latency = 0.0
        n_escalations = 0

        final_arm = 0
        is_correct = False

        for arm in range(self.n_actions):
            final_arm = arm
            data = query_outcomes.get(arm, {"correct": False, "total_tokens": 100.0, "latency": 0.5, "confidence": 0.5})
            cum_tokens += data.get("total_tokens", 100.0)
            cum_latency += data.get("latency", 0.5)
            is_correct = bool(data.get("correct", False))
            conf = float(data.get("confidence", 0.9 if is_correct else 0.4))

            # If confidence exceeds threshold or we reached the most capable arm, stop
            if conf >= self.threshold or arm == self.n_actions - 1:
                break

            n_escalations += 1

        self.escalation_counts += n_escalations
        self.total_steps += 1

        return final_arm, is_correct, cum_tokens, cum_latency, n_escalations

    def update(self, state: np.ndarray, action: int, reward: float):
        """FrugalGPT uses static rule-based thresholds; no dynamic weight updates."""
        pass

