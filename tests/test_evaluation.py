"""Tests for statistical significance and regret analysis modules."""
import numpy as np
import pandas as pd
import pytest
from src.evaluation.statistical_tests import (
    paired_ttest, wilcoxon_test, cohens_d, effect_size_label,
    bonferroni_correction, pairwise_comparison,
    paired_bootstrap_ci, pareto_frontier_auc,
)
from src.evaluation.regret import (
    compute_cumulative_regret, compute_average_regret,
    compute_oracle_rewards,
)


# --- Statistical Tests ---

def test_paired_ttest_identical():
    """Identical scores should give high p (no difference)."""
    scores = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    result = paired_ttest(scores, scores)
    # t-test on identical arrays may return NaN — either way, not significant
    assert result["significant_005"] == False


def test_paired_ttest_different():
    """Clearly different scores should give low p-value."""
    a = np.array([1.0] * 50)
    b = np.array([0.0] * 50)
    result = paired_ttest(a, b)
    assert result["p_value"] < 0.01
    assert result["significant_001"] == True


def test_wilcoxon_identical():
    scores = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    result = wilcoxon_test(scores, scores)
    assert result["p_value"] == 1.0


def test_cohens_d_large_effect():
    a = np.array([1.0] * 50)
    b = np.array([0.0] * 50)
    d = cohens_d(a, b)
    assert abs(d) > 0.8, f"Should be a large effect size, got {d}"
    assert effect_size_label(d) == "large"


def test_cohens_d_negligible():
    np.random.seed(42)
    a = np.random.randn(100) + 10
    b = a + np.random.randn(100) * 0.01  # tiny difference
    d = cohens_d(a, b)
    assert abs(d) < 0.2


def test_bonferroni():
    p_values = [0.01, 0.03, 0.06]
    corrected = bonferroni_correction(p_values, alpha=0.05)
    assert corrected[0] == 0.03  # 0.01 * 3
    assert corrected[1] == 0.09  # 0.03 * 3
    assert corrected[2] == 0.18  # 0.06 * 3


def test_bonferroni_caps_at_1():
    p_values = [0.5, 0.8]
    corrected = bonferroni_correction(p_values)
    assert corrected[0] == 1.0
    assert corrected[1] == 1.0


def test_pairwise_comparison():
    """Pairwise comparison between 3 agents should produce 3 comparisons."""
    np.random.seed(42)
    results_dict = {
        "agent_a": pd.DataFrame({"correct": np.random.binomial(1, 0.8, 50)}),
        "agent_b": pd.DataFrame({"correct": np.random.binomial(1, 0.3, 50)}),
        "agent_c": pd.DataFrame({"correct": np.random.binomial(1, 0.5, 50)}),
    }
    comparisons = pairwise_comparison(results_dict, metric="correct")
    assert len(comparisons) == 3  # C(3,2) = 3
    for comp in comparisons:
        assert "ttest_p" in comp
        assert "cohens_d" in comp
        assert "ttest_p_corrected" in comp


# --- Regret Analysis ---

def test_cumulative_regret_perfect_agent():
    """Perfect agent (matches oracle) should have zero regret."""
    agent = [1.0, 0.8, 0.5, 1.0]
    oracle = [1.0, 0.8, 0.5, 1.0]
    regret = compute_cumulative_regret(agent, oracle)
    assert np.allclose(regret, 0.0)


def test_cumulative_regret_grows():
    """Suboptimal agent should accumulate positive regret."""
    agent =  [0.0, 0.0, 0.0, 0.0]
    oracle = [1.0, 1.0, 1.0, 1.0]
    regret = compute_cumulative_regret(agent, oracle)
    assert regret[-1] == 4.0
    assert np.all(np.diff(regret) >= 0), "Cumulative regret should be non-decreasing"


def test_average_regret_converges():
    """Average regret for a consistently suboptimal agent."""
    cum_regret = np.array([1.0, 2.0, 3.0, 4.0])
    avg = compute_average_regret(cum_regret)
    assert np.allclose(avg, [1.0, 1.0, 1.0, 1.0])


def test_oracle_rewards():
    """Oracle should pick the max reward per problem."""
    df = pd.DataFrame({
        "problem_id": ["p1", "p1", "p1", "p2", "p2", "p2"],
        "strategy": ["direct", "cot", "tot", "direct", "cot", "tot"],
        "reward": [0.5, 0.9, 0.3, 0.1, 0.2, 0.8],
    })
    oracle = compute_oracle_rewards(df)
    assert oracle["p1"] == 0.9
    assert oracle["p2"] == 0.8


def test_paired_bootstrap_ci_and_pareto_auc():
    rng = np.random.RandomState(42)
    a = rng.binomial(1, 0.85, size=500).astype(float)
    b = rng.binomial(1, 0.65, size=500).astype(float)
    res = paired_bootstrap_ci(a, b, n_boot=500, seed=42)
    assert res["ci_low"] > 0.0 and res["mean_diff"] > 0.10
    assert res["p_value"] >= 1.0 / 501.0

    # Higher frontier strictly gives higher Pareto AUC
    c = np.array([0.05, 0.15, 0.30])
    acc_high = np.array([75.0, 80.0, 82.0])
    acc_low = np.array([70.0, 74.0, 76.0])
    assert pareto_frontier_auc(c, acc_high, 0.05, 0.30) > pareto_frontier_auc(c, acc_low, 0.05, 0.30)
