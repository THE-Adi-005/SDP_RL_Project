"""Statistical significance tests for research paper evaluation.

Provides paired comparisons between agents/strategies with:
- Paired t-test (parametric)
- Wilcoxon signed-rank test (non-parametric)
- Bonferroni correction for multiple comparisons
- Cohen's d effect size
"""
import numpy as np
from scipy import stats
from itertools import combinations


def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Paired two-sided t-test between two sets of per-problem scores.

    Args:
        scores_a: Per-problem metric for method A (e.g., correctness 0/1).
        scores_b: Per-problem metric for method B.

    Returns:
        dict with t-statistic, p-value, and significance at α=0.05.
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return {
        "test": "paired_ttest",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    More robust when normality assumption is violated (common with binary
    correctness data).
    """
    # Handle case where all differences are zero
    diff = scores_a - scores_b
    if np.all(diff == 0):
        return {
            "test": "wilcoxon",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_005": False,
            "significant_001": False,
        }

    try:
        stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    except ValueError:
        # Too few non-zero differences
        return {
            "test": "wilcoxon",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_005": False,
            "significant_001": False,
        }

    return {
        "test": "wilcoxon",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Cohen's d effect size for paired samples.

    Interpretation:
        |d| < 0.2  → negligible
        0.2 ≤ |d| < 0.5 → small
        0.5 ≤ |d| < 0.8 → medium
        |d| ≥ 0.8 → large
    """
    diff = scores_a - scores_b
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    return float(d)


def effect_size_label(d: float) -> str:
    """Human-readable label for Cohen's d."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_correction(p_values: list, alpha: float = 0.05) -> list:
    """Apply Bonferroni correction for multiple comparisons.

    Adjusted p-value = min(p * n_comparisons, 1.0)
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def pairwise_comparison(results_dict: dict, metric: str = "correct") -> list:
    """Run all pairwise statistical tests between agents.

    Args:
        results_dict: {agent_name: DataFrame with per-problem results}
        metric: Column name to compare (e.g., "correct", "reward").

    Returns:
        List of dicts with pairwise test results.
    """
    agent_names = list(results_dict.keys())
    comparisons = []

    for name_a, name_b in combinations(agent_names, 2):
        scores_a = results_dict[name_a][metric].values.astype(float)
        scores_b = results_dict[name_b][metric].values.astype(float)

        # Ensure same length (align by problem_id if needed)
        min_len = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:min_len]
        scores_b = scores_b[:min_len]

        ttest = paired_ttest(scores_a, scores_b)
        wilcox = wilcoxon_test(scores_a, scores_b)
        d = cohens_d(scores_a, scores_b)

        comparisons.append({
            "agent_a": name_a,
            "agent_b": name_b,
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
            "ttest_p": ttest["p_value"],
            "wilcoxon_p": wilcox["p_value"],
            "cohens_d": d,
            "effect_size": effect_size_label(d),
            "ttest_significant": ttest["significant_005"],
            "wilcoxon_significant": wilcox["significant_005"],
        })

    # Apply Bonferroni correction
    p_values_t = [c["ttest_p"] for c in comparisons]
    p_values_w = [c["wilcoxon_p"] for c in comparisons]
    corrected_t = bonferroni_correction(p_values_t)
    corrected_w = bonferroni_correction(p_values_w)

    for i, comp in enumerate(comparisons):
        comp["ttest_p_corrected"] = corrected_t[i]
        comp["wilcoxon_p_corrected"] = corrected_w[i]
        comp["ttest_significant_corrected"] = corrected_t[i] < 0.05
        comp["wilcoxon_significant_corrected"] = corrected_w[i] < 0.05

    return comparisons


def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Paired non-parametric bootstrap over queries for mean difference (A - B).

    Reports mean difference, empirical 95% CI, and bounded two-sided bootstrap p-value
    (lower-bounded at 1/(n_boot+1), avoiding misleading asymptotic 1e-275 values).
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    diff = a - b
    n = len(diff)
    obs_diff = float(np.mean(diff))

    rng = np.random.RandomState(seed)
    # Vectorized chunked bootstrap for speed
    chunk = 1000
    boot_means = np.empty(n_boot, dtype=np.float64)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        idx = rng.randint(0, n, size=(m, n))
        boot_means[s : s + m] = diff[idx].mean(axis=1)

    alpha = 1.0 - ci
    ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    # Centered bootstrap null p-value + sign-proportion check
    centered = boot_means - obs_diff
    p_two = float((np.sum(np.abs(centered) >= abs(obs_diff)) + 1.0) / (n_boot + 1.0))

    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": obs_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": max(p_two, 1.0 / (n_boot + 1.0)),
        "significant_005": (ci_low > 0.0) or (ci_high < 0.0),
    }


def pareto_frontier_auc(
    costs: np.ndarray,
    accuracies: np.ndarray,
    cost_min: float = None,
    cost_max: float = None,
    left_fill: float = None,
) -> float:
    """Compute normalized Area Under the Monotone Pareto Frontier (AURC / AUC-Acc)
    over a matched [cost_min, cost_max] domain.

    Builds the upper non-dominated accuracy envelope as cost increases (monotone
    non-decreasing step/piecewise-linear frontier) and divides integrated accuracy
    by (cost_max - cost_min) so the resulting AUC sits directly on the [0, 100]%
    accuracy scale for easy interpretation.
    """
    c = np.asarray(costs, dtype=np.float64)
    a = np.asarray(accuracies, dtype=np.float64)
    order = np.argsort(c)
    c_sorted = c[order]
    a_sorted = a[order]

    # Upper non-dominated envelope: monotone non-decreasing accuracy with cost
    env_c = []
    env_a = []
    best_a = -np.inf
    for ci_val, ai_val in zip(c_sorted, a_sorted):
        if ai_val >= best_a:
            best_a = ai_val
            if env_c and np.isclose(env_c[-1], ci_val):
                env_a[-1] = max(env_a[-1], best_a)
            else:
                env_c.append(ci_val)
                env_a.append(best_a)

    c_lo = float(c_sorted[0]) if cost_min is None else float(cost_min)
    c_hi = float(c_sorted[-1]) if cost_max is None else float(cost_max)
    if c_hi <= c_lo:
        return float(np.max(a_sorted))

    # Clamp / extend piecewise-linear upper hull over [c_lo, c_hi]
    xs = np.linspace(c_lo, c_hi, 500)
    # Right-continuous / linear interpolation (optionally 0.0 below lowest feasible cost)
    l_val = env_a[0] if left_fill is None else float(left_fill)
    ys = np.interp(xs, env_c, env_a, left=l_val, right=env_a[-1])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area = float(_trapz(ys, xs))
    return area / (c_hi - c_lo)

