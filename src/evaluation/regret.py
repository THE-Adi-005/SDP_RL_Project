"""Regret analysis for contextual bandit evaluation.

Computes:
- Oracle policy: picks the strategy that gives the highest reward per problem
  (computed post-hoc from baseline results)
- Cumulative regret: Σ(oracle_reward - agent_reward) over time
- Average regret: cumulative / t
- Normalized regret per dataset

This is a key metric for bandit papers — shows how quickly
the agent converges to the oracle's performance.
"""
import numpy as np
import pandas as pd


def compute_oracle_rewards(baseline_results: pd.DataFrame) -> dict:
    """Compute the oracle reward per problem — the best strategy's reward for each problem.

    The oracle has perfect hindsight: for each problem, it knows which strategy
    would have yielded the highest reward, and always picks that one.

    Args:
        baseline_results: DataFrame with columns [problem_id, strategy, reward].
            Must contain results for ALL strategies on ALL problems.

    Returns:
        Dict mapping problem_id → oracle_reward (the max reward across strategies).
    """
    oracle = baseline_results.groupby("problem_id")["reward"].max()
    return oracle.to_dict()


def compute_cumulative_regret(agent_rewards: list, oracle_rewards_per_step: list) -> np.ndarray:
    """Compute cumulative regret over time.

    Regret_t = oracle_reward_t - agent_reward_t
    Cumulative_regret_T = Σ_{t=1}^{T} Regret_t

    Args:
        agent_rewards: List of rewards the agent received at each step.
        oracle_rewards_per_step: List of oracle rewards at each step (same order).

    Returns:
        np.ndarray of cumulative regret at each time step.
    """
    agent = np.array(agent_rewards, dtype=float)
    oracle = np.array(oracle_rewards_per_step, dtype=float)
    per_step_regret = oracle - agent
    return np.cumsum(per_step_regret)


def compute_average_regret(cumulative_regret: np.ndarray) -> np.ndarray:
    """Average regret at each time step: cumulative_regret_t / t."""
    t = np.arange(1, len(cumulative_regret) + 1)
    return cumulative_regret / t


def analyze_regret(agent_results: pd.DataFrame, oracle_map: dict) -> dict:
    """Full regret analysis for one agent's run.

    Args:
        agent_results: DataFrame with columns [problem_id, reward] in order.
        oracle_map: Dict from compute_oracle_rewards().

    Returns:
        Dict with regret metrics and per-step arrays.
    """
    agent_rewards = agent_results["reward"].values
    oracle_rewards = np.array([
        oracle_map.get(pid, 0.0) for pid in agent_results["problem_id"]
    ])

    cum_regret = compute_cumulative_regret(agent_rewards, oracle_rewards)
    avg_regret = compute_average_regret(cum_regret)

    return {
        "total_regret": float(cum_regret[-1]) if len(cum_regret) > 0 else 0.0,
        "final_avg_regret": float(avg_regret[-1]) if len(avg_regret) > 0 else 0.0,
        "cumulative_regret": cum_regret,
        "average_regret": avg_regret,
        "per_step_regret": oracle_rewards - agent_rewards,
        "oracle_rewards": oracle_rewards,
        "agent_rewards": agent_rewards,
    }


def compute_baseline_rewards(baseline_results: pd.DataFrame,
                              reward_cfg: dict) -> pd.DataFrame:
    """Add reward column to baseline results if missing.

    Uses the same reward function as the RL environment:
    R = α * correct - β * norm_tokens - γ * norm_latency

    Args:
        baseline_results: DataFrame with [correct, total_tokens, latency, strategy, problem_id].
        reward_cfg: Dict with keys alpha, beta, gamma.

    Returns:
        DataFrame with 'reward' column added.
    """
    df = baseline_results.copy()
    alpha = reward_cfg.get("alpha", 1.0)
    beta = reward_cfg.get("beta", 0.3)
    gamma = reward_cfg.get("gamma", 0.1)

    max_tokens = df["total_tokens"].max() if df["total_tokens"].max() > 0 else 1
    max_latency = df["latency"].max() if df["latency"].max() > 0 else 1

    df["norm_tokens"] = df["total_tokens"] / max_tokens
    df["norm_latency"] = df["latency"] / max_latency
    df["reward"] = (alpha * df["correct"].astype(float)
                    - beta * df["norm_tokens"]
                    - gamma * df["norm_latency"])
    return df
