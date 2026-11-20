"""Unified experiment runner — runs all RL agents and baselines across datasets.

[SDP: FACADE PATTERN]
This script acts as a Facade. It provides a simplified, unified interface 
(a single CLI command) to the complex subsystems of the project 
(LLM initialization, RL Agent loading, Dataset processing, Feature extraction).
The user doesn't need to manually wire the Caching layer to the Strategy layer 
and then to the RL environment; the Facade handles this complex orchestration.

Usage:
    python run_experiment.py --datasets gsm8k_sample,logiqa_sample --agents all --limit 50
    python run_experiment.py --datasets gsm8k_sample --agents linucb,epsilon_greedy --limit 20
    python run_experiment.py --datasets gsm8k_sample --agents all --seeds 42,123,456 --limit 50
"""
import argparse
import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

from src.llm.client import LLMClient
from src.strategies.direct import DirectStrategy
from src.strategies.cot import CoTStrategy
from src.strategies.got import GoTStrategy
from src.strategies.tot import ToTStrategy
from src.evaluation.evaluator import evaluate
from src.features.extractor import extract_features
from src.rl.bandit import EpsilonGreedyBandit
from src.rl.ucb import UCB1
from src.rl.linucb import LinUCB
from src.rl.thompson import ThompsonSamplingLinear
from src.rl.dqn import DQNAgent

ACTIONS = ["direct", "cot", "got", "tot"]

# PCA transformer for Thompson Sampling (reduces 389-dim to 20-dim)
_pca_fitted = None

def get_pca_state(state, n_components=20):
    """Reduce state dimensionality via PCA for Thompson Sampling."""
    global _pca_fitted
    if _pca_fitted is None:
        # Fit PCA on first call — just project this single vector
        _pca_fitted = PCA(n_components=min(n_components, len(state)))
        _pca_fitted.fit(state.reshape(1, -1))  # Will update as more data comes
    return _pca_fitted.transform(state.reshape(1, -1)).flatten()


AGENT_REGISTRY = {
    "epsilon_greedy": lambda: EpsilonGreedyBandit(n_actions=4, epsilon=0.3,
                                                    epsilon_decay=0.995, epsilon_min=0.05),
    "ucb1": lambda: UCB1(n_actions=4, c=1.414),
    "linucb": lambda: LinUCB(n_actions=4, alpha=1.0),
    "thompson": lambda: ThompsonSamplingLinear(n_actions=4, v_squared=0.5),
    "dqn": lambda: DQNAgent(n_actions=4, lr=1e-3, epsilon=0.3,
                              epsilon_decay=0.995, epsilon_min=0.05, batch_size=32),
}


def load_configs():
    """Load all YAML configuration files."""
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/strategies.yaml") as f:
        strat_cfg = yaml.safe_load(f)
    with open("configs/reward.yaml") as f:
        reward_cfg = yaml.safe_load(f)
    return model_cfg, strat_cfg, reward_cfg


def build_strategies(llm, strat_cfg):
    """Instantiate all 4 reasoning strategies."""
    def load_prompt(p):
        with open(p) as f:
            return f.read()

    return {
        "direct": DirectStrategy(llm, load_prompt("prompts/direct.txt")),
        "cot": CoTStrategy(llm, load_prompt("prompts/cot.txt")),
        "got": GoTStrategy(llm, load_prompt("prompts/got.txt"),
                            **(strat_cfg.get("got") or {})),
        "tot": ToTStrategy(llm, load_prompt("prompts/tot.txt"),
                            **(strat_cfg.get("tot") or {})),
    }


def compute_reward(metrics, reward_cfg, max_tokens, max_latency):
    """Compute composite reward from evaluation metrics."""
    alpha = reward_cfg["alpha"]
    beta = reward_cfg["beta"]
    gamma = reward_cfg["gamma"]

    norm_tokens = metrics["total_tokens"] / max(max_tokens, 1)
    norm_latency = metrics["latency"] / max(max_latency, 1.0)

    return alpha * float(metrics["correct"]) - beta * norm_tokens - gamma * norm_latency


def run_fixed_strategy(strategy_name, strategies, df, model_name):
    """Run a single fixed strategy on all problems (baseline)."""
    strategy = strategies[strategy_name]
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                        desc=f"fixed_{strategy_name}"):
        sr = strategy.solve(row["question"])
        metrics = evaluate(sr, str(row["expected_answer"]), row["category"])
        metrics.update({
            "problem_id": row["problem_id"],
            "category": row["category"],
            "strategy": strategy_name,
            "agent": f"always_{strategy_name}",
            "model": model_name,
        })
        results.append(metrics)

    return results


def run_random_policy(strategies, df, reward_cfg, model_name):
    """Random policy — selects strategy uniformly at random."""
    results = []
    max_tokens, max_latency = 1000, 10.0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="random"):
        action = np.random.randint(len(ACTIONS))
        strategy_name = ACTIONS[action]
        strategy = strategies[strategy_name]

        sr = strategy.solve(row["question"])
        metrics = evaluate(sr, str(row["expected_answer"]), row["category"])

        max_tokens = max(max_tokens, metrics["total_tokens"])
        max_latency = max(max_latency, metrics["latency"])
        reward = compute_reward(metrics, reward_cfg, max_tokens, max_latency)

        metrics.update({
            "problem_id": row["problem_id"],
            "category": row["category"],
            "strategy": strategy_name,
            "agent": "random",
            "model": model_name,
            "reward": reward,
        })
        results.append(metrics)

    return results


def run_rule_based_policy(strategies, df, reward_cfg, model_name):
    """Rule-based policy from src/policies/rule_based_policy.py."""
    results = []
    max_tokens, max_latency = 1000, 10.0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="rule_based"):
        length = len(str(row["question"]).split())
        category = row["category"]

        # Rule: short → CoT, logic → ToT, else GoT
        if length < 15:
            action = 1  # CoT
        elif category == "logic":
            action = 3  # ToT
        else:
            action = 2  # GoT

        strategy_name = ACTIONS[action]
        strategy = strategies[strategy_name]
        sr = strategy.solve(row["question"])
        metrics = evaluate(sr, str(row["expected_answer"]), row["category"])

        max_tokens = max(max_tokens, metrics["total_tokens"])
        max_latency = max(max_latency, metrics["latency"])
        reward = compute_reward(metrics, reward_cfg, max_tokens, max_latency)

        metrics.update({
            "problem_id": row["problem_id"],
            "category": row["category"],
            "strategy": strategy_name,
            "agent": "rule_based",
            "model": model_name,
            "reward": reward,
        })
        results.append(metrics)

    return results


def run_rl_agent(agent_name, agent, strategies, df, reward_cfg, model_name,
                 use_pca=False):
    """Run an RL agent on a dataset — agent selects strategy per problem."""
    global _pca_fitted
    results = []
    cumulative_reward = 0.0
    max_tokens, max_latency = 1000, 10.0

    # For PCA-Thompson: collect states first, fit PCA, then use
    if use_pca:
        all_states = []
        for _, row in df.iterrows():
            all_states.append(extract_features(row["question"], row["category"]))
        all_states = np.array(all_states)
        n_comp = min(20, all_states.shape[1], all_states.shape[0])
        _pca_fitted = PCA(n_components=n_comp)
        _pca_fitted.fit(all_states)

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                          desc=agent_name)):
        # Extract features for RL agent
        state = extract_features(row["question"], row["category"])
        agent_state = get_pca_state(state) if use_pca else state

        # Agent selects action
        action = agent.select_action(agent_state)
        strategy_name = ACTIONS[action]
        strategy = strategies[strategy_name]

        # Execute strategy
        sr = strategy.solve(row["question"])
        metrics = evaluate(sr, str(row["expected_answer"]), row["category"])

        # Compute reward
        max_tokens = max(max_tokens, metrics["total_tokens"])
        max_latency = max(max_latency, metrics["latency"])
        reward = compute_reward(metrics, reward_cfg, max_tokens, max_latency)

        # Agent learns
        agent.update(agent_state, action, reward)

        cumulative_reward += reward
        metrics.update({
            "problem_id": row["problem_id"],
            "category": row["category"],
            "strategy": strategy_name,
            "agent": agent_name,
            "model": model_name,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "step": idx,
        })

        # Log exploration rate if available
        if hasattr(agent, "epsilon"):
            metrics["epsilon"] = agent.epsilon

        results.append(metrics)

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument("--datasets", required=True,
                        help="Comma-separated dataset names (e.g. gsm8k_sample,logiqa_sample)")
    parser.add_argument("--agents", default="all",
                        help="Comma-separated agent names or 'all'")
    parser.add_argument("--baselines", action="store_true", default=True,
                        help="Run fixed-strategy baselines")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit problems per dataset")
    parser.add_argument("--run-name", default="experiment",
                        help="Name prefix for output files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for single run)")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds for multi-seed runs (e.g. 42,123,456,789,1024)")
    args = parser.parse_args()

    # Determine seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]

    model_cfg, strat_cfg, reward_cfg = load_configs()

    # Build LLM client and strategies
    llm = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model_name"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
        timeout=model_cfg["timeout_seconds"],
    )
    strategies = build_strategies(llm, strat_cfg)

    # Determine which agents to run
    if args.agents == "all":
        agent_names = list(AGENT_REGISTRY.keys())
    else:
        agent_names = [a.strip() for a in args.agents.split(",")]

    datasets = [d.strip() for d in args.datasets.split(",")]
    all_results = []

    for seed_idx, seed in enumerate(seeds):
        np.random.seed(seed)
        if len(seeds) > 1:
            print(f"\n{'#'*60}")
            print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
            print(f"{'#'*60}")

        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")

            df = pd.read_csv(f"data/processed/{dataset_name}.csv")
            if args.limit:
                df = df.head(args.limit)
            print(f"  Problems: {len(df)}")

            # --- Fixed-strategy baselines (only on first seed — deterministic) ---
            if args.baselines and seed_idx == 0:
                for strat_name in ACTIONS:
                    print(f"\n  Running fixed baseline: always_{strat_name}")
                    results = run_fixed_strategy(strat_name, strategies, df,
                                                  model_cfg["model_name"])
                    for r in results:
                        r["dataset"] = dataset_name
                        r["seed"] = seed
                    all_results.extend(results)

                # Random policy
                print(f"\n  Running random policy")
                results = run_random_policy(strategies, df, reward_cfg,
                                             model_cfg["model_name"])
                for r in results:
                    r["dataset"] = dataset_name
                    r["seed"] = seed
                all_results.extend(results)

                # Rule-based policy
                print(f"\n  Running rule-based policy")
                results = run_rule_based_policy(strategies, df, reward_cfg,
                                                 model_cfg["model_name"])
                for r in results:
                    r["dataset"] = dataset_name
                    r["seed"] = seed
                all_results.extend(results)

            # --- RL agents ---
            for agent_name in agent_names:
                print(f"\n  Running RL agent: {agent_name} (seed={seed})")
                agent = AGENT_REGISTRY[agent_name]()
                # Use PCA for Thompson Sampling to fix dimensionality issue
                use_pca = (agent_name == "thompson")
                results = run_rl_agent(agent_name, agent, strategies, df,
                                        reward_cfg, model_cfg["model_name"],
                                        use_pca=use_pca)
                for r in results:
                    r["dataset"] = dataset_name
                    r["seed"] = seed
                all_results.extend(results)

    # Save all results
    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{args.run_name}_results.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved {len(all_results)} total rows to {out_path}")
    print(f"Seeds used: {seeds}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
