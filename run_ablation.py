"""Ablation study runner — tests sensitivity to key hyperparameters.

Three ablation experiments:
1. Reward weight sensitivity: vary (α, β, γ)
2. Feature ablation: embedding-only vs structural-only vs full
3. LinUCB exploration parameter sensitivity: vary α

Usage:
    python run_ablation.py --dataset gsm8k_sample --limit 50
"""
import argparse
import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.llm.client import LLMClient
from src.strategies.direct import DirectStrategy
from src.strategies.cot import CoTStrategy
from src.strategies.got import GoTStrategy
from src.strategies.tot import ToTStrategy
from src.evaluation.evaluator import evaluate
from src.features.extractor import extract_features, get_embedder, CATEGORY_MAP
from src.rl.linucb import LinUCB


ACTIONS = ["direct", "cot", "got", "tot"]


def load_configs():
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/strategies.yaml") as f:
        strat_cfg = yaml.safe_load(f)
    return model_cfg, strat_cfg


def build_strategies(llm, strat_cfg):
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


def compute_reward(metrics, alpha, beta, gamma, max_tokens, max_latency):
    norm_tokens = metrics["total_tokens"] / max(max_tokens, 1)
    norm_latency = metrics["latency"] / max(max_latency, 1.0)
    return alpha * float(metrics["correct"]) - beta * norm_tokens - gamma * norm_latency


def extract_features_ablation(question, category, mode="full"):
    """Extract features with ablation modes."""
    embedder = get_embedder()

    if mode == "embedding_only":
        return embedder.encode(question).astype(np.float32)

    elif mode == "structural_only":
        length = len(question.split())
        n_sentences = question.count(".") + question.count("?") + 1
        category_onehot = np.zeros(len(CATEGORY_MAP))
        category_onehot[CATEGORY_MAP.get(category, 0)] = 1.0
        return np.array([length, n_sentences, *category_onehot], dtype=np.float32)

    else:  # full
        return extract_features(question, category)


def run_agent_with_config(agent, strategies, df, alpha, beta, gamma, feature_mode="full"):
    """Run a LinUCB agent with specific reward weights and feature mode."""
    results = []
    max_tokens, max_latency = 1000, 10.0
    cumulative_reward = 0.0

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="  running")):
        state = extract_features_ablation(row["question"], row["category"], feature_mode)
        action = agent.select_action(state)
        strategy_name = ACTIONS[action]
        strategy = strategies[strategy_name]

        sr = strategy.solve(row["question"])
        metrics = evaluate(sr, str(row["expected_answer"]), row["category"])

        max_tokens = max(max_tokens, metrics["total_tokens"])
        max_latency = max(max_latency, metrics["latency"])
        reward = compute_reward(metrics, alpha, beta, gamma, max_tokens, max_latency)

        agent.update(state, action, reward)
        cumulative_reward += reward

        metrics.update({
            "problem_id": row["problem_id"],
            "category": row["category"],
            "strategy": strategy_name,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "step": idx,
        })
        results.append(metrics)

    return results


def ablation_reward_weights(strategies, df, out_dir):
    """Ablation 1: Vary reward weights (α, β, γ)."""
    configs = {
        "accuracy_only":  (1.0, 0.0, 0.0),
        "cost_heavy":     (0.5, 0.5, 0.0),
        "latency_heavy":  (0.5, 0.0, 0.5),
        "balanced":       (1.0, 0.3, 0.1),  # default
        "equal":          (0.33, 0.33, 0.33),
    }

    all_results = []
    for name, (alpha, beta, gamma) in configs.items():
        print(f"\n  Reward config: {name} (α={alpha}, β={beta}, γ={gamma})")
        agent = LinUCB(n_actions=4, alpha=1.0)
        results = run_agent_with_config(agent, strategies, df, alpha, beta, gamma)
        for r in results:
            r["ablation"] = "reward_weights"
            r["config"] = name
            r["alpha_w"] = alpha
            r["beta_w"] = beta
            r["gamma_w"] = gamma
        all_results.extend(results)

    pd.DataFrame(all_results).to_csv(
        os.path.join(out_dir, "ablation_reward_weights.csv"), index=False)
    print(f"  Saved reward weight ablation results")
    return all_results


def ablation_features(strategies, df, out_dir):
    """Ablation 2: Compare feature representations."""
    modes = ["embedding_only", "structural_only", "full"]

    all_results = []
    for mode in modes:
        print(f"\n  Feature mode: {mode}")
        agent = LinUCB(n_actions=4, alpha=1.0)
        results = run_agent_with_config(agent, strategies, df,
                                         alpha=1.0, beta=0.3, gamma=0.1,
                                         feature_mode=mode)
        for r in results:
            r["ablation"] = "features"
            r["config"] = mode
        all_results.extend(results)

    pd.DataFrame(all_results).to_csv(
        os.path.join(out_dir, "ablation_features.csv"), index=False)
    print(f"  Saved feature ablation results")
    return all_results


def ablation_exploration(strategies, df, out_dir):
    """Ablation 3: LinUCB exploration parameter α."""
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    all_results = []
    for alpha_explore in alpha_values:
        print(f"\n  LinUCB α (exploration): {alpha_explore}")
        agent = LinUCB(n_actions=4, alpha=alpha_explore)
        results = run_agent_with_config(agent, strategies, df,
                                         alpha=1.0, beta=0.3, gamma=0.1)
        for r in results:
            r["ablation"] = "exploration"
            r["config"] = f"alpha={alpha_explore}"
            r["linucb_alpha"] = alpha_explore
        all_results.extend(results)

    pd.DataFrame(all_results).to_csv(
        os.path.join(out_dir, "ablation_exploration.csv"), index=False)
    print(f"  Saved exploration ablation results")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. gsm8k_sample)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablations", default="all",
                        help="Comma-separated: reward_weights,features,exploration or 'all'")
    args = parser.parse_args()

    np.random.seed(args.seed)

    model_cfg, strat_cfg = load_configs()
    llm = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model_name"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
        timeout=model_cfg["timeout_seconds"],
    )
    strategies = build_strategies(llm, strat_cfg)

    df = pd.read_csv(f"data/processed/{args.dataset}.csv")
    if args.limit:
        df = df.head(args.limit)

    out_dir = "data/results"
    os.makedirs(out_dir, exist_ok=True)

    ablations = args.ablations.split(",") if args.ablations != "all" \
        else ["reward_weights", "features", "exploration"]

    print(f"Running ablations on {args.dataset} ({len(df)} problems)")
    print(f"Ablations: {ablations}")

    if "reward_weights" in ablations:
        print("\n" + "="*50)
        print("Ablation 1: Reward Weight Sensitivity")
        print("="*50)
        ablation_reward_weights(strategies, df, out_dir)

    if "features" in ablations:
        print("\n" + "="*50)
        print("Ablation 2: Feature Representation")
        print("="*50)
        ablation_features(strategies, df, out_dir)

    if "exploration" in ablations:
        print("\n" + "="*50)
        print("Ablation 3: Exploration Parameter (LinUCB α)")
        print("="*50)
        ablation_exploration(strategies, df, out_dir)

    print("\n✓ All ablation studies complete!")


if __name__ == "__main__":
    main()
