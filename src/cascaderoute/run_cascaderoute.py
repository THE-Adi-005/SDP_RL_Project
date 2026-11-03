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
from src.features.extractor import extract_features
from src.rl.linucb import LinUCB

from src.cascaderoute.budget_tracker import BudgetTracker
from src.cascaderoute.confidence_gate import ConfidenceGate

ACTIONS = ["3b_direct", "3b_cot", "7b_direct", "7b_cot", "7b_got", "7b_tot"]

def load_prompt(p):
    with open(p) as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default="cascaderoute")
    args = parser.parse_args()
    
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/strategies.yaml") as f:
        strat_cfg = yaml.safe_load(f)
    with open("configs/reward.yaml") as f:
        reward_cfg = yaml.safe_load(f)

    # Initialize two LLM clients
    llm_3b = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["fallback_model"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
        timeout=model_cfg["timeout_seconds"],
    )
    llm_7b = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model_name"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
        timeout=model_cfg["timeout_seconds"],
    )

    strategies = {
        "3b_direct": DirectStrategy(llm_3b, load_prompt("prompts/direct.txt")),
        "3b_cot": CoTStrategy(llm_3b, load_prompt("prompts/cot.txt")),
        "7b_direct": DirectStrategy(llm_7b, load_prompt("prompts/direct.txt")),
        "7b_cot": CoTStrategy(llm_7b, load_prompt("prompts/cot.txt")),
        "7b_got": GoTStrategy(llm_7b, load_prompt("prompts/got.txt"), **(strat_cfg.get("got") or {})),
        "7b_tot": ToTStrategy(llm_7b, load_prompt("prompts/tot.txt"), **(strat_cfg.get("tot") or {})),
    }

    gate = ConfidenceGate(strategies)
    budget = BudgetTracker(args.budget)
    bandit = LinUCB(n_actions=len(ACTIONS), alpha=1.0)
    
    datasets = [d.strip() for d in args.datasets.split(",")]
    all_results = []
    
    for dataset_name in datasets:
        df = pd.read_csv(f"data/processed/{dataset_name}.csv")
        if args.limit:
            df = df.head(args.limit)
            
        cumulative_reward = 0.0
        max_tokens, max_latency = 1000, 10.0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"CascadeRoute ({dataset_name})"):
            # 1. State formulation (Features + Budget Pct)
            features = extract_features(row["question"], row["category"])
            budget_pct = budget.get_remaining_pct()
            state = np.append(features, budget_pct)
            
            # 2. Bandit selects action
            action_idx = bandit.select_action(state)
            initial_action_name = ACTIONS[action_idx]
            
            # 3. Confidence Gate executes with escalation
            res, final_action_name, esc_count = gate.solve_with_cascade(initial_action_name, row["question"])
            
            # 4. Evaluation and budget update
            metrics = evaluate(res, str(row["expected_answer"]), row["category"])
            budget.add_expense(metrics["total_tokens"])
            
            # 5. Reward computation & Bandit update
            max_tokens = max(max_tokens, metrics["total_tokens"])
            max_latency = max(max_latency, metrics["latency"])
            
            alpha = reward_cfg["alpha"]
            beta = reward_cfg["beta"]
            gamma = reward_cfg["gamma"]
            norm_tokens = metrics["total_tokens"] / max(max_tokens, 1)
            norm_latency = metrics["latency"] / max(max_latency, 1.0)
            reward = alpha * float(metrics["correct"]) - beta * norm_tokens - gamma * norm_latency
            
            bandit.update(state, action_idx, reward)
            cumulative_reward += reward
            
            metrics.update({
                "problem_id": row["problem_id"],
                "category": row["category"],
                "initial_action": initial_action_name,
                "final_action": final_action_name,
                "escalations": esc_count,
                "budget_remaining_pct": budget_pct,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "step": idx,
                "dataset": dataset_name
            })
            all_results.append(metrics)
            
    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{args.run_name}_results.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"Saved {len(all_results)} rows to {out_path}")

if __name__ == "__main__":
    main()
