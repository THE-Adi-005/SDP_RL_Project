import argparse
import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import ray
except ImportError:
    ray = None

from src.llm.client import LLMClient
from src.strategies.direct import DirectStrategy
from src.strategies.cot import CoTStrategy
from src.strategies.got import GoTStrategy
from src.strategies.tot import ToTStrategy
from src.evaluation.evaluator import evaluate

ACTIONS = ["3b_direct", "3b_cot", "7b_direct", "7b_cot", "7b_got", "7b_tot"]

def load_prompt(p):
    with open(p) as f:
        return f.read()

def init_strategies():
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/strategies.yaml") as f:
        strat_cfg = yaml.safe_load(f)

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
    return strategies

def run_single_eval(row_dict, policy, strategies_dict):
    if policy == "random":
        action = np.random.choice(ACTIONS)
    else:
        action = policy
        
    strategy = strategies_dict[action]
    res = strategy.solve(row_dict["question"])
    metrics = evaluate(res, str(row_dict["expected_answer"]), row_dict["category"])
    
    metrics.update({
        "problem_id": row_dict["problem_id"],
        "dataset": row_dict["dataset"],
        "category": row_dict["category"],
        "policy": policy,           
        "action_taken": action      
    })
    return metrics

if ray is not None:
    @ray.remote
    class EvaluatorActor:
        def __init__(self):
            self.strategies = init_strategies()
            
        def evaluate(self, row_dict, policy):
            return run_single_eval(row_dict, policy, self.strategies)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--save-interval", type=int, default=10, help="Save to CSV every N queries")
    parser.add_argument("--use-ray", action="store_true", help="Enable ray distributed execution")
    parser.add_argument("--num-actors", type=int, default=4, help="Number of ray actors to spawn (if using ray)")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    policies_to_test = ACTIONS + ["random"]

    os.makedirs("data/results", exist_ok=True)
    out_path = "data/results/full_baselines_results.csv"

    if os.path.exists(out_path):
        all_results = pd.read_csv(out_path).to_dict('records')
        print(f"Resuming from {len(all_results)} existing records...")
        completed_keys = set((r["dataset"], r["problem_id"], r["policy"]) for r in all_results)
    else:
        all_results = []
        completed_keys = set()

    # Collect all tasks
    tasks = []
    for dataset_name in datasets:
        df = pd.read_csv(f"data/processed/{dataset_name}.csv")
        for policy in policies_to_test:
            
            # Subsample expensive policies to cut inference time drastically
            if policy in ["7b_tot", "7b_got"]:
                policy_df = df.head(400)
            else:
                policy_df = df
                
            for _, row in policy_df.iterrows():
                if (dataset_name, row["problem_id"], policy) not in completed_keys:
                    row_dict = row.to_dict()
                    row_dict["dataset"] = dataset_name
                    tasks.append((row_dict, policy))

    print(f"Found {len(tasks)} tasks to run.")
    if len(tasks) == 0:
        print("Nothing to do!")
        return

    if args.use_ray:
        if ray is None:
            raise ImportError("Ray is not installed. Run `pip install ray`.")
        
        ray.init(ignore_reinit_error=True, runtime_env={"working_dir": "."})
        print(f"Connected to Ray cluster. Spawning {args.num_actors} evaluator actors...")
        actors = [EvaluatorActor.remote() for _ in range(args.num_actors)]
        
        # Dispatch tasks
        futures = []
        for i, (row_dict, policy) in enumerate(tasks):
            actor = actors[i % len(actors)]
            futures.append(actor.evaluate.remote(row_dict, policy))
            
        # Process results as they complete
        pbar = tqdm(total=len(futures), desc="Evaluating via Ray")
        unready = futures
        while unready:
            ready, unready = ray.wait(unready, num_returns=min(args.save_interval, len(unready)), timeout=1.0)
            if ready:
                results = ray.get(ready)
                all_results.extend(results)
                pd.DataFrame(all_results).to_csv(out_path, index=False)
                pbar.update(len(ready))
        pbar.close()

    else:
        strategies = init_strategies()
        pbar = tqdm(tasks, desc="Evaluating sequentially")
        for i, (row_dict, policy) in enumerate(pbar):
            metrics = run_single_eval(row_dict, policy, strategies)
            all_results.append(metrics)
            if len(all_results) % args.save_interval == 0:
                pd.DataFrame(all_results).to_csv(out_path, index=False)
                
        pd.DataFrame(all_results).to_csv(out_path, index=False)

    print(f"\nCompleted! Saved {len(all_results)} total rows to {out_path}")

if __name__ == "__main__":
    main()
