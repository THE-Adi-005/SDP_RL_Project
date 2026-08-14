import argparse
import os
import yaml
import pandas as pd
from tqdm import tqdm
from src.llm.client import LLMClient
from src.strategies.direct import DirectStrategy
from src.strategies.cot import CoTStrategy
from src.strategies.got import GoTStrategy
from src.strategies.tot import ToTStrategy
from src.evaluation.evaluator import evaluate

STRATEGY_MAP = {
    "direct": (DirectStrategy, "prompts/direct.txt"),
    "cot": (CoTStrategy, "prompts/cot.txt"),
    "got": (GoTStrategy, "prompts/got.txt"),
    "tot": (ToTStrategy, "prompts/tot.txt"),
}

def load_prompt(path):
    with open(path) as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. gsm8k_sample")
    parser.add_argument("--strategies", required=True, help="comma-separated: direct,cot,got,tot")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/strategies.yaml") as f:
        strat_cfg = yaml.safe_load(f)

    llm = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model_name"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
        timeout=model_cfg["timeout_seconds"],
    )

    df = pd.read_csv(f"data/processed/{args.dataset}.csv")
    if args.limit:
        df = df.head(args.limit)

    results = []
    strategies = args.strategies.split(",")
    for strat_name in strategies:
        cls, prompt_path = STRATEGY_MAP[strat_name]
        prompt_template = load_prompt(prompt_path)
        kwargs = strat_cfg.get(strat_name, {}) or {}
        strategy = cls(llm, prompt_template, **kwargs)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=strat_name):
            sr = strategy.solve(row["question"])
            metrics = evaluate(sr, str(row["expected_answer"]), row["category"])
            metrics.update({
                "problem_id": row["problem_id"],
                "category": row["category"],
                "difficulty": row.get("difficulty", "unknown"),
                "strategy": strat_name,
                "model": model_cfg["model_name"],
            })
            results.append(metrics)

    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{args.dataset}_baseline_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved {len(results)} rows to {out_path}")

if __name__ == "__main__":
    main()
