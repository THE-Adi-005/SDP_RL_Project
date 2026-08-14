import argparse
import os
import pandas as pd
from tqdm import tqdm
from src.rl.environment import ReasoningBanditEnv, ACTIONS
from src.rl.bandit import EpsilonGreedyBandit
from src.features.extractor import extract_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(f"data/processed/{args.dataset}.csv")
    if args.limit:
        df = df.head(args.limit)

    env = ReasoningBanditEnv(df)
    bandit = EpsilonGreedyBandit(n_actions=len(ACTIONS))

    logs = []
    cumulative_reward = 0.0
    for _ in tqdm(range(len(df)), desc="bandit training"):
        state, row = env.reset()
        action = bandit.select_action(state)
        reward, metrics, done = env.step(action, row)
        bandit.update(state, action, reward)

        cumulative_reward += reward
        metrics["cumulative_reward"] = cumulative_reward
        metrics["epsilon"] = bandit.epsilon
        logs.append(metrics)
        if done:
            break

    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{args.dataset}_bandit_results.csv"
    pd.DataFrame(logs).to_csv(out_path, index=False)
    print(f"Saved {len(logs)} rows to {out_path}")

if __name__ == "__main__":
    main()
