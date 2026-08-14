import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)

    summary = df.groupby("strategy").agg(
        accuracy=("correct", "mean"),
        avg_tokens=("total_tokens", "mean"),
        median_latency=("latency", "median"),
    )
    print(summary)
    summary.to_csv("reports/tables/summary.csv")

    fig, ax = plt.subplots()
    summary["accuracy"].plot(kind="bar", ax=ax, title="Accuracy by Strategy")
    plt.tight_layout()
    plt.savefig("reports/figures/accuracy_by_strategy.png")

    fig, ax = plt.subplots()
    ax.scatter(summary["avg_tokens"], summary["accuracy"])
    for name, row in summary.iterrows():
        ax.annotate(name, (row["avg_tokens"], row["accuracy"]))
    ax.set_xlabel("Avg Tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Token Cost")
    plt.tight_layout()
    plt.savefig("reports/figures/accuracy_vs_cost.png")

    if "cumulative_reward" in df.columns:
        fig, ax = plt.subplots()
        df["cumulative_reward"].plot(ax=ax, title="Cumulative Reward Over Time")
        plt.tight_layout()
        plt.savefig("reports/figures/cumulative_reward.png")

    print("Saved figures to reports/figures/")

if __name__ == "__main__":
    main()
