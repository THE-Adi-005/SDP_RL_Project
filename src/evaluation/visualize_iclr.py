import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Set ICLR paper styling for matplotlib
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/results/cascaderoute_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)

    # 1. Generate ICLR-Formatted LaTeX Table
    group_col = "final_action" if "final_action" in df.columns else "strategy"
    summary = df.groupby(group_col).agg(
        Count=(group_col, "count"),
        Accuracy=("correct", "mean"),
        AvgTokens=("total_tokens", "mean"),
        MedianLatency=("latency", "median"),
    ).reset_index()
    
    # Sort logically
    summary = summary.sort_values(by="AvgTokens")
    
    # Format for LaTeX
    latex_df = summary.copy()
    latex_df["Accuracy"] = (latex_df["Accuracy"] * 100).map("{:.1f}\%".format)
    latex_df["AvgTokens"] = latex_df["AvgTokens"].map("{:.0f}".format)
    latex_df["MedianLatency"] = latex_df["MedianLatency"].map("{:.2f}s".format)
    latex_df.rename(columns={group_col: "Strategy / Model"}, inplace=True)
    
    latex_code = latex_df.to_latex(index=False, escape=False, caption="Performance Metrics by Action Type", label="tab:performance_metrics", column_format="lcccc")
    
    with open("reports/tables/iclr_performance_table.tex", "w") as f:
        f.write(latex_code)
    print("Saved LaTeX table to reports/tables/iclr_performance_table.tex")

    # 2. Pareto Frontier Scatter Plot (Accuracy vs Cost)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=summary, x="AvgTokens", y="Accuracy", s=150, hue=group_col, palette="deep", edgecolor="black", alpha=0.9)
    
    # Annotate points
    for i, row in summary.iterrows():
        plt.annotate(row[group_col], (row["AvgTokens"], row["Accuracy"]), 
                     xytext=(8, 5), textcoords='offset points', fontsize=10)
        
    plt.title("Pareto Trade-off: Accuracy vs. Token Cost")
    plt.xlabel("Average Tokens Consumed (Cost)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Action", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("reports/figures/iclr_pareto_frontier.png")
    plt.close()
    
    # 3. Budget Frugality Plot (Strategy selection over time/budget)
    if "budget_remaining_pct" in df.columns:
        plt.figure(figsize=(10, 5))
        # Bin the budget remaining into 10% buckets
        df["budget_bin"] = pd.cut(df["budget_remaining_pct"], bins=10, labels=[f"{i}0%" for i in range(1, 11)])
        
        # Calculate percentage of each strategy within each budget bin
        budget_dist = df.groupby(["budget_bin", group_col]).size().unstack(fill_value=0)
        budget_dist_pct = budget_dist.div(budget_dist.sum(axis=1), axis=0) * 100
        
        # Plot stacked bar chart
        ax = budget_dist_pct.plot(kind="bar", stacked=True, colormap="viridis", figsize=(10, 5), edgecolor="black")
        plt.title("Action Distribution as Budget Depletes")
        plt.xlabel("Remaining Budget (%)")
        plt.ylabel("Percentage of Actions Selected (%)")
        plt.legend(title="Action", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        plt.savefig("reports/figures/iclr_budget_frugality.png")
        plt.close()

    print("Saved ICLR visualizations to reports/figures/")

if __name__ == "__main__":
    main()
