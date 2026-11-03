import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/results/cascaderoute_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    os.makedirs("reports/tables", exist_ok=True)

    if "dataset" not in df.columns:
        print("Error: 'dataset' column not found in results.")
        return

    # Group by dataset and action
    group_col = "final_action" if "final_action" in df.columns else "strategy"
    
    # Overall Domain Performance
    domain_summary = df.groupby("dataset").agg(
        Count=("dataset", "count"),
        OverallAccuracy=("correct", "mean"),
        AvgTokens=("total_tokens", "mean"),
        MedianLatency=("latency", "median")
    ).reset_index()

    # Formatted for LaTeX
    latex_domain = domain_summary.copy()
    latex_domain["OverallAccuracy"] = (latex_domain["OverallAccuracy"] * 100).map("{:.1f}\%".format)
    latex_domain["AvgTokens"] = latex_domain["AvgTokens"].map("{:.0f}".format)
    latex_domain["MedianLatency"] = latex_domain["MedianLatency"].map("{:.2f}s".format)
    
    latex_code = latex_domain.to_latex(index=False, escape=False, caption="Overall Performance by Dataset Domain", label="tab:domain_perf", column_format="lcccc")
    
    with open("reports/tables/iclr_domain_perf_table.tex", "w") as f:
        f.write(latex_code)
    print("Saved Domain Performance table to reports/tables/iclr_domain_perf_table.tex")
    
    # Detailed breakdown: What actions were picked for each dataset?
    # Get the top 2 actions per dataset
    breakdown = df.groupby(["dataset", group_col]).size().reset_index(name="Count")
    breakdown["Percentage"] = breakdown.groupby("dataset")["Count"].transform(lambda x: x / x.sum() * 100)
    
    # Sort and take top 2
    top_actions = breakdown.sort_values(["dataset", "Count"], ascending=[True, False]).groupby("dataset").head(2)
    
    print("\n--- Top Actions Selected by Bandit per Domain ---")
    for dataset in top_actions["dataset"].unique():
        print(f"\nDataset: {dataset}")
        subset = top_actions[top_actions["dataset"] == dataset]
        for _, row in subset.iterrows():
            print(f"  - {row[group_col]}: {row['Percentage']:.1f}% of queries")

if __name__ == "__main__":
    main()
