"""Verify Token Savings of CascadeRoute vs 7B-ToT Baseline.

This script loads stored evaluation dataset results and computes the exact 
token reduction percentage (27.4%) claimed in the research report.
"""
import pandas as pd
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    cascade_file = root / "data" / "results" / "cascaderoute_results.csv"
    baseline_file = root / "data" / "results" / "full_baselines_results.csv"

    if not cascade_file.exists() or not baseline_file.exists():
        print(f"Error: Results CSV files not found in {root / 'data' / 'results'}")
        return

    df_cascade = pd.read_csv(cascade_file)
    df_base = pd.read_csv(baseline_file)

    cascade_tokens = df_cascade["total_tokens"].mean()
    cascade_acc = (df_cascade["correct"].astype(str).str.lower() == "true").mean() * 100

    tot_sub = df_base[df_base["policy"] == "7b_tot"]
    tot_tokens = tot_sub["total_tokens"].mean()
    tot_acc = (tot_sub["correct"].astype(str).str.lower() == "true").mean() * 100

    token_savings_pct = ((tot_tokens - cascade_tokens) / tot_tokens) * 100

    print("=" * 75)
    print("      CASCADEROUTE TOKEN SAVINGS REPRODUCIBILITY VERIFICATION")
    print("=" * 75)
    print(f"Evaluation Dataset Queries : {len(df_cascade):,} queries")
    print("-" * 75)
    print(f"7B-ToT Baseline Avg Tokens  : {tot_tokens:,.2f} tokens | Accuracy: {tot_acc:.1f}%")
    print(f"CascadeRoute Avg Tokens    : {cascade_tokens:,.2f} tokens | Accuracy: {cascade_acc:.1f}%")
    print("-" * 75)
    print(f"Absolute Token Savings     : {tot_tokens - cascade_tokens:,.2f} tokens / query")
    print(f"Relative Token Savings (%) : {token_savings_pct:.1f}% reduction")
    print("=" * 75)

if __name__ == "__main__":
    main()
