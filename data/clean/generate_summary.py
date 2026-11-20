import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import chi2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CLEAN_DIR = os.path.dirname(os.path.abspath(__file__))
OUTCOMES_CSV = os.path.join(CLEAN_DIR, "outcomes.csv")
SUMMARY_MD = os.path.join(PROJECT_ROOT, "CLEAN_SUMMARY.md")
FINAL_SUMMARY_MD = os.path.join(PROJECT_ROOT, "FINAL_CLEAN_SUMMARY.md")

ACTIONS_ORDER = ["3b_direct", "7b_direct", "3b_cot", "7b_cot", "7b_got", "7b_tot"]

OVERTHINKING_PAIRS = [
    ("3b_direct", "3b_cot", "3B Direct vs 3B CoT"),
    ("7b_direct", "7b_cot", "7B Direct vs 7B CoT"),
    ("7b_cot", "7b_got", "7B CoT vs 7B GoT"),
    ("7b_got", "7b_tot", "7B GoT vs 7B ToT"),
]

def generate_summary(final=False):
    if not os.path.exists(OUTCOMES_CSV):
        print(f"Error: {OUTCOMES_CSV} does not exist.")
        return
        
    df = pd.read_csv(OUTCOMES_CSV)
    if len(df) == 0:
        print("Error: outcomes.csv is empty.")
        return
        
    policies = [p for p in ACTIONS_ORDER if p in df["policy"].unique()]
    datasets = df["dataset"].unique().tolist() if "dataset" in df.columns else []
    
    report_lines = []
    header_title = "FINAL Clean Evaluation Benchmark Summary (`FINAL_CLEAN_SUMMARY.md`)" if final else "Clean Evaluation Benchmark Summary (`CLEAN_SUMMARY.md`)"
    report_lines.append(f"# {header_title}\n")
    report_lines.append(f"Generated at: `{datetime.now().isoformat()}`\n")
    report_lines.append(f"Total Evaluated Outcome Rows: **{len(df):,}**\n")
    report_lines.append(f"Evaluated Policies: `{', '.join(policies)}`\n\n")
    
    # ---------------------------------------------------------
    # 1. Overall Policy Benchmarks Table
    # ---------------------------------------------------------
    report_lines.append("## 1. Overall Policy Performance & Extraction Metrics\n")
    report_lines.append("| Policy | Evaluated N | Primary Acc (%) | Acc Excl Extr Fail (%) | Extr Failures (%) | Errors | Avg Tokens | Avg Lat (s) | P50 Lat (s) | P95 Lat (s) | P99 Lat (s) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for pol in policies:
        sub = df[df["policy"] == pol]
        n = len(sub)
        acc_primary = sub["correct"].mean() * 100.0
        
        # Extraction failure stats
        ext_fails = sub["extraction_failed"].sum() if "extraction_failed" in sub.columns else 0
        ext_fail_rate = (ext_fails / n) * 100.0
        
        valid_extr = sub[sub["extraction_failed"] == False] if "extraction_failed" in sub.columns else sub
        acc_no_ext_fail = valid_extr["correct"].mean() * 100.0 if len(valid_extr) > 0 else 0.0
        
        err_cnt = sub["error"].dropna().count()
        avg_tok = sub["total_tokens"].mean()
        avg_lat = sub["latency"].mean()
        p50_lat = sub["latency"].quantile(0.50)
        p95_lat = sub["latency"].quantile(0.95)
        p99_lat = sub["latency"].quantile(0.99)
        
        report_lines.append(
            f"| **{pol}** | {n:,} | **{acc_primary:.2f}%** | {acc_no_ext_fail:.2f}% | {ext_fails} ({ext_fail_rate:.2f}%) | {err_cnt} | {avg_tok:.1f} | {avg_lat:.2f}s | {p50_lat:.2f}s | {p95_lat:.2f}s | {p99_lat:.2f}s |"
        )
    report_lines.append("\n")
    
    # ---------------------------------------------------------
    # 2. Accuracy per 1,000 Tokens (Per Dataset & Policy)
    # ---------------------------------------------------------
    report_lines.append("## 2. Accuracy per 1,000 Tokens Efficiency (Per Dataset)\n")
    report_lines.append("| Dataset | Policy | Primary Acc (%) | Avg Tokens | Acc per 1,000 Tokens |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for d in datasets:
        for pol in policies:
            sub = df[(df["policy"] == pol) & (df["dataset"] == d)]
            if len(sub) == 0:
                continue
            acc = sub["correct"].mean() * 100.0
            avg_tok = sub["total_tokens"].mean()
            acc_per_1k = (acc / max(avg_tok, 1.0)) * 1000.0
            report_lines.append(f"| `{d}` | **{pol}** | {acc:.2f}% | {avg_tok:.1f} | **{acc_per_1k:.2f}** |")
    report_lines.append("\n")
    
    # ---------------------------------------------------------
    # 3. Overthinking Analysis Table
    # ---------------------------------------------------------
    report_lines.append("## 3. Overthinking Analysis (Cheaper vs More Expensive Policies)\n")
    report_lines.append("Analyzes queries where adding reasoning complexity or parameter scale hurts correctness.\n\n")
    report_lines.append("| Comparison (Cheaper vs Expensive) | Common N | Cheaper RIGHT / Expensive WRONG (Overthinking) | Cheaper WRONG / Expensive RIGHT (Helpful Complexity) | Net Delta |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    piv_corr = df.pivot_table(index="query_id", columns="policy", values="correct", aggfunc="first")
    for p_cheap, p_exp, label in OVERTHINKING_PAIRS:
        if p_cheap in piv_corr.columns and p_exp in piv_corr.columns:
            valid = piv_corr[[p_cheap, p_exp]].dropna()
            c_cheap = valid[p_cheap].values.astype(bool)
            c_exp = valid[p_exp].values.astype(bool)
            
            overthinking_cnt = np.sum(c_cheap & (~c_exp))
            helpful_cnt = np.sum((~c_cheap) & c_exp)
            net_delta = helpful_cnt - overthinking_cnt
            
            report_lines.append(
                f"| **{label}** (`{p_cheap}` vs `{p_exp}`) | {len(valid):,} | **{overthinking_cnt:,}** ({overthinking_cnt/len(valid)*100:.1f}%) | **{helpful_cnt:,}** ({helpful_cnt/len(valid)*100:.1f}%) | {net_delta:+d} |"
            )
    report_lines.append("\n")
    
    # ---------------------------------------------------------
    # 4. Oracle Performance & Cheapest-Correct-Action Cost
    # ---------------------------------------------------------
    report_lines.append("## 4. Oracle Upper Bound & Cheapest-Correct-Action Cost\n")
    if len(piv_corr) > 0:
        oracle_acc = (piv_corr.sum(axis=1) > 0).mean() * 100.0
        
        # Cheapest correct action token calculation
        piv_tok = df.pivot_table(index="query_id", columns="policy", values="total_tokens", aggfunc="first")
        cheapest_tokens = []
        for q_id, row_corr in piv_corr.iterrows():
            correct_policies = row_corr[row_corr == True].index.tolist()
            if correct_policies:
                min_tok = piv_tok.loc[q_id, correct_policies].min()
                cheapest_tokens.append(min_tok)
            else:
                # If no policy was correct, oracle takes min tokens among evaluated
                min_tok = piv_tok.loc[q_id].min()
                cheapest_tokens.append(min_tok)
                
        avg_oracle_cost = np.mean(cheapest_tokens) if cheapest_tokens else 0.0
        report_lines.append(f"- **Oracle Accuracy (Hindsight Best Action)**: **{oracle_acc:.2f}%** across {len(piv_corr):,} unique queries.")
        report_lines.append(f"- **Cheapest-Correct-Action Oracle Token Cost**: **{avg_oracle_cost:.1f} tokens per query** (vs 1,959.3 tokens for static 7B-ToT).\n\n")

    # ---------------------------------------------------------
    # 5. Pareto Table with 95% Bootstrap Confidence Intervals
    # ---------------------------------------------------------
    report_lines.append("## 5. Pareto Frontier Table (Accuracy vs Token Overhead with 95% Bootstrap CIs)\n")
    report_lines.append("| Policy | Evaluated N | Accuracy (%) [95% CI] | Avg Tokens per Query [95% CI] | Avg Latency (s) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    np.random.seed(42)
    for pol in policies:
        sub = df[df["policy"] == pol]
        n = len(sub)
        corr_vals = sub["correct"].values.astype(int)
        tok_vals = sub["total_tokens"].values.astype(float)
        lat_val = sub["latency"].mean()
        
        acc_mean = corr_vals.mean() * 100.0
        tok_mean = tok_vals.mean()
        
        # Bootstrap 95% CI
        boot_acc = []
        boot_tok = []
        for _ in range(1000):
            idx = np.random.choice(n, size=n, replace=True)
            boot_acc.append(corr_vals[idx].mean() * 100.0)
            boot_tok.append(tok_vals[idx].mean())
            
        acc_ci = (np.percentile(boot_acc, 2.5), np.percentile(boot_acc, 97.5))
        tok_ci = (np.percentile(boot_tok, 2.5), np.percentile(boot_tok, 97.5))
        
        report_lines.append(
            f"| **{pol}** | {n:,} | **{acc_mean:.2f}%** [{acc_ci[0]:.2f}%, {acc_ci[1]:.2f}%] | **{tok_mean:.1f}** [{tok_ci[0]:.1f}, {tok_ci[1]:.1f}] | {lat_val:.2f}s |"
        )
    report_lines.append("\n")

    # ---------------------------------------------------------
    # 6. Anomalies & Quality Control Audit Report
    # ---------------------------------------------------------
    report_lines.append("## 6. Anomalies & Quality Control Audit Report\n")
    
    fast_outputs = df[df["latency"] < 0.01]
    empty_outputs = df[df["raw_output"].isna() | (df["raw_output"] == "")]
    disagreements = df[df["token_disagreement_flag"] == True] if "token_disagreement_flag" in df.columns else pd.DataFrame()
    err_rows = df[df["error"].notna()]
    
    report_lines.append(f"- **Suspiciously Fast Outputs (Latency < 0.01s)**: **{len(fast_outputs)}** rows.")
    report_lines.append(f"- **Empty Model Responses (`raw_output == \"\"`)**: **{len(empty_outputs)}** rows.")
    report_lines.append(f"- **Token Count Disagreements (>5% OpenAI vs Ollama Native)**: **{len(disagreements)}** rows.")
    report_lines.append(f"- **Execution Error / Timeout Rows**: **{len(err_rows)}** rows.\n")
    
    summary_text = "\n".join(report_lines)
    
    # Save to CLEAN_SUMMARY.md
    with open(SUMMARY_MD, "w") as f:
        f.write(summary_text)
    print(f"Generated {SUMMARY_MD}")
    
    # If final run complete, also save to FINAL_CLEAN_SUMMARY.md
    if final:
        with open(FINAL_SUMMARY_MD, "w") as f:
            f.write(summary_text)
        print(f"Generated {FINAL_SUMMARY_MD}")

if __name__ == "__main__":
    generate_summary()
