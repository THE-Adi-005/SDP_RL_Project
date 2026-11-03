"""Publication-quality analysis — generates all tables and figures for the paper.

Usage:
    python analyze_paper.py --results data/results/experiment_results.csv
    python analyze_paper.py --results data/results/ --all-csvs
"""
import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.figsize": (7, 5),
})

AGENT_ORDER = [
    "always_direct", "always_cot", "always_got", "always_tot",
    "random", "rule_based",
    "epsilon_greedy", "ucb1", "linucb", "thompson", "dqn",
]

ACTIONS = ["direct", "cot", "got", "tot"]

AGENT_COLORS = {
    "always_direct": "#95a5a6", "always_cot": "#bdc3c7",
    "always_got": "#7f8c8d", "always_tot": "#636e72",
    "random": "#fdcb6e", "rule_based": "#e17055",
    "epsilon_greedy": "#74b9ff", "ucb1": "#0984e3",
    "linucb": "#e84393", "thompson": "#6c5ce7", "dqn": "#00b894",
}


def load_results(path):
    """Load results from a single CSV or all CSVs in a directory."""
    if os.path.isdir(path):
        csvs = glob.glob(os.path.join(path, "*.csv"))
        dfs = [pd.read_csv(f) for f in csvs if os.path.getsize(f) > 0]
        if not dfs:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return pd.concat(dfs, ignore_index=True)
    return pd.read_csv(path)


def table_dataset_statistics(df, out_dir):
    """Table 1: Dataset statistics."""
    stats = []
    for dataset in df["dataset"].unique() if "dataset" in df.columns else ["all"]:
        sub = df[df["dataset"] == dataset] if "dataset" in df.columns else df
        for cat in sub["category"].unique():
            cat_df = sub[sub["category"] == cat]
            n = len(cat_df["problem_id"].unique()) if "problem_id" in cat_df.columns else len(cat_df)
            stats.append({
                "Dataset": dataset,
                "Category": cat,
                "Problems": n,
            })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(out_dir, "table1_dataset_stats.csv"), index=False)
    print(f"  Table 1: {os.path.join(out_dir, 'table1_dataset_stats.csv')}")
    return stats_df


def table_agent_comparison(df, out_dir):
    """Table 3: Agent comparison — accuracy, tokens, latency, reward."""
    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]

    rows = []
    for agent in agents_present:
        sub = df[df["agent"] == agent]
        row = {
            "Agent": agent,
            "Accuracy (%)": round(sub["correct"].mean() * 100, 2),
            "Avg Tokens": round(sub["total_tokens"].mean(), 1),
            "Median Latency (s)": round(sub["latency"].median(), 2),
        }
        if "reward" in sub.columns:
            row["Avg Reward"] = round(sub["reward"].mean(), 4)
        if "n_llm_calls" in sub.columns:
            row["Avg LLM Calls"] = round(sub["n_llm_calls"].mean(), 1)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "table3_agent_comparison.csv"), index=False)
    print(f"  Table 3: {os.path.join(out_dir, 'table3_agent_comparison.csv')}")
    print(summary.to_string(index=False))
    return summary


def table_per_category(df, out_dir):
    """Table 4: Per-category accuracy breakdown."""
    if "category" not in df.columns or "agent" not in df.columns:
        return None

    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    pivot = df.pivot_table(
        values="correct", index="agent", columns="category",
        aggfunc="mean"
    ).reindex(agents_present) * 100

    pivot = pivot.round(2)
    pivot.to_csv(os.path.join(out_dir, "table4_per_category.csv"))
    print(f"  Table 4: {os.path.join(out_dir, 'table4_per_category.csv')}")
    return pivot


def fig_accuracy_by_agent(df, out_dir):
    """Fig 2: Grouped bar chart — accuracy by agent."""
    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    acc = df.groupby("agent")["correct"].mean().reindex(agents_present) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [AGENT_COLORS.get(a, "#333") for a in agents_present]
    bars = ax.bar(range(len(agents_present)), acc.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(agents_present)))
    ax.set_xticklabels([a.replace("always_", "fixed_") for a in agents_present],
                        rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Agent / Policy")
    ax.set_ylim(0, 100)
    ax.axhline(y=acc.mean(), color="red", linestyle="--", alpha=0.5, label="Mean")
    ax.legend()

    for bar, val in zip(bars, acc.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_accuracy_by_agent.png"))
    plt.close()
    print(f"  Fig 2: accuracy_by_agent.png")


def fig_pareto_front(df, out_dir):
    """Fig 3: Pareto front — accuracy vs token cost per agent."""
    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]

    fig, ax = plt.subplots(figsize=(8, 6))
    for agent in agents_present:
        sub = df[df["agent"] == agent]
        acc = sub["correct"].mean() * 100
        tokens = sub["total_tokens"].mean()
        color = AGENT_COLORS.get(agent, "#333")
        ax.scatter(tokens, acc, s=120, color=color, edgecolors="black",
                    linewidth=0.5, zorder=5)
        ax.annotate(agent.replace("always_", "fixed_"),
                    (tokens, acc), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)

    ax.set_xlabel("Average Tokens per Problem")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Token Cost (Pareto Front)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_pareto_front.png"))
    plt.close()
    print(f"  Fig 3: pareto_front.png")


def fig_cumulative_reward(df, out_dir):
    """Fig 5: Cumulative reward over time for RL agents."""
    rl_agents = [a for a in AGENT_ORDER
                  if a in df["agent"].unique() and "cumulative_reward" in df.columns]

    if not rl_agents:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for agent in rl_agents:
        sub = df[(df["agent"] == agent) & df["cumulative_reward"].notna()]
        if len(sub) == 0:
            continue
        color = AGENT_COLORS.get(agent, "#333")
        ax.plot(range(len(sub)), sub["cumulative_reward"].values,
                label=agent, color=color, linewidth=1.5)

    ax.set_xlabel("Problem Index (Training Step)")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Over Training")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_cumulative_reward.png"))
    plt.close()
    print(f"  Fig 5: cumulative_reward.png")


def fig_strategy_selection_heatmap(df, out_dir):
    """Fig 6: Heatmap — strategy selection frequency per category per agent."""
    rl_agents = ["epsilon_greedy", "ucb1", "linucb", "thompson", "dqn"]
    rl_agents = [a for a in rl_agents if a in df["agent"].unique()]

    if not rl_agents or "category" not in df.columns or "strategy" not in df.columns:
        return

    fig, axes = plt.subplots(1, len(rl_agents), figsize=(4 * len(rl_agents), 4),
                              sharey=True)
    if len(rl_agents) == 1:
        axes = [axes]

    for ax, agent in zip(axes, rl_agents):
        sub = df[df["agent"] == agent]
        ct = pd.crosstab(sub["category"], sub["strategy"], normalize="index") * 100
        # Reorder columns
        cols = [c for c in ACTIONS if c in ct.columns]
        ct = ct.reindex(columns=cols, fill_value=0)

        sns.heatmap(ct, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                    vmin=0, vmax=100, cbar=False)
        ax.set_title(agent, fontsize=10)
        ax.set_ylabel("" if ax != axes[0] else "Category")
        ax.set_xlabel("Strategy")

    plt.suptitle("Strategy Selection Frequency (%) by Category", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_strategy_heatmap.png"))
    plt.close()
    print(f"  Fig 6: strategy_heatmap.png")


def fig_reward_distribution(df, out_dir):
    """Fig 7: Box plot — reward distribution per agent."""
    if "reward" not in df.columns:
        return

    agents_present = [a for a in AGENT_ORDER
                      if a in df["agent"].unique() and
                      df[df["agent"] == a]["reward"].notna().any()]

    if not agents_present:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df[df["agent"] == a]["reward"].dropna().values for a in agents_present]
    colors = [AGENT_COLORS.get(a, "#333") for a in agents_present]

    bp = ax.boxplot(data, tick_labels=[a.replace("always_", "f_") for a in agents_present],
                     patch_artist=True, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 5})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Reward")
    ax.set_title("Reward Distribution by Agent")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_reward_distribution.png"))
    plt.close()
    print(f"  Fig 7: reward_distribution.png")


def fig_radar_chart(df, out_dir):
    """Fig 9: Radar chart — multi-metric comparison for top agents."""
    rl_agents = ["linucb", "thompson", "epsilon_greedy", "dqn"]
    rl_agents = [a for a in rl_agents if a in df["agent"].unique()]

    if len(rl_agents) < 2 or "reward" not in df.columns:
        return

    metrics = {}
    for agent in rl_agents:
        sub = df[df["agent"] == agent]
        metrics[agent] = {
            "Accuracy": sub["correct"].mean(),
            "Low Cost": 1 - (sub["total_tokens"].mean() /
                              max(df["total_tokens"].max(), 1)),
            "Low Latency": 1 - (sub["latency"].mean() /
                                  max(df["latency"].max(), 1)),
            "Reward": (sub["reward"].mean() - df["reward"].min()) /
                       max(df["reward"].max() - df["reward"].min(), 1e-6),
        }

    labels = list(list(metrics.values())[0].keys())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for agent in rl_agents:
        values = [metrics[agent][l] for l in labels]
        values += values[:1]
        color = AGENT_COLORS.get(agent, "#333")
        ax.plot(angles, values, "o-", linewidth=1.5, label=agent, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Agent Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig9_radar_chart.png"))
    plt.close()
    print(f"  Fig 9: radar_chart.png")


# ======================================================================
# IMPROVEMENT 4: Statistical significance tests
# ======================================================================
def table_statistical_significance(df, out_dir, reference_agent="linucb"):
    """Table 5: Pairwise statistical tests — compares reference_agent vs all others."""
    from scipy.stats import ttest_rel, wilcoxon

    if "reward" not in df.columns:
        print("  Table 5: Skipped (no reward column)")
        return None

    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()
                      and a != reference_agent]
    ref = df[df["agent"] == reference_agent]

    if len(ref) == 0:
        print(f"  Table 5: Skipped (no data for {reference_agent})")
        return None

    rows = []
    for agent in agents_present:
        comp = df[df["agent"] == agent]

        # Align by problem_id to get paired samples
        if "problem_id" in df.columns:
            merged = ref[["problem_id", "reward"]].merge(
                comp[["problem_id", "reward"]],
                on="problem_id", suffixes=("_ref", "_comp")
            )
            if len(merged) < 5:
                continue
            ref_rewards = merged["reward_ref"].values
            comp_rewards = merged["reward_comp"].values
        else:
            # Fallback: assume same ordering
            n = min(len(ref), len(comp))
            ref_rewards = ref["reward"].values[:n]
            comp_rewards = comp["reward"].values[:n]

        # Skip if all identical
        diff = ref_rewards - comp_rewards
        if np.std(diff) < 1e-10:
            rows.append({
                "Agent": agent,
                "Mean Reward": round(comp_rewards.mean(), 4),
                "Delta Reward": round(np.mean(diff), 4),
                "t-statistic": "N/A",
                "p-value (t-test)": "N/A",
                "p-value (Wilcoxon)": "N/A",
                "Cohen's d": 0.0,
                "Significant (a=0.05)": "N/A",
            })
            continue

        # Paired t-test
        t_stat, p_ttest = ttest_rel(ref_rewards, comp_rewards)

        # Wilcoxon signed-rank
        try:
            _, p_wilcox = wilcoxon(diff)
        except ValueError:
            p_wilcox = 1.0

        # Cohen's d
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

        # Bonferroni correction
        n_comparisons = len(agents_present)
        bonferroni_alpha = 0.05 / n_comparisons

        rows.append({
            "Agent": agent,
            "Mean Reward": round(comp_rewards.mean(), 4),
            "Delta Reward": round(np.mean(diff), 4),
            "t-statistic": round(t_stat, 3),
            "p-value (t-test)": round(p_ttest, 6),
            "p-value (Wilcoxon)": round(p_wilcox, 6),
            "Cohen's d": round(d, 3),
            "Significant (a=0.05)": "Yes" if p_ttest < bonferroni_alpha else "No",
        })

    if rows:
        sig_df = pd.DataFrame(rows)
        sig_df.to_csv(os.path.join(out_dir, "table5_statistical_tests.csv"), index=False)
        print(f"  Table 5: {os.path.join(out_dir, 'table5_statistical_tests.csv')}")
        print(f"  Reference agent: {reference_agent} | Bonferroni alpha = {0.05/len(agents_present):.4f}")
        print(sig_df.to_string(index=False))
        return sig_df
    return None


# ======================================================================
# IMPROVEMENT 8: Oracle upper-bound
# ======================================================================
def table_oracle_upperbound(df, out_dir):
    """Table 6: Oracle performance — picks best strategy per problem in hindsight."""

    # Only use fixed-strategy baselines to determine oracle
    fixed_agents = [a for a in df["agent"].unique() if a.startswith("always_")]

    if len(fixed_agents) < 2:
        print("  Table 6: Skipped (need fixed baselines)")
        return None

    fixed_df = df[df["agent"].isin(fixed_agents)].copy()
    fixed_df["strategy_from_agent"] = fixed_df["agent"].str.replace("always_", "")

    # For each problem, find the strategy that was correct (prefer cheaper if tied)
    oracle_rows = []
    problem_ids = fixed_df["problem_id"].unique() if "problem_id" in fixed_df.columns else []

    for pid in problem_ids:
        problem_data = fixed_df[fixed_df["problem_id"] == pid]
        # Best = correct first, then fewest tokens
        correct = problem_data[problem_data["correct"] == True]
        if len(correct) > 0:
            best = correct.sort_values("total_tokens").iloc[0]
        else:
            best = problem_data.sort_values("total_tokens").iloc[0]

        oracle_rows.append({
            "problem_id": pid,
            "oracle_strategy": best["strategy_from_agent"],
            "correct": best["correct"],
            "total_tokens": best["total_tokens"],
            "latency": best["latency"],
            "category": best.get("category", "unknown"),
        })

    if not oracle_rows:
        print("  Table 6: Skipped (no problem_id data)")
        return None

    oracle_df = pd.DataFrame(oracle_rows)

    # Summary
    oracle_acc = oracle_df["correct"].mean() * 100
    oracle_tokens = oracle_df["total_tokens"].mean()
    oracle_latency = oracle_df["latency"].median()

    # Strategy distribution
    strat_dist = oracle_df["oracle_strategy"].value_counts(normalize=True) * 100

    summary_rows = [{
        "Agent": "oracle (upper bound)",
        "Accuracy (%)": round(oracle_acc, 2),
        "Avg Tokens": round(oracle_tokens, 1),
        "Median Latency (s)": round(oracle_latency, 2),
        "Strategy Distribution": "; ".join(f"{s}: {p:.0f}%" for s, p in strat_dist.items()),
    }]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "table6_oracle.csv"), index=False)
    print(f"  Table 6: {os.path.join(out_dir, 'table6_oracle.csv')}")
    print(f"  Oracle accuracy: {oracle_acc:.1f}% | Avg tokens: {oracle_tokens:.0f}")
    print(f"  Oracle strategy distribution: {dict(strat_dist.round(1))}")

    # Per-category oracle
    for cat in oracle_df["category"].unique():
        cat_df = oracle_df[oracle_df["category"] == cat]
        cat_acc = cat_df["correct"].mean() * 100
        print(f"    {cat}: {cat_acc:.1f}%")

    return summary_df


# ======================================================================
# Multi-seed reporting helper
# ======================================================================
def table_multiseed_summary(df, out_dir):
    """If data has a 'seed' column with multiple seeds, report mean ± std."""
    if "seed" not in df.columns:
        return None
    seeds = df["seed"].unique()
    if len(seeds) <= 1:
        return None

    print(f"\n  Multi-seed summary ({len(seeds)} seeds: {sorted(seeds)})")

    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    rows = []
    for agent in agents_present:
        sub = df[df["agent"] == agent]
        # Per-seed accuracy
        per_seed_acc = sub.groupby("seed")["correct"].mean() * 100
        per_seed_reward = sub.groupby("seed")["reward"].mean() if "reward" in sub.columns else None
        per_seed_tokens = sub.groupby("seed")["total_tokens"].mean()

        row = {
            "Agent": agent,
            "Accuracy (mean ± std)": f"{per_seed_acc.mean():.1f} ± {per_seed_acc.std():.1f}",
            "Tokens (mean ± std)": f"{per_seed_tokens.mean():.0f} ± {per_seed_tokens.std():.0f}",
        }
        if per_seed_reward is not None:
            row["Reward (mean ± std)"] = f"{per_seed_reward.mean():.3f} ± {per_seed_reward.std():.3f}"
        rows.append(row)

    ms_df = pd.DataFrame(rows)
    ms_df.to_csv(os.path.join(out_dir, "table7_multiseed.csv"), index=False)
    print(f"  Table 7: {os.path.join(out_dir, 'table7_multiseed.csv')}")
    print(ms_df.to_string(index=False))
    return ms_df


def main():
    parser = argparse.ArgumentParser(description="Publication analysis")
    parser.add_argument("--results", required=True,
                        help="Path to results CSV or directory")
    args = parser.parse_args()

    fig_dir = "reports/figures"
    tbl_dir = "reports/tables"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    print("Loading results...")
    df = load_results(args.results)
    print(f"  Total rows: {len(df)}")
    print(f"  Agents: {df['agent'].unique().tolist() if 'agent' in df.columns else 'N/A'}")

    print("\n--- Generating Tables ---")
    table_dataset_statistics(df, tbl_dir)
    table_agent_comparison(df, tbl_dir)
    table_per_category(df, tbl_dir)
    table_statistical_significance(df, tbl_dir)
    table_oracle_upperbound(df, tbl_dir)
    table_multiseed_summary(df, tbl_dir)

    print("\n--- Generating Figures ---")
    fig_accuracy_by_agent(df, fig_dir)
    fig_pareto_front(df, fig_dir)
    fig_cumulative_reward(df, fig_dir)
    fig_strategy_selection_heatmap(df, fig_dir)
    fig_reward_distribution(df, fig_dir)
    fig_radar_chart(df, fig_dir)

    print(f"\n[OK] All outputs saved to {fig_dir}/ and {tbl_dir}/")


if __name__ == "__main__":
    main()

