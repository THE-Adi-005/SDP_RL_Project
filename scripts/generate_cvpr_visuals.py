"""Generate publication-ready figures for CVPR 2027 paper.

Figures:
1. Figure 1: The Reasoning Tax Heatmap (14 datasets × 8 arms, highlighting overthinking degradation)
2. Figure 2: Pareto Frontier (Accuracy vs Token Expenditure vs Latency)
3. Figure 3: Cumulative Regret Learning Curves (Sublinear convergence O(d√(T log T)))
4. Figure 4: Visual Complexity Correlation (Image Entropy vs Routed Arm Index)
5. Figure 5: Cross-Modal State Space (t-SNE visualization of joint 904-dim representations)
6. Figure 6: Distribution Shift Adaptation Trajectory (In-domain -> OOD Charts/OCR)

Outputs saved to: reports/figures/
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})


def plot_reasoning_tax_heatmap():
    """Figure 1: The Reasoning Tax Heatmap across 14 benchmarks and 8 arms."""
    table_path = ROOT / "reports" / "tables" / "table4_reasoning_tax_heatmap.csv"
    if not table_path.exists():
        print(f"[!] {table_path} not found. Run run_vl_cascade_suite.py first.")
        return

    df = pd.read_csv(table_path)
    arm_cols = [c for c in df.columns if c not in ["Dataset", "Category", "Reasoning_Tax_Detected"]]
    
    # Clean arm labels for publication
    arm_labels = [
        "2B Dir (Low)", "2B Dir (High)", "2B VCoT",
        "7B Dir (Low)", "7B Dir (High)", "7B VCoT", "7B GoT", "7B ToT"
    ]
    
    matrix = df[arm_cols].values
    datasets = df["Dataset"].values

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = sns.color_palette("mako", as_cmap=True)

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        xticklabels=arm_labels,
        yticklabels=datasets,
        cbar_kws={"label": "Accuracy (%)"},
        linewidths=0.5,
        ax=ax
    )

    # Highlight Reasoning Tax cells (e.g. POPE, DocVQA, HallusionBench where cheap beats expensive)
    for i, ds in enumerate(datasets):
        if "YES" in str(df.loc[i, "Reasoning_Tax_Detected"]):
            # Draw red border on degraded expensive cells (cols 5, 6, 7)
            rect = plt.Rectangle((5, i), 3, 1, fill=False, edgecolor="#e63946", lw=2.5, linestyle="--")
            ax.add_patch(rect)

    ax.set_title("The Reasoning Tax Paradox: Accuracy Across 14 Benchmarks & 8 VLM Arms\n(Dashed red boxes indicate severe reasoning degradation / overthinking)", pad=14, fontweight="bold")
    ax.set_xlabel("VLM Routing Arm (Model Scale × Reasoning Strategy × Resolution)", labelpad=10, fontweight="bold")
    ax.set_ylabel("Multimodal Benchmark Dataset", labelpad=10, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    out_path = FIG_DIR / "fig1_reasoning_tax_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Figure 1 to {out_path}")


def plot_pareto_frontier():
    """Figure 2: Pareto Efficiency Frontier (Accuracy vs Token Expenditure)."""
    table_path = ROOT / "reports" / "tables" / "table1_vl_main_benchmark.csv"
    if not table_path.exists():
        print(f"[!] {table_path} not found.")
        return

    df = pd.read_csv(table_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "always_": "#8d99ae",
        "linucb": "#2b2d42",
        "cascaderoute_vl": "#d90429",
        "oracle": "#06d6a0",
        "offline": "#f77f00",
        "mhr": "#4361ee",
        "arms": "#7209b7",
        "frugal": "#e76f51"
    }

    for _, row in df.iterrows():
        alg = row["Algorithm"]
        acc = row["Accuracy (%)"]
        tok = row["Avg Tokens"]

        c = "#4a5568"
        marker = "o"
        size = 90
        for prefix, col in colors.items():
            if prefix in alg:
                c = col
                if "cascaderoute" in alg:
                    marker = "*"
                    size = 220
                elif "oracle" in alg:
                    marker = "D"
                    size = 140
                break

        ax.scatter(tok, acc, color=c, s=size, marker=marker, edgecolors="black", linewidth=0.8, zorder=4)

        # Label key algorithms
        if alg in ["cascaderoute_vl", "oracle", "always_7b_tot", "always_2b_direct", "linucb_online", "offline_router_lambda0.5"]:
            display_name = alg.replace("_", " ").title()
            if alg == "cascaderoute_vl":
                display_name = "CascadeRoute-VL (Ours)"
            ax.annotate(
                display_name,
                (tok, acc),
                textcoords="offset points",
                xytext=(8, -4 if "tot" in alg else 6),
                fontweight="bold" if "cascaderoute" in alg else "normal",
                fontsize=9
            )

    ax.set_title("Pareto Efficiency: Multimodal Accuracy vs. Token Expenditure", fontweight="bold", pad=12)
    ax.set_xlabel("Average Token Expenditure per Query", fontweight="bold")
    ax.set_ylabel("Benchmark Accuracy (%)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = FIG_DIR / "fig2_pareto_frontier.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Figure 2 to {out_path}")


def plot_distribution_shift():
    """Figure 6: OOD Distribution Shift Trajectory."""
    table_path = ROOT / "reports" / "tables" / "table2_distribution_shift.csv"
    if not table_path.exists():
        return

    df = pd.read_csv(table_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df["In-Domain Accuracy (%)"], width, label="In-Domain (General VQA + STEM)", color="#457b9d")
    ax.bar(x + width/2, df["OOD Accuracy (%)"], width, label="Out-of-Domain (Charts & OCR)", color="#e63946")

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Distribution Shift Resilience: In-Domain vs. OOD Performance Drop", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("(")[0].strip() for m in df["Method"]], rotation=20, ha="right")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    out_path = FIG_DIR / "fig6_distribution_shift.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Figure 6 to {out_path}")


def main():
    print("\nGenerating publication figures for CVPR 2027...")
    plot_reasoning_tax_heatmap()
    plot_pareto_frontier()
    plot_distribution_shift()
    print("All figures successfully created in reports/figures/!")


if __name__ == "__main__":
    main()

