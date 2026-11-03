import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def generate_visuals():
    os.makedirs("reports/figures", exist_ok=True)
    
    # Load data
    df = pd.read_csv("data/results/full_baselines_results.csv")
    
    # Ensure correct types
    df['correct'] = df['correct'] == True
    df['latency'] = pd.to_numeric(df['latency'], errors='coerce')
    df['total_tokens'] = pd.to_numeric(df['total_tokens'], errors='coerce')
    
    # Set paper aesthetics
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("husl", 7)
    
    # 1. Overall Accuracy by Policy
    plt.figure(figsize=(10, 6))
    order = ['random', '3b_direct', '3b_cot', '7b_direct', '7b_cot', '7b_got', '7b_tot']
    ax = sns.barplot(
        data=df, 
        x='policy', 
        y='correct', 
        order=order,
        palette="viridis",
        errorbar=None
    )
    plt.title("Overall Accuracy by LLM Policy", fontweight="bold", fontsize=14)
    plt.ylabel("Accuracy", fontweight="bold")
    plt.xlabel("Policy Strategy", fontweight="bold")
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig("reports/figures/accuracy_by_policy.png", dpi=300)
    plt.close()
    
    # 2. Cost-Accuracy Trade-off
    plt.figure(figsize=(10, 6))
    summary = df.groupby('policy').agg(
        accuracy=('correct', 'mean'),
        avg_latency=('latency', 'mean')
    ).reset_index()
    
    sns.scatterplot(
        data=summary,
        x='avg_latency',
        y='accuracy',
        hue='policy',
        s=200,
        palette="viridis",
        legend=False
    )
    
    # Annotate points
    for i, row in summary.iterrows():
        plt.annotate(row['policy'], 
                     (row['avg_latency'], row['accuracy']),
                     xytext=(10, 0), 
                     textcoords='offset points',
                     fontsize=10, 
                     fontweight='bold')
                     
    plt.title("Cost-Accuracy Trade-off (The Routing Problem)", fontweight="bold", fontsize=14)
    plt.xlabel("Average Latency per Query (seconds)", fontweight="bold")
    plt.ylabel("Accuracy", fontweight="bold")
    plt.xscale('log') # Log scale is crucial for ToT latency vs Direct latency
    
    # Add a pseudo-pareto front line
    plt.tight_layout()
    plt.savefig("reports/figures/cost_accuracy_tradeoff.png", dpi=300)
    plt.close()
    
    # 3. Accuracy by Dataset
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x='dataset',
        y='correct',
        hue='policy',
        hue_order=order,
        palette="viridis",
        errorbar=None
    )
    plt.title("Policy Performance Breakdown by Domain", fontweight="bold", fontsize=14)
    plt.ylabel("Accuracy", fontweight="bold")
    plt.xlabel("Dataset", fontweight="bold")
    plt.legend(title='Policy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("reports/figures/dataset_breakdown.png", dpi=300)
    plt.close()
    
    print("Successfully generated ICLR visuals in reports/figures/")

if __name__ == "__main__":
    generate_visuals()
