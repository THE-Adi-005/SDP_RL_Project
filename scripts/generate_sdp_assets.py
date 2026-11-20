import os
import matplotlib.pyplot as plt
import numpy as np

# Output directories
os.makedirs("SDP_Ppt", exist_ok=True)
os.makedirs("SDP/report/figures", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# 1. Architectural Inference Latency (sdp_latency_improvement.png)
# -------------------------------------------------------------
def plot_latency():
    labels = ['Static 7B-ToT\n(Sequential)', 'Static 7B-CoT', 'AdaptInfer\n(Ray Distributed)', 'Static 3B-Direct']
    latencies = [45.2, 17.4, 14.8, 4.4]
    colors = ['#c0392b', '#e67e22', '#27ae60', '#2980b9']

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    bars = ax.bar(labels, latencies, color=colors, width=0.55, edgecolor='#2c3e50', linewidth=1.2)
    
    ax.set_title('AdaptInfer: Architectural Inference Latency Optimization', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Median Response Latency (Seconds)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 52)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.2,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Annotation arrow showing 67.2% reduction
    ax.annotate('67.2% Latency Reduction\n(Slashed from 45.2s to 14.8s)',
                xy=(2, 16.5), xytext=(0.8, 36),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=2, headwidth=8),
                fontsize=10.5, fontweight='bold', color='#1e8449',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#eafaf1', edgecolor='#27ae60', lw=1.2))

    plt.tight_layout()
    plt.savefig('SDP_Ppt/sdp_latency_improvement.png', dpi=300)
    plt.savefig('SDP/report/figures/sdp_latency_improvement.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 2. Token Cost & Systems Efficiency (sdp_token_savings.png)
# -------------------------------------------------------------
def plot_token_savings():
    policies = ['Static 3B-Direct', 'Static 7B-CoT', 'AdaptInfer (Ours)', 'Static 7B-ToT']
    tokens = [108, 379, 1421, 1959]
    colors = ['#7f8c8d', '#f39c12', '#2980b9', '#c0392b']

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=300)
    bars = ax.bar(policies, tokens, color=colors, width=0.55, edgecolor='#2c3e50', linewidth=1.2)
    
    ax.set_title('AdaptInfer: Architectural Token Efficiency (Cost Reduction)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Average Tokens per Query', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 2300)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 45,
                f'{height:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Token savings annotation
    ax.annotate('27.5% Token Savings\n(-538 Tokens / Query vs ToT)',
                xy=(2, 1500), xytext=(1.05, 1850),
                arrowprops=dict(facecolor='#2980b9', shrink=0.08, width=2, headwidth=8),
                fontsize=10.5, fontweight='bold', color='#1b4f72',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ebf5fb', edgecolor='#2980b9', lw=1.2))

    plt.tight_layout()
    plt.savefig('SDP_Ppt/sdp_token_savings.png', dpi=300)
    plt.savefig('SDP/report/figures/sdp_token_savings.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 3. Strategy Comparison (Accuracy vs Latency Matrix)
# -------------------------------------------------------------
def plot_strategy_comparison():
    strategies = ['3B-Direct', '3B-CoT', '7B-Direct', '7B-CoT', '7B-GoT', '7B-ToT', 'AdaptInfer\n(Adaptive)']
    accuracies = [53.9, 64.2, 67.5, 80.0, 80.8, 82.0, 81.4]
    colors = ['#95a5a6', '#bdc3c7', '#3498db', '#2980b9', '#9b59b6', '#e74c3c', '#27ae60']

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    bars = ax.bar(strategies, accuracies, color=colors, width=0.55, edgecolor='#2c3e50', linewidth=1.2)
    
    ax.set_title('Accuracy Comparison Across Reasoning Strategies (3,712 Queries)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Evaluation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(82.0, color='#e74c3c', linestyle=':', linewidth=1.5, label='7B-ToT Upper Bound (82.0%)')
    ax.axhline(81.4, color='#27ae60', linestyle='--', linewidth=1.5, label='AdaptInfer Parity (81.4%)')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', frameon=True, fontsize=10)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.8,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig('SDP_Ppt/sdp_strategy_comparison.png', dpi=300)
    plt.savefig('SDP/report/figures/sdp_strategy_comparison.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 4. Workload Dataset Breakdown (sdp_dataset_distribution.png)
# -------------------------------------------------------------
def plot_dataset_breakdown():
    labels = ['GSM8K (Math)\n1,319 Queries', 'LogiQA (Logic)\n1,172 Queries', 'CommonsenseQA\n1,221 Queries']
    sizes = [1319, 1172, 1221]
    colors = ['#3498db', '#e67e22', '#2ecc71']
    explode = (0.04, 0.04, 0.04)

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=300)
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                      startangle=140, colors=colors,
                                      textprops=dict(color="#2c3e50", fontweight='bold', fontsize=11),
                                      wedgeprops=dict(edgecolor='#2c3e50', linewidth=1.2))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax.set_title('AdaptInfer Evaluation Benchmark (Total: 3,712 Queries)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('SDP_Ppt/sdp_dataset_distribution.png', dpi=300)
    plt.savefig('SDP/report/figures/sdp_dataset_distribution.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 5. Systems Pareto Tradeoff: Accuracy vs Latency (sdp_system_tradeoff.png)
# -------------------------------------------------------------
def plot_system_tradeoff():
    latencies = [4.4, 8.6, 7.8, 17.4, 28.5, 45.2, 14.8]
    accuracies = [53.9, 64.2, 67.5, 80.0, 80.8, 82.0, 81.4]
    names = ['3B-Direct', '3B-CoT', '7B-Direct', '7B-CoT', '7B-GoT', '7B-ToT', 'AdaptInfer (Ours)']
    colors = ['#7f8c8d', '#95a5a6', '#3498db', '#2980b9', '#9b59b6', '#c0392b', '#27ae60']
    sizes = [120, 140, 140, 180, 200, 250, 320]

    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=300)
    
    for i in range(len(names)):
        marker = '*' if 'AdaptInfer' in names[i] else 'o'
        ax.scatter(latencies[i], accuracies[i], color=colors[i], s=sizes[i],
                   edgecolors='#2c3e50', linewidths=1.5, zorder=4, marker=marker)
        offset_y = 1.2 if i != 6 else -2.2
        offset_x = 0.8 if i != 5 else -6.5
        ax.annotate(names[i], (latencies[i] + offset_x, accuracies[i] + offset_y),
                    fontsize=10.5, fontweight='bold', color=colors[i])

    # Connecting static curve
    sorted_static = sorted([(latencies[i], accuracies[i]) for i in range(6)])
    sx, sy = zip(*sorted_static)
    ax.plot(sx, sy, linestyle='--', color='#bdc3c7', alpha=0.7, label='Static Strategy Frontier', zorder=2)

    ax.set_title('Systems Pareto Efficiency: Latency vs. Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Median Response Latency (Seconds) [Lower is Better]', fontsize=12, fontweight='bold')
    ax.set_ylabel('System Accuracy (%) [Higher is Better]', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 52)
    ax.set_ylim(50, 88)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', frameon=True, fontsize=10)

    # Highlight box for AdaptInfer
    ax.annotate('Optimal Systems Balance:\n81.4% Acc @ 14.8s Latency\n(Matches ToT @ 3x Speed)',
                xy=(14.8, 81.4), xytext=(22, 70),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=2, headwidth=8),
                fontsize=10.5, fontweight='bold', color='#1e8449',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#eafaf1', edgecolor='#27ae60', lw=1.2))

    plt.tight_layout()
    plt.savefig('SDP_Ppt/sdp_system_tradeoff.png', dpi=300)
    plt.savefig('SDP/report/figures/sdp_system_tradeoff.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 6. AdaptInfer Systems Architecture Diagram (sdp_architecture_diagram.png)
# -------------------------------------------------------------
def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=300)
    ax.axis('off')

    # Color definitions
    c_client = '#3498db'
    c_gateway = '#9b59b6'
    c_engine = '#e67e22'
    c_worker = '#2ecc71'
    c_guard = '#e74c3c'
    c_cache = '#1abc9c'

    from matplotlib.patches import FancyBboxPatch
    # Box drawing helper
    def draw_box(x, y, w, h, title, subtitle, color, text_color='white'):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                              facecolor=color, edgecolor='#2c3e50',
                              linewidth=1.8, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2., y + h*0.65, title, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=text_color, zorder=4)
        ax.text(x + w/2., y + h*0.3, subtitle, ha='center', va='center',
                fontsize=9.0, color=text_color, zorder=4)

    # 1. Client Layer
    draw_box(0.04, 0.72, 0.22, 0.18, "Client / Dashboard SPA", "Vanilla JS + Chart.js\nREST & SSE Streaming", c_client)

    # 2. API Gateway Layer
    draw_box(0.34, 0.72, 0.28, 0.18, "FastAPI Microservice", "Uvicorn ASGI Gateway (Port 8000)\n/api/analyze | /api/recommend", c_gateway)

    # 3. Cache & Budget Proxy
    draw_box(0.69, 0.72, 0.27, 0.18, "Proxy / Caching Service", "LRU Cache + BudgetTracker\nSession Token Monitoring", c_cache)

    # 4. Orchestration Engine
    draw_box(0.34, 0.38, 0.28, 0.20, "Dynamic Strategy Router", "Polymorphic Strategy Selection\nContextual Feature Vector (390-d)", c_engine)

    # 5. Confidence Guardrail (Chain of Responsibility)
    draw_box(0.69, 0.38, 0.27, 0.20, "ConfidenceGate Guardrail", "Chain of Responsibility\nSelf-Consistency (T=0.2 vs 0.8)", c_guard)

    # 6. Ray Distributed Scheduler
    draw_box(0.18, 0.08, 0.28, 0.18, "Ray Distributed Actor Pool", "Parallel LLM Task Allocation\nNon-blocking Worker Scheduling", '#34495e')

    # 7. Local Hardware Worker Engines
    draw_box(0.54, 0.08, 0.42, 0.18, "Ollama Execution Engines", "Local Workers: Qwen2.5-3B & 7B Instruct\nIsolated GPUs / Zero Cloud Jitter", c_worker)

    # Connecting arrows
    arrow_props = dict(facecolor='#2c3e50', edgecolor='#2c3e50', arrowstyle='->', lw=2.0)
    
    # Client <-> Gateway
    ax.annotate('', xy=(0.34, 0.81), xytext=(0.26, 0.81), arrowprops=arrow_props)
    ax.annotate('', xy=(0.26, 0.78), xytext=(0.34, 0.78), arrowprops=arrow_props)
    
    # Gateway <-> Cache
    ax.annotate('', xy=(0.69, 0.81), xytext=(0.62, 0.81), arrowprops=arrow_props)
    
    # Gateway -> Router
    ax.annotate('', xy=(0.48, 0.58), xytext=(0.48, 0.72), arrowprops=arrow_props)
    
    # Router <-> ConfidenceGate
    ax.annotate('', xy=(0.69, 0.48), xytext=(0.62, 0.48), arrowprops=arrow_props)
    
    # Router -> Ray Pool
    ax.annotate('', xy=(0.32, 0.26), xytext=(0.42, 0.38), arrowprops=arrow_props)
    
    # Ray Pool -> Ollama Workers
    ax.annotate('', xy=(0.54, 0.17), xytext=(0.46, 0.17), arrowprops=arrow_props)

    # Legend / Title banner
    ax.text(0.5, 0.96, "AdaptInfer: Distributed Microservices Architecture & Orchestration Pipeline",
            ha='center', va='center', fontsize=14, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    plt.savefig('SDP_Ppt/sdp_architecture_diagram.png', dpi=300)
    plt.savefig('SDP/report/figures/sdp_architecture_diagram.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_latency()
    plot_token_savings()
    plot_strategy_comparison()
    plot_dataset_breakdown()
    plot_system_tradeoff()
    plot_architecture_diagram()
    print("Successfully generated all 6 AdaptInfer SDP figures in SDP_Ppt/ and SDP/report/figures/!")
