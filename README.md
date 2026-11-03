# CascadeRoute-VL: Taming the Reasoning Tax in Vision-Language Models via Online Contextual Bandit Routing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-red.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-42%2F42%20passed-brightgreen.svg)](tests/)
[![Target](https://img.shields.io/badge/Target-CVPR%202027-purple.svg)](cvpr2027/)

> **Paper Title:** CascadeRoute-VL: Taming the Reasoning Tax in Vision-Language Models via Online Contextual Bandit Routing  
> **Target Track:** CVPR 2027 (Computer Vision and Pattern Recognition)  
> **Key Deadlines:** Abstract: November 10, 2026 | Full Paper: November 16, 2026 | Supplementary: November 23, 2026

---

## 🔬 Core Discoveries & Research Highlights

1. **The VLM Reasoning Tax:** On perceptual hallucination benchmarks (POPE) and fine-grained OCR (DocVQA), smaller direct-perception models consistently outperform heavy reasoning paradigms (Tree-of-Thought, Visual CoT) by up to **20.0%** while being **13.1× cheaper**. Excessive visual chain-of-thought causes frontier VLMs to substitute real visual evidence with linguistic priors.
2. **Empirical Validation on 30,540 Real Benchmark Queries:**
   - Single Best Frontier Model (`Qwen2.5-VL-72B-Instruct`): **78.21%** overall accuracy across 17 models.
   - Oracle Upper Bound: **95.45%** accuracy (**+17.24% routing opportunity gap**).
   - **Smaller Models Rescue Frontier Failures:** Out of 6,655 failure queries of the 72B model, smaller models (`Kimi-VL-A3B`, `deepseek_vl2_tiny`, `SmolVLM2`, `Janus-Pro-1B`) solve **5,266 (79.1%)** of them!
3. **Offline Routers Suffer Fatal Non-Stationary Collapse:** Offline classifiers (e.g. VL-RouterBench linear routers) drop by **-49.5%** under distribution shift (General VQA $\to$ Charts/OCR) with zero recovery. CascadeRoute-VL's online contextual bandit updates recover within $\sim 49$ queries.
4. **Visual Resolution as a 3D Action Dimension:** Expanding routing from (Model $\times$ Strategy) to (Model $\times$ Strategy $\times$ Resolution) yields a **+7.71% accuracy boost**, proving visual resolution is essential in VLM routing.

---

## 🏛️ System Architecture

```
                    ┌────────────────────────────────────────┐
                    │ Multimodal Query (Image I_t, Text q_t) │
                    └───────────────────┬────────────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌───────────────────────────────┐               ┌─────────────────────────────────────┐
│ 904-dim Joint Context Extractor│               │ Visual Perturbation Confidence Gate │
│ - SigLIP Visual (512-dim)     │               │ - Mild Gaussian / Crop perturbation │
│ - MiniLM Semantic (384-dim)   │               │ - Overthinking Score: S_OT          │
│ - Perceptual Entropy & Edges  │               │ - Chain-of-Responsibility Escalation│
└────────────┬──────────────────┘               └──────────────────┬──────────────────┘
             │                                                     │
             ▼                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      Online Contextual Bandit Engine (LinUCB)                       │
│  - Ridge Regression with O(d^2) Sherman-Morrison rank-1 recursive inverse           │
│  - Dynamic Budget-Annealing Exploration: α_t = α_0 · b_t                            │
│  - Hybrid Warm-Start from Offline Logs: Zero Cold-Start Regret                      │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
             ┌───────────────────────────────────────────────────────────┐
             │ 8-Arm Multimodal Action Space (Scale × Strategy × Res)    │
             │ - 2B Direct Low-Res       - 7B Direct High-Res            │
             │ - 2B Direct High-Res      - 7B VCoT High-Res              │
             │ - 2B VCoT High-Res        - 7B Graph-of-Thought (GoT)     │
             │ - 7B Direct Low-Res       - 7B Tree-of-Thought (ToT)      │
             └───────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup & Execution Guide (Lab System)

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/THE-Adi-005/SDP_RL_Project.git
cd SDP_RL_Project

# Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify Test Suite
Run the 42 unit tests covering all bandit algorithms, baselines, and feature extractors:
```bash
pytest tests/ -v
# Output: 42 passed in ~10s
```

### 3. Run Benchmark Evaluation
#### Option A: Full Live 30,540 Real Data Benchmark
Evaluates CascadeRoute-VL against 16 baselines on the official VL-RouterBench records:
```bash
python scripts/adapt_real_vlrouterbench.py
```
Outputs generated in `reports/tables/`:
- `real_table1_main_benchmark.csv`: Full comparative metrics (Accuracy, Tokens, Latency, Cost).
- `real_table2_distribution_shift.csv`: In-domain vs OOD Charts/OCR degradation.
- `real_table3_modality_ablation.csv`: Text-only vs Vision-only vs Multimodal.
- `real_table5_ips_counterfactual.csv`: Counterfactual IPS and SNIPS with 95% bootstrap CI.

#### Option B: CVPR Master Simulation Suite
```bash
# Main 14-benchmark comparative suite
python scripts/run_vl_cascade_suite.py --mode main

# Distribution shift test
python scripts/run_vl_cascade_suite.py --mode distribution_shift

# Action space ablation (3D vs 2D vs 1D)
python scripts/run_vl_cascade_suite.py --mode arm_ablation

# Reasoning tax heatmap
python scripts/run_vl_cascade_suite.py --mode reasoning_tax

# IPS off-policy counterfactual evaluation
python scripts/run_vl_cascade_suite.py --mode ips_ope
```

### 4. Generate Publication Figures
Render camera-ready CVPR 2027 visual figures:
```bash
python scripts/generate_cvpr_visuals.py
```
Generated figures in `reports/figures/`:
- `fig1_reasoning_tax_heatmap.png`: High-resolution 14×8 dataset heatmap illustrating overthinking degradation.
- `fig2_pareto_frontier.png`: Accuracy vs Token Expenditure Pareto Frontier.
- `fig6_distribution_shift.png`: In-domain vs Out-of-Distribution degradation bar chart.

### 5. Compile CVPR 2027 Manuscript
The full 8-page double-blind submission draft with proofs in the Supplementary Appendix:
```bash
cd cvpr2027
pdflatex CascadeRoute_VL_CVPR2027.tex
bibtex CascadeRoute_VL_CVPR2027
pdflatex CascadeRoute_VL_CVPR2027.tex
pdflatex CascadeRoute_VL_CVPR2027.tex
```

---

## 📂 Repository Structure

```
SDP_RL_Project/
├── cvpr2027/                   # CVPR 2027 Paper Manuscript & Proofs
│   ├── CascadeRoute_VL_CVPR2027.tex
│   └── cvpr.sty
├── data/
│   └── vl_routerbench_processed/  # Preprocessed benchmark matrices (30,540 samples)
│       ├── BENCHMARKS/             # Per-task prompt & sample records
│       ├── ORACLE/score/           # 14 benchmark Parquet score files
│       ├── SPLITS/                 # Train / Dev / Test indices
│       ├── registry/               # Metadata & model indices
│       └── matrices/               # Y.npz (accuracy), C.npz (cost), X_multimodal_904_f16.npz
├── reports/
│   ├── tables/                 # Publication benchmark CSV tables
│   └── figures/                # Publication-quality rendered figures
├── scripts/
│   ├── adapt_real_vlrouterbench.py   # Live 30,540 real benchmark evaluation
│   ├── run_vl_cascade_suite.py       # Master simulation & ablation harness
│   ├── generate_cvpr_visuals.py      # High-res publication figure generator
│   └── download_vlrouterbench.py     # Script to fetch raw Hugging Face data
├── src/
│   ├── cascaderoute/           # Confidence gating & visual perturbation logic
│   ├── features/               # 904-dim Multimodal context extractor
│   ├── rl/                     # Bandit & RL agents (LinUCB, Thompson, DQN, ARMS, MHR, FrugalGPT)
│   ├── evaluation/             # Regret, statistical significance, and metric suites
│   └── llm/                    # Local VLM clients (Ollama / HuggingFace)
├── tests/                      # 42 Unit tests (100% passing)
├── vl_routerbench_code/        # Adapted VL-RouterBench benchmark tools
├── EXECUTION_LOG.md            # Detailed chronological research & engineering log
└── requirements.txt            # Python dependencies
```

---

## 📜 Theoretical Proofs (Included in Paper Appendix)
- **Theorem 1 (Regret Bound):** Proves that CascadeRoute-VL with multimodal state dimension $d=904$ satisfies sublinear regret $R(T) = \mathcal{O}(d \sqrt{T \ln T})$.
- **Theorem 2 (Budget Feasibility):** Proves that under dynamic exploration annealing $\alpha_t = \alpha_0 \cdot b_t$, total token expenditure never exceeds budget $B$ with high probability ($1 - \delta$).

---

## 🤝 Citation & Contact
For research questions, please contact the author at `adithyakrs2408@gmail.com`.

