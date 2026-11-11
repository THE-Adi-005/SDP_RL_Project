# CascadeRoute-VL: Taming the Reasoning Tax in Vision-Language Models via Online Contextual Bandit Routing

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-red.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-45%2F45%20passed-brightgreen.svg)](tests/)
[![Target](https://img.shields.io/badge/Target-CVPR%202027-purple.svg)](cvpr2027/)

> **Title:** *CascadeRoute-VL: Taming the Reasoning Tax in Vision-Language Models via Online Contextual Bandit Routing*  
> **Keywords:** Vision-Language Models, Model Routing, Contextual Bandits, Multimodal Reasoning, Test-Time Compute Scaling, Efficient Inference  
> **Manuscripts:** [`cvpr2027/CascadeRoute_VL_CVPR2027.tex`](cvpr2027/CascadeRoute_VL_CVPR2027.tex) & [`cvpr2027/Supplementary.tex`](cvpr2027/Supplementary.tex)

---

## Core Discoveries & Verified Experimental Results (`30,540` Real Queries, `14` Benchmarks, `17` VLMs)

All quantitative claims in this repository are measured directly on **VL-RouterBench** (`30,540` queries $\times$ `17` contemporary VLMs across `238` raw inference transcripts; splits `Train=21,378`, `Dev=3,054`, `Test=6,108`), with strictly zero benchmark-label leakage and zero test peeking:

1. **The Multimodal Reasoning Tax (+17.24% Untapped Gap):**
   - Across all `30,540` queries, the strongest static monolithic model (**`Qwen2.5-VL-72B`**) reaches **78.21%** overall accuracy and fails on **6,655** queries.
   - An oracle selector across the 17-VLM pool achieves **95.45%** accuracy (**+17.24% gap**), while **compact models ($\le 8\text{B}$) rescue 4,557 of those 6,655 failures (68.5%)**—led by `Kimi-VL-A3B-Thinking` (`2,324`), `Qianfan-VL-8B` (`1,789`), and `DeepSeek-VL2-Tiny` (`1,689`).
   - On `Charts_OCR`, compact `Qianfan-VL-8B` reaches **86.48%** on `ChartQA` (1.2 out tok), **87.42%** on `DocVQA` (2.7 tok), and **85.70%** on `OCRBench` (4.6 tok), whereas explicit reasoning loops (`MiMo-VL-7B-RL`, emitting `<think>` on 100% of queries, mean `237.4` tok) degrade accuracy on OCR/hallucination or hit downstream extraction collapse (`0.0%` extracted on 4 OCR datasets).

2. **Held-Out Test Performance (`N=6,108`, 5 Seeds, `17` Arms):**
   - **`CascadeRoute-VL (Balanced, gamma=0.08)`:** **78.59% ± 0.12%** accuracy at **\$0.1598 / 1k queries** — **+1.30%** over monolithic `Qwen2.5-VL-72B` (`77.29%` at `\$0.2605`) with **38.7% USD saved** (`p = 9.49e-275`), **1.26×** latency speedup (`1.550s` vs `1.959s`), and Bonferroni-significant gains over `Cold LinUCB` (`+5.91%`, `p = 1.07e-22`), `MHR-Router` (`+7.11%`, `p = 5.17e-32`), `DQN-Router` (`+5.63%`), and `FrugalGPT-Cascade` (`+5.36%`, `p = 3.34e-25`).
   - **`CascadeRoute-VL (Frugal, gamma=0.30)`:** **76.13% ± 0.21%** accuracy at **\$0.0876 / 1k queries** and **0.824s** latency — **66.4% USD saved**, **2.38× speedup** vs `72B`, and `+2.90%` above `FrugalGPT-Cascade` (`73.23% @ \$0.1704`) at nearly half the cost.

3. **Zero-Leakage OOD Distribution Shift (`General_VQA + STEM` $\to$ Unseen `Charts_OCR + Hallucination`, `N=17,601`, 5 Seeds):**
   - Static offline supervised routers suffer fatal non-stationary collapse: `MLP-Router` plunges **-15.80%** (`72.77% -> 56.97%`), `ARMS-Router` drops `-9.48%` (`76.06% -> 66.58%`), and `Offline-Linear` drops `-8.24%` (`73.64% -> 65.41%`) with zero recovery.
   - `CascadeRoute-VL` adapts continuously via online Sherman-Morrison updates to **80.68%** OOD accuracy (**+4.70%** shift gain, **+23.71%** above OOD `MLP-Router`), surpassing `MLP-Router` within **`~51` unseen queries** (4× faster than `Cold LinUCB`'s `~191` queries).

4. **Emergent Zero-Label Domain Specialization & Unbiased OPE:**
   - Despite receiving **zero** dataset or domain identifiers, `CascadeRoute-VL` directs **63.2%** of `STEM` traffic (`259.6` mean out tok) to thinking specialists (`Kimi-3B-Think` `36.79%`, `MiMo-VL-RL` `26.36%`) while reducing `MiMo-VL-RL` to **0.09%** on `Charts_OCR` and shifting `44.7%` of queries to `Qianfan-8B` / `DeepSeek-VL2` (`4.9` mean out tok) — a **53× dynamic compute scaling factor**.
   - **Counterfactual OPE (`1,000` bootstrap resamples):** Against stationary warm-start target accuracy `0.7878`, IPS estimates `0.8356 [0.7332, 0.9367]`, SNIPS estimates **`0.7919 [0.7461, 0.8386]`**, and Doubly Robust (DR) estimates **`0.7833 [0.7395, 0.8281]`** (`ESS=306.4`).

---

## System Architecture

```text
                  ┌─────────────────────────────────────────────┐
                  │  Unlabelled Query (Image I_t, Question Q_t) │
                  └──────────────────────┬──────────────────────┘
                                         ▼
           ┌──────────────────────────────────────────────────────────┐
           │ Zero-Leakage 904-d Context Extractor (||s_t||_2 <= 0.894)│
           │  • Linguistic (384-d): all-MiniLM-L6-v2 (L2-norm, w_q=.50)│
           │  • Dual-Scale Vision (512-d): DINOv2-ViT-S/14 (384-d)    │
           │    + Orthogonal JL ResNet-50 (128-d)   (L2-norm, w_v=.55)│
           │  • Unsupervised Cues (8-d in [0,1]^8, w_p=0.235):        │
           │    Entropy, Sobel edge density, log aspect ratio, log    │
           │    area, white-bg ratio, words, symbolic/MCQ cue, sents  │
           └─────────────────────────────┬────────────────────────────┘
                                         ▼
           ┌──────────────────────────────────────────────────────────┐
           │   Warm-Started Disjoint LinUCB (17 Real VLM Arms)        │
           │  • Ridge Prior: A_a^(0) = λI + w_0 Σ s_i s_i^T           │
           │  • Budget Annealing: α_t = α_0 · clip((B_max-C)/B_max)   │
           │  • Sherman-Morrison O(d^2) Online Covariance Updates     │
           └─────────────────────────────┬────────────────────────────┘
                                         ▼
           ┌──────────────────────────────────────────────────────────┐
           │   Confidence-Guided Escalation Gate (Compact -> Pool F)  │
           │  • Triggers iff b_t > 0.15, μ_a0 < p_cap, Δ_eff >= τ_m,  │
           │    and risk score ρ_t >= τ_u (zero oracle peeking)       │
           └──────────────────────────────────────────────────────────┘
```

---

## Setup & Reproduction Guide

### 1. Environment Installation
```bash
git clone https://github.com/THE-Adi-005/SDP_RL_Project.git
cd SDP_RL_Project

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Unit Tests (`45/45` Passing)
```bash
pytest tests/ -v
```

### 3. Reconstruct Real Matrices & Evaluate All 20 Baselines / Variants
*(Pre-built real matrices `Y.npz`, `C.npz`, `O_tok.npz`, `T_tot.npz`, `L.npz`, and compressed `X_multimodal_904_f16.npz` are already included in `data/vl_routerbench_processed/matrices/`. Re-running feature extraction from scratch requires the raw VL-RouterBench tarball downloaded via `scripts/download_vlrouterbench.py`.)*

```bash
# Optional: Rebuild cost/token/tax stats & qualitative JSONs from raw data
python scripts/build_real_cost_and_tax_stats.py

# Run complete real 30,540-query benchmark, Pareto sweep, OOD shift, ablations, stats & OPE
python scripts/adapt_real_vlrouterbench.py

# Regenerate all 5 CVPR publication figures in reports/figures/
python scripts/generate_cvpr_visuals.py
```

Key generated outputs in `reports/tables/`:
- `real_table1_main_benchmark.csv`, `real_pareto_sweep.csv`, `real_statistical_tests.csv`
- `real_table2_distribution_shift.csv`, `real_ood_trajectory.csv`
- `real_table3_modality_ablation.csv`, `real_component_ablation.csv`
- `real_domain_breakdown.csv`, `real_arm_selection_counts.csv`
- `real_table4_14x17_accuracy.csv`, `real_table4_14x17_tokens.csv`
- `real_model_profiles.csv`, `real_perceptual_stats_by_dataset.csv`, `real_reasoning_tax_by_dataset.csv`
- `real_table5_ips_counterfactual.csv`

Generated figures in `reports/figures/`:
- `fig1_reasoning_tax_teaser.png`
- `fig2_architecture.png`
- `fig3_pareto_frontier.png`
- `fig4_ood_shift_adaptation.png`
- `fig5_ablations_and_tokens.png`

### 4. Compile CVPR 2027 Manuscripts
Using **Tectonic** (recommended single-command build):
```bash
cd cvpr2027
tectonic CascadeRoute_VL_CVPR2027.tex
tectonic Supplementary.tex
```
*(Or using standard `pdflatex` + `bibtex` twice per `.tex` file.)*

---

## Repository Layout

```text
SDP_RL_Project/
├── cvpr2027/                             # Double-blind CVPR 2027 manuscripts & style
│   ├── CascadeRoute_VL_CVPR2027.tex      # Main paper (<= 8 body pages + refs)
│   ├── Supplementary.tex                 # Full proofs (S1), 14x17 data (S2), gallery
│   ├── CascadeRoute_VL_CVPR2027.pdf      # Compiled main paper PDF
│   ├── Supplementary.pdf                 # Compiled supplementary PDF
│   ├── references.bib                    # 37 non-anonymous BibTeX citations
│   ├── cvpr.sty
│   └── ieeenat_fullname.bst
├── data/vl_routerbench_processed/        # Standardized 30,540-query VL-RouterBench data
│   ├── matrices/                         # Y.npz, C.npz, L.npz, T_tot.npz, O_tok.npz,
│   │                                     # has_think.npz, X_multimodal_904_f16.npz, P_raw_8.npz
│   ├── registry/                         # meta.parquet, model_index.pkl, sample_index.pkl
│   ├── ORACLE/                           # Ground-truth score Parquet & raw stats
│   └── SPLITS/                           # Train (21,378) / Dev (3,054) / Test (6,108)
├── reports/
│   ├── figures/                          # fig1..fig5 PNGs + qualitative_raw/ images
│   └── tables/                           # All real_*.csv benchmark & OPE outputs
├── scripts/                              # Evaluation, stats, feature & visual pipelines
├── src/                                  # Core router, bandit, gate & evaluation modules
│   ├── cascaderoute/                     # confidence_gate.py, budget_tracker.py, runner
│   ├── features/                         # multimodal_extractor.py, extractor.py
│   ├── rl/                               # linucb, arms_router, mhr_router, frugalgpt, dqn, ...
│   └── evaluation/                       # statistical_tests.py, regret.py, evaluator.py
├── tests/                                # 45 pytest unit tests (100% passing)
├── vl_routerbench_code/                  # Upstream VL-RouterBench reference utilities
├── CASCADE_ROUTE_VL_MASTER_METHODOLOGY.md# Self-contained mathematical & technical reference
├── EXECUTION_LOG.md                      # Chronological execution & experiment log
└── requirements.txt                      # Python package requirements
```

---

## Theoretical Guarantees (Proven in `cvpr2027/Supplementary.tex`)
- **Lemma 1 (Bounded Context Space):** Analytical bound $\|s_t\|_2 \le 0.9972 < 1$ under $(w_q, w_v, w_p) = (0.50, 0.55, 0.235)$; empirically verified across all `30,540` queries ($\min 0.7611$, mean $0.8116$, max $0.8943$).
- **Lemma 2 (Sub-Gaussian Reward Noise):** Bounded reward $r_{t,a} \in [-2.5\gamma, 1]$ gives conditional $\nu$-sub-Gaussian noise with $\nu = (1 + 2.5\gamma)/2$.
- **Theorem 1 (Disjoint Multimodal Regret Bound & OOD Prior Decay):** Self-normalized confidence ellipsoids + elliptical potential lemma establish high-probability cumulative regret $R_T = \tilde{\mathcal{O}}(d\sqrt{KT})$, while offline warm-start source bias decays as $\mathcal{O}(w_0 \Delta_{\text{shift}} / (\lambda + n_{t,a}))$ during online OOD adaptation.
- **Theorem 2 (Asymptotic Budget Feasibility):** Annealed exploration $\alpha_t = \alpha_0 b_t$ with escalation guard $b_t > 0.15$ collapses exploration and escalation cleanly as $b_t \to 0^+$.
