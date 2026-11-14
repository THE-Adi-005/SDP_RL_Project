# CascadeRoute-VL: Taming the Reasoning Tax in Vision-Language Models via Online Contextual Bandit Routing

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-red.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-50%2F50%20passed-brightgreen.svg)](tests/)
[![Target](https://img.shields.io/badge/Target-CVPR%202027-purple.svg)](cvpr2027/)

> **Title:** *CascadeRoute-VL: Taming the Reasoning Tax in Vision-Language Models via Online Contextual Bandit Routing*  
> **Keywords:** Vision-Language Models, Model Routing, Contextual Bandits, Multimodal Reasoning, Test-Time Compute Scaling, Efficient Inference  
> **Manuscripts:** [`cvpr2027/CascadeRoute_VL_CVPR2027.tex`](cvpr2027/CascadeRoute_VL_CVPR2027.tex) & [`cvpr2027/Supplementary.tex`](cvpr2027/Supplementary.tex)

---

## Core Discoveries & Verified Experimental Results (`30,540` Real Queries, `14` Benchmarks, `17` VLMs)

All quantitative claims in this repository are measured directly on **VL-RouterBench** (`30,540` queries $\times$ `17` contemporary VLMs across `238` raw inference transcripts; splits `Train=21,378`, `Dev=3,054`, `Test=6,108`), plus a live 4-bit NF4 CUDA hardware evaluation (`600` runs on `Qwen3-VL-2B-Instruct` & `-Thinking` across `12` configurations and unseen external `AMBER`), without using any oracle task/benchmark labels:

1. **The Multimodal Reasoning Tax & Chance-Corrected Verification (`N=30,540`):**
   - Across the full matrix (`N=30,540`), the strongest static monolithic model (**`Qwen2.5-VL-72B`**) reaches **78.21%** (Held-Out Test `N=6,108`: **77.29%**; General+STEM In-Dist Test `N=2,523`: **69.24%**) and fails on **6,655** queries.
   - An oracle selector across the 17-VLM pool achieves **95.45%** accuracy (**+17.24% raw gap** vs. **99.99%** independent-chance union control). Genuine complementarity is verified via failure rescues: strictly **dense $\le 8\text{B}$** models solve **63.2%** (`4,209/6,655` all; **57.6%** strictly Non-MCQ) of 72B failures, rising to **68.5%** all / **63.6%** Non-MCQ including 3B-active/16B-total MoE `Kimi-VL-A3B`.
   - On `Charts_OCR`, compact `Qianfan-VL-8B` reaches **86.48%** on `ChartQA` (1.2 out tok), **87.42%** on `DocVQA` (2.7 tok), and **85.70%** on `OCRBench` (4.6 tok), whereas `MiMo-VL-7B-RL`'s upstream `0.0%` OCR score in raw `Y.npz` is an unstripped `<think>` parsing failure that recovers to **74.16%–88.67%** (`35.83% -> 77.76%` global) once `<think>` is stripped (`Y_repaired.npz`).

2. **Held-Out Test Performance & Matched-Cost Pareto (`N=6,108`, 5 Seeds, 5-Split CV, 10k Bootstrap CIs):**
   - **`CascadeRoute-VL (Balanced, gamma=0.08, 896-d default)`:** **78.59% ± 0.12%** (5-Split CV: **79.56% ± 0.56%**) at **\$0.1598 / 1k queries** — **+1.20%** seed-42 paired bootstrap over `Qwen2.5-VL-72B` (95% CI `[+0.23, +2.14]`, `p=0.0152`, **38.7% USD saved** `p=1e-4`), beating `Per-Domain Best Static` (`+0.38%`, `-\$0.0283/1k`), `Cold LinUCB` (`72.68%`), `NeuralUCB` (`75.97%`, `+3.39%`), and `NeuralTS` (`75.02%`, `+2.78%`), with Cost-AUC **78.15%** and Lat-AUC **76.96%**.
   - **`CascadeRoute-VL (Frugal, gamma=0.30)`:** **76.13% ± 0.21%** at **\$0.0876 / 1k** (**66.4% USD saved**, **2.38× speedup** vs. 72B).
   - **Deterministic Horizon Reserve (Proposition 1 vs. BwK):** Enforces $C_T \le B_{\max} + c_{\max}$ with **0/6 violations** and **0.00% soft overrun** across `0.25x..1.25x B_max`, beating `BwK Primal-Dual` (5/6 overruns) on accuracy at every tier. Clamped regret bounded by Theorem 1.

3. **Unified-$w_0=0.20$ OOD Shift, Continuous Drift, & Hot Onboarding (`N_OOD=17,601`):**
   - Trained only on `General_VQA + STEM`, frozen offline routers collapse on unseen `Charts_OCR + Hallucination` (`MLP-Router` `-15.80%` to `56.97%`; `ARMS` `-9.48%` to `66.58%`).
   - Cutting 8 OOD-fragile heuristic cues (`896-d` vs `904-d`) boosts OOD accuracy by **+2.41%** (`p=0.0002`); unified-$w_0=0.20$ `CascadeRoute-VL` reaches **81.59%** (`32.0%` cheaper & `1.35×` faster than 72B) and `D-LinUCB (beta=0.996)` reaches **80.54%** (`44.3%` cheaper & `1.46×` faster than 72B).
   - Survives 10k-query Gradual Drift (**79.45%** vs MLP `64.75%`), Repeated Switches (**78.30%** vs MLP `65.58%`), `3×` API Price Shocks (`18.1%` vs `28.7%` expensive picks), and live Test-Time Onboarding (`Kimi-Think` added at `t=500`: first pick at `0` queries post-insert, `62.0% -> 4.9%` steady-state specialization vs `0.0%` frozen).

4. **Calibrated Gate, Live 12-Config GPU Routing, Partial/Noisy Feedback, & OPE Audit:**
   - Dev-calibrated isotonic/conformal gate ($\tau_{\text{cal}}=0.04$) is neutral on the unconstrained 17-arm pool (`+0.02%`) while delivering **+8.69%** accuracy (`+13.82%/$`) in Compact-Primary (`<=8B`) -> Paid Escalation tiers (isolating `both_arms` vs `final_only` feedback).
   - **Controlled Live 4-Bit CUDA Multi-Config Study (`Qwen3-VL-2B`, `RTX 4050`, `N=600` runs):** Same-backbone `-Thinking` drops accuracy `52.0% -> 42.0%` (`-10.0%`) at `6.24×` tokens and `100×` p50 latency (`0.430s -> 43.065s`). Expanding `2` mode arms $\to$ `6` token budgets (`B32/128/512`) $\to$ `12` `(mode × resolution × budget)` arms raises 5-fold Pareto Latency-AUC **`45.38% -> 55.66%` (`+10.28%`)**, cuts matched latency `2.84×`, and reaches **`90.0% @ 0.350s`** on unseen external **`AMBER`** (`38.3×` faster than static Thinking). Combined 3-encoder + LinUCB overhead is **`52.66ms` p50 / `68.21ms` p95** (`OLS` latency calibration $r \ge 0.9894$, `3.5%` Thinking MAPE).
   - Beats frozen MLP on unseen Leave-One-Dataset-Out tasks, transfers across disjoint arm pools A(9)->B(8) (`48.7%` USD saved vs cold), holds `80.22%` OOD at 25% partial feedback and `78.51%` under 20% judge noise, and validates both stationary and **Online Adaptive Trajectory** OPE in Supplementary Sec. S4 (`10/10` 95% CI coverage, `ESS=397.7`).

---

## System Architecture

```text
                  ┌─────────────────────────────────────────────┐
                  │  Unlabelled Query (Image I_t, Question Q_t) │
                  └──────────────────────┬──────────────────────┘
                                         ▼
           ┌──────────────────────────────────────────────────────────┐
           │ Domain-Label-Free 896-d State (||s_t||_2 <= 0.743 < 1)   │
           │  • Linguistic (384-d): all-MiniLM-L6-v2 (L2-norm, w_q=.50)│
           │  • Dual-Scale Vision (512-d): DINOv2-ViT-S/14 (384-d)    │
           │    + Orthogonal JL ResNet-50 (128-d)   (L2-norm, w_v=.55)│
           │  • Optional 8-d Cues (904-d ablation; cut by default)    │
           └─────────────────────────────┬────────────────────────────┘
                                         ▼
           ┌──────────────────────────────────────────────────────────┐
           │  Warm-Started & Discounted Disjoint LinUCB + Guardrail   │
           │  • Ridge Prior: A_a^(0) = λI + w_0 Σ s_i s_i^T           │
           │  • Clamped Annealing: α_t = max(α_min, α_0 b_t)          │
           │  • Horizon Reserve (Thm 2): C_T <= B_max + c_max (0/6)   │
           │  • O(d^2) Sherman-Morrison + Beta-Discounted Tracking    │
           └─────────────────────────────┬────────────────────────────┘
                                         ▼
           ┌──────────────────────────────────────────────────────────┐
           │   Dev-Calibrated Deferral Gate (Regime-Scoped)           │
           │  • Single isotonic/conformal threshold τ_cal = 0.04      │
           │  • Compact-Primary -> Paid Escalation (+8.69%, +13.8%/\$)│
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

### 2. Verify Unit Tests (`50/50` Passing)
```bash
pytest tests/ -v
```

### 3. Reconstruct Real Matrices & Evaluate All 20 Baselines / Variants
*(Pre-built real matrices `Y.npz`, `C.npz`, `O_tok.npz`, `T_tot.npz`, `L.npz`, and compressed `X_multimodal_904_f16.npz` are already included in `data/vl_routerbench_processed/matrices/`. Re-running feature extraction from scratch requires the raw VL-RouterBench tarball downloaded via `scripts/download_vlrouterbench.py`.)*

```bash
# Run complete real 30,540-query base benchmark & Pareto/OOD/OPE scripts
python scripts/adapt_real_vlrouterbench.py

# Run Claim 1-8 & cross-cutting experimental suite (30,540-query + 600 live CUDA runs):
python scripts/exp1_claim1_oracle_and_mimo_repair.py
python scripts/exp2_live_vlm_configurations.py 10
python scripts/exp3_bandits_budgets_gates.py
python scripts/exp4_ood_drift_onboarding_ope_generalization.py

# Regenerate all 7 CVPR publication figures in reports/figures/
python scripts/generate_cvpr_visuals.py
```

Key generated outputs in `reports/tables/`:
- Base matrix & Pareto/OOD tables: `real_table1_main_benchmark.csv` .. `real_table5_ips_counterfactual.csv`
- Claim 1 & 2 / Live GPU (`exp1_*.csv`, `exp2_*.csv`): `exp1_chance_corrected_oracle.csv`, `exp1_mimo_parser_repair.csv`, `exp1_rescue_nonmcq_and_dense_compact.csv`, `exp2_same_backbone_qwen3vl.csv`, `exp2_multiconfig_pareto.csv`, `exp2_amber_external_eval.csv`, `exp2_real_latency_and_sim_cal.csv`, `exp2_router_overhead.csv`, `exp2_raw_live_traces.csv`
- Claim 3–5 (`exp3_*.csv`): `exp3_claim3_baselines_and_splits.csv`, `exp3_5splits_summary.csv`, `exp3_claim3_bootstrap_ci.csv`, `exp3_claim3_modality_ablation_boot.csv`, `exp3_claim4_alpha_sensitivity.csv`, `exp3_claim4_budget_compliance.csv`, `exp3_claim5_calibrated_gate.csv`, `exp3_claim5_tau_sensitivity.csv`
- Claim 6–7 & Cross-Cutting (`exp4_*.csv`): `exp4_claim6_ood_unified_w0.csv`, `exp4_claim6_continuous_drift.csv`, `exp4_claim6_hot_onboarding.csv`, `exp4_claim7_ope_audit.csv`, `exp4_cross_lodo_benchmarks.csv`, `exp4_cross_pool_transfer.csv`, `exp4_cross_partial_and_noisy_fb.csv`

Generated figures in `reports/figures/`:
- `fig1_reasoning_tax_teaser.png`
- `fig2_architecture.png`
- `fig3_pareto_frontier.png`
- `fig4_ood_shift_adaptation.png`
- `fig5_ablations_and_tokens.png`
- `fig6_budgets_and_gates.png`
- `fig7_live_configs_and_drift.png`

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
│   ├── figures/                          # fig1..fig7 PNGs + qualitative_raw/ images
│   └── tables/                           # All real_*.csv & exp1..exp4_*.csv outputs
├── scripts/                              # Evaluation, stats, feature & visual pipelines
├── src/                                  # Core router, bandit, gate & evaluation modules
│   ├── cascaderoute/                     # confidence_gate, budget_tracker, runner
│   ├── features/                         # multimodal_extractor.py, extractor.py
│   ├── rl/                               # linucb, neural_bandit, arms_router, frugalgpt, ...
│   └── evaluation/                       # statistical_tests.py, regret.py, evaluator.py
├── tests/                                # 50 pytest unit tests (100% passing)
├── vl_routerbench_code/                  # Upstream VL-RouterBench reference utilities
├── REVIEWER_RESPONSE_AND_CHANGES_LOG.md  # Complete Claim 1-8 & cross-cutting log
├── CASCADE_ROUTE_VL_MASTER_METHODOLOGY.md# Self-contained mathematical & technical reference
├── EXECUTION_LOG.md                      # Chronological execution & experiment log
└── requirements.txt                      # Python package requirements
```

---

## Theoretical Guarantees (Proven in `cvpr2027/Supplementary.tex`)
- **Lemma 1 (Bounded Context Space):** Analytical bound $\|s_t\|_2 \le 0.7433 < 1$ ($896$-d default) and $\le 0.9972 < 1$ ($904$-d ablation; empirically $\min 0.7611$, mean $0.8116$, max $0.8943$).
- **Lemma 2 (Sub-Gaussian Reward Noise):** Bounded reward $r_{t,a} \in [-2.5\gamma, 1]$ gives conditional $\nu$-sub-Gaussian noise with $\nu = (1 + 2.5\gamma)/2$ ($\nu = 0.60$ at $\gamma = 0.08$).
- **Theorem 1 (Clamped-Annealed Disjoint Regret & OOD Prior Decay):** Under $\alpha_t = \max(\alpha_{\min}, \alpha_0 b_t)$ with $\alpha_{\min}$ satisfying the Abbasi-Yadkori ellipsoid width ($\alpha_{\min}^{\text{theory}} = 42.67$ for $d{=}904, K{=}17, T{=}6108, \delta{=}0.05$; empirically tuned $\alpha_0{=}0.10, \alpha_{\min}{=}0.01$), cumulative pseudo-regret satisfies $R_T = \tilde{\mathcal{O}}(d\sqrt{KT})$ while source bias decays as $\mathcal{O}(w_0 \Delta_{\text{shift}} / (\lambda + n_{t,a}))$.
- **Theorem 2 (Deterministic Budget Feasibility):** Horizon-reserve cheapest-arm fallback $a_{\min}$ guarantees $C_T \le B_{\max} + c_{\max}$ deterministically ($0/6$ violations vs.\ Bandits-with-Knapsacks across $0.25\times$–$1.25\times B_{\max}$).
