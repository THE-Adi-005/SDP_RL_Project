# CascadeRoute-VL (CVPR 2027) — Complete Claim-by-Claim Changes & Quantitative Output Log

This document records every methodological, theoretical, empirical, and manuscript change implemented across **Claims 1–8** and **Cross-Cutting Fixes**, paired directly with the exact observed outputs saved in `reports/tables/` and integrated into `cvpr2027/CascadeRoute_VL_CVPR2027.tex` and `cvpr2027/Supplementary.tex`.

---

## Claim 1: “+17.24% oracle gap, compact models rescue 68.5%”

### Changes Implemented
1. **Chance-Corrected Control & Split Clarity (`scripts/exp1_claim1_oracle_and_mimo_repair.py`):**
   - Added a Monte Carlo independent-chance oracle control (`global` and `dataset-stratified` within per-benchmark accuracy bins) over 17 arms to test whether raw 17-arm hindsight union accuracy exceeds independent coin-flip expectation.
   - Explicitly labelled exact dataset split next to every reported number across Abstract, Introduction, Sec. 3, and Supplementary Sec. S2:
     - Full set ($N{=}30{,}540$): `Qwen2.5-VL-72B` = `78.21%`
     - Held-Out Test ($N{=}6{,}108$): `77.29%`
     - Non-MCQ Open-Ended/OCR/Math ($N{=}22{,}659$): `77.32%`
     - General+STEM In-Dist Test ($N{=}2{,}523$): `69.24%` (`73.13%` best single model in-dist)
2. **Dense $\le 8\text{B}$ vs. 3B-Active MoE (`16B` Total) & Non-MCQ Rescue Breakdown:**
   - Separated strictly dense $\le 8\text{B}$ checkpoints (`Janus-Pro-1B`, `DeepSeek-VL2-Tiny`, `SmolVLM2`, `Phi-3.5-Vision`, `LLaVA-NeXT-Vicuna-7B`, `Janus-Pro-7B`, `MiMo-VL-7B-RL`, `Qianfan-VL-8B`) from 3B-active/16B-total MoE `Kimi-VL-A3B-Thinking-2506`.
   - Computed 72B failure rescue rates separately on **All 14 Benchmarks**, **Non-MCQ Only** ($N{=}22{,}659$, 5,138 72B failures), and **MCQ Only**.
3. **Robust `<think>` Parser Repair & Upstream Bug Diagnosis (`Y_repaired.npz`):**
   - Inspected raw VLMEvalKit transcripts for `MiMo-VL-7B-RL`, stripping closed/unclosed `<think>` preambles (`strip_think_and_extract`), and audited upstream `Y.npz` scores across all 238 logs, exposing both `MiMo-VL` `<think>` extraction failures and upstream `int(ANLS < 1.0)` truncation (`SmolVLM2`, `Janus-Pro`). Evaluated routers on **both Raw `Y.npz` and Repaired matrices**.
4. **Controlled Same-Backbone Reasoning Tax Study (`scripts/exp2_live_vlm_configurations.py`):**
   - Deployed `Qwen3-VL-2B-Instruct` vs. `Qwen3-VL-2B-Thinking` live in 4-bit NF4 CUDA (`RTX 4050`) across identical prompts/resolutions/budgets.

### Observed Outputs
- **Chance-Corrected Oracle Control (`reports/tables/exp1_chance_corrected_oracle.csv`):**
  - Full Raw ($N{=}30{,}540$): Real Oracle `95.45%` vs. `78.21%` (`+17.24%` raw gap); Stratified Independent Chance = **`99.99%`** (`global 100.00%`).
  - Held-Out Test ($N{=}6{,}108$): Real Oracle `95.17%` vs. `77.29%` (`+17.88%` raw gap); Chance = **`100.00%`**.
  - Non-MCQ ($N{=}22{,}659$): Real Oracle `94.43%` vs. `77.32%` (`+17.10%`); Chance = **`99.99%`**.
  - Finding honestly stated in paper: raw 17-arm hindsight union is *smaller* than independent-chance union (`95.45% < 99.99%`) due to shared hardness across models; genuine complementarity must be evidenced by specific failure rescues and live same-backbone comparison.
- **Dense $\le 8\text{B}$ & Non-MCQ Failure Rescues (`reports/tables/exp1_rescue_nonmcq_and_dense_compact.csv`):**
  - Strictly Dense $\le 8\text{B}$ Rescues: **`63.2%`** (`4,209 / 6,655`) All Benchmarks Raw (**`57.6%`**, `2,962 / 5,138` strictly Non-MCQ Raw); **`67.6%`** All / **`63.3%`** Non-MCQ after MiMo repair; **`70.6%`** All / **`66.4%`** Non-MCQ at ANLS $\ge 0.5$.
  - Dense $\le 8\text{B}$ + 3B-MoE (`16B` total): **`68.5%`** All / **`63.6%`** Non-MCQ Raw (**`71.9%`** / **`68.0%`** MiMo-repaired).
- **Upstream Parser Repair (`reports/tables/exp1_mimo_parser_repair.csv`):**
  - `MiMo-VL-7B-RL` (`0.0%` raw in `Y.npz` explicitly labelled **unstripped `<think>` parsing failure**; `0.0–0.3%` unclosed): recovers to `ChartQA 84.72%`, `DocVQA 88.67% exact / 95.51% (ANLS>=0.5, 94.24% cont.)`, `InfoVQA 79.76% exact / 85.18% (>=0.5)`, `TextVQA 74.16% exact / 78.60% (>=0.5)` (**`35.83% -> 77.76%` global**).
  - Upstream `int(ANLS < 1.0)` truncation also flagged: `SmolVLM2 InfoVQA` (`0.0% -> 49.66% >=0.5, 37.75% cont.`), `Janus-Pro-1B/7B DocVQA` (`0.60%/1.08% -> 46.91%/52.91% >=0.5`).
- **Same-Backbone Study (`reports/tables/exp2_same_backbone_qwen3vl.csv`, 4-bit NF4 RTX 4050 CUDA, `N=50` stratified across 5 datasets):**
  - Overall `ALL` ($N{=}50$, `HighRes 392^2, B512`): `Qwen3-VL-2B-Instruct` = **`52.0%`** (`49.8` OutTok, **`0.430s`** p50) vs. `Qwen3-VL-2B-Thinking` = **`42.0%`** (`-10.0%`, `311.1` OutTok = **`6.24×` tokens**, **`43.065s`** p50 = **`100×` median slowdown**).
  - Per-dataset breakdown: `OCRBench` (`60.0% -> 70.0%`, `+10.0%`, `26.13×` tok), `ChartQA_TEST` (`60.0% -> 40.0%`, `-20.0%`, `79.11×` tok), `MathVista_MINI` (`10.0% -> 10.0%`, `2.15×` tok), `HallusionBench` (`40.0% -> 0.0%`, `-40.0%`, `174.55×` tok), and unseen external `AMBER` (`90.0% -> 90.0%`, `49.10×` tok). Confirms the Reasoning Tax exists on the **exact same backbone** without cross-model confounding.

---

## Claim 2: “dynamically selects cost-efficient VLM configurations”

### Changes Implemented
1. **Corrected Parameter Accounting:** Re-classified `Kimi-VL-A3B-Thinking-2506` everywhere as **`3B-active MoE (16B total)`**, reserving `<=8B` strictly for dense $\le 8\text{B}$ models.
2. **Multi-Configuration Arm Expansion (`scripts/exp2_live_vlm_configurations.py`):** Expanded live VLM arms to explicit **12 `(Backbone/Mode × Image Resolution × Max-Token Budget)` configurations**: `{Instruct, Thinking}` × `{LowRes 196×196, HighRes 392×392}` × `{B=32, B=128, B=512}` evaluated live in 4-bit CUDA with 5-fold cross-validation (`896-d` state).
3. **Per-Knob Pareto Gain:** Measured progressive 5-fold Pareto Latency-AUC & peak/matched-latency accuracy across: (1) Mode Only `[2 arms]` $\to$ (2) + Token Budget `[6 arms]` $\to$ (3) + Image Resolution `[12 arms]`.

### Observed Outputs
- **12-Configuration Per-Knob Pareto Expansion (`reports/tables/exp2_multiconfig_pareto.csv`, `600` live 4-bit CUDA runs, 5-Fold CV, `left_fill=0.0`):**
  1. `Mode Only (HighRes, B512) [2 arms]`: `52.0%` peak / `52.0%` matched ($\gamma{=}0.12$) @ **`6.673s`**, Pareto Latency-AUC = **`45.38%`**.
  2. `+ Token Budget Knob (B32, B128, B512) [6 arms]`: peak **`56.0%`** (`+4.0%`), matched `52.0%` @ **`2.353s`** (**`2.84×` faster**), min latency **`1.341s`**, Pareto Latency-AUC = **`55.64%`** (**`+10.26%`**).
  3. `+ Image Resolution Knob (LowRes 196^2, HighRes 392^2) [12 arms]`: peak **`56.0%`**, matched `52.0%` @ **`2.353s`**, min latency reduced further to **`1.327s`**, Pareto Latency-AUC = **`55.66%`** (**`+10.28%`** over Mode Only). Within `Instruct B128`, `HighRes` lifts `ChartQA` `20% -> 60%` (`54% vs 48%` overall).
- **Per-Configuration Trace Matrix:** Complete 12-arm live CUDA trace breakdown (`Acc%`, `OutTok`, `p50/mean Lat`) added in Supplementary Table `tab:supp_exp2_live` (`reports/tables/exp2_raw_live_traces.csv`).

---

## Claim 3: “Disjoint LinUCB, Sherman-Morrison, offline Warm-Start”

### Changes Implemented (`src/rl/*`, `src/evaluation/statistical_tests.py`, `scripts/exp3_bandits_budgets_gates.py`)
1. Added **`Greedy Warm-Start (alpha=0)`** baseline under both Held-Out Test (`N=6,108`) and 5-Split CV (`70/10/20`), explicitly diagnosing what UCB exploration buys vs. greedy exploitation.
2. Added **cost-aware offline & neural bandit baselines**: `KNN-Reward` (MSE cost-penalized reward), `MLP-Reward` (MSE cost-penalized reward), `Per-Domain Best Static` (domain oracle skyline), **`NeuralUCB (Warm+Online)`**, and **`NeuralTS (Warm+Online)`** (implemented in `src/rl/neural_bandit.py`).
3. Computed full matched-cost & matched-latency Pareto curves and normalized **Cost-AUC (`$0.05–$0.28/1k`)** & **Latency-AUC (`0.3–1.9s`)** via trapezoidal integration (`pareto_frontier_auc`).
4. Redid feature/modality ablation under **Cold i.i.d.**, **Warm i.i.d.**, and **OOD Online** with **10,000-resample paired bootstrap CIs**. Cut the 8 OOD-fragile heuristic cues by default (**`896-d` Text+Vision default vs. `904-d` ablation**).

### Observed Outputs (`reports/tables/exp3_*.csv`)
- **Held-Out Test (`N=6,108`, 5 Seeds) & 5-Split CV (`exp3_claim3_baselines_and_splits.csv`, `exp3_5splits_summary.csv`, `exp3_claim3_bootstrap_ci.csv`):**
  - `Qwen2.5-VL-72B`: `77.29% @ $0.2605/1k, 1.959s` (5-Split: `78.21 ± 0.53% @ $0.2626`)
  - `Per-Domain Best Static` (Oracle Domain): `78.11% @ $0.1868/1k, 1.088s`
  - `Offline-Linear`: `78.21% @ $0.1777/1k` (Cost-AUC `77.24%` / Lat-AUC `74.01%`)
  - `KNN-Reward`: `78.19% @ $0.1134/1k` (Cost-AUC `78.81%` / Lat-AUC `77.37%`)
  - `ARMS-Router`: `78.80% @ $0.2190/1k` (Cost-AUC `77.27%` / Lat-AUC `76.38%`)
  - `MLP-BCE` (i.i.d. static): `79.81% @ $0.1858/1k` (5-Split: `80.61 ± 0.66% @ $0.2194`, Cost-AUC `78.66%` / Lat-AUC `75.54%`)
  - `MLP-Reward (MSE)` (i.i.d. static): `79.65% @ $0.1790/1k` (Cost-AUC **`78.89%`** / Lat-AUC `76.75%`)
  - `NeuralUCB (Warm+Online)`: `75.97 ± 0.74%` (`76.14%` Repaired) @ `$0.1925/1k` (`-3.39%` vs. CR, `p=1e-4`)
  - `NeuralTS (Warm+Online)`: `75.02 ± 1.01%` (`75.25%` Repaired) @ `$0.1721/1k` (`-2.78%` vs. CR, `p=1e-4`)
  - `Greedy Warm-Start (alpha=0)`: `78.52 ± 0.22% @ $0.1593/1k` (5-Split: `79.60 ± 0.58%`, Cost-AUC `78.18%` / Lat-AUC `76.96%`)
  - **`CascadeRoute-VL (gamma=0.08)`**: **`78.59 ± 0.12% @ $0.1598/1k, 1.550s`** (5-Split: `79.56 ± 0.56%`, Cost-AUC **`78.15%`** / Lat-AUC **`76.96%`**; beats static 17-VLM AUC `74.54%` & Offline-Linear `77.24%`; paired boot vs. 72B `+1.20%` CI `[+0.23, +2.14]`, `p=0.0152`, USD saved `+$0.1021/1k` `[+0.0967, +0.1073]`, `p=1e-4`).
  - **What UCB Buys vs. Greedy:** On dense stationary i.i.d. logs, Greedy matches mean accuracy (`+0.20%` boot diff, `p=0.0674`), while UCB cuts seed variance nearly in half (`0.12%` vs `0.22%`) and becomes essential for OOD recovery and live hot onboarding (Claim 6).
- **Modality & 8-Cue Ablation (`exp3_claim3_modality_ablation_boot.csv`):**
  - `Text+Vision (896-d, Cues Cut)` vs. `Full Joint (904-d)`: Warm i.i.d. **`78.68%` vs `78.49%`** (`p=0.3613`); Cold i.i.d. **`72.33%` vs `72.51%`** (`p=0.0794`); **OOD Online `82.48%` vs `80.07%` (`+2.41%` improvement from cutting cues, 95% CI `[+1.94, +2.90]`, $p{=}0.0002$)**.

---

## Claim 4: “Budget-Aware Annealing with sublinear regret and asymptotic budget feasibility”

### Changes Implemented (`src/rl/linucb.py`, `src/rl/neural_bandit.py`, `cvpr2027/`)
1. **Clamped Annealing Proof & Numerical Constants (Theorem 1):** Proved sublinear regret for clamped $\alpha_t = \max(\alpha_{\min}, \alpha_0 b_t)$ with $\alpha_{\min}$ meeting the Abbasi-Yadkori confidence width condition, computed exact operational constants, explicitly separated theoretical worst-case $\alpha_{\min}^{\text{theory}}$ from Dev-tuned scaling $\alpha_0{=}0.10$ ($\alpha_{\min}{=}0.01$), and plotted $\alpha_0 \in [0, 20]$ sensitivity.
2. **Deterministic Horizon-Reserve Guardrail (Theorem 2):** Added `select_action_budgeted()` / horizon-reserve feasibility check $\mathcal{A}_t^{\text{feas}}$ in `src/rl/linucb.py` forcing cheapest fallback $a_{\min}$ (`DeepSeek-VL2-Tiny`) whenever remaining headroom falls below $(T-t) c_{\min}^{\text{safe}}$, proving deterministic feasibility $C_T \le B_{\max} + c_{\max}$.
3. **Primal-Dual Baseline (`BwKPrimalDualLinUCB`):** Implemented Bandits-with-Knapsacks primal-dual multiplicative pacing and evaluated budget compliance & accuracy across $B_{\max} \in \{0.25\times, 0.40\times, 0.60\times, 0.80\times, 1.00\times, 1.25\times\}$ nominal spend (`$0.2443`–`$1.2216`).

### Observed Outputs (`exp3_claim4_alpha_sensitivity.csv`, `exp3_claim4_budget_compliance.csv`)
- **Explicit Regret Constants:** For $d{=}904, K{=}17, T{=}6108, \lambda{=}4.0, \nu{=}0.60, S{=}3.329, \delta{=}0.05$, Abbasi-Yadkori gives **$\alpha_{\min}^{\text{theory}} = 42.67$** ($42.51$ at $d{=}896$). Empirically sweeping $\alpha_0 \in [0, 20]$ shows optimal plateau around $\alpha_0 \in [0.10, 0.50]$ (`78.49–78.59%` i.i.d.; `80.85–80.96%` OOD at `0.10`), whereas $\alpha_0{=}20$ degrades OOD to `77.15%`.
- **Budget Compliance vs. BwK (`0.25x..1.25x B_max`):**
  - `Unguarded Annealed LinUCB`: violates `4/6` tight budgets (up to `3.97× B_max`, `+296.95%` overrun).
  - `BwK Primal-Dual LinUCB`: violates **`5/6`** soft limits (`+30.23%, +16.70%, +9.29%, +5.39%, +3.24%, +1.48%` overruns) with lower accuracy (`65.28%..77.00%`).
  - **`Hard-Guarded CascadeRoute (Theorem 2)`**: achieves **`0/6` hard violations (`0.00%` soft overrun)** and **beats BwK accuracy at all 6 budgets**: `65.32%` ($0.25\times$), `68.37%` ($0.40\times$), `71.69%` ($0.60\times$), `75.95%` ($0.80\times$), `78.45%` ($1.00\times$), `78.52%` ($1.25\times$).

---

## Claim 5: “Confidence-Guided Escalation Gate”

### Changes Implemented (`src/cascaderoute/confidence_gate.py`, `scripts/exp3_bandits_budgets_gates.py`)
1. Replaced the previous ~8 heuristic constants with a **`CalibratedDeferralGate`** (isotonic regression + split-conformal margin fitted strictly on `Dev`) controlled by a single scalar threshold $\tau_{\text{cal}}{=}0.04$, plus full $\tau_{\text{cal}} \in [-0.08, +0.14]$ sensitivity sweep (`Fig. 6(b)`).
2. Evaluated across four distinct regimes: Balanced 17-Pool ($\gamma{=}0.08$), Frugal 17-Pool ($\gamma{=}0.30$), OOD 17-Pool, and **`Compact-Primary (<=8B) -> Paid Escalation`**.
3. Explicitly isolated **dual-arm (`both_arms`)** vs. **strictly single-arm (`final_only`)** feedback accounting in Algorithm 1, Sec. 4.3, and Sec. 5.3, reporting exact `ΔAcc (%) per Extra $`.

### Observed Outputs (`exp3_claim5_calibrated_gate.csv`, `exp3_claim5_tau_sensitivity.csv`)
- **Unconstrained 17-Pool (Balanced/Frugal/OOD — Honestly Reported as Neutral Ablation):** Triggers on only `0.13%–0.97%` of queries and yields `+0.02%–+0.08%` net ΔAcc (`$0.0019–$0.0093` extra spend).
- **Compact-Primary ($\le 8\text{B}$) $\to$ Paid Escalation Tier ($\tau_{\text{cal}}{=}0.04$):**
  - No Gate: `69.27% @ $0.0451/1k`.
  - With Gate (`both_arms` feedback): **`77.90% @ $0.1482/1k`** (`41.91%` esc rate, **`+8.69%` net accuracy gain**, **`+13.82%` accuracy gain per extra dollar**).
  - With Gate (`final_only` single-arm feedback): **`77.85% @ $0.1481/1k`** (`41.01%` esc rate, **`+8.38%` net accuracy**, **`+13.40%` ΔAcc/$**).

---

## Claim 6: “OOD stress tests, static offline routers collapse, CascadeRoute-VL adapts online”

### Changes Implemented (`src/rl/linucb.py`, `scripts/exp4_ood_drift_onboarding_ope_generalization.py`)
1. **Unified Prior Weight ($w_0{=}0.20$):** Removed the inconsistency between `w0=0.05` (OOD) and `w0=0.20` (main) — re-ran all OOD stress tests with unified **`w0=0.20`** (`5 seeds`), reporting OOD **Cost/1k** and **Latency** alongside 72B.
2. **Non-Stationary Variants:** Added **Discounted LinUCB (`D-LinUCB`, $\beta{=}0.996$)** and **Sliding-Window LinUCB (`SW-LinUCB`, $W{=}400$)** with fast GPU/CPU covariance tracking.
3. **Continuous Drift & Price Shock Streams (`10,000` queries):** Tested Gradual Mixture Drift, Repeated Regime Switches, and a `3×` API Price Shock ($t > 2000$).
4. **Live Test-Time Hot Model Onboarding (`hot_onboard_arm(prior_strength=1.0, init_reg=0.5)`):** Added 17th arm `Kimi-VL-A3B-Thinking` live at $t{=}500$ during inference without any offline retraining (`init_reg=0.5`, $\alpha{=}0.35$).

### Observed Outputs (`exp4_claim6_*.csv`)
- **Unified $w_0{=}0.20$ OOD Shift (`General+STEM` $\to$ `Charts+OCR+Hall`, $N{=}17{,}601$, 5 Seeds):**
  - `Qwen2.5-VL-72B` (Static): `83.81% @ $0.2223/1k, 0.801s`.
  - `MLP-Router` (Frozen): `56.97%` (`-15.80%` collapse); `ARMS` (Frozen): `66.58%`.
  - `Cold LinUCB (w0=0)`: `74.01 ± 4.37% @ $0.1768/1k`.
  - **`CascadeRoute-VL Standard (g=0.08, 896-d)`**: **`81.59 ± 0.77% @ $0.1512/1k, 0.591s`** (**32.0% lower USD cost** & **1.35× faster latency** than 72B).
  - **`Discounted CascadeRoute (beta=0.996, 896-d)`**: **`80.54 ± 0.47% @ $0.1238/1k, 0.547s`** (**44.3% lower USD cost** & **1.46× faster** than 72B).
  - `Sliding-Window (W=400, 904-d)`: `80.68 ± 0.50% @ $0.1300/1k, 0.551s` (`41.5%` USD savings, `1.45×` speedup).
- **Continuous Drift (`10k`) & Price Shock (`exp4_claim6_continuous_drift.csv`):**
  - Gradual Mixture Drift: `CascadeRoute` **`79.45%`**, `D-LinUCB` **`78.67%`**, `SW-LinUCB` `78.22%` vs. `Frozen MLP` `64.75%`.
  - Repeated Switches: `CascadeRoute` **`78.30%`**, `D-LinUCB` **`78.11%`** vs. `Frozen MLP` `65.58%`.
  - `3×` API Price Shock ($t > 2000$): `D-LinUCB` cuts expensive picks **`28.7% -> 18.1%`** (`$0.1508/1k` vs. frozen `MLP` `25.9%` at `$0.1859/1k`).
- **Live Test-Time Hot Onboarding (`exp4_claim6_hot_onboarding.csv`):**
  - Before insert `[0, 500)`: `0.0%`. Immediately at `t=500`: first pick after **`0` queries**, **`62.00%`** exploration across `[500, 550)`, **`25.33%`** (`80.00%` acc) in `[550, 700)`, settling cleanly to **`11.12% -> 5.80% -> 4.91%`** specialization (vs. Frozen Offline Routers **`0.0%`** without retraining).

---

## Claim 7: “counterfactual IPS/SNIPS”

### Changes Implemented (`exp4_ood_drift_onboarding_ope_generalization.py`, `cvpr2027/Supplementary.tex` Sec. S4)
1. **Moved Counterfactual OPE from Main Paper to Supplementary Sec. S4:** Noted explicitly that the full $30{,}540 \times 17$ outcome matrix is available so primary accuracy evaluation is direct, framing OPE strictly as a logged-data validation audit.
2. **Evaluated BOTH Snapshot AND Online Adaptive Trajectory + Explained 72B Coverage Failure:** Added `CascadeRoute (Online Adaptive Trajectory)` evaluation alongside the Warm-Start Snapshot across 10 independent behavior loggers ($\epsilon{=}0.75$ uniformized logging), and explicitly disclosed in text why skewed star-logger propensities ($p_{\log} < 0.01$) caused `Qwen2.5-VL-72B` SNIPS/DR CIs to under-cover previously vs. uniformized support.

### Observed Outputs (`exp4_claim7_ope_audit.csv`)
- **`CascadeRoute (Online Adaptive Trajectory)`:** True Acc `0.7845`, IPS `0.7921 [0.7125, 0.8793]`, SNIPS `0.7862 [0.7476, 0.8263]`, DR **`0.7894 [0.7496, 0.8255]`**, Mean DR (10 loggers) `0.7808`, **`10/10` 95% CI coverage**, Mean ESS **`397.7`**.
- `Always-Qwen-72B` (Uniformized Support): True `0.7729`, DR `0.7835 [0.7386, 0.8246]`, **`10/10`** coverage (vs. prior skewed-logger exclusion explicitly documented).
- All 7 audited policies hit `9/10`–`10/10` 10-logger 95% CI coverage.

---

## Claim 8: “substantially reducing token consumption and inference latency”

### Changes Implemented (`scripts/exp2_live_vlm_configurations.py`, `cvpr2027/CascadeRoute_VL_CVPR2027.tex`)
1. Measured real **GPU wall-clock p50 & p95 inference latency** (`Qwen3-VL-2B-Instruct` & `-Thinking` in 4-bit NF4 CUDA on RTX 4050) and timed **all three encoders individually** (`all-MiniLM-L6-v2`, `DINOv2-ViT-S/14`, `ResNet-50`, optional `8-d` cues, + `LinUCB` selection).
2. Fitted OLS latency calibration $L = \text{TTFT} + O/\text{TPS}$ against live measured hardware traces and reported **MAE, MAPE%, and Pearson $r$** validation error, explicitly disclosing 4-bit HF `transformers` scope (not vLLM/SGLang continuous batching).
3. Replaced every instance of `"53x dynamic compute scaling"` with **`"53× output-token ratio"`** (`259.6` vs. `4.9` tokens).

### Observed Outputs (`exp2_real_latency_and_sim_cal.csv`, `exp2_router_overhead.csv`)
- **Real 4-Bit CUDA Wall-Clock Latency & OLS Calibration (`reports/tables/exp2_real_latency_and_sim_cal.csv`, `N=600` live hardware runs on RTX 4050):**
  - `Qwen3-VL-2B-Instruct` ($N{=}300$): Mean `4.203s`, **p50 `0.490s`**, **p95 `19.643s`**; OLS `TTFT=0.050s, Decode=6.4 TPS` -> **`MAE=0.598s`, `MAPE=14.0%`, Pearson `r=0.9894`**.
  - `Qwen3-VL-2B-Thinking` ($N{=}300$): Mean `20.512s`, **p50 `13.174s`**, **p95 `66.555s`**; OLS `TTFT=0.195s, Decode=7.8 TPS` -> **`MAE=0.661s`, `MAPE=3.5%`, Pearson `r=0.9985`**.
  - Honest scope note added to Sec. 6 (`Limitations`): measurements run on 4-bit HF `transformers` (`RTX 4050`), not vLLM/SGLang continuous batching.
- **All-Three-Encoder + LinUCB Overhead (`reports/tables/exp2_router_overhead.csv`, `N=50` CUDA queries):**
  - `MiniLM-L6` (`p50 13.87ms / p95 18.54ms`), `DINOv2-ViT-S/14` (`15.20ms / 18.86ms`), `ResNet-50 JL` (`16.50ms / 19.81ms`), optional `8-d cues` (`4.37ms / 10.29ms`), `LinUCB selection` (`1.01ms / 1.45ms`).
  - **Combined `896-d` Overhead:** **`52.66ms p50 / 68.21ms p95`** (`< 0.4%` of `-Thinking` p50; `904-d` encoding p50 `55.18ms`).

---

## Cross-Cutting Fixes

### Changes Implemented
1. **Honest Feedback & Terminology (`CascadeRoute_VL_CVPR2027.tex`, `Supplementary.tex`, `README.md`):**
   - Replaced all instances of `"zero-label"` with **`"no domain labels"`** / `"domain-label-free"` and clarified executed-arm reward updates (and dual-arm observation when gating).
   - Replaced `"provably"` and `"strictly zero-leakage"` with exact formal scope disclosures (`no task/dataset IDs` vs. historical warm-start priors and Dev calibration).
2. **Paired Bootstrap & Removed `p=1e-275` (`src/evaluation/statistical_tests.py`):**
   - Switched primary statistical claims to 10,000-resample paired bootstrap CIs over Test queries and multi-split 5-fold CV, removing `p=9.49e-275` from Main & Supplementary.
3. **Partial Feedback & Noisy LLM-Judge Experiments (`exp4_cross_partial_and_noisy_fb.csv`):**
   - Evaluated label observability `5%..100%` and symmetric LLM-judge flip noise `0%..20%`.
4. **Leave-One-Dataset-Out, External `AMBER`, & Cross-Pool Transfer (`exp4_cross_lodo_benchmarks.csv`, `exp4_cross_pool_transfer.csv`, `exp2_amber_external_eval.csv`):**
   - Tested LODO held-out benchmarks (`InfoVQA_VAL`, `MathVision_MINI`, `HallusionBench`), disjoint Pool A (`9` arms) $\to$ Pool B (`8` unseen arms) scalar prior transfer, and held-out external `AMBER` live GPU eval.
5. **Open Release (`README.md`, `data/vl_routerbench_processed/`, `scripts/`):**
   - Documented release of raw/repaired outcome tensors (`Y.npz`, `Y_repaired.npz`), all 238 VLMEvalKit transcripts metadata, unit tests (`50/50`), and reproduction scripts.

### Observed Outputs
- **Partial Feedback (`5%..100%`, `exp4_cross_partial_and_noisy_fb.csv`):** Warm i.i.d. holds `78.72%–78.80%` (`5%..25%` feedback); OOD D-LinUCB rises from `77.22%` (`5%`) to **`80.22%`** (`25%`) and `80.84%` (`50%`).
- **Noisy Judge (`0%..20%` flips):** Warm i.i.d. degrades gracefully `78.85% -> 78.60% (10%) -> 77.82% (20%)`; OOD D-LinUCB holds `80.93% -> 79.71% (10%) -> 78.51% (20%)`.
- **Leave-One-Dataset-Out (vs. Frozen MLP):** `InfoVQA_VAL` **`79.97%`** vs. `76.79%`; `MathVision_MINI` **`42.11% @ $0.3762/1k`** vs. `41.12% @ $0.7161/1k` (higher acc, ~half cost, beating 72B `38.82%`); `HallusionBench` **`72.66%`** vs. `70.66%`.
- **Cross-Pool Transfer A(9) $\to$ B(8 Unseen Arms):** Cold LinUCB collapses onto 78B (`76.18% @ $0.3012/1k`), while Pool A$\to$B prior transfer hits **`75.10% @ $0.1544/1k`** (**48.7% USD savings**).
- **External `AMBER` Live GPU Eval (`reports/tables/exp2_amber_external_eval.csv`, zero `AMBER` training):** `Static Qwen3-VL-2B-Instruct`: `90.0% @ 0.350s`; `Static Qwen3-VL-2B-Thinking`: `90.0% @ 13.389s`; **`CascadeRoute-VL`**: **`90.0% @ 0.350s`** (**`38.3×` faster** than static Thinking); `12-Config Oracle`: `100.0% @ 0.325s`.

---

## Verification & Final Submission Checklist

- **Unit Tests (`pytest tests/ -v`):** **`50/50` passed** (`tests/test_algorithms.py`, `tests/test_correctness.py`, `tests/test_evaluation.py`, `tests/test_linucb_upgrades.py`, `tests/test_new_baselines.py`).
- **All Experimental Suites Executed End-to-End:**
  - `scripts/exp1_claim1_oracle_and_mimo_repair.py` $\to$ `reports/tables/exp1_*.csv`, `data/vl_routerbench_processed/matrices/Y_repaired.npz`
  - `scripts/exp2_live_vlm_configurations.py 10` (`600` live 4-bit CUDA runs on `RTX 4050`) $\to$ `reports/tables/exp2_*.csv`
  - `scripts/exp3_bandits_budgets_gates.py` $\to$ `reports/tables/exp3_*.csv`
  - `scripts/exp4_ood_drift_onboarding_ope_generalization.py` $\to$ `reports/tables/exp4_*.csv`
  - `scripts/generate_cvpr_visuals.py` $\to$ all 7 CVPR figures (`reports/figures/fig1..fig7.png`) visually inspected with zero overlapping annotations.
- **CVPR 2027 Compiled Manuscripts & Overleaf Archives (`cvpr2027/`):**
  - `cvpr2027/CascadeRoute_VL_CVPR2027.pdf`: **10 pages total** — Main body, all 7 figures, all 3 tables, and Limitations/Conclusion fit cleanly within **Pages 1–8**; **References start on Page 9** (`0` undefined `??`, compiled cleanly via Tectonic `Exit=0`).
  - `cvpr2027/Supplementary.pdf`: **7 pages total** (`0` undefined `??`, `Exit=0`).
  - **Verified Standalone Overleaf ZIPs (extracted and test-compiled cleanly via Tectonic, `Exit=0`, zero `??`):**
    - `cvpr2027/CascadeRoute_VL_Main_Overleaf.zip` (`3.47 MB`)
    - `cvpr2027/CascadeRoute_VL_Supp_Overleaf.zip` (`0.42 MB`)
    - `cvpr2027/CascadeRoute_VL_Overleaf.zip` (`3.87 MB`, combined)


