# CascadeRoute-VL: Master Methodology & Technical Reference
## Taming the Reasoning Tax in Vision-Language Models via Online Contextual Bandit Routing

> **Target Venue:** CVPR 2027 (IEEE / CVF Computer Vision and Pattern Recognition)  
> **Registered Title:** *CascadeRoute-VL: Taming the Reasoning Tax in Vision-Language Models via Online Contextual Bandit Routing*  
> **Keywords:** Vision-Language Models, Model Routing, Contextual Bandits, Multimodal Reasoning, Test-Time Compute Scaling, Efficient Inference  
> **Repository:** `https://github.com/THE-Adi-005/SDP_RL_Project` (`main`)  
> **Author:** Adithya R (`adithyakrs2408@gmail.com`)  

---

## 1. Scientific Foundation: The Multimodal Reasoning Tax

We analyze the complete ground-truth outcome matrix $Y \in \{0,1\}^{30{,}540 \times 17}$ reconstructed from all **238 VLMEvalKit inference transcripts** across **14 benchmarks** and **17 contemporary VLMs** in **VL-RouterBench** (519,180 query-model cells, $>99.9\%$ exact cell coverage), using exact `cl100k_base` tokenization ($+256$ visual tokens per query), benchmark token pricing schedules ($c_{90} = \$0.000329$), and deterministic TTFT + decode throughput latency profiling:

- **Strongest Monolithic Frontier VLM (`Qwen2.5-VL-72B-Instruct`):** `78.21%` accuracy across all `30,540` queries (`$0.2626 / 1k Q`), failing on `6,655` queries (`77.29%` on `6,108` held-out Test @ `$0.2605 / 1k Q`, `1.959s` latency).
- **17-VLM Oracle Upper Bound:** `95.45%` overall (`95.17%` on Test), revealing a **+17.24% routing opportunity gap** and rescuing `5,266 / 6,655` (`79.1%`) frontier failures.
- **Compact Model Rescue Phenomenon ($\le 8\text{B}$):** Compact VLM architectures solve **4,557 of those 6,655 failures (68.5%)**—led by `Kimi-VL-A3B-Thinking` (`2,324`), `Qianfan-VL-8B` (`1,789`), and `DeepSeek-VL2-Tiny` (`1,689`).
- **Two Opposing Regimes (Reasoning Gain vs. Perceptual Inversion):**
  - *STEM & Diagrammatic Reasoning:* Thinking specialist `Kimi-VL-A3B-Thinking` attains `79.60%` on `MathVista_MINI` and `47.04%` on `MathVision_MINI`, outperforming `Qwen2.5-VL-72B` (`73.90%` / `38.82%`).
  - *Charts & Dense OCR:* Perception specialist `Qianfan-VL-8B` delivers `86.48%` on `ChartQA_TEST` (`1.2` mean output tokens), `87.42%` on `DocVQA_VAL` (`2.7` tok), and `85.70%` on `OCRBench` (`4.6` tok) at $\sim 5\times$ lower dollar cost and $\sim 8.8\times$ lower latency than `72B`. Meanwhile, unconstrained `<think>` chains in `MiMo-VL-7B-RL` (emitting `<think>` on `100%` of queries, averaging `237.4` output tokens, up to `1,401.7` on `MathVision`) degrade OCR fidelity or disrupt downstream answer parsing (`0.0%` extracted accuracy across `ChartQA`, `DocVQA`, `InfoVQA`, and `TextVQA`).

---

## 2. Architecture & Mathematical Formulation

### 2.1 Strictly Zero-Leakage 904-d Multimodal Context State ($s_t \in \mathbb{R}^{904}$)
Implemented in [`src/features/multimodal_extractor.py`](src/features/multimodal_extractor.py) and cached for `30,540` queries via [`scripts/extract_real_multimodal_features.py`](scripts/extract_real_multimodal_features.py). **Dataset IDs, task tags, and domain one-hots are strictly forbidden.**
$$s_t = \left[\, w_q \hat{q}_t \;\|\; w_v \hat{v}_t \;\|\; w_p p_t \,\right] \in \mathbb{R}^{904}, \qquad (w_q, w_v, w_p) = (0.50, 0.55, 0.235).$$

1. **Linguistic Semantics ($\hat{q}_t \in \mathbb{R}^{384}$):** $L_2$-normalized `all-MiniLM-L6-v2` prompt embedding (`||\hat{q}_t||_2 = 1`).
2. **Dual-Scale Vision ($\hat{v}_t \in \mathbb{R}^{512}$):** $L_2$-normalized concatenation of frozen `DINOv2-ViT-S/14` `[CLS]` (`384-d`) + fixed orthogonal Johnson-Lindenstrauss projection (`2048 -> 128`) of `ResNet-50` global average pooled texture/layout features (`||\hat{v}_t||_2 = 1`).
3. **Unsupervised Perceptual & Syntactic Cues ($p_t \in [0, 1]^8$):**
   - Luminance Shannon entropy $H(I_t)/8$ (`MathVerse` `2.00` bits vs. `TextVQA` `7.10` bits)
   - Sobel spatial edge gradient density $\nabla(I_t)$ (`InfoVQA` `0.281` vs. `ChartQA` `0.095`)
   - Logarithmic aspect ratio $\frac{1}{\pi}\arctan(\ln(W/H)) + 0.5$
   - Clipped log pixel area $\text{clip}(\log_{10}(WH)/7, 0, 1)$
   - Bright white background pixel fraction ($L > 235/255$; `MathVerse` `0.887` vs. `RealWorldQA` `0.005`)
   - Clipped prompt word ratio $\min(w, 128)/128$
   - Unsupervised symbolic / digit / MCQ cue density $\psi_{\text{math}}(Q_t)$
   - Sentence count ratio $\min(s, 16)/16$

**Norm Guarantee (Lemma 1):**
$$\|s_t\|_2^2 = w_q^2 + w_v^2 + w_p^2 \|p_t\|_2^2 \le 0.50^2 + 0.55^2 + 8(0.235^2) = 0.9943 \implies \|s_t\|_2 \le 0.9972 < 1.$$
Across all `30,540` encoded queries: $\min_t \|s_t\|_2 = 0.7611$, $\text{mean}_t \|s_t\|_2 = 0.8116$, $\max_t \|s_t\|_2 = 0.8943 < 1.0$.

### 2.2 Cost-Aware Scalar Reward & Warm-Started Disjoint LinUCB
Implemented in [`src/rl/linucb.py`](src/rl/linucb.py) with PyTorch GPU acceleration + CPU fallback:
$$r_{t,a} = y_{t,a} - \gamma \tilde{c}_{t,a}, \qquad \tilde{c}_{t,a} = \text{clip}\!\left(\frac{c_{t,a}}{c_{90}},\, 0,\, 2.5\right) \in [0, 2.5], \qquad c_{90} = \$0.000329.$$

- **Offline Warm-Start Prior ($\lambda = 4.0$, $w_0 = 0.20$ for i.i.d., $w_0 = 0.05$ for OOD shift):**
  $$A_a^{(0)} = \lambda I_d + w_0 \sum_{i:\,a_i=a} s_i s_i^\top, \qquad b_a^{(0)} = w_0 \sum_{i:\,a_i=a} r_i s_i, \qquad \hat{\theta}_a = A_a^{-1} b_a.$$
- **Online $O(d^2)$ Sherman-Morrison Covariance Update:**
  $$A_a^{-1} \leftarrow A_a^{-1} - \frac{(A_a^{-1} s_t)(A_a^{-1} s_t)^\top}{1 + s_t^\top A_a^{-1} s_t}, \qquad b_a \leftarrow b_a + r_{t,a} s_t.$$

### 2.3 Budget-Aware Exploration Annealing & Confidence Escalation Gate
Implemented in [`src/cascaderoute/confidence_gate.py`](src/cascaderoute/confidence_gate.py) and [`src/cascaderoute/budget_tracker.py`](src/cascaderoute/budget_tracker.py):
- **Budget Fraction & Exploration Annealing:** Given remaining USD reserve fraction $b_t = \text{clip}((B_{\max} - C_{t-1})/B_{\max}, 0, 1)$, exploration scales linearly as $\alpha_t = \alpha_0 b_t$ ($\alpha_0 = 0.10$):
  $$a_0 = \arg\max_{a \in \mathcal{A}} \left(\hat{\theta}_a^\top s_t + \alpha_t \sqrt{s_t^\top A_a^{-1} s_t}\right).$$
- **Zero-Oracle Escalation Gate:** Activated only when primary selection $a_0 \in \mathcal{A}_{\text{compact}}$ ($\le 8\text{B}$ direct tier) and remaining budget satisfies $b_t > b_{\min} = 0.15$. Recovering quality estimates $\hat{\mu}_a = \hat{\theta}_a^\top s_t + \gamma \bar{c}_a$, it selects candidate fallback $a^* \in \mathcal{F} = \{\text{Kimi-3B-Think}, \text{Qwen2.5-72B}, \text{InternVL2.5-78B}, \text{GeminiFlash2.5}\}$ maximizing $\hat{\mu}_a - \kappa \gamma \bar{c}_a$ ($\kappa = 0.22$). With effective utility gain $\Delta_{\text{eff}} = (\hat{\mu}_{a^*} - \hat{\mu}_{a_0}) - \kappa \gamma \max(0, \bar{c}_{a^*} - \bar{c}_{a_0})$, the gate escalates iff $\hat{\mu}_{a_0} < 0.78$, $\Delta_{\text{eff}} \ge 0.025$, and:
  $$\rho_t = \Delta_{\text{eff}} + 0.25\,\sigma_{a_0}(s_t) + 0.06\,\nabla(I_t) + 0.06\,\psi_{\text{math}}(Q_t) + 0.05\,\mathbb{I}[o_{t,a_0} > 15] \ge 0.12.$$
  All primary + escalated tokens and latencies are truthfully billed and both executed arms updated online via Sherman-Morrison. *(In live inference mode, `ConfidenceGate` also exposes image-perturbation consistency $S_{\text{OT}} = 1 - \text{Jaccard}(A_1, A_2)$.)*

---

## 3. Theoretical Guarantees (`cvpr2027/Supplementary.tex` Sec. S1)

- **Lemma 1 (Bounded Context Space):** $\|s_t\|_2 \le 0.9972 < 1$ analytically ($\le 0.8943$ across all `30,540` queries).
- **Lemma 2 (Sub-Gaussian Reward Noise):** Bounded $r_{t,a} \in [-2.5\gamma, 1]$ yields conditionally $\nu$-sub-Gaussian noise with $\nu = (1 + 2.5\gamma)/2$ ($\nu = 0.60$ at $\gamma = 0.08$).
- **Theorem 1 (Disjoint Multimodal Regret Bound & OOD Prior Decay):** Self-normalized martingale bounds (Abbasi-Yadkori et al., 2011) + Cauchy-Schwarz across $K$ disjoint arm counts $\sum_{a=1}^K \sqrt{T_a} \le \sqrt{KT}$ + Determinant-Trace Elliptical Potential yield high-probability cumulative regret:
  $$R_T \le 2\alpha_0 \sqrt{2 d K T \ln\!\left(1 + \frac{T}{d\lambda}\right)} + \mathcal{O}(K) = \tilde{\mathcal{O}}(d\sqrt{KT}).$$
  Furthermore, under domain shift $\theta_a^{\text{src}} \to \theta_a^{\text{ood}}$, frozen supervised routers suffer linear regret $\Omega(\Delta_{\text{shift}} T)$, whereas warm-started ridge prior bias decays as $\mathcal{O}\!\left(\frac{w_0 \|\theta_a^{\text{src}} - \theta_a^{\text{ood}}\|}{\lambda + n_{t,a}}\right)$ under live online Sherman-Morrison updates.
- **Theorem 2 (Asymptotic Budget Feasibility):** Under $b_t \to 0^+$, $b_t \le 0.15$ unconditionally disables escalation and $\alpha_t = \alpha_0 b_t \to 0$ collapses decisions onto pure cost-penalized ridge exploitation $\arg\max_a \hat{\theta}_a^\top s_t$.

---

## 4. Verified Experimental Results (`N=6,108` Held-Out Test & `N=17,601` OOD, 5 Seeds)

### 4.1 Main Benchmark (`reports/tables/real_table1_main_benchmark.csv`)

| Method | Category | Acc (%) | Avg Tok | Out Tok | Latency (s) | Cost / 1k ($) | USD Saved | Cum. Regret |
|---|---|---|---|---|---|---|---|---|
| `Qwen2.5-VL-72B` | Static VLM | 77.29 | 309.8 | 31.7 | 1.959 | \$0.2605 | 0.0% | 1092.0 |
| `InternVL2.5-78B` | Static VLM | 76.20 | 293.5 | 15.4 | 1.328 | \$0.3012 | -15.6% | 1159.0 |
| `Kimi-VL-3B-Think` | Static VLM | 72.86 | 335.3 | 57.1 | 0.741 | \$0.0699 | 73.2% | 1363.0 |
| `Qianfan-VL-8B` | Static VLM | 69.88 | 282.2 | 4.1 | 0.228 | \$0.0511 | 80.4% | 1545.0 |
| `DeepSeek-VL2-Tiny` | Static VLM | 62.57 | 285.5 | 7.3 | 0.144 | \$0.0143 | 94.5% | 1991.0 |
| `Offline-Linear` | Offline Router | 78.21 | 384.3 | 106.2 | 1.721 | \$0.1777 | 31.8% | 1036.0 |
| `KNN-Router` ($k=15$) | Offline Router | 78.31 | 361.8 | 83.7 | 1.444 | \$0.1306 | 49.9% | 1030.0 |
| `ARMS-Router` | Offline Router | 78.80 | 369.0 | 90.8 | 1.306 | \$0.2190 | 16.0% | 1000.0 |
| `MLP-Router` (GPU) | Offline Router | **79.85** | 400.4 | 122.2 | 1.844 | \$0.1854 | 28.8% | **936.0** |
| `FrugalGPT-Cascade` | Seq. Cascade | 73.23 | 725.8 | 69.7 | 1.719 | \$0.1704 | 34.6% | 1340.0 |
| `MHR-Router` | Retrieval Bandit | 71.48 ± 0.51 | 346.4 | 68.2 | 1.333 | \$0.1247 | 52.1% | 1446.8 |
| `DQN-Router` | Online Bandit | 72.96 ± 1.02 | 335.5 | 57.3 | 1.421 | \$0.1845 | 29.2% | 1356.4 |
| `LinUCB (Cold Online)` | Online Bandit | 72.68 ± 1.71 | 363.0 | 84.9 | 1.535 | \$0.3220 | -23.6% | 1373.8 |
| **`CascadeRoute-VL (Frugal γ=0.30)`** | **Proposed** | 76.13 ± 0.21 | **360.5** | **51.1** | **0.824** | **\$0.0876** | **66.4%** | 1162.8 |
| **`CascadeRoute-VL (Full, γ=0.08)`** | **Proposed** | **78.59 ± 0.12** | 370.8 | 91.9 | 1.550 | \$0.1598 | 38.7% | **1013.0** |
| `Oracle (Hindsight 17-Arm)` | Upper Bound | 95.17 | 296.5 | 18.3 | 0.339 | \$0.0348 | 86.6% | 0.0 |

### 4.2 Zero-Leakage OOD Distribution Shift (`General+STEM` -> Unseen `Charts+OCR+Hall`, `N=17,601`, 5 Seeds)

| Method | In-Dist Test (`N=2,523`) | Unseen OOD (`N=17,601`) | Shift $\Delta$ | Exceeds Static MLP |
|---|---|---|---|---|
| `Qwen2.5-VL-72B` (Static) | 69.24% | 83.81% | +14.56% | Ref. |
| `Offline-Linear` | 73.64% | 65.41% | -8.24% | Never |
| `ARMS-Router` | 76.06% | 66.58% | -9.48% | Never |
| `MLP-Router` | 72.77% | 56.97% | **-15.80%** | Baseline |
| `LinUCB (Cold Online)` | 74.44% | 75.72% | +1.28% | $\sim 191$ queries |
| **`CascadeRoute-VL (Ours)`** | **75.98%** | **80.68%** | **+4.70%** | **$\sim 51$ queries** |
| `Oracle` (Upper Bound) | 95.48% | 95.36% | -0.12% | Instant |

### 4.3 Counterfactual OPE (`real_table5_ips_counterfactual.csv`, `1,000` Bootstraps) & Ablations
- **OPE on Warm-Start Target Policy (`True = 0.7878`):** IPS = `0.8356 [0.7332, 0.9367]`, SNIPS = **`0.7919 [0.7461, 0.8386]`**, Doubly Robust = **`0.7833 [0.7395, 0.8281]`** (`ESS = 306.4`).
- **Component & Modality Gains:** Adding offline warm-start lifts cold online LinUCB from `72.68%` (`\$0.3220`) to `78.58%` (`\$0.1590`); adding the 8 unsupervised perceptual cues boosts cold online `Text+Vision` from `68.82%` to `72.68%` (`+3.86%`).
- **Zero-Label Emergent Specialization:** `63.2%` thinking picks on `STEM` (`259.6` out tok; `74.62%` acc vs `67.08%` for 72B) vs `0.09% MiMo-VL-RL` and `44.7% Qianfan-8B/DeepSeek-VL2` on `Charts_OCR` (`4.9` out tok) = **53× dynamic compute scaling**.
