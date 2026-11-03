# ICLR 2027 Paper Proposal: Novel Architecture for Adaptive LLM Reasoning

> **Deadline:** Abstract — September 18, 2026 | Paper — September 25, 2026
> **Time remaining:** ~4 weeks (1–2 weeks for implementation, 2 weeks for paper writing)

---

## 1. Landscape Analysis: Where Your Project Sits

I have analyzed the following key papers in this domain:

| Paper | Year/Venue | What They Do | Key Limitation |
|-------|------------|-------------|----------------|
| **RouteLLM** (Ong et al.) | ICLR 2025 | Routes queries between a strong model and weak model using preference data. Uses BERT/MF/SW classifiers. | Only routes between **2 models** (binary). Does NOT route between reasoning **strategies** (CoT, ToT). |
| **FrugalGPT** (Chen et al.) | 2023 | LLM Cascade: tries cheap model first, escalates on low confidence. | Sequential only. No RL. No reasoning strategy selection. |
| **AutoMix** (Aggarwal et al.) | 2024 | Self-verification: small model checks its own answer before escalating. | Single-strategy (no CoT/ToT routing). |
| **Route-To-Reason (RTR)** | arXiv 2025 | **Closest competitor.** Jointly selects model + reasoning strategy under budget constraints. | Requires offline training with labeled performance data. Not online/adaptive. Uses fixed embeddings — no bandit exploration. |
| **DOTS** | 2025 | Searches for optimal "reasoning action trajectories" per question. | Requires fine-tuning the LLM itself. Very expensive. |
| **DeepSeek-R1** | 2025 | RL-trained reasoning within the LLM. | Trains the model weights. Not a routing/orchestration framework. |

### The Critical Gap (Your Opportunity)

**RouteLLM** routes across **models** but ignores reasoning strategies.
**Route-To-Reason** routes across **models + strategies** but uses offline supervised learning — it cannot adapt online.
**Nobody** has combined:
1. Joint model x strategy routing
2. Online RL adaptation (bandit learning from live feedback)
3. Budget-constrained optimization
4. Confidence-gated cascading as a fallback

**This is your novel contribution.**

---

## 2. Proposed Paper Title

> **"CascadeRoute: Budget-Aware Online Routing Across Models and Reasoning Strategies for Cost-Efficient LLM Inference"**

---

## 3. Proposed Architecture

```
                    CascadeRoute Framework
  
    Input       Feature         Budget
    Query  ---> Extractor  ---> Tracker
                (Sentence       (remaining
                 Transformer    tokens,
                 + metadata)    $ spent)
                     |              |
                     v              v
               Contextual Bandit Router
               (LinUCB / Thompson)
  
               Action Space:
               Model x Strategy
  
               Qwen-3b + Direct   (0)
               Qwen-3b + CoT      (1)
               Qwen-7b + Direct   (2)
               Qwen-7b + CoT      (3)
               Qwen-7b + GoT      (4)
               Qwen-7b + ToT      (5)
  
               State = [features, budget_%]
                         |
                         v
                  Confidence Gate
                  (Chain of Resp.)
  
                  IF confidence >= threshold:
                     RETURN answer
                  ELSE:
                     ESCALATE to next strategy
                         |
                         v
                  Semantic Cache
                  (Proxy Pattern)
                  Check before LLM
                         |
                         v
                  LLM Execution
                  (Strategy Pattern)
                         |
                         v
                  Reward Calculator
                  reward = a*correct
                  - b*(tokens/max)
                  - g*(latency/max)
                         |
                         v
                  Online Update
                  (Bandit learns)
```

### What Makes This Novel vs. Existing Work

| Dimension | RouteLLM | Route-To-Reason | **CascadeRoute (Ours)** |
|-----------|----------|-----------------|------------------------|
| Routes across models | Yes (2 models) | Yes (N models) | Yes (2+ models) |
| Routes across strategies | No | Yes (CoT, ToT, etc.) | Yes (Direct, CoT, GoT, ToT) |
| Learning paradigm | Offline supervised | Offline supervised | **Online RL (bandit)** |
| Budget constraint | No | Yes (static) | Yes **(dynamic, depleting)** |
| Confidence cascading | No | No | Yes **(novel fallback)** |
| Caching / cost reduction | No | No | Yes (Semantic Proxy) |
| Adapts at inference time | No | No | Yes **(key differentiator)** |

---

## 4. The Three Novel Contributions (for the paper)

### Contribution 1: Joint Model x Strategy Bandit with Budget State
Unlike RouteLLM (binary model routing) and RTR (offline optimization), we formulate the routing problem as a **contextual bandit** where:
- The **action space** is the Cartesian product of models x strategies (e.g., 6 actions for 2 models x 3 strategies)
- The **state vector** includes query features AND the remaining budget percentage
- The agent learns **online** — it gets smarter with every query, no offline training data needed

### Contribution 2: Confidence-Gated Cascade (Chain of Responsibility)
After the bandit selects an action, a **Confidence Gate** evaluates the output:
- Compute confidence via **self-consistency**: ask the LLM the same question twice with different temperatures. If answers agree, high confidence. If they disagree, low confidence, then escalate.
- This is a safety net that catches the bandit's mistakes without requiring expensive ToT on every query.
- Architecturally, this implements the **Chain of Responsibility** design pattern.

### Contribution 3: Dynamic Budget Tracking
The system operates under a **fixed token budget** (e.g., 100,000 tokens for 200 questions). As the budget depletes:
- The bandit's state vector includes `remaining_budget_pct`
- Early in the session, the agent can afford expensive strategies (ToT on 7b model)
- Late in the session, the agent automatically shifts to cheap strategies (Direct on 3b model)
- This models **real-world API billing constraints** that no other routing paper addresses dynamically

---

## 5. Feasibility Assessment (1-2 Week Timeline)

### Week 1: Implementation
| Day | Task | Details |
|-----|------|---------|
| 1-2 | Expand action space to Model x Strategy | Modify `run_experiment.py` to support 2 models (3b + 7b) x 4 strategies = 8 actions. Update LinUCB `n_actions=8`. |
| 3 | Implement Budget Tracker | Add `BudgetTracker` class: tracks cumulative tokens, injects `remaining_pct` into bandit state. |
| 4 | Implement Confidence Gate | Add `ConfidenceGate`: runs the selected strategy twice at temp=0.2 and temp=0.8. If answers match, accept. If not, escalate to next-tier strategy. |
| 5-6 | Run Full Experiments | 3 datasets x 500 questions x 3 seeds x all agents. This will take significant compute time. |
| 7 | Generate Figures and Tables | Pareto front, ablation tables, cost analysis, learning curves. |

### Week 2: Paper Writing
| Day | Task |
|-----|------|
| 1 | Abstract + Introduction + Related Work |
| 2 | Method Section (Architecture, Bandit formulation, Cascade) |
| 3 | Experiments + Results Tables |
| 4 | Analysis + Ablations + Discussion |
| 5 | Polish, figures, references, supplementary |

---

## 6. Required Experiments for the Paper

### Main Results Table
- Compare **CascadeRoute** (LinUCB with budget + cascade) vs:
  - `always_direct`, `always_cot`, `always_tot` (static baselines)
  - `random` routing
  - `RouteLLM-style` (binary model routing without strategy selection)
  - `RTR-style` (offline optimal, no online learning)
- Metrics: **Accuracy**, **Avg Tokens**, **Avg Latency**, **Composite Reward**, **Budget Utilization %**

### Ablation Table
| Ablation | What It Proves |
|----------|---------------|
| No budget tracking (remove budget from state) | Budget-awareness improves end-of-session performance |
| No confidence cascade (disable escalation) | Cascade catches ~X% of bandit mistakes |
| No cache (disable Proxy) | Cache reduces latency by X% |
| Single model only (remove model routing) | Joint model+strategy > strategy-only |
| Random features (replace extractor) | Learned features improve routing accuracy |

### Cost Analysis Table
| Agent | Accuracy | Tokens Used | Simulated API Cost ($) | Cost Savings vs. always_tot |
|-------|----------|-------------|----------------------|---------------------------|
| always_direct | -- | -- | -- | -- |
| always_tot | -- | -- | -- | baseline |
| CascadeRoute | -- | -- | -- | **X% savings** |

### Learning Curve
- Plot cumulative reward over time for CascadeRoute vs. baselines
- Show that the bandit improves within the first 50-100 queries

---

## 7. Paper Structure (ICLR Format)

```
1. Abstract (250 words)
2. Introduction
   - LLM cost problem
   - Limitation of static prompting
   - Our contribution: online joint routing with budget + cascade
3. Related Work
   - RouteLLM, FrugalGPT, AutoMix, RTR, DOTS
   - Position our work clearly
4. Method
   4.1 Problem Formulation (contextual bandit + budget)
   4.2 Joint Action Space (model x strategy)
   4.3 Confidence-Gated Cascade
   4.4 Budget-Aware State Representation
5. Experiments
   5.1 Setup (datasets, models, baselines)
   5.2 Main Results
   5.3 Ablation Studies
   5.4 Cost Analysis
   5.5 Learning Dynamics
6. Discussion and Limitations
7. Conclusion
```

---

## 8. Key References to Cite

1. **RouteLLM** — Ong et al., "Learning to Route LLMs with Preference Data," ICLR 2025. arXiv:2406.18665
2. **FrugalGPT** — Chen et al., "How to Use Large Language Models While Reducing Cost," 2023. arXiv:2305.05176
3. **AutoMix** — Aggarwal et al., "Automatically Mixing Language Models," 2024. arXiv:2310.12963
4. **Route-To-Reason** — "Dynamic Reasoning Strategy and Expert Model Allocation," 2025. arXiv:2505.19435
5. **DeepSeek-R1** — "Incentivizing Reasoning Capability in LLMs via RL," 2025. arXiv:2501.12948
6. **Chain-of-Thought** — Wei et al., NeurIPS 2022
7. **Tree-of-Thoughts** — Yao et al., NeurIPS 2023
8. **Graph-of-Thoughts** — Besta et al., AAAI 2024
9. **LinUCB** — Li et al., "A Contextual-Bandit Approach to Personalized News," WWW 2010
10. **C3PO** — "Cost Controlled Cascaded Prediction Optimization," 2025

---

## 9. Honest Risk Assessment

| Factor | Assessment |
|--------|-----------|
| **Novelty** | Strong. Online RL + joint routing + budget + cascade is genuinely unexplored. |
| **Experimental rigor** | Medium risk. Local Ollama models (Qwen 3b/7b) are not GPT-4 vs Mixtral. Reviewers may want larger model gaps. Mitigate by showing the framework is model-agnostic. |
| **Scale concern** | Medium risk. 3 datasets x 500 questions is acceptable but not overwhelming. GSM8K + ARC + CommonsenseQA is a solid trio. |
| **ICLR fit** | Strong fit for the "Efficient ML" or "Applications" track. |
| **Timeline risk** | Tight. 1 week for implementation + 1 week for writing is aggressive but doable if you focus on the 3 contributions above and don't over-engineer. |

> **Bottom line:** This architecture is publishable at ICLR if the experiments are rigorous and the paper is well-written. The combination of online RL + joint routing + budget awareness + cascade is a clear gap in the literature that no existing paper fills.
