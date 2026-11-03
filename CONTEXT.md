# CONTEXT.md — CascadeRoute: RL-Driven Adaptive LLM Routing

> **This file is the single source of truth for any LLM assistant, developer, or collaborator joining this project.**
> Read this file first before exploring any other file.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project Name** | CascadeRoute |
| **Full Title** | CascadeRoute: Reinforcement Learning-Driven Cascade Routing for Cost-Efficient LLM Reasoning |
| **Author** | Adithya (ADITHYA_23001) |
| **Type** | Research Project (SDP — Software Design Patterns) |
| **Status** | Experiments complete; two paper submissions in progress |
| **Paper 1** | BVICAM National Conference (Springer CCIS) — `BVICAM/CascadeRoute_Springer.tex` |
| **Paper 2** | ICLR 2027 Conference — `iclr2027/CascadeRoute_ICLR2027.tex` |

---

## 2. What is CascadeRoute?

CascadeRoute is an **RL-based adaptive routing system** that decides, for each incoming LLM query, which combination of:
- **Model size** (3B vs 7B parameters — using Qwen2.5 via Ollama locally), and
- **Prompting strategy** (Direct / Chain-of-Thought / Tree-of-Thought / Graph-of-Thought)

…will maximize **reasoning accuracy while minimizing token cost and latency**.

### Core Research Contribution

Instead of always using a large expensive model, CascadeRoute learns (using a **LinUCB contextual bandit**) which strategy works best for different query types. It achieves near-Oracle accuracy at a fraction of the cost — solving the **cost-accuracy Pareto frontier** problem.

### Key Innovation: The Confidence Gate

After the bandit selects a strategy, a **Confidence Gate** runs the query **twice** at different temperatures:
- If both answers agree → return confidently
- If they disagree → **automatically escalate** to a stronger model/strategy (cascade)

This means the system **self-corrects** without human intervention.

---

## 3. Architecture Overview

```
User Query + Category
        │
        ▼
[Feature Extractor]  ← SentenceTransformers (384-dim embedding + 5 metadata dims)
        │
        ▼
[LinUCB Bandit]  ← Selects 1 of 6 actions based on contextual state
        │
   Action Space:
   ┌──────────────┐
   │ 3B_Direct    │
   │ 3B_CoT       │
   │ 7B_Direct    │
   │ 7B_CoT       │
   │ 7B_GoT       │
   │ 7B_ToT       │
   └──────────────┘
        │
        ▼
[Confidence Gate]  ← Runs strategy twice, checks self-consistency
     ┌──┴──┐
  Match?   No → Escalate to stronger strategy
     │
     ▼
[Evaluator]  ← Checks answer correctness, counts tokens
     │
     ▼
[Reward Function]  ← accuracy - token_penalty - latency_penalty
     │
     ▼
[Bandit Update]  ← LinUCB learns from reward
```

---

## 4. Directory Structure

```
SDP_RL_Project/
│
├── CONTEXT.md                  ← YOU ARE HERE — read this first
├── requirements.txt            ← Python dependencies
├── .env.example                ← Environment config template
├── .env                        ← Real config (gitignored — create from .env.example)
├── .gitignore
├── architecture_diagram.png    ← High-level system architecture visual
│
├── src/                        ← ALL core Python source code
│   ├── __init__.py
│   ├── cascaderoute/           ← Main orchestration module
│   │   ├── run_cascaderoute.py ← Primary experiment runner logic
│   │   ├── confidence_gate.py  ← Self-consistency check + cascade escalation
│   │   └── budget_tracker.py   ← Token budget management
│   ├── rl/                     ← Reinforcement Learning algorithms
│   │   ├── bandit.py           ← Base bandit class
│   │   ├── linucb.py           ← LinUCB contextual bandit (PRIMARY ALGORITHM)
│   │   ├── ucb.py              ← UCB1 bandit
│   │   ├── thompson.py         ← Thompson Sampling bandit
│   │   ├── dqn.py              ← Deep Q-Network (ablation/comparison)
│   │   └── environment.py      ← OpenAI Gym-style RL environment
│   ├── llm/                    ← LLM client abstraction
│   │   ├── client.py           ← Ollama API client (OpenAI-compatible)
│   │   └── parser.py           ← Answer extraction from LLM responses
│   ├── strategies/             ← Prompting strategy implementations
│   │   ├── base.py             ← Abstract base strategy
│   │   ├── direct.py           ← Direct answer prompting
│   │   ├── cot.py              ← Chain-of-Thought prompting
│   │   ├── tot.py              ← Tree-of-Thought prompting
│   │   └── got.py              ← Graph-of-Thought prompting
│   ├── evaluation/             ← Metrics, analysis, and visualization
│   │   ├── evaluator.py        ← Core evaluator (correctness + cost)
│   │   ├── correctness.py      ← Answer correctness checker
│   │   ├── cost.py             ← Cost/token metric computation
│   │   ├── regret.py           ← Bandit regret analysis
│   │   ├── statistical_tests.py← Wilcoxon, Mann-Whitney significance tests
│   │   ├── analyze_domain.py   ← Per-domain performance breakdown
│   │   ├── run_full_baselines.py← Baseline comparison runner
│   │   └── visualize_iclr.py   ← ICLR-quality figure generation
│   ├── features/
│   │   └── extractor.py        ← Query feature extraction (embedding + metadata)
│   ├── policies/               ← Simple baseline policies for comparison
│   │   ├── always_x_policy.py  ← Always picks a fixed strategy (ablation)
│   │   ├── random_policy.py    ← Random strategy selection
│   │   └── rule_based_policy.py← Heuristic rule-based policy
│   └── utils/
│       └── caching.py          ← LLM response caching (avoid redundant API calls)
│
├── configs/                    ← YAML configuration files
│   ├── datasets.yaml           ← Dataset paths and split sizes
│   ├── experiment.yaml         ← Experiment hyperparameters
│   ├── model.yaml              ← Model names and API endpoints
│   ├── reward.yaml             ← Reward function weights
│   └── strategies.yaml         ← Strategy action space definitions
│
├── prompts/                    ← Raw LLM prompt templates
│   ├── direct.txt              ← Direct answer prompt
│   ├── cot.txt                 ← Chain-of-Thought prompt
│   ├── tot.txt                 ← Tree-of-Thought prompt
│   └── got.txt                 ← Graph-of-Thought prompt
│
├── data/
│   ├── processed/              ← Pre-processed dataset CSVs (ready for experiments)
│   │   ├── gsm8k.csv           ← GSM8K math reasoning dataset
│   │   ├── gsm8k_sample.csv    ← Sampled subset for quick runs
│   │   ├── logiqa.csv          ← LogiQA logical reasoning dataset
│   │   ├── logiqa_sample.csv
│   │   ├── commonsenseqa.csv   ← CommonsenseQA dataset
│   │   └── commonsenseqa_sample.csv
│   └── results/                ← Raw experiment output CSVs
│       ├── cascaderoute_results.csv      ← Main CascadeRoute experiment results
│       └── full_baselines_results.csv    ← All baseline algorithm results
│
├── experiments/                ← (Placeholder for future experiment configs)
│
├── reports/                    ← Generated analysis reports
│   ├── CascadeRoute_Deep_Dive.md         ← Deep technical analysis
│   ├── cascaderoute_summary.md           ← Executive summary
│   ├── figures/                ← All generated plots (14 PNG files)
│   │   ├── accuracy_by_policy.png
│   │   ├── cost_accuracy_tradeoff.png
│   │   ├── fig2_accuracy_by_agent.png
│   │   ├── fig3_pareto_front.png
│   │   ├── fig5_cumulative_reward.png
│   │   ├── fig6_strategy_heatmap.png
│   │   ├── iclr_pareto_frontier.png
│   │   └── ... (14 total)
│   └── tables/                 ← LaTeX tables and CSV data tables
│       ├── iclr_performance_table.tex
│       ├── table3_agent_comparison.csv
│       └── ...
│
├── docs/                       ← Developer documentation
│   ├── build_guide.md          ← Full step-by-step setup guide (Ollama + local models)
│   ├── architecture_and_execution.md ← Detailed architecture + execution walkthrough
│   ├── CASCADEROUTE_COMMANDS.md      ← Quick command reference
│   ├── Datasets_EDA.md              ← Dataset exploratory data analysis
│   ├── dashboard_analysis_guide.md  ← Dashboard usage guide
│   └── ICLR_2027_proposal.md        ← Full ICLR 2027 research proposal
│
├── tests/                      ← Pytest test suite
│   ├── conftest.py
│   ├── test_algorithms.py      ← RL algorithm unit tests
│   ├── test_correctness.py     ← Answer correctness tests
│   └── test_evaluation.py      ← Evaluator unit tests
│
├── dashboard/                  ← Interactive results dashboard (Flask)
│   ├── app.py                  ← Flask backend with REST API
│   └── static/
│       ├── index.html          ← Dashboard frontend
│       ├── main.js             ← Charts and interactivity (Chart.js)
│       └── style.css           ← Dashboard styling
│
├── BVICAM/                     ← Springer CCIS paper (BVICAM National Conference)
│   ├── CascadeRoute_Springer.tex   ← Main LaTeX paper (ACTIVE DOCUMENT)
│   ├── references.bib              ← Bibliography
│   ├── sn-jnl.cls                  ← Springer journal class file
│   └── *.png                       ← Paper figures (4 PNGs)
│
└── iclr2027/                   ← ICLR 2027 conference paper
    ├── CascadeRoute_ICLR2027.tex   ← Main ICLR paper draft
    ├── math_commands.tex           ← Shared LaTeX math macros
    ├── iclr2027_conference.bib     ← ICLR bibliography
    ├── iclr2027_conference.sty     ← ICLR style file
    ├── iclr2027_conference.bst     ← ICLR bibliography style
    ├── fancyhdr.sty / natbib.sty   ← LaTeX packages
    └── *.png                       ← Paper figures (4 PNGs)
```

---

## 5. Key Entry Points

| Task | Command |
|---|---|
| **Run main experiment** | `python run_experiment.py` |
| **Run ablation study** | `python run_ablation.py` |
| **Analyze results** | `python analyze_results.py` |
| **Analyze paper metrics** | `python analyze_paper.py` |
| **Generate ICLR figures** | `python generate_iclr_visuals.py` |
| **Download datasets** | `python download_datasets.py` |
| **Run dashboard** | `python -m dashboard.app` (then open `http://localhost:5000`) |
| **Run tests** | `pytest tests/` |

---

## 6. Setup Instructions

### Prerequisites
- Python 3.10 or 3.11
- [Ollama](https://ollama.com) installed and running locally
- GPU recommended (6GB+ VRAM); CPU is possible but slow

### Step 1: Install Ollama models
```bash
ollama pull qwen2.5:7b-instruct-q4_K_M   # 7B model (primary)
ollama pull qwen2.5:3b-instruct-q4_K_M   # 3B model (fast/cheap)
ollama serve                               # Start if not auto-running
```

### Step 2: Create Python environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Configure environment
```bash
cp .env.example .env
# Edit .env if needed — default values work with local Ollama
```

`.env` contents:
```
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
MODEL_NAME=qwen2.5:7b-instruct-q4_K_M
```

### Step 4: Run experiment
```bash
python run_experiment.py
```

---

## 7. Datasets Used

| Dataset | Domain | Task | Size (processed) |
|---|---|---|---|
| **GSM8K** | Mathematics | Grade-school math word problems | ~1,319 rows |
| **LogiQA** | Logic | Multiple-choice logical reasoning | ~651 rows |
| **CommonsenseQA** | Commonsense | Multiple-choice commonsense | ~1,221 rows |

All datasets are pre-processed and stored in `data/processed/`. Each row contains `question`, `answer`, `category`.

---

## 8. RL Algorithms Implemented

| Algorithm | File | Type | Description |
|---|---|---|---|
| **LinUCB** | `src/rl/linucb.py` | Contextual Bandit | **PRIMARY** — uses query embedding as context |
| **UCB1** | `src/rl/ucb.py` | Multi-armed Bandit | Context-free upper confidence bound |
| **Thompson Sampling** | `src/rl/thompson.py` | Multi-armed Bandit | Bayesian exploration |
| **DQN** | `src/rl/dqn.py` | Deep RL | Neural network Q-learning (ablation) |
| **Random** | `src/policies/random_policy.py` | Baseline | Random strategy selection |
| **Rule-based** | `src/policies/rule_based_policy.py` | Baseline | Heuristic routing |

---

## 9. Action Space (6 Strategies)

| Action ID | Model | Strategy | Relative Cost |
|---|---|---|---|
| 0 | 3B | Direct | ⭐ Cheapest |
| 1 | 3B | CoT | ⭐⭐ |
| 2 | 7B | Direct | ⭐⭐ |
| 3 | 7B | CoT | ⭐⭐⭐ |
| 4 | 7B | GoT | ⭐⭐⭐⭐ |
| 5 | 7B | ToT | ⭐⭐⭐⭐⭐ Most Expensive |

---

## 10. Reward Function

```
reward = accuracy_score
       - (token_cost / budget) * token_penalty_weight
       - latency_penalty_weight * latency
```

Weights are configured in `configs/reward.yaml`.

---

## 11. Experimental Results Summary

From `data/results/cascaderoute_results.csv` and `reports/tables/`:

- **CascadeRoute (LinUCB)** achieves **~85–90% accuracy** across all datasets
- Token cost reduced by **~40–60%** vs always using 7B_ToT
- Outperforms all baselines (random, rule-based, UCB, Thompson) on cost-accuracy Pareto frontier
- Statistical significance confirmed via Wilcoxon signed-rank tests (p < 0.05)

---

## 12. Paper Status

### BVICAM (Springer CCIS)
- File: `BVICAM/CascadeRoute_Springer.tex`
- Template: `BVICAM/sn-jnl.cls` (Springer Nature journal class)
- Status: Drafted, under revision

### ICLR 2027
- File: `iclr2027/CascadeRoute_ICLR2027.tex`
- Template: `iclr2027/iclr2027_conference.sty`
- Status: Full draft complete
- Proposal: `docs/ICLR_2027_proposal.md`

---

## 13. Software Design Patterns Used

This is also an SDP (Software Design Patterns) course project. Patterns implemented:

| Pattern | Where Used |
|---|---|
| **Strategy Pattern** | `src/strategies/` — swappable prompting strategies |
| **Chain of Responsibility** | `src/cascaderoute/confidence_gate.py` — cascade escalation |
| **Facade Pattern** | `src/cascaderoute/run_cascaderoute.py` — unified experiment interface |
| **Template Method** | `src/strategies/base.py` — base strategy with abstract methods |
| **Observer Pattern** | `src/cascaderoute/budget_tracker.py` — monitors token usage |

---

## 14. Important Notes for LLMs

1. **No OpenAI API needed** — uses local Ollama models (qwen2.5). The `OLLAMA_API_KEY=ollama` is a dummy key for the OpenAI-compatible API format.
2. **The `.env` file is gitignored** — always copy from `.env.example` on a new machine.
3. **Data files are gitignored** — `data/raw/`, `data/processed/`, `data/results/` are not tracked. CSVs are included in this export ZIP.
4. **The primary RL algorithm is LinUCB** (`src/rl/linucb.py`) — everything else is a baseline for comparison.
5. **The active paper** being edited is `BVICAM/CascadeRoute_Springer.tex`.
6. **Run `docs/build_guide.md`** for full step-by-step setup from scratch on a new machine.
7. **All configs** live in `configs/*.yaml` — experiment parameters, model names, reward weights are all tunable there without touching source code.

---

*Generated: 2026-09-04 | Context file for CascadeRoute SDP_RL_Project*
