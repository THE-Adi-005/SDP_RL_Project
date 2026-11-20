# AdaptInfer: Distributed Microservices Architecture for LLM Strategy Orchestration

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Ray](https://img.shields.io/badge/Ray-Distributed-028CF0.svg)](https://www.ray.io/)
[![Springer Nature](https://img.shields.io/badge/Journal-Springer%20Nature%20(sn--jnl)-orange.svg)](journal/)
[![Tests Passing](https://img.shields.io/badge/tests-50%2F50%20passed-brightgreen.svg)](tests/)

> **Journal Publication Draft:** *AdaptInfer: A Scalable Distributed Microservices Architecture and Dynamic Strategy Orchestration Engine for Cost-Aware LLM Inference*  
> **Coursework Stream:** Software Development Project (SDP) & Reinforcement Learning (RL)  
> **Manuscript Location:** [`journal/sn-article.tex`](journal/sn-article.tex) (Compiled: [`journal/sn-article.pdf`](journal/sn-article.pdf))  
> **Comprehensive Report:** [`report/SDP_Report.md`](report/SDP_Report.md) & [`report/SDP_Report.tex`](report/SDP_Report.tex)  
> **Slide Deck:** [`presentation/SLIDE_CONTENT_AND_LAYOUT.md`](presentation/SLIDE_CONTENT_AND_LAYOUT.md)  

---

## 1. Executive Overview

Modern LLM deployment faces an acute systems engineering bottleneck: statically assigning exhaustive reasoning strategies (e.g., Tree-of-Thoughts) across all incoming traffic incurs prohibitive median latencies (**45.2s**) and massive token overhead (**1,959 tokens/query**), whereas static zero-shot prompting on lightweight models suffers a **46.1% error rate** on complex multi-step reasoning.

**AdaptInfer** solves this trade-off through an asynchronous distributed microservices architecture that dynamically orchestrates model parameter tier (`Qwen2.5-3B` vs. `Qwen2.5-7B`) and reasoning pathway (`Direct`, `Chain-of-Thought`, `Graph-of-Thoughts`, `Tree-of-Thoughts`) under hard session token budget constraints.

### Key Empirical Results (3,712 Benchmark Queries: GSM8K, LogiQA, CommonsenseQA)
- **Accuracy Parity:** Reaches **81.4% accuracy**, matching the theoretical Tree-of-Thoughts skyline (**82.0%**).
- **Latency Reduction:** Slashes average response latency by **67.2%** (**14.8s** vs. **45.2s**).
- **Token Efficiency:** Saves **27.5% in token expenditure** (**1,421** vs. **1,959 tokens/query**).

---

## 2. System Architecture & Design Patterns

```text
                           ┌──────────────────────────────────────────────┐
                           │      Incoming Query + Session Budget B_max   │
                           └──────────────────────┬───────────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │  FastAPI Asynchronous Gateway       │
                               │  (/api/solve, /api/history, /docs)  │
                               └──────────────────┬──────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │   390-d Context Vector Extractor    │
                               │   • 384-d all-MiniLM-L6-v2 Semantic │
                               │   • Structural & Domain Indicators  │
                               │   • Remaining Budget Ratio b_t      │
                               └──────────────────┬──────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │  LinUCB Contextual Decision Core    │
                               │  Argmax [ theta_a^T s_t + alpha*unc]│
                               └──────────────────┬──────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │  Chain of Responsibility Guardrail  │
                               │  Confidence Gate (T=0.2 vs. T=0.8)  │
                               └─────────┬───────────────────┬───────┘
                     Consensus (y1 == y2)│                   │ Divergence (y1 != y2)
                                         ▼                   ▼
                          ┌──────────────────────┐  ┌──────────────────────┐
                          │ Execute Selected a_t │  │ Escalate to High-Cap │
                          └──────────┬───────────┘  └──────────┬───────────┘
                                     └───────────┬─────────────┘
                                                 ▼
                               ┌─────────────────────────────────────┐
                               │ Ray Distributed Worker Pool         │
                               │ (Ollama Local Inference Runtime)    │
                               └─────────────────────────────────────┘
```

The system strictly adheres to classic object-oriented software engineering patterns:
1. **Strategy Pattern (`src/strategies/`):** Polymorphic reasoning executors (`DirectStrategy`, `CoTStrategy`, `GoTStrategy`, `ToTStrategy`) inheriting from `BaseStrategy`.
2. **Chain of Responsibility Pattern (`src/cascaderoute/confidence_gate.py`):** Temperature-scaled dual-sampling consensus gate that intercepts uncertainty and escalates queries.
3. **Adapter Pattern (`src/llm/client.py`):** Encapsulates local Ollama HTTP REST endpoints behind a unified generation interface.

---

## 3. Directory Structure

```text
F:\ADI\SDP/
├── journal/               # Springer Nature journal paper source, bib, styles, and figures
│   ├── sn-article.tex     # Main Springer Nature LaTeX article source
│   ├── sn-article.pdf     # Compiled publication PDF
│   ├── sn-bibliography.bib# Bibliography entries
│   ├── sn-jnl.cls         # Springer Nature document class
│   └── *.png, *.eps       # Architecture, Pareto, latency, and token comparison charts
├── report/                # Coursework report & alternate journal drafts
│   ├── SDP_Report.tex     # Comprehensive 67 KB LaTeX coursework report
│   ├── SDP_Report.md      # Markdown report
│   └── SDP_Paper_IJCS.tex # IJCS submission version
├── presentation/          # Defense slide deck materials & figures
│   └── SLIDE_CONTENT_AND_LAYOUT.md # Slide by slide narration and diagram placements
├── dashboard/             # Full-Stack Interactive Telemetry Single-Page Application
│   ├── app.py             # FastAPI backend server
│   └── static/            # Frontend (index.html, main.js, style.css)
├── src/                   # Core Python implementation
│   ├── strategies/        # BaseStrategy, Direct, CoT, GoT, ToT implementations
│   ├── llm/               # Ollama client adapter and structured parser
│   ├── cascaderoute/      # Confidence Gate & Dynamic Budget Tracker
│   ├── rl/                # LinUCB, UCB1, Thompson Sampling, DQN bandits
│   ├── features/          # Sentence-BERT semantic feature extractor
│   └── evaluation/        # Cost, latency, correctness evaluators
├── configs/               # System configurations (strategies, models, budgets)
├── prompts/               # Prompt templates (direct, cot, got, tot)
├── scripts/               # Utility scripts & asset generators
│   └── generate_sdp_assets.py # Generates all 6 paper/report figures
├── data/                  # Cleaned benchmark data (clean, results, processed)
├── docs/                  # Design documents, workflows, and developer guides
└── tests/                 # Unit test suite (50 passing tests)
```

---

## 4. How to Compile the Journal Manuscript

To compile the Springer Nature journal paper locally:
```bash
cd journal
tectonic sn-article.tex
# Or with standard TeX Live:
# pdflatex sn-article
# bibtex sn-article
# pdflatex sn-article
# pdflatex sn-article
```
Output PDF will be updated at `journal/sn-article.pdf`.

---

## 5. Running the Test Suite & Dashboard

```bash
# 1. Run all unit tests
pytest tests/ -v

# 2. Launch interactive telemetry dashboard
uvicorn dashboard.app:app --reload --port 8000
```
Open `http://localhost:8000` to inspect live strategy racing and telemetry.
