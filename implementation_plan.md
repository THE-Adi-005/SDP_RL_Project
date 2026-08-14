# Adaptive LLM Reasoning — Implementation Plan

Build the complete project from [build_guide.md](file:///c:/Users/Adithya R/SDP_RL/build_guide.md) in the workspace `c:\Users\Adithya R\SDP_RL`.

## Summary

This project builds an **Adaptive LLM Reasoning** system that uses **reinforcement learning** (epsilon-greedy bandit) to dynamically select the best reasoning strategy (Direct, Chain-of-Thought, Graph-of-Thought, Tree-of-Thought) for each problem, optimizing a composite reward of correctness, token cost, and latency.

The system uses **Ollama** (local LLM inference) with **Qwen 2.5** models, benchmarked across **GSM8K** (math), **LogiQA** (logic), and **CommonsenseQA** (commonsense) datasets.

## User Review Required

> [!IMPORTANT]
> **Ollama Prerequisite**: This project requires Ollama to be installed and running locally (`ollama serve`) with models pulled (`qwen2.5:7b-instruct-q4_K_M` and `qwen2.5:3b-instruct-q4_K_M`). I will **not** install Ollama or pull models — please confirm those are available before running experiments.

> [!IMPORTANT]
> **GPU/Hardware**: The guide targets 6GB VRAM / 24GB RAM. The LLM inference (baseline runs, bandit training) will be slow without a GPU. Confirm you're ready for this.

> [!WARNING]
> **Dataset Downloads**: Phase 5 downloads datasets from HuggingFace (GSM8K, LogiQA, CommonsenseQA). This requires internet access and may take a few minutes.

## Open Questions

1. **Should I run the actual experiments** (baseline, bandit) after building the code, or just build the codebase and leave execution to you? Running them requires Ollama with models loaded and will take significant time.
2. **Python version**: The guide specifies Python 3.11. Should I use your system's default Python, or do you have 3.11 specifically installed?

## Proposed Changes

I will build all 20 phases from the guide as source files in `c:\Users\Adithya R\SDP_RL\`.

---

### Phase 2 — Environment Setup

#### [NEW] [requirements.txt](file:///c:/Users/Adithya R/SDP_RL/requirements.txt)
All Python dependencies (pandas, numpy, scikit-learn, torch, stable-baselines3, openai, sentence-transformers, etc.)

#### [NEW] [.env.example](file:///c:/Users/Adithya R/SDP_RL/.env.example)
Ollama connection config template.

#### [NEW] [.env](file:///c:/Users/Adithya R/SDP_RL/.env)
Copy of .env.example with defaults.

#### [NEW] [.gitignore](file:///c:/Users/Adithya R/SDP_RL/.gitignore)
Standard Python gitignore + project-specific exclusions.

---

### Phase 3 — Directory Structure

Create all directories and `__init__.py` files:
- `configs/`, `data/raw/`, `data/processed/`, `data/results/`
- `prompts/`
- `src/llm/`, `src/strategies/`, `src/evaluation/`, `src/features/`, `src/rl/`, `src/policies/`, `src/utils/`
- `experiments/baseline/`, `experiments/bandit/`, `experiments/rl/`, `experiments/ablation/`
- `notebooks/`, `reports/figures/`, `reports/tables/`, `tests/`

---

### Phase 4 — Config Files

#### [NEW] configs/model.yaml, strategies.yaml, reward.yaml, datasets.yaml, experiment.yaml

---

### Phase 5 — Dataset Scripts

#### [NEW] [download_datasets.py](file:///c:/Users/Adithya R/SDP_RL/download_datasets.py)
#### [NEW] [sample_datasets.py](file:///c:/Users/Adithya R/SDP_RL/sample_datasets.py)

---

### Phase 6 — Prompt Templates

#### [NEW] prompts/direct.txt, cot.txt, got.txt, tot.txt

---

### Phase 7 — LLM Client

#### [NEW] [src/llm/client.py](file:///c:/Users/Adithya R/SDP_RL/src/llm/client.py)
OpenAI-compatible client wrapping Ollama's API.

#### [NEW] [src/llm/parser.py](file:///c:/Users/Adithya R/SDP_RL/src/llm/parser.py)
Answer extraction utilities.

---

### Phase 8 — Caching Layer

#### [NEW] [src/utils/caching.py](file:///c:/Users/Adithya R/SDP_RL/src/utils/caching.py)

---

### Phase 9 — Reasoning Strategies

#### [NEW] src/strategies/base.py, direct.py, cot.py, tot.py, got.py

---

### Phase 10 — Evaluation Modules

#### [NEW] src/evaluation/correctness.py, cost.py, evaluator.py

---

### Phase 11 — Baseline Runner

#### [NEW] [run_baseline.py](file:///c:/Users/Adithya R/SDP_RL/run_baseline.py)

---

### Phase 12 — Feature Extractor

#### [NEW] [src/features/extractor.py](file:///c:/Users/Adithya R/SDP_RL/src/features/extractor.py)

---

### Phase 13–14 — RL Components

#### [NEW] [src/rl/bandit.py](file:///c:/Users/Adithya R/SDP_RL/src/rl/bandit.py)
#### [NEW] [src/rl/environment.py](file:///c:/Users/Adithya R/SDP_RL/src/rl/environment.py)

---

### Phase 15 — Bandit Runner

#### [NEW] [run_bandit.py](file:///c:/Users/Adithya R/SDP_RL/run_bandit.py)

---

### Phase 16 — Non-RL Baseline Policies

#### [NEW] src/policies/random_policy.py, always_x_policy.py, rule_based_policy.py

---

### Phase 17 — Analysis & Plotting

#### [NEW] [analyze_results.py](file:///c:/Users/Adithya R/SDP_RL/analyze_results.py)

---

### Phase 18 — Tests

#### [NEW] [tests/test_correctness.py](file:///c:/Users/Adithya R/SDP_RL/tests/test_correctness.py)

---

## Verification Plan

### Automated Tests
```bash
pytest tests/ -v
```

### Manual Verification
- Confirm all files created with correct structure
- Confirm `pip install -r requirements.txt` succeeds
- Confirm dataset download and sampling works
- Confirm feature extractor produces correct shape output
- Actual LLM experiments require Ollama running (user-driven)

## Execution Order

1. Create all directories and `__init__.py` files
2. Create config files (requirements.txt, .env, .gitignore, YAML configs)
3. Create prompt templates
4. Create all source modules (Phases 7–16)
5. Create runner scripts (run_baseline.py, run_bandit.py, analyze_results.py)
6. Create tests
7. Set up virtual environment and install dependencies
8. Download and sample datasets
9. Run tests to verify
