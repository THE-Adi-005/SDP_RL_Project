# CascadeRoute: Project Briefing & Hand-off Document

## 1. PROJECT OVERVIEW
**Goal:** Solve the Pareto Frontier trade-off between inference accuracy and computational cost in Large Language Models (LLMs). Simple queries can be solved cheaply via Direct prompting, while complex queries demand expensive Tree-of-Thoughts (ToT).
**Core Idea:** Formulate LLM routing as a Stateless Environment and use a Reinforcement Learning (RL) Contextual Bandit (LinUCB) to dynamically route queries to the optimal Model $\times$ Strategy combination, minimizing token cost while maximizing accuracy.
**Deliverable Scope:** The repository serves a dual academic purpose. 
*   **RL Track:** Focuses on algorithmic intelligence (LinUCB formulation, state/action spaces, learning speed). Target: ICLR 2027 format (`iclr2027/`).
*   **SDP Track:** Focuses on Software Development Systems Architecture (Ray distributed backend, API/UI, Object-Oriented patterns like Confidence Gate). Target: BVICAM Journal via Springer Nature (`SDP_Journal/`).

## 2. REPO MAP
*   `dashboard/` - FastAPI/HTML/JS UI Application layer. Entry point: `cd dashboard && python -m uvicorn app:app --reload --port 8080`.
*   `src/` - Core logic.
    *   `src/rl/linucb.py` - Contains the LinUCB contextual bandit algorithm.
    *   `src/strategies/` - Strategy pattern definitions for `direct`, `cot`, `got`, and `tot`.
    *   `src/cascaderoute/` - Pipeline orchestrator (Confidence Gate).
*   `data/` - Holds raw JSON datasets and generated `.csv` results.
*   `RL/` - Contains tailored Markdown reports and PPT outlines for the RL class defense.
*   `SDP/` - Contains tailored Markdown reports and PPT outlines for the SDP class defense.
*   `SDP_Journal/` - The academic manuscript (`sn-article.tex`) formatted for Springer.
*   `iclr2027/` - The academic manuscript formatted for ICLR.

## 3. TECHNICAL DESIGN
*   **Architecture (SDP):** A decoupled architecture using the **Ray** framework. The RL Agent acts as a central coordinator, dispatching LLM queries to distributed worker nodes running local Ollama instances to eliminate cloud latency variance. Features a `ConfidenceGate` safety net using Chain of Responsibility (samples at T=0.2 and T=0.8; if divergent, escalates the query).
*   **Models:** `qwen2.5:3b` and `qwen2.5:7b-instruct-q4_K_M` hosted locally via Ollama.
*   **RL Algorithm (RL):** Contextual Bandit `LinUCB`. We do not use Deep RL (DQN/PPO) because LLM routing is a stateless environment.
    *   *State Space:* 384-dim Sentence-BERT (`all-MiniLM-L6-v2`) + remaining Session Budget (%).
    *   *Action Space:* 6 actions (3B-Direct, 3B-CoT, 7B-Direct, 7B-CoT, 7B-GoT, 7B-ToT).
    *   *Reward:* $r_t = 100 \cdot \mathbb{I}(\text{correct}) - 10 \cdot (\text{Tokens}/\text{Tokens}_{\max}) - 5 \cdot (\text{Latency}/\text{Latency}_{\max})$.

## 4. DATA
*   **Datasets:** 3,712 total queries evaluated across 3 datasets:
    1.  `GSM8K` (Mathematical logic)
    2.  `LogiQA` (Complex reasoning)
    3.  `CommonsenseQA` (General knowledge)
*   **Features:** Raw text is preprocessed into 384-dimensional dense vectors via Sentence-BERT.

## 5. EXPERIMENTS AND RESULTS
**RL Ablations (Algorithm Selection):**
*   *DQN/PPO:* Failed. Highly unstable in a stateless environment.
*   *UCB1 (Stateless Bandit):* Achieved 68.7% accuracy. Converged on safe average models (7B-CoT) but failed to adapt contextually.
*   *LinUCB (Ours):* Achieved 81.4% accuracy (matching the 82.0% upper bound of 7B-ToT) while saving massive compute.

**SDP Ablations (Architectural Savings):**
*   *Static 7B-ToT (Baseline):* 82.0% Acc | 1,959 Tokens | 45.2s Latency.
*   *CascadeRoute (Ours):* 81.4% Acc | 1,421 Tokens | 14.8s Latency.
*   *Result:* Saved 538 tokens per query and reduced system average latency by 67%.

## 6. INFRASTRUCTURE AND REPRODUCIBILITY
*   **Environment:** Python 3.10+, FastAPI, Uvicorn, Ray, Ollama.
*   **Hardware Setup:** Requires local Ollama backend with pulled `qwen2.5:3b` and `7b` models (`ollama pull ...`). Uses Ray for distributed processing.
*   **Reproducibility:** A `HOWTO_RUN.md` file exists detailing how to boot the Ray head node, worker nodes, and run the FastAPI dashboard on `port 8080`. Note: Currently, a Uvicorn process for a CV project is hogging port 8000 on the host machine.

## 7. REPORT / PAPER STATE
*   `SDP_Journal/sn-article.tex`: Complete. Formatted for Springer Nature / BVICAM. Includes math formulas and latency evaluations.
*   `iclr2027/CascadeRoute_ICLR2027.tex`: Complete.
*   `RL/report/RL_Report.md`: Complete. Details LinUCB Ridge Regression updates and exploration mechanics.
*   `SDP/report/SDP_Report.md`: Complete. Details Ray infrastructure and OOP software design patterns.
*   **UI Status:** Dashboard successfully integrates live charts (Chart.js), cumulative stat tracking, and live "CascadeRoute Efficiency" comparison panels showing direct token/time savings against baseline models.

## 8. OPEN ISSUES
*   None identified. The system gracefully handles Ollama connection failures (recently patched in `app.py` to return clean JSON 500 errors instead of crashing the UI parser) and handles missing Sentence-BERT feature extraction (patched in `app.py:186` to prevent empty sequence `np.argmax` crashes).

## 9. ANSWERS I CANNOT GET FROM THE REPO
1.  Are there any strict constraints on presentation duration for the RL or SDP classes that require cutting down the generated PPT structures?
2.  Should the actual publication manuscripts (`sn-article.tex`) be compiled to PDF, and if so, what LaTeX engine does the user prefer?
