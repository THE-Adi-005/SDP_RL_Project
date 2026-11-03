# 🧠 Adaptive LLM Reasoning: Project Guide (Explained for a 5-Year-Old!)

Welcome to **Adaptive LLM Reasoning** (also known as **CascadeRoute**)! 

This guide explains what this project does, why it is super smart, and how all the code files work together — using fun analogies that anyone can understand!

---

## 🎈 Part 1: The Story (Understanding the Project from Scratch)

### 1. The Big Problem: Small Brains vs. Giant Brains 🤖

Imagine you have two robot helpers:

* 🏎️ **Little Robot (3B Model):** Super fast! It answers instantly and costs almost no energy (tokens). But if you ask it a tricky riddle or a hard math problem, it gets confused and gives the wrong answer.
* 🚀 **Giant Super-Robot (7B Model + Tree-of-Thoughts):** Super smart! It can solve the hardest puzzles by thinking step-by-step and trying many paths. But it takes **45 seconds** and uses **1,500 energy tokens**!

> **The Dilemma:**
> * If you use the **Giant Super-Robot** for *every single question* (even easy ones like `"15 + 27"`), you waste huge amounts of time and energy money!
> * If you use the **Little Robot** for *everything*, you fail all your hard homework!

---

### 2. The Solution: Meet CascadeRoute, The Smart Traffic Director 🚦

Instead of always using the fast robot or always using the giant robot, we built **CascadeRoute**!

Think of **CascadeRoute** as a clever **Traffic Director** sitting at a desk. 
When a new question arrives, the Traffic Director inspects the question:
* *"Is this easy math?"* ➡️ Sends it to the fast **3B Direct** robot!
* *"Is this a hard multi-step logic puzzle?"* ➡️ Sends it to the **7B Tree-of-Thoughts** super-robot!

This way, we get **81.4% top accuracy** while saving **538 tokens per query**! 🎉

---

### 3. The 6 Superhero Abilities (The Strategy Actions) ⚡

The Traffic Director can choose from **6 different strategies**:

| Action Name | Model Used | Reasoning Method | Best For | Speed | Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`3b_direct`** | 3B (Qwen 2.5) | Instant answer | Simple facts & easy arithmetic | ⚡ Ultra Fast | 🪙 1x |
| **`3b_cot`** | 3B (Qwen 2.5) | Chain-of-Thought (thinks step-by-step) | Medium arithmetic & simple logic | 🏎️ Fast | 🪙 3x |
| **`7b_direct`** | 7B (Qwen 2.5) | Instant answer | Harder knowledge questions | 🚗 Medium | 🪙 2x |
| **`7b_cot`** | 7B (Qwen 2.5) | Chain-of-Thought | Standard math & logic puzzles | 🐢 Slow | 🪙 5x |
| **`7b_got`** | 7B (Qwen 2.5) | Graph-of-Thoughts (connects idea nodes) | Complex interconnected riddles | 🐢 Slow | 🪙 10x |
| **`7b_tot`** | 7B (Qwen 2.5) | Tree-of-Thoughts (explores branch choices) | Ultra-hard multi-step reasoning | 🐌 Slower | 🪙 15x |

---

### 4. How the Traffic Director Learns (Reinforcement Learning) 🎮

How does the Traffic Director know which robot to pick? It learns by playing a game with a **Scorecard (Reward Formula)**:

$$\text{Reward} = \text{Correctness Score} - \text{Time Tax} - \text{Energy Tax}$$

1. **If the answer is WRONG:** Score is **0**! (Big punishment).
2. **If the answer is CORRECT:** Score is **1.0**, but we subtract a tiny tax for time spent and tokens used.
3. **The Lesson:** The director gets maximum points by getting the answer **CORRECT** while spending the **LEAST time and energy**!

We use a contextual bandit algorithm called **LinUCB** (Linear Upper Confidence Bound). It looks at features of the question (length, word types, domain) and picks the best strategy with confidence bars!

---

### 5. The Interactive Dashboard 📊

We built a live web dashboard (`http://127.0.0.1:8080`) where you can:
1. Type any question (e.g. `"What is 15 + 27?"`).
2. Watch all 6 strategies race against each other in real-time.
3. See which strategy the RL Agent recommended.
4. Inspect reasoning traces and dynamic Chart.js historical performance tabs!

---

## 📂 Part 2: Focused Directory Structure (RL & Dashboard Work)

Below is the file structure showing **only** the code responsible for Reinforcement Learning, strategy definitions, evaluation harness, and the interactive web dashboard:

```
SDP_RL_Project/
├── dashboard/                              # 🌐 Interactive FastAPI Web Dashboard
│   ├── app.py                              # FastAPI backend server (Ollama routing, LinUCB endpoint, /api/history)
│   ├── sample_questions.json               # 🧪 Machine-readable benchmark test suite (9 questions)
│   ├── sample_questions.md                 # 📖 Human-readable test suite with domain breakdown
│   └── static/                             # 🎨 Frontend Web User Interface
│       ├── index.html                      # Single Page Application layout (Live charts, history tabs, traces)
│       ├── main.js                         # Dynamic Chart.js renderers, cumulative stats tracker, API fetchers
│       └── style.css                       # Modern dark-mode styling & glassmorphism theme
│
├── src/                                    # 🛠️ Core Source Code Engine
│   ├── llm/                                # 🤖 LLM Client Interface
│   │   └── client.py                       # Unified Ollama API client (handles retries, timeouts, seed)
│   │
│   ├── strategies/                         # 💡 Reasoning Strategy Implementations
│   │   ├── base.py                         # Base class defining strategy output contract
│   │   ├── direct.py                       # Direct prompting strategy (3B & 7B)
│   │   ├── cot.py                          # Chain-of-Thought step-by-step strategy
│   │   ├── got.py                          # Graph-of-Thoughts node aggregation strategy
│   │   └── tot.py                          # Tree-of-Thoughts depth-first branch evaluation strategy
│   │
│   ├── rl/                                 # 🧠 Reinforcement Learning & Bandit Agents
│   │   ├── linucb.py                       # LinUCB Contextual Bandit implementation (Ridge regression + UCB)
│   │   ├── ucb.py                          # Multi-Armed Bandit UCB1 baseline
│   │   ├── thompson.py                     # Thompson Sampling bandit baseline
│   │   ├── dqn.py                          # Deep Q-Network (DQN) PyTorch baseline agent
│   │   ├── environment.py                  # OpenAI Gym / Gymnasium environment for strategy routing
│   │   └── bandit.py                       # Simple multi-armed bandit environment wrapper
│   │
│   ├── features/                           # 🔍 Feature Extractor
│   │   └── extractor.py                    # Converts raw text questions into 384-dim embedding state vectors
│   │
│   └── evaluation/                         # 📊 Evaluation & Verification Engine
│       ├── correctness.py                  # Strict answer verification logic (Math, Logic, CSQA)
│       ├── evaluator.py                    # Evaluation runner for single strategies
│       ├── run_full_baselines.py           # Fixed-policy baseline benchmark runner
│       └── statistical_tests.py            # Significance testing & confidence intervals
│
├── data/                                   # 💾 Dataset & Results Storage
│   ├── clean/                              # 🧹 Verified Evaluation Harness
│   │   ├── clean_eval.py                   # Clean benchmark runner across 3,712 queries (seed 42, zero retries)
│   │   └── generate_summary.py             # Generates aggregated accuracy, token, and latency metrics
│   └── results/                            # 📈 Output Performance Data
│       ├── full_baselines_results.csv      # Performance logs for static baseline strategies
│       └── cascaderoute_results.csv        # Performance logs for RL CascadeRoute agent
│
└── reports/                                # 🖼️ Generated Charts & Artifacts
    └── figures/                            # Exported charts (Pareto front, accuracy bars, radar plots)
```

---

## 🛠️ Part 3: Component Responsibilities Quick Reference

| Directory / File | Key Responsibility |
| :--- | :--- |
| **`dashboard/app.py`** | Serves the web UI on port 8080. Connects local Ollama models (`qwen2.5:3b` and `qwen2.5:7b`) to the frontend. |
| **`dashboard/static/main.js`** | Renders live Chart.js graphs, updates cumulative live stats, and displays step-by-step reasoning traces. |
| **`src/rl/linucb.py`** | The core RL algorithm that estimates upper confidence bounds to pick the best model/strategy. |
| **`src/features/extractor.py`** | Extracts feature representations from user questions so the RL agent can recognize domain difficulty. |
| **`src/strategies/`** | Contains code for `direct`, `cot`, `got`, and `tot` reasoning engines. |
| **`data/clean/clean_eval.py`** | Runs reproducible, non-flaky benchmark evaluations across all 3,712 test queries. |

---

## 🚀 How to Run the Project

1. **Ensure Ollama is running:**
   ```bash
   ollama list
   ```

2. **Start the Interactive Dashboard:**
   ```bash
   cd dashboard
   python -m uvicorn app:app --reload --port 8080
   ```

3. **Open in Browser:**
   Navigate to `http://127.0.0.1:8080` in your web browser and test questions from [sample_questions.md](file:///c:/Users/amma/ADITHYA_23001/SDP_RL_Project/dashboard/sample_questions.md)!
