# 🖥️ Software Development Project (SDP) — Defense Presentation Slides

## Title: AdaptInfer — A Distributed Systems Architecture for Dynamic LLM Strategy Orchestration
**Presenter:** Adithya R, Aniketh Reddy J, AmanReddy J, Naishadha B  
**Track:** 22CS401 – Software Development Project (SDP) Systems & Software Engineering Track

---

## 📽️ Slide 1: Title & Engineering Problem Statement

### 📌 Slide Content:
* **Project Name:** AdaptInfer: Distributed Microservices Architecture for Dynamic LLM Strategy Orchestration
* **The Engineering Problem:**
  - Modern Large Language Models (LLMs) offer high reasoning capability (e.g., Tree-of-Thoughts), but at exorbitant computational cost (45s+ latency, 1,950+ tokens per query).
  - Conversely, fast zero-shot models (e.g., 3B-Direct) execute in ~4s, but suffer a 46% error rate on complex logic.
  - **Static Allocation Failure:** Hardcoding a single model/strategy for all queries leads to either system infrastructure collapse or poor user experience.
* **Objective:** Engineer a scalable, asynchronous, microservice architecture that dynamically routes queries to the optimal Model $\times$ Strategy pair in real-time under session budget constraints.

### 🎤 Speaker Script:
> *"Good morning respected evaluators. Today we present AdaptInfer, a distributed systems architecture engineered to solve the inference latency and token cost bottleneck in Large Language Models. In production software systems, routing every query to an expensive Tree-of-Thoughts pipeline wastes computational resources, while routing everything to a fast 3B model causes high error rates. AdaptInfer introduces a dynamic, context-aware orchestration system to achieve optimal performance balance."*

---

## 📽️ Slide 2: Systems Architecture & Distributed Processing

### 📌 Slide Content:
* **Asynchronous Microservices Layer:**
  - **FastAPI Backend (`app.py`):** Provides REST API endpoints (`/api/analyze`, `/api/recommend`, `/api/history`) operating asynchronously via Uvicorn on port 8080.
  - **Single Page Application (SPA):** Reactive frontend built with Vanilla JS and Chart.js for real-time telemetry and strategy comparison.
* **Distributed Task Scheduling (Ray Framework):**
  - Decouples decision state updates and API request handling from long-running synchronous LLM inference.
  - Distributes LLM prompt evaluation across Ray worker nodes.
* **Local Hardware Hosting (Ollama Engine):**
  - Runs local instances of `Qwen2.5-3B-Instruct` and `Qwen2.5-7B-Instruct`.
  - Eliminates cloud network latency jitter, ensuring latency benchmarks measure true computational inference.

```
   [Dashboard SPA UI]  <--->  [FastAPI Microservice (app.py)]
                                         |
                                         v
                              [Ray Distributed Engine]
                                         |
                 +-----------------------+-----------------------+
                 v                                               v
     [Ollama Worker: Qwen 3B]                         [Ollama Worker: Qwen 7B]
```

### 🎤 Speaker Script:
> *"To ensure our backend never locks under heavy LLM generation workloads, we decoupled the architecture using the Ray distributed computing framework. FastAPI receives user requests asynchronously, while Ray farms execution out to isolated local worker processes hosting Qwen 3B and 7B models via Ollama. This guarantees deterministic latency benchmarking free from cloud network jitter."*

---

## 📽️ Slide 3: Software Design Patterns & Object-Oriented Blueprint

### 📌 Slide Content:
1. **Strategy Pattern (`src/strategies/`):**
   - Implements a polymorphic `BaseStrategy` interface defining a unified `.solve(question)` method.
   - Encapsulates `DirectStrategy`, `CoTStrategy`, `GoTStrategy`, and `ToTStrategy`.
   - Allows the router to dynamically swap reasoning algorithms at runtime without changing client-side invocation logic.
2. **Chain of Responsibility / Guardrail Pattern (`ConfidenceGate`):**
   - Evaluates LLM responses across low ($T=0.2$) and high ($T=0.8$) sampling temperatures.
   - If answers diverge, the system flags a potential hallucination and automatically escalates the query to `7B_ToT` for safety.
3. **Adapter Pattern (`LLMClient`):**
   - Wraps the low-level Ollama REST API into a clean, reusable interface with configurable timeout (120s), zero retries, and fixed seed (42).

### 🎤 Speaker Script:
> *"From a software engineering perspective, we strictly adhered to classic Object-Oriented design patterns. We implemented the Strategy Pattern to make reasoning methods polymorphic and dynamically hot-swappable. We also implemented a Confidence Gate using the Chain of Responsibility pattern—if temperature sampling detects answer divergence, the system overrides the simple router and escalates the query to our highest-accuracy fallback strategy."*

---

## 📽️ Slide 4: Interactive Dashboard & Real-Time Telemetry

### 📌 Slide Content:
* **Real-Time Strategy Race:** Evaluates all 6 strategies (`3b_direct`, `3b_cot`, `7b_direct`, `7b_cot`, `7b_got`, `7b_tot`) in parallel.
* **LinUCB Recommendation & Confidence Bars:** Visualizes decision engine strategy action probabilities.
* **AdaptInfer Efficiency Box:** Displays immediate token savings and latency reduction for the current query vs. static ToT.
* **Dynamic Agent Performance Summary Table:** Displays live session metrics (Accuracy, Avg Tokens, Median Latency, Avg Reward, Avg LLM Calls) updating dynamically in real-time.

### 🎤 Speaker Script:
> *"We constructed a full-stack, single-page application dashboard to make the system's routing decisions completely transparent. End-users can enter any query, watch all six strategies execute live, view step-by-step reasoning traces, and analyze dynamic telemetry that proves exact token and latency savings."*

---

## 📽️ Slide 5: Empirical Evaluation & System Savings (3,712 Queries)

### 📌 Slide Content:
* **Benchmark Scope:** 3,712 queries evaluated across **GSM8K**, **LogiQA**, and **CommonsenseQA**.

| Policy | Accuracy (%) | Tokens / Query | Median Latency (s) | System Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Always 3B-Direct** | 53.9% | 108 | 4.4s | High Error |
| **Always 7B-CoT** | 80.0% | 379 | 17.4s | Expensive |
| **Always 7B-ToT** | **82.0%** | **1,959** | **45.2s** | Latency Bottleneck |
| **AdaptInfer (Ours)** | **81.4%** | **1,421** | **14.8s** | **Optimal System Balance** |

* **Architectural Results:**
  - ⚡ **67.2% Latency Reduction:** Cut median response latency from 45.2 seconds down to **14.8 seconds**.
  - 🪙 **27.5% Token Savings:** Saved **538 tokens per query** compared to static ToT while matching its 82% accuracy ceiling.

### 🎤 Speaker Script:
> *"Our empirical evaluation on 3,712 queries proves the architectural efficiency of AdaptInfer. While static Tree-of-Thoughts achieves 82% accuracy at a massive cost of 45 seconds per query, AdaptInfer matches top accuracy at 81.4% while reducing system latency by 67.2% and saving 538 tokens per query."*

---

## 📽️ Slide 6: Conclusion & Engineering Takeaways

### 📌 Slide Content:
* **Core Takeaway:** Dynamic strategy orchestration is a critical systems engineering necessity for LLM production pipelines.
* **Software Contributions:**
  1. A decoupled asynchronous microservice architecture using Ray, FastAPI, and Ollama.
  2. Production-grade OOP design patterns (Strategy, Adapter, Chain of Responsibility).
  3. Interactive real-time dashboard for explainable telemetry and policy monitoring.
* **Future Work:** Extending worker nodes to multi-GPU clusters and adding dynamic SLA budget constraints.

### 🎤 Speaker Script:
> *"In conclusion, AdaptInfer demonstrates that intelligent dynamic strategy orchestration bridges the gap between LLM accuracy and system infrastructure performance. Thank you, and we welcome any questions."*
