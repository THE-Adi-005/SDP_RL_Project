# AdaptInfer: Software Development Project (SDP) Presentation Guide

**Course:** 22CS401 – Software Development Project (SDP)  
**Track:** Systems & Software Engineering  
**Project Title:** AdaptInfer: A Distributed Microservices Architecture and Dynamic Strategy Orchestration Engine for Cost-Aware LLM Inference  
**Authors:** Adithya R (CH.SC.U4AIE23001), Aniketh Reddy J (CH.SC.U4AIE23020), AmanReddy J (CH.SC.U4AIE23023), Naishadha Badithala (CH.SC.U4AIE23036)  
**Supervisor:** Dr. R. Prasanna Kumar, Department of Computer Science & Engineering  
**Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai  

---

## Visual Assets in this Directory (`SDP_Ppt/`)

All generated visual outputs are saved directly in this folder (`SDP_Ppt/`):
1. `sdp_architecture_diagram.png` — End-to-end distributed microservice architecture (FastAPI, Ray, Ollama, Strategy Pattern, ConfidenceGate).
2. `sdp_latency_improvement.png` — Latency optimization bar chart (45.2s down to 14.8s, -67.2% reduction).
3. `sdp_token_savings.png` — Token expenditure efficiency (1,959 down to 1,421 tokens, -27.5% reduction).
4. `sdp_strategy_comparison.png` — Accuracy matrix across all 6 reasoning strategies vs AdaptInfer parity (81.4% vs 82.0%).
5. `sdp_system_tradeoff.png` — Systems Pareto efficiency scatter plot (Accuracy vs Response Latency).
6. `sdp_dataset_distribution.png` — Workload characterization pie chart (3,712 queries across GSM8K, LogiQA, CommonsenseQA).

---

## 📽️ Slide 1: Title & Engineering Problem Statement

### 🖼️ Visual Placement:
* **Layout:** Left 55% Text / Right 45% Visual Graphic.
* **Image to Place:** `sdp_dataset_distribution.png`
* **Placement Instruction:** Position `sdp_dataset_distribution.png` on the right side to establish the large-scale benchmark scope (3,712 queries across 3 reasoning domains).

### 📝 Slide Content (Minimal Bullets):
* **AdaptInfer:** Distributed microservices architecture for dynamic LLM reasoning strategy orchestration.
* **The Production Trilemma:** Severe tension between inference accuracy, token expenditure, and response latency.
* **Static Allocation Failure:**
  * Fast zero-shot 3B: ~4s latency, but 46.1% failure rate on complex logic.
  * Heavy Tree-of-Thoughts (ToT) 7B: 82.0% accuracy, but 45.2s median latency and 1,959 tokens/query.
* **System Goal:** Dynamically route incoming queries to the optimal Model $\times$ Strategy pair in real time to minimize latency and token overhead while preserving top-tier accuracy.

### 🎤 Speaker Script:
> *"Respected evaluators, modern LLMs face a critical systems engineering trade-off: deep reasoning strategies like Tree-of-Thoughts deliver high accuracy but incur unacceptable 45-second latencies and massive token costs. Conversely, cheap direct prompting fails on complex logic. AdaptInfer solves this by decoupling API handling, distributed scheduling, and dynamic strategy orchestration into a high-throughput microservices architecture."*

---

## 📽️ Slide 2: Distributed Systems Architecture

### 🖼️ Visual Placement:
* **Layout:** Top 25% Key Architecture Bullets / Bottom 75% Full Architecture Diagram.
* **Image to Place:** `sdp_architecture_diagram.png`
* **Placement Instruction:** Center `sdp_architecture_diagram.png` prominently across the lower two-thirds of the slide. Ensure all 6 subsystem blocks (FastAPI, Ray, Ollama Workers, Strategy Router, ConfidenceGate, Cache) are clearly visible.

### 📝 Slide Content (Minimal Bullets):
* **Decoupled Three-Tier Microservices Topology:**
  * **API Gateway (`dashboard/app.py`):** Asynchronous FastAPI + Uvicorn server providing non-blocking REST & SSE endpoints.
  * **Distributed Actor Pool (Ray Framework):** Distributes heavy LLM prompt generation asynchronously across isolated worker nodes.
  * **Local Execution Engines (Ollama):** Dedicated local instances of `Qwen2.5-3B` and `Qwen2.5-7B` eliminating cloud network jitter.
* **Zero-Lock Concurrency:** Background tasks execute in parallel without stalling client-facing API responses.

### 🎤 Speaker Script:
> *"As shown in our system diagram, AdaptInfer decouples synchronous LLM execution from client HTTP request handling. FastAPI receives incoming queries asynchronously, while Ray schedules task workers across local Ollama instances hosting Qwen 3B and 7B models. This decoupled topology prevents API thread locking and guarantees deterministic latency measurements."*

---

## 📽️ Slide 3: Software Design Patterns & Object-Oriented Blueprint

### 🖼️ Visual Placement:
* **Layout:** Left 60% Pattern Descriptions / Right 40% Architecture Snippet.
* **Image to Place:** `sdp_architecture_diagram.png` (or UML pattern block).
* **Placement Instruction:** Place pattern cards on the left with the architecture diagram zoomed to the engine block on the right.

### 📝 Slide Content (Minimal Bullets):
* **Strategy Pattern (`src/strategies/`):**
  * Abstract base interface `BaseStrategy.solve(question)` implemented by `DirectStrategy`, `CoTStrategy`, `GoTStrategy`, and `ToTStrategy`.
  * Allows runtime hot-swapping of reasoning algorithms with zero client coupling.
* **Chain of Responsibility Pattern (`ConfidenceGate`):**
  * Tiered escalation chain: `3B_Direct` $\to$ `3B_CoT` and `7B_Direct` $\to$ `7B_CoT` $\to$ `7B_GoT` $\to$ `7B_ToT`.
  * Detects answer divergence between low ($T=0.2$) and high ($T=0.8$) temperatures, automatically escalating uncertain queries.
* **Adapter Pattern (`LLMClient`):** Standardizes local Ollama REST interactions with deterministic seeds (42) and configurable timeouts.
* **Proxy / Caching Pattern:** In-memory LRU cache and budget tracker to avoid redundant LLM invocations.

### 🎤 Speaker Script:
> *"From a software engineering standpoint, AdaptInfer strictly enforces classic design patterns. We use the Strategy Pattern to make reasoning algorithms polymorphic and interchangeable at runtime. To prevent hallucinations, we implemented the Confidence Gate using the Chain of Responsibility pattern: if low and high temperature responses disagree, the system seamlessly escalates execution to higher-tier reasoning strategies."*

---

## 📽️ Slide 4: Real-Time Telemetry & Interactive Single-Page Application

### 🖼️ Visual Placement:
* **Layout:** Top 40% Architecture Capabilities / Bottom 60% Comparative Visualization.
* **Image to Place:** `sdp_strategy_comparison.png`
* **Placement Instruction:** Place `sdp_strategy_comparison.png` on the right half or lower center to illustrate how the system benchmarks all 6 strategies side-by-side in real time.

### 📝 Slide Content (Minimal Bullets):
* **Reactive Full-Stack Frontend:** Single-Page Application built with Vanilla JS, CSS3 glassmorphism, and Chart.js.
* **Live Telemetry & Strategy Racing:**
  * Parallel execution traces for all 6 model-strategy combinations.
  * Real-time visual confidence gauges and recommendation likelihood bars.
* **Dynamic Systems Telemetry:**
  * Instant token expenditure breakdown (input vs. output tokens).
  * Real-time latency tracking and cumulative cost efficiency ledger.

### 🎤 Speaker Script:
> *"To ensure operational observability, we engineered a reactive Single-Page Application dashboard. Users can submit arbitrary queries, observe all six strategies race concurrently, inspect intermediate reasoning chains, and review live latency and token telemetry in real time."*

---

## 📽️ Slide 5: Empirical Latency Optimization (3,712 Queries)

### 🖼️ Visual Placement:
* **Layout:** Left 45% Numerical Findings & Analysis / Right 55% Latency Chart.
* **Image to Place:** `sdp_latency_improvement.png`
* **Placement Instruction:** Position `sdp_latency_improvement.png` prominently on the right. Point out the large green callout box highlighting the **67.2% reduction**.

### 📝 Slide Content (Minimal Bullets):
* **Extensive Workload Benchmark:** 3,712 total queries (GSM8K: 1,319, LogiQA: 1,172, CommonsenseQA: 1,221).
* **System Latency Results:**
  * Static 7B Tree-of-Thoughts (ToT): **45.2 seconds** (severe production bottleneck).
  * Static 7B Chain-of-Thought (CoT): **17.4 seconds**.
  * **AdaptInfer Distributed Ray Engine: 14.8 seconds.**
* **Key Achievement:** Slashed end-to-end response latency by **67.2%** compared to static multi-step reasoning while maintaining top reasoning quality.

### 🎤 Speaker Script:
> *"Here are our empirical latency findings across 3,712 queries. Running static Tree-of-Thoughts requires a crushing 45.2 seconds per query. Through distributed actor scheduling and dynamic strategy selection, AdaptInfer slashes response latency down to 14.8 seconds—a massive 67.2% speedup that makes advanced reasoning feasible in production."*

---

## 📽️ Slide 6: Token Cost Efficiency & System Savings

### 🖼️ Visual Placement:
* **Layout:** Left 45% Token Metrics Table / Right 55% Token Bar Chart.
* **Image to Place:** `sdp_token_savings.png`
* **Placement Instruction:** Place `sdp_token_savings.png` on the right side. Emphasize the blue callout box showing **27.5% Token Savings (-538 tokens per query)**.

### 📝 Slide Content (Minimal Bullets):
* **Token Expenditure per Query:**
  * Static 7B-ToT: **1,959 tokens/query** (exhaustive branch exploration).
  * AdaptInfer: **1,421 tokens/query** (selective reasoning depth).
  * Static 7B-CoT: 379 tokens/query.
  * Static 3B-Direct: 108 tokens/query.
* **System Cost Savings:**
  * **27.5% Reduction in Token Cost** compared to static ToT.
  * Direct saving of **538 tokens per query**, preventing GPU VRAM exhaustion and reducing inference energy usage.

### 🎤 Speaker Script:
> *"Computational cost in LLM systems is governed by token volume. As shown on the right, static Tree-of-Thoughts consumes nearly 2,000 tokens per query. AdaptInfer cuts this by 27.5%—saving 538 tokens on every single query by intelligently invoking lighter strategies whenever high-depth search is redundant."*

---

## 📽️ Slide 7: Systems Pareto Frontier & Accuracy Parity

### 🖼️ Visual Placement:
* **Layout:** Left 40% Performance Summary / Right 60% Pareto Tradeoff Scatter Plot.
* **Image to Place:** `sdp_system_tradeoff.png`
* **Placement Instruction:** Place `sdp_system_tradeoff.png` on the right side. Point out the green star representing AdaptInfer breaking through the static frontier curve.

### 📝 Slide Content (Minimal Bullets):
* **Accuracy vs. Latency Performance Summary:**

| System Configuration | Accuracy (%) | Tokens / Query | Latency (s) | System Status |
| :--- | :---: | :---: | :---: | :--- |
| **Static 3B-Direct** | 53.9% | 108 | 4.4s | High Error Rate |
| **Static 7B-CoT** | 80.0% | 379 | 17.4s | Suboptimal Logic |
| **Static 7B-ToT** | **82.0%** | **1,959** | **45.2s** | Latency Bottleneck |
| **AdaptInfer (Ours)** | **81.4%** | **1,421** | **14.8s** | **Optimal Pareto Balance** |

* **Accuracy Parity:** Retains **99.3% of the ToT accuracy ceiling** (81.4% vs 82.0%) at **1/3rd the latency**.

### 🎤 Speaker Script:
> *"This Pareto frontier illustrates the core systems contribution of AdaptInfer. Conventional static architectures force engineers to choose between fast error-prone models or slow expensive models. AdaptInfer breaks this frontier: our system achieves 81.4% accuracy—virtually identical to the 82.0% ToT ceiling—while running three times faster."*

---

## 📽️ Slide 8: Conclusion & Future Engineering Roadmap

### 🖼️ Visual Placement:
* **Layout:** Left 50% Engineering Contributions / Right 50% Roadmap & Architecture Thumbnail.
* **Image to Place:** `sdp_architecture_diagram.png` (Small thumbnail) + Thank You / Q&A Box.
* **Placement Instruction:** Position contributions on the left and future enhancements on the right.

### 📝 Slide Content (Minimal Bullets):
* **Core Systems Engineering Contributions:**
  1. **Decoupled Architecture:** High-throughput microservice integrating FastAPI, Ray distributed task scheduling, and local Ollama workers.
  2. **Robust Design Patterns:** Strategy, Adapter, Chain of Responsibility, and Proxy patterns implemented for production maintainability.
  3. **Verified Efficiency:** Slashed latency by 67.2% and token usage by 27.5% across 3,712 benchmark queries with statistical accuracy parity.
  4. **Observable Telemetry:** Full-stack SPA dashboard enabling live strategy tracing and cost auditing.
* **Future Work:** Multi-node Ray GPU clustering and dynamic SLA latency deadline enforcement.

### 🎤 Speaker Script:
> *"In summary, AdaptInfer proves that intelligent systems engineering and dynamic strategy orchestration can overcome the performance barriers of modern LLM inference. We thank our faculty guide Dr. R. Prasanna Kumar, our department, and the review committee. We now welcome your questions."*

---

## 📋 Quick Slide-to-Asset Mapping Checklist

| Slide # | Slide Title | Visual Asset File to Embed | Layout Position on Slide |
| :---: | :--- | :--- | :--- |
| **1** | Title & Problem Statement | `sdp_dataset_distribution.png` | Right half (45% width) |
| **2** | Distributed Microservices Architecture | `sdp_architecture_diagram.png` | Center-bottom (70% width) |
| **3** | Software Design Patterns | `sdp_architecture_diagram.png` (Crop) | Right column (40% width) |
| **4** | Real-Time Telemetry & SPA Dashboard | `sdp_strategy_comparison.png` | Right half (50% width) |
| **5** | Empirical Latency Optimization | `sdp_latency_improvement.png` | Right half (55% width) |
| **6** | Token Cost Savings & Efficiency | `sdp_token_savings.png` | Right half (55% width) |
| **7** | Systems Pareto Frontier & Accuracy | `sdp_system_tradeoff.png` | Right half (60% width) |
| **8** | Conclusion & Engineering Contributions | `sdp_architecture_diagram.png` (Small) | Lower right corner |
