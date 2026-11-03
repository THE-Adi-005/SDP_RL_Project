# Project Scope Split: SDP vs. RL

The `CascadeRoute` framework is a massive dual-purpose project. To effectively manage it, we are splitting it into two clear, distinct coursework tracks: **Software Development Project (SDP)** and **Reinforcement Learning (RL)**.

Below is the clear differentiator defining the boundaries, deliverables, and focus of each track.

---

## 1. SDP Track: Systems Engineering & Journal Publication
**Core Focus:** The architecture, distributed systems engineering, and academic publication of the framework.

*   **Primary Deliverable:** The 15-page Springer Nature Journal Article (`SDP_Journal/`).
*   **What it Covers:**
    *   **Architecture Design:** Designing the asynchronous microservice architecture that allows the Bandit to operate separately from the LLM inference engine.
    *   **Distributed Systems:** The implementation of the **Ray** distributed computing framework to orchestrate 24,000+ LLM calls across multiple GPUs efficiently.
    *   **Fallback Mechanics:** The "Chain of Responsibility" software design pattern used to implement the Confidence-Gated fallback (temperature scaling).
    *   **Data Pipeline:** The end-to-end pipeline of scraping GSM8K/LogiQA, vectorizing with Sentence-BERT, and tracking the global token budget mathematically.
*   **The "Why":** SDP focuses on *how* the system was built to be robust, scalable, and academically rigorous enough for publication.

---

## 2. RL Track: Demo, UI, and Applied Agentic Behavior
**Core Focus:** The Contextual Bandit agent, the real-time application perspective, and the interactive UI dashboard.

*   **Primary Deliverable:** The Interactive `CascadeRoute` Dashboard (`dashboard/app.py`).
*   **What it Covers:**
    *   **The Brain (`linucb.py`):** The actual Ridge Regression mathematics and Upper Confidence Bound (UCB) exploration logic that learns routing weights on the fly.
    *   **State vs. Action Spaces:** Defining the stateless environment constraint that necessitated a Contextual Bandit over a Deep RL (MDP-based) algorithm like PPO.
    *   **The Application Demo:** The live FastAPI/Uvicorn dashboard that visually demonstrates the Pareto Frontier being broken (showing Cost vs. Accuracy in real-time).
    *   **User Experience (UX):** How a developer or end-user would actually interact with `CascadeRoute` as an API or application layer to save money.
*   **The "Why":** The RL track focuses on the *intelligence* of the agent and demonstrating its practical value through a live, visual application.
