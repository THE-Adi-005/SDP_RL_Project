# Reinforcement Learning - Presentation Slides (PPT Content)
## Title: CascadeRoute - A Contextual Bandit for LLM Routing

### Slide 1: The Problem
*   **The Dilemma:** LLMs have a strict Cost vs. Accuracy trade-off (Pareto Frontier).
*   **Simple Queries:** Direct Prompting is cheap but fails on complex logic.
*   **Complex Queries:** Tree-of-Thoughts (ToT) is highly accurate but burns thousands of tokens and takes 60+ seconds.
*   **The Goal:** Build an RL agent to dynamically route queries to the perfect Model $\times$ Strategy combination based on difficulty.

### Slide 2: Why Contextual Bandits? (Not Deep RL)
*   **Stateless Environment:** LLM routing is entirely stateless. Routing query A doesn't change the world for query B.
*   **MDP Failure:** Deep RL (PPO/DQN) assumes Markov Decision Processes. Using them here causes hyperparameter instability and wastes compute trying to learn non-existent transitions.
*   **The Solution:** We used the **LinUCB (Linear Upper Confidence Bound)** contextual bandit.

### Slide 3: The RL Formulation
*   **State Space ($s_t$):** 384-dimensional Sentence-BERT vector + Dynamic Budget Tracker remaining %.
*   **Action Space ($\mathcal{A}$):** 6 joint actions ranging from 3B-Direct up to 7B-ToT.
*   **Reward Function:** Multi-objective scalar heavily rewarding correctness ($\lambda=100$) while penalizing token consumption ($\lambda=10$) and latency ($\lambda=5$).

### Slide 4: Online Learning Mechanics
*   LinUCB does not require a massive offline pre-training phase.
*   It maintains a **Ridge Regression** matrix for each action.
*   It balances **Exploitation** (expected reward) and **Exploration** (upper confidence bound for unseen math/logic patterns).
*   **Update Rule:** Linear algebra matrix addition allows instant, real-time learning after every query.

### Slide 5: Experimental Results
*   **Dataset:** 3,700+ queries across GSM8K (Math), LogiQA (Logic), and CommonsenseQA (General).
*   **Result:** The RL agent achieved **81.4% accuracy** (statistically tied with the 7B-ToT ceiling of 82.0%).
*   **Frugality:** The agent slashed average token consumption by **27.4%**, proving it successfully learned to avoid expensive strategies on simple queries.
