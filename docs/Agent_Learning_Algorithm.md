# The Intelligence: How the RL Agent Learns

This document provides absolute clarity on the Reinforcement Learning (RL) algorithm powering `CascadeRoute`, why we chose it, and exactly how it learns without needing offline training.

## 1. Why Not Deep RL (PPO/DQN)?
When most people hear "Reinforcement Learning," they think of Deep RL algorithms like PPO (Proximal Policy Optimization) or DQN (Deep Q-Networks). 
We did *not* use these, and here is why: Deep RL is designed for **Markov Decision Processes (MDPs)**. In an MDP, taking an action changes the state of the world (e.g., moving a chess piece changes the board state for the next turn). 
LLM routing is a **Stateless Environment**. Routing Question A to a 3B model has zero impact on the text of Question B. Using Deep RL here would waste massive compute trying to learn non-existent transition dynamics.

## 2. The Algorithm: Contextual Bandits (LinUCB)
Because our environment is stateless, the mathematically optimal algorithm is a **Contextual Bandit**. 
We specifically implemented **LinUCB (Linear Upper Confidence Bound)**. 

### How it works:
Unlike Multi-Armed Bandits (which blindly pull levers to see which pays out the most), a *Contextual* Bandit looks at the "Context" (the State) before pulling a lever.
1.  **The Context ($s_t$):** The 384-dimensional Sentence-BERT vector of the user's question.
2.  **The Arms (Actions):** The 6 routing strategies (e.g., `3B_Direct`, `7B_ToT`).

LinUCB maintains a **Ridge Regression model** for *every single action*. 
For each action $a$, it maintains:
*   A matrix $A_a$ (capturing the covariance of features).
*   A vector $b_a$ (capturing the cumulative rewards).

## 3. The Decision (Exploration vs. Exploitation)
When a new question arrives, the agent calculates a score for each action using the UCB formula:

$$ \text{Score}_a = \underbrace{\theta_a^\top s_t}_{\text{Expected Reward}} + \underbrace{\alpha \sqrt{s_t^\top A_a^{-1} s_t}}_{\text{Exploration Bonus}} $$

*   **Exploitation (Expected Reward):** The left side of the equation is pure Ridge Regression. It asks: "Based on my past experience with questions that have a similar Sentence-BERT vector, what is my predicted reward if I use this strategy?"
*   **Exploration (Bonus):** The right side of the equation tracks uncertainty. If the agent hasn't tried a specific strategy on this *type* of math question before, the uncertainty is high, so it gets a mathematical boost. This guarantees the agent explores unknown territory to find hidden optimal routes.

## 4. How Learning Actually Happens (The Update Rule)
The beauty of LinUCB is that it does not require a GPU-heavy backpropagation loop to learn. It uses closed-form linear algebra.

Once the chosen strategy finishes and the system calculates the final Reward $r_t$ (factoring in correctness, tokens, and latency), the agent learns instantly via these two equations:

$$ A_a \leftarrow A_a + s_t s_t^\top $$
$$ b_a \leftarrow b_a + r_t s_t $$

By simply adding the outer product of the state vector to the $A_a$ matrix, the agent permanently updates its understanding of the world. 
**This is how it learns live.** Every single time a user asks a question, the agent gets slightly smarter, dynamically shifting its weights to prefer cheap models for simple questions and expensive models for hard questions, breaking the Pareto frontier.
