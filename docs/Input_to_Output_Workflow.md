
# CascadeRoute: Full Input to Output Workflow

This document provides crystal clear clarity on exactly how the `CascadeRoute` framework processes a single question, from the moment a user hits "Submit" to the final answer.

## Step 1: The Input (The User Query)
The system receives a raw text question from the user (e.g., *"What is 100 times 500?"*). In a real production system, this would be an API call. In our UI, this is the text typed into the dashboard.

## Step 2: Feature Extraction (Sentence-BERT)
Before the RL agent can make a decision, it needs to "understand" the question. 
We pass the raw text into a pre-trained **Sentence-BERT** model (`all-MiniLM-L6-v2`).
*   **Output of this step:** A dense 384-dimensional mathematical vector (an array of 384 floating-point numbers) representing the semantic meaning, cognitive complexity, and domain of the question.

## Step 3: Contextual State Formulation
The 384-dimensional vector is combined with the **Session Budget**. 
*   If the user has 100,000 tokens left, the budget state is `1.0`.
*   If the user has 5,000 tokens left, the budget state is `0.05`.
This combined vector represents the $State$ for our RL agent.

## Step 4: The RL Agent Decision (LinUCB)
The $State$ vector is fed into the **LinUCB Contextual Bandit** agent. 
The agent evaluates the state against 6 possible actions (Model $\times$ Strategy):
1. `3B_Direct`
2. `3B_CoT`
3. `7B_Direct`
4. `7B_CoT`
5. `7B_GoT` (Graph-of-Thoughts)
6. `7B_ToT` (Tree-of-Thoughts)

The agent uses Ridge Regression matrices to calculate the "Expected Reward" for each of the 6 actions. It selects the action with the highest expected reward (balancing exploration vs. exploitation).
*   **Example Output of this step:** The agent predicts that `7B_CoT` is the best choice for this specific question.

## Step 5: Execution & The Confidence Gate (Safety Net)
The chosen action is passed to the LLM (via Ollama). However, RL agents can make mistakes. We run the query through a **Confidence Gate** (using the Chain-of-Responsibility design pattern).
1. We ask the LLM the question at a low temperature ($T=0.2$).
2. We ask the LLM the question at a high temperature ($T=0.8$).
3. **Self-Consistency Check:** If the answers are semantically identical, the model is confident. If the answers are completely different, the model is hallucinating/guessing because the question is too hard.
4. **Fallback:** If the model is guessing, the Confidence Gate overrides the RL agent and escalates to a harder strategy (e.g., bumping from `7B_CoT` to `7B_ToT`).

## Step 6: Final Answer & Reward Calculation
The final answer is presented to the user. Behind the scenes, the system calculates a **Reward Score**:
*   **+100 Points:** If the answer is correct.
*   **-X Points:** Penalty based on how many tokens were burned.
*   **-Y Points:** Penalty based on how many seconds it took (latency).

## Step 7: Online Learning (Weight Update)
The reward is fed back into the LinUCB agent. The agent instantly updates its internal Ridge Regression matrices using linear algebra. This means the agent learns **in real-time**. The next time a similar question comes in, it will make a smarter, more cost-efficient routing decision.
