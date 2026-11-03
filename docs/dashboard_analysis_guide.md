# Dashboard Analysis & Inference Guide

This document explains every label, graph, metric, and feature present in the Adaptive LLM Reasoning UI. Use this guide to interpret the results of your experiments and understand what the RL system is doing under the hood.

---

## 1. Top-Level Summary Metrics (Header)

*   **Experiments (Total Rows):** The total number of problem-solving attempts recorded in the backend database (`experiment_results.csv`).
*   **Best Agent:** The reinforcement learning agent or fixed baseline that achieved the highest overall composite reward across all test runs.
*   **Oracle Accuracy:** This is a theoretical ceiling. If you had a "perfect" routing engine that magically knew *exactly* which strategy would get the right answer every single time, this is the accuracy it would achieve. If Oracle is 90% and your best agent (e.g., LinUCB) is 85%, it means your RL agent is performing very close to perfectly.

---

## 2. "Analyze a Question" Module (Real-Time Inference)

This section allows you to test the architecture live.

### Input Labels
*   **Question:** The raw prompt or puzzle.
*   **Category:** The domain of the problem (Math, Logic, Commonsense). This is a crucial "feature" fed into the RL state.
*   **Expected Answer (Optional):** Used strictly to calculate the Reward score after generation to see if the LLM got it right.

### Output / Inference Labels
*   **Recommended Strategy:** The strategy (e.g., `cot` or `direct`) that the trained LinUCB agent thinks is best for this specific question.
*   **Scores & Confidence:** These are the internal mathematical predictions of the RL agent. 
    *   *Scores* represent the expected "Reward" for each strategy based on the question's features (length, category). 
    *   *Confidence* translates those scores into a percentage (e.g., "I am 80% sure CoT is the best choice here").
*   **Tokens (Input/Output):** Input tokens measure how long the prompt was. Output tokens measure how much the LLM talked. *Higher output tokens heavily penalize the reward score because they cost more API money.*
*   **Latency:** Time in seconds the LLM took to generate the answer. *High latency penalizes the reward score.*
*   **Reward:** The final composite score. Calculated as: `(Correctness) - (Token Penalty) - (Latency Penalty)`.

---

## 3. Experiment History Dashboard (The Graphs)

These graphs visualize the data from the offline experiments (`run_experiment.py`).

### A. Accuracy by Agent / Policy (Bar Chart)
*   **What it is:** A simple bar chart showing the percentage of questions each agent got right.
*   **What to look for:** Compare the RL agents (like `linucb` or `thompson`) against the static baselines (`always_direct`, `always_tot`). If your RL agent matches the accuracy of `always_tot` (Tree-of-Thought), that is a massive success.

### B. Pareto Front (Scatter Plot: Accuracy vs. Cost)
*   **What it is:** The X-axis represents average Latency or Token Cost. The Y-axis represents Accuracy. Each dot is an Agent.
*   **What to look for:** You want dots that are in the **Top-Left Corner** (High Accuracy, Low Cost). 
*   **Significance:** `always_tot` will usually be in the Top-Right (High Accuracy, High Cost). `always_direct` will be Bottom-Left (Low Accuracy, Low Cost). A successful RL agent will find the "sweet spot" (Pareto optimal point) near the Top-Left, balancing the two extremes.

### C. Strategy Heatmap / Distribution (Stacked Bar)
*   **What it is:** Shows what percentage of the time an RL agent chose each strategy. (e.g., Direct: 50%, CoT: 30%, ToT: 20%).
*   **What to look for:** If the agent chooses `direct` 100% of the time, it hasn't learned anything (it's just being lazy). If it distributes its choices intelligently (e.g., using `tot` for logic, `direct` for commonsense), the RL architecture is working properly.

### D. Reward Distribution (Box Plot)
*   **What it is:** Shows the variance and spread of the composite reward for each agent.
*   **What to look for:** The line in the middle of the box is the **Median** reward. The height of the box (Interquartile Range) shows consistency. A tall, stretched box means the agent's performance is erratic. A tight box high up on the Y-axis means the agent consistently gets good rewards.

### E. Cumulative Reward (Line Chart / Learning Curve)
*   **What it is:** The X-axis is "Time" (number of questions answered). The Y-axis is the total accumulated reward.
*   **What to look for:** The lines should curve upwards. If the line is straight or flat, the agent is not learning. If an RL agent's line crosses and overtakes a static baseline's line, it proves that "dynamic routing" is mathematically superior to "static prompting" over time.

### F. Radar Chart (Multi-dimensional Comparison)
*   **What it is:** A spider-web chart comparing 1-3 agents across multiple axes simultaneously (e.g., Accuracy, Token Efficiency, Speed).
*   **Significance:** This visually proves the trade-off management of the Software Architecture. It easily demonstrates to viewers that while `always_tot` might max out the "Accuracy" axis, the RL Agent maxes out the overall volume of the radar shape by balancing all three axes effectively.
