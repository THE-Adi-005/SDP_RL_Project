# CascadeRoute: Architecture and Execution Flow

## 1. Architecture Diagram

The system follows a highly modular design leveraging Software Design Patterns (Strategy, Facade, Chain of Responsibility). 

```mermaid
flowchart TD
    %% Define Nodes
    Query[User Query & Category]
    FE[Feature Extractor<br/>SentenceTransformers]
    BT[Budget Tracker]
    StateVector[State Vector<br/>384-dim text + 5-dim metadata]
    Bandit[LinUCB Bandit<br/>Joint Action Selection]
    CG[Confidence Gate<br/>Chain of Responsibility]
    Eval[Evaluator & Metric Logger]
    Reward[Reward Function<br/>Accuracy - Cost - Latency]

    %% Strategies
    subgraph Action Space (6 Options)
        3B_D[3B Direct]
        3B_C[3B CoT]
        7B_D[7B Direct]
        7B_C[7B CoT]
        7B_G[7B GoT]
        7B_T[7B ToT]
    end

    %% Execution Flow
    Query --> FE
    FE -->|Embedding & Length| StateVector
    BT -.->|Remaining Budget %| StateVector
    StateVector --> Bandit
    
    Bandit -->|Selects Action| Action Space
    Action Space -->|e.g., 7B_CoT| CG

    %% Inside Confidence Gate
    subgraph Confidence Gate Execution
        Run1[Run 1: Temp 0.2]
        Run2[Run 2: Temp 0.8]
        Match{Do answers<br/>match?}
        Escalate[Escalate to Next Tier<br/>e.g., 7B_GoT]
    end

    CG --> Run1
    CG --> Run2
    Run1 --> Match
    Run2 --> Match
    Match -- Yes --> Eval
    Match -- No --> Escalate
    Escalate --> Eval

    Eval -->|Metrics| Reward
    Eval -->|Deduct Tokens| BT
    Reward -.->|Update| Bandit
```

---

## 2. Step-by-Step Execution Example

Let's walk through exactly what happens when the framework processes a single question from the GSM8K dataset.

**Scenario:** 
- The system is halfway through the dataset. 
- The `Budget Tracker` has 45% of the token budget remaining.
- The question is: *"If a train travels 60 mph for 2 hours, and then 40 mph for 1 hour, how far did it travel in total?"*

### Step 1: Feature Extraction
The script passes the question to `extract_features()`.
1. The `SentenceTransformer` converts the text into a 384-dimensional vector representing its semantic meaning (e.g., recognizing it as a time/distance math problem).
2. It extracts metadata: Word count = 20, Sentence count = 1, Category = Math.
3. The `Budget Tracker` provides `0.45` (45%).
4. **Result:** A 389-dimensional state array is created.

### Step 2: Bandit Action Selection
The `LinUCB` bandit receives the 389-dimensional state vector. 
1. It calculates the expected reward for all 6 possible actions based on past experiences with similar math questions.
2. Because the budget is getting low (45%) and this question seems relatively short (20 words), the bandit calculates that an expensive strategy like `7b_tot` would result in a massive token penalty.
3. It determines the highest expected reward lies with `3b_cot` (cheap model, but allows step-by-step thinking).
4. **Result:** Bandit selects action `3b_cot`.

### Step 3: Confidence Gate Execution (The Cascade)
The selected action `3b_cot` is passed to the `ConfidenceGate`.

**Run 1 (Temperature = 0.2):**
- The 3B model generates a CoT response: *"Distance = Speed x Time. 60 x 2 = 120. 40 x 1 = 40. 120 + 40 = 160."*
- Extracted Answer: `160`

**Run 2 (Temperature = 0.8):**
- The 3B model generates a slightly different reasoning path but concludes: *"Total distance is 160 miles."*
- Extracted Answer: `160`

**Self-Consistency Check:**
- The answers match (`160 == 160`). The system determines it has **high confidence** in the result.
- *If the answers had mismatched (e.g., Run 1 said 160, Run 2 said 120), the Confidence Gate would have escalated the query to `7b_direct` or `7b_cot` automatically.*

### Step 4: Evaluation and Budget Deduction
1. The `Evaluator` compares the final predicted answer (`160`) to the ground truth (`160`). It marks `correct = True`.
2. It counts the tokens used in *both* runs. (e.g., 200 input tokens + 150 output tokens = 350 tokens).
3. The `Budget Tracker` deducts 350 tokens from the remaining budget.

### Step 5: Reward Calculation and Learning
The `compute_reward` function aggregates the metrics:
- Accuracy: +1.0 (It got it right)
- Token Penalty: -0.05 (It used 350 tokens, which is very cheap)
- Latency Penalty: -0.02 (It was fast)
- **Final Reward = +0.93**

Finally, the `LinUCB` agent updates its internal matrices using the state vector and the reward (+0.93). 
**Learning outcome:** The agent slightly adjusts its internal weights to remember: *"For short math questions at 45% budget, 3b_cot is a highly effective choice."*
