# 🧪 Dashboard Test Suite: Domain & Difficulty Benchmark

This file contains **9 benchmark test queries** split across 3 cognitive domains (**Math**, **Logic**, **Commonsense**) and 3 difficulty levels (**Simple**, **Medium**, **Hard**). 

You can copy and paste these directly into the **Analyze a Question** panel on the UI dashboard (`http://127.0.0.1:8080`).

---

## 🧮 1. Math Domain

| Difficulty | Question | Expected Answer | Rationale / Strategy Insight |
| :--- | :--- | :--- | :--- |
| **Simple** | `What is 15 + 27?` | `42` | Basic arithmetic; solvable via **3B_DIRECT**. |
| **Medium** | `A train travels at 60 mph for 2 hours and then at 80 mph for 1.5 hours. What is the total distance traveled in miles?` | `240` | Requires multi-step arithmetic; ideal for **3B_COT** or **7B_DIRECT**. |
| **Hard** | `If a principal of $1,000 is invested at a compound annual interest rate of 10% compounded annually, what will be the total value in dollars after 3 years?` | `1331` | Multi-stage exponentiation; benefits from **7B_COT** or **7B_TOT**. |

---

## 🧩 2. Logic Domain

| Difficulty | Question | Expected Answer | Rationale / Strategy Insight |
| :--- | :--- | :--- | :--- |
| **Simple** | `All humans are mortal. Socrates is a human. Is Socrates mortal? Answer with Yes or No.` | `Yes` | Categorical syllogism; handled efficiently by **3B_DIRECT**. |
| **Medium** | `Five people (A, B, C, D, E) sit in a row from left to right. A is in seat 1, B is in seat 2, C is in seat 3, D is in seat 4, and E is in seat 5. Who is sitting in the middle position (3rd seat)?` | `C` | Spatial positioning logic; test simple reasoning models. |
| **Hard** | `In a group of 30 people, 18 speak French, 15 speak Spanish, and 5 speak neither language. How many people speak both French and Spanish?` | `8` | Inclusion-Exclusion set logic; triggers **7B_COT** or **7B_GOT**. |

---

## 💡 3. Commonsense Domain

| Difficulty | Question | Expected Answer | Rationale / Strategy Insight |
| :--- | :--- | :--- | :--- |
| **Simple** | `If you leave an ice cube on a hot sunny outdoor table, what state of matter does it turn into first?` | `Liquid` | Physical state change; fast answer with **3B_DIRECT**. |
| **Medium** | `John put his chocolate bar inside a hot car trunk on a summer afternoon. When he opened it two hours later, why was the chocolate messy to eat?` | `It melted` | Physical causal reasoning; fast answer with **3B_COT** / **7B_DIRECT**. |
| **Hard** | `A farmer must take a wolf, a goat, and a cabbage across a river in a boat that can only carry the farmer and one item. If left alone, the wolf eats the goat, and the goat eats the cabbage. Which item must the farmer take across FIRST?` | `Goat` | Classic river-crossing constraint puzzle; benefits from **7B_GOT** or **7B_TOT**. |

---

### JSON Version
The machine-readable JSON version is saved in:
[sample_questions.json](file:///c:/Users/amma/ADITHYA_23001/SDP_RL_Project/dashboard/sample_questions.json)
