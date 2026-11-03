# Clean Evaluation Benchmark Summary (`CLEAN_SUMMARY.md`)

Generated at: `2026-10-07T23:20:02.167962`

Total Evaluated Outcome Rows: **18,610**

Evaluated Policies: `3b_direct, 7b_direct, 3b_cot, 7b_cot, 7b_got, 7b_tot`


## 1. Overall Policy Performance & Extraction Metrics

| Policy | Evaluated N | Primary Acc (%) | Acc Excl Extr Fail (%) | Extr Failures (%) | Errors | Avg Tokens | Avg Lat (s) | P50 Lat (s) | P95 Lat (s) | P99 Lat (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **3b_direct** | 3,712 | **53.96%** | 54.33% | 25 (0.67%) | 0 | 107.8 | 0.08s | 0.08s | 0.14s | 0.18s |
| **7b_direct** | 3,712 | **60.75%** | 61.13% | 23 (0.62%) | 0 | 107.3 | 0.28s | 0.25s | 0.45s | 0.61s |
| **3b_cot** | 3,712 | **72.25%** | 78.49% | 295 (7.95%) | 0 | 425.1 | 2.26s | 1.94s | 4.40s | 7.92s |
| **7b_cot** | 3,712 | **84.94%** | 87.44% | 106 (2.86%) | 0 | 396.0 | 3.46s | 3.34s | 5.23s | 6.40s |
| **7b_got** | 3,712 | **82.06%** | 82.75% | 31 (0.84%) | 0 | 1282.6 | 4.81s | 4.46s | 8.41s | 10.64s |
| **7b_tot** | 50 | **84.00%** | 84.00% | 0 (0.00%) | 0 | 2019.7 | 9.31s | 8.44s | 14.57s | 19.12s |


## 2. Accuracy per 1,000 Tokens Efficiency (Per Dataset)

| Dataset | Policy | Primary Acc (%) | Avg Tokens | Acc per 1,000 Tokens |
| :--- | :--- | :--- | :--- | :--- |
| `logiqa` | **3b_direct** | 78.92% | 117.6 | **671.00** |
| `logiqa` | **7b_direct** | 87.54% | 117.5 | **745.35** |
| `logiqa` | **3b_cot** | 72.10% | 453.4 | **159.02** |
| `logiqa` | **7b_cot** | 87.54% | 434.5 | **201.49** |
| `logiqa` | **7b_got** | 88.31% | 1283.6 | **68.80** |
| `logiqa` | **7b_tot** | 100.00% | 2280.5 | **43.85** |
| `commonsenseqa` | **3b_direct** | 77.07% | 91.3 | **844.32** |
| `commonsenseqa` | **7b_direct** | 81.90% | 91.4 | **896.53** |
| `commonsenseqa` | **3b_cot** | 69.29% | 366.5 | **189.03** |
| `commonsenseqa` | **7b_cot** | 79.36% | 366.0 | **216.83** |
| `commonsenseqa` | **7b_got** | 79.77% | 1061.9 | **75.12** |
| `commonsenseqa` | **7b_tot** | 77.27% | 1683.6 | **45.90** |
| `gsm8k` | **3b_direct** | 10.39% | 114.3 | **90.86** |
| `gsm8k` | **7b_direct** | 17.36% | 113.1 | **153.53** |
| `gsm8k` | **3b_cot** | 75.13% | 454.2 | **165.43** |
| `gsm8k` | **7b_cot** | 87.79% | 389.5 | **225.39** |
| `gsm8k` | **7b_got** | 78.62% | 1486.0 | **52.91** |
| `gsm8k` | **7b_tot** | 81.25% | 2286.1 | **35.54** |


## 3. Overthinking Analysis (Cheaper vs More Expensive Policies)

Analyzes queries where adding reasoning complexity or parameter scale hurts correctness.


| Comparison (Cheaper vs Expensive) | Common N | Cheaper RIGHT / Expensive WRONG (Overthinking) | Cheaper WRONG / Expensive RIGHT (Helpful Complexity) | Net Delta |
| :--- | :--- | :--- | :--- | :--- |
| **3B Direct vs 3B CoT** (`3b_direct` vs `3b_cot`) | 3,712 | **365** (9.8%) | **1,044** (28.1%) | +679 |
| **7B Direct vs 7B CoT** (`7b_direct` vs `7b_cot`) | 3,712 | **179** (4.8%) | **1,077** (29.0%) | +898 |
| **7B CoT vs 7B GoT** (`7b_cot` vs `7b_got`) | 3,712 | **323** (8.7%) | **216** (5.8%) | -107 |
| **7B GoT vs 7B ToT** (`7b_got` vs `7b_tot`) | 50 | **2** (4.0%) | **1** (2.0%) | -1 |


## 4. Oracle Upper Bound & Cheapest-Correct-Action Cost

- **Oracle Accuracy (Hindsight Best Action)**: **94.85%** across 3,712 unique queries.
- **Cheapest-Correct-Action Oracle Token Cost**: **204.4 tokens per query** (vs 1,959.3 tokens for static 7B-ToT).


## 5. Pareto Frontier Table (Accuracy vs Token Overhead with 95% Bootstrap CIs)

| Policy | Evaluated N | Accuracy (%) [95% CI] | Avg Tokens per Query [95% CI] | Avg Latency (s) |
| :--- | :--- | :--- | :--- | :--- |
| **3b_direct** | 3,712 | **53.96%** [52.34%, 55.60%] | **107.8** [107.0, 108.5] | 0.08s |
| **7b_direct** | 3,712 | **60.75%** [59.24%, 62.34%] | **107.3** [106.5, 108.1] | 0.28s |
| **3b_cot** | 3,712 | **72.25%** [70.82%, 73.68%] | **425.1** [422.0, 428.3] | 2.26s |
| **7b_cot** | 3,712 | **84.94%** [83.84%, 86.07%] | **396.0** [393.2, 398.8] | 3.46s |
| **7b_got** | 3,712 | **82.06%** [80.85%, 83.35%] | **1282.6** [1272.7, 1292.5] | 4.81s |
| **7b_tot** | 50 | **84.00%** [72.00%, 94.00%] | **2019.7** [1889.6, 2171.3] | 9.31s |


## 6. Anomalies & Quality Control Audit Report

- **Suspiciously Fast Outputs (Latency < 0.01s)**: **0** rows.
- **Empty Model Responses (`raw_output == ""`)**: **0** rows.
- **Token Count Disagreements (>5% OpenAI vs Ollama Native)**: **0** rows.
- **Execution Error / Timeout Rows**: **0** rows.
