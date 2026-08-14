"""Download GSM8K, LogiQA, CommonsenseQA and write unified CSVs to data/processed/."""
import pandas as pd
from datasets import load_dataset

# --- GSM8K (math) ---
gsm8k = load_dataset("openai/gsm8k", "main", split="test")
rows = []
for i, ex in enumerate(gsm8k):
    answer = ex["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({
        "problem_id": f"gsm8k_{i}",
        "question": ex["question"],
        "expected_answer": answer,
        "category": "math",
        "difficulty": "unknown",
    })
pd.DataFrame(rows).to_csv("data/processed/gsm8k.csv", index=False)
print(f"GSM8K: {len(rows)} problems saved")

# --- LogiQA (logic, MCQ) ---
logiqa = load_dataset("lucasmccabe/logiqa", split="test")
rows = []
for i, ex in enumerate(logiqa):
    options = ex["options"]
    correct_idx = ex["correct_option"]
    letters = ["A", "B", "C", "D"]
    formatted_q = ex["query"] + " " + ex["context"] + "\n" + \
        "\n".join(f"{letters[j]}. {opt}" for j, opt in enumerate(options))
    rows.append({
        "problem_id": f"logiqa_{i}",
        "question": formatted_q,
        "expected_answer": letters[correct_idx],
        "category": "logic",
        "difficulty": "unknown",
    })
pd.DataFrame(rows).to_csv("data/processed/logiqa.csv", index=False)
print(f"LogiQA: {len(rows)} problems saved")

# --- CommonsenseQA ---
csqa = load_dataset("tau/commonsense_qa", split="validation")
rows = []
for i, ex in enumerate(csqa):
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    formatted_q = ex["question"] + "\n" + \
        "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    rows.append({
        "problem_id": f"csqa_{i}",
        "question": formatted_q,
        "expected_answer": ex["answerKey"],
        "category": "commonsense",
        "difficulty": "unknown",
    })
pd.DataFrame(rows).to_csv("data/processed/commonsenseqa.csv", index=False)
print(f"CommonsenseQA: {len(rows)} problems saved")
