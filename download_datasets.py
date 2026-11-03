"""Download GSM8K, LogiQA v2, CommonsenseQA and write unified CSVs to data/processed/."""
import os
import pandas as pd
from datasets import load_dataset

os.makedirs("data/processed", exist_ok=True)

# --- GSM8K (math) ---
try:
    print("Downloading GSM8K...")
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
except Exception as e:
    print(f"GSM8K download failed: {e}")

# --- LogiQA (logic, MCQ) ---
try:
    print("Downloading LogiQA...")
    # Try the working parquet-based version first
    logiqa = load_dataset("di-dimitrov/logiqa", split="test")
    rows = []
    letters = ["A", "B", "C", "D"]
    for i, ex in enumerate(logiqa):
        options = ex["options"]
        correct_idx = ex["correct_option"]
        context = ex.get("context", "")
        query = ex.get("query", "")
        formatted_q = query + " " + context + "\n" + \
            "\n".join(f"{letters[j]}. {opt}" for j, opt in enumerate(options))
        rows.append({
            "problem_id": f"logiqa_{i}",
            "question": formatted_q,
            "expected_answer": letters[correct_idx] if isinstance(correct_idx, int) else str(correct_idx),
            "category": "logic",
            "difficulty": "unknown",
        })
    pd.DataFrame(rows).to_csv("data/processed/logiqa.csv", index=False)
    print(f"LogiQA: {len(rows)} problems saved")
except Exception as e:
    print(f"LogiQA primary download failed: {e}")
    # Fallback: Use ARC-Challenge (science reasoning MCQ) as logic proxy
    try:
        print("Trying ARC-Challenge as logic fallback...")
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        rows = []
        for i, ex in enumerate(arc):
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            formatted_q = ex["question"] + "\n" + \
                "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            rows.append({
                "problem_id": f"logiqa_{i}",
                "question": formatted_q,
                "expected_answer": ex["answerKey"],
                "category": "logic",
                "difficulty": "unknown",
            })
        pd.DataFrame(rows).to_csv("data/processed/logiqa.csv", index=False)
        print(f"ARC-Challenge (logic fallback): {len(rows)} problems saved")
    except Exception as e2:
        print(f"All logic dataset attempts failed: {e2}")

# --- CommonsenseQA ---
try:
    print("Downloading CommonsenseQA...")
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
except Exception as e:
    print(f"CommonsenseQA download failed: {e}")

print("\nDone! Check data/processed/ for CSV files.")

