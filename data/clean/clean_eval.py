import os
import sys
import time
import json
import subprocess
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from openai import OpenAI

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.strategies.direct import DirectStrategy
from src.strategies.cot import CoTStrategy
from src.strategies.got import GoTStrategy
from src.strategies.tot import ToTStrategy
from src.evaluation.correctness import check_correct
from src.evaluation.cost import estimate_cost
from src.llm.parser import extract_final_answer, extract_number

CLEAN_DIR = os.path.dirname(os.path.abspath(__file__))
OUTCOMES_CSV = os.path.join(CLEAN_DIR, "outcomes.csv")
MANIFEST_JSON = os.path.join(CLEAN_DIR, "manifest.json")
LOG_MD = os.path.join(PROJECT_ROOT, "CLEAN_RUN_LOG.md")
PLAN_MD = os.path.join(PROJECT_ROOT, "CLEAN_RUN_PLAN.md")

ACTIONS_ORDER = ["3b_direct", "7b_direct", "3b_cot", "7b_cot", "7b_got", "7b_tot"]

MODEL_TAGS = {
    "3b_direct": "qwen2.5:3b-instruct-q4_K_M",
    "3b_cot": "qwen2.5:3b-instruct-q4_K_M",
    "7b_direct": "qwen2.5:7b-instruct-q4_K_M",
    "7b_cot": "qwen2.5:7b-instruct-q4_K_M",
    "7b_got": "qwen2.5:7b-instruct-q4_K_M",
    "7b_tot": "qwen2.5:7b-instruct-q4_K_M",
}

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency: float
    error: str = ""
    attempts: int = 1
    prompt_eval_count: int = 0
    eval_count: int = 0

class CleanLLMClient:
    def __init__(self, base_url="http://localhost:11434/v1", model="qwen2.5:7b-instruct-q4_K_M",
                 temperature=0.2, max_tokens=512, timeout=900.0, seed=42):
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            max_retries=0  # Zero internal SDK retries
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.seed = seed

    def generate(self, prompt: str) -> LLMResponse:
        start = time.time()
        last_err = ""
        
        for attempt in range(1, 3):  # Max 2 attempts (1 initial + 1 retry pass)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    extra_body={
                        "options": {"seed": self.seed},
                        "keep_alive": -1  # Keep warm in VRAM
                    }
                )
                latency = time.time() - start
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                
                in_tok = usage.prompt_tokens if usage else 0
                out_tok = usage.completion_tokens if usage else 0
                
                return LLMResponse(
                    text=text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency=latency,
                    attempts=attempt,
                    prompt_eval_count=in_tok,
                    eval_count=out_tok
                )
            except Exception as e:
                last_err = str(e)
                time.sleep(1.0)
                
        latency = time.time() - start
        return LLMResponse(text="", input_tokens=0, output_tokens=0, latency=latency, error=last_err, attempts=2)

def log_gpu_and_ollama(event_description: str):
    try:
        nvismi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
    except Exception as e:
        nvismi = f"nvidia-smi error: {e}"
        
    try:
        ollama_ps = subprocess.check_output(["ollama", "ps"], stderr=subprocess.STDOUT).decode()
    except Exception as e:
        ollama_ps = f"ollama ps error: {e}"
        
    entry = f"\n### System Snapshot: {event_description}\n"
    entry += f"Timestamp: `{datetime.now().isoformat()}`\n\n"
    entry += f"**`nvidia-smi` Output:**\n```\n{nvismi.strip()}\n```\n\n"
    entry += f"**`ollama ps` Output:**\n```\n{ollama_ps.strip()}\n```\n\n---\n"
    
    with open(LOG_MD, "a") as f:
        f.write(entry)

def log_progress_entry(message: str):
    with open(LOG_MD, "a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] {message}\n")

def warm_model(client: CleanLLMClient):
    warmup_prompts = [
        "What is 1+1? Answer: 2.",
        "What is the capital of France? Answer: Paris.",
        "Explain brevity in one sentence."
    ]
    for p in warmup_prompts:
        client.generate(p)

def check_extraction_failed(raw_text: str, extracted_answer: str, category: str) -> bool:
    if not raw_text or not raw_text.strip():
        return True
    if not extracted_answer or not extracted_answer.strip():
        return True
    if category in ("logic", "commonsense"):
        char = extracted_answer.strip().upper()[:1]
        if char not in ("A", "B", "C", "D", "E"):
            return True
    elif category == "math":
        num = extract_number(extracted_answer)
        if num is None:
            return True
    return False

def load_all_queries(seed=42):
    dfs = []
    for d in ["commonsenseqa", "gsm8k", "logiqa"]:
        p = os.path.join(PROJECT_ROOT, f"data/processed/{d}.csv")
        df = pd.read_csv(p)
        df["dataset"] = d
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def load_prompt(p):
    with open(os.path.join(PROJECT_ROOT, p)) as f:
        return f.read()

def get_strategy_instance(policy_name: str, client: CleanLLMClient):
    if policy_name in ("3b_direct", "7b_direct"):
        return DirectStrategy(client, load_prompt("prompts/direct.txt"))
    elif policy_name in ("3b_cot", "7b_cot"):
        return CoTStrategy(client, load_prompt("prompts/cot.txt"))
    elif policy_name == "7b_got":
        return GoTStrategy(client, load_prompt("prompts/got.txt"), max_nodes=4, aggregate_top_k=2)
    elif policy_name == "7b_tot":
        return ToTStrategy(client, load_prompt("prompts/tot.txt"), branches=2, depth=2, eval_top_k=2)
    else:
        raise ValueError(f"Unknown policy {policy_name}")

def append_outcomes(rows):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    header = not os.path.exists(OUTCOMES_CSV)
    df_new.to_csv(OUTCOMES_CSV, mode="a", index=False, header=header)

def load_existing_outcomes():
    if not os.path.exists(OUTCOMES_CSV):
        return set()
    try:
        df = pd.read_csv(OUTCOMES_CSV)
        if "query_id" in df.columns and "policy" in df.columns:
            return set(zip(df["query_id"].astype(str), df["policy"].astype(str)))
    except Exception:
        pass
    return set()

def run_full_clean_evaluation():
    all_queries_df = load_all_queries(seed=42)
    existing_keys = load_existing_outcomes()
    
    log_progress_entry(f"Starting Clean Evaluation. Total Query Pool: {len(all_queries_df)}. Finished keys in CSV: {len(existing_keys)}")
    
    for policy in ACTIONS_ORDER:
        model_name = MODEL_TAGS[policy]
        client = CleanLLMClient(model=model_name, seed=42, timeout=900.0)
        
        # 1. Warm model with 3 throwaway queries
        warm_model(client)
        
        # 2. Log System Snapshot at start of policy run
        log_gpu_and_ollama(f"Start of Policy Execution: `{policy}` ({model_name})")
        
        strategy = get_strategy_instance(policy, client)
        
        # Remaining queries to evaluate for this policy
        remaining_queries = [row for _, row in all_queries_df.iterrows()
                             if (str(row["problem_id"]), policy) not in existing_keys]
        
        total_remaining = len(remaining_queries)
        print(f"\n=======================================================")
        print(f"Executing Policy: {policy} | Model: {model_name}")
        print(f"Remaining Queries to Process: {total_remaining} / {len(all_queries_df)}")
        print(f"=======================================================")
        
        if total_remaining == 0:
            print(f"Policy {policy} already completely evaluated across all 3,712 queries.")
            continue
            
        policy_start_time = time.time()
        recent_200_errors = 0
        policy_batch = []
        
        for idx, q in enumerate(remaining_queries, 1):
            q_id = str(q["problem_id"])
            dataset = q["dataset"]
            question = q["question"]
            expected = str(q["expected_answer"])
            category = q["category"]
            
            res = strategy.solve(question)
            
            # Answer extraction check
            ext_failed = check_extraction_failed(res.raw_text, res.predicted_answer, category) if not res.error else True
            is_corr = check_correct(res.predicted_answer, expected, category) if (not res.error and not ext_failed) else False
            
            if res.error:
                recent_200_errors += 1
                
            # Token disagreement check (compare OpenAI SDK vs Ollama native)
            prompt_eval_count = getattr(res, "prompt_eval_count", res.input_tokens)
            eval_count = getattr(res, "eval_count", res.output_tokens)
            
            disagree_in = abs(res.input_tokens - prompt_eval_count) / max(prompt_eval_count, 1) > 0.05
            disagree_out = abs(res.output_tokens - eval_count) / max(eval_count, 1) > 0.05
            token_disagreement_flag = disagree_in or disagree_out
            
            row_dict = {
                "query_id": q_id,
                "dataset": dataset,
                "policy": policy,
                "correct": is_corr,
                "extracted_answer": res.predicted_answer,
                "raw_output": res.raw_text[:500] if hasattr(res, "raw_text") else "",
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "total_tokens": res.input_tokens + res.output_tokens,
                "latency": res.latency,
                "n_calls": res.n_llm_calls,
                "error": res.error,
                "attempts": getattr(res, "attempts", 1),
                "seed": 42,
                "timestamp": datetime.now().isoformat(),
                "model_tag": model_name,
                "ollama_options": json.dumps({"seed": 42, "keep_alive": -1}),
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "token_disagreement_flag": token_disagreement_flag,
                "extraction_failed": ext_failed,
            }
            policy_batch.append(row_dict)
            existing_keys.add((q_id, policy))
            
            # Append every 10 rows
            if len(policy_batch) >= 10:
                append_outcomes(policy_batch)
                policy_batch = []
                
            # System Snapshot every 500 queries
            if idx % 500 == 0:
                log_gpu_and_ollama(f"Mid-Run Snapshot: Policy `{policy}` step {idx} / {total_remaining}")
                
            # Error Rate Check every 200 queries
            if idx % 200 == 0:
                err_rate = (recent_200_errors / 200.0) * 100.0
                if err_rate > 1.0:
                    cause_msg = f"PAUSED JOB: Policy `{policy}` exceeded 1% error rate limit in last 200 queries ({err_rate:.2f}% errors, {recent_200_errors} errors)."
                    print(f"\nCRITICAL: {cause_msg}")
                    log_progress_entry(cause_msg)
                    if policy_batch:
                        append_outcomes(policy_batch)
                    return False
                recent_200_errors = 0  # Reset window
                
            # Print console progress
            if idx % 50 == 0 or idx == total_remaining:
                elapsed = time.time() - policy_start_time
                avg_time_per_q = elapsed / idx
                eta_min = (total_remaining - idx) * avg_time_per_q / 60.0
                print(f"[{policy}] Progress: {idx}/{total_remaining} done ({idx/total_remaining*100:.1f}%) | Avg Lat: {avg_time_per_q:.2f}s | ETA: {eta_min:.1f}m")
                
        if policy_batch:
            append_outcomes(policy_batch)
            policy_batch = []
            
        elapsed_policy = time.time() - policy_start_time
        log_progress_entry(f"Policy `{policy}` completed in {elapsed_policy/60.0:.2f} minutes.")
        
        # Regenerate CLEAN_SUMMARY.md after each policy finishes
        try:
            from data.clean.generate_summary import generate_summary
            generate_summary()
        except Exception as e:
            print(f"Error regenerating summary: {e}")
            
    # Write FINAL_CLEAN_SUMMARY.md when all 6 policies complete
    try:
        from data.clean.generate_summary import generate_summary
        generate_summary(final=True)
    except Exception as e:
        print(f"Error generating final summary: {e}")
        
    print("\nALL SIX POLICIES COMPLETED SUCCESSFULLY!")
    return True

if __name__ == "__main__":
    run_full_clean_evaluation()
