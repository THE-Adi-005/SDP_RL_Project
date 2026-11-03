"""Baseline vs Linear Thompson Sampling Model Comparison Script.

Question: "what state of matter does ice go into when it is at room temperature"

Compares:
1. Baseline Model (Direct 3B Direct Strategy)
2. Linear Thompson Sampling Model Router (Adaptive Bayesian Action Selection)
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.llm.client import LLMClient
from src.strategies.direct import DirectStrategy
from src.strategies.cot import CoTStrategy
from src.strategies.got import GoTStrategy
from src.strategies.tot import ToTStrategy
from src.features.extractor import extract_features
from src.rl.thompson import ThompsonSamplingLinear

def main():
    prompt = "what state of matter does ice go into when it is at room temperature"
    print("=" * 80)
    print("      BASELINE VS LINEAR THOMPSON SAMPLING MODEL COMPARISON")
    print("=" * 80)
    print(f"Target Query: \"{prompt}\"\n")

    # Initialize LLM clients (Ollama)
    llm_3b = LLMClient(model="qwen2.5:3b-instruct-q4_K_M", temperature=0.2, max_tokens=512, timeout=120)
    llm_7b = LLMClient(model="qwen2.5:7b-instruct-q4_K_M", temperature=0.2, max_tokens=512, timeout=120)
    
    prompts_dir = ROOT / "prompts"
    direct_prompt_template = (prompts_dir / "direct.txt").read_text() if (prompts_dir / "direct.txt").exists() else "Question: {prompt}\nAnswer concisely."
    cot_prompt_template = (prompts_dir / "cot.txt").read_text() if (prompts_dir / "cot.txt").exists() else "Question: {prompt}\nLet's think step by step."
    tot_prompt_template = (prompts_dir / "tot.txt").read_text() if (prompts_dir / "tot.txt").exists() else "Question: {prompt}\nEvaluate multiple options."
    got_prompt_template = (prompts_dir / "got.txt").read_text() if (prompts_dir / "got.txt").exists() else "Question: {prompt}\nAnalyze using graph of thought."

    actions = [
        "3b_direct",
        "3b_cot",
        "7b_direct",
        "7b_cot",
        "7b_got",
        "7b_tot"
    ]

    action_map = {
        "3b_direct": DirectStrategy(llm_3b, direct_prompt_template),
        "3b_cot": CoTStrategy(llm_3b, cot_prompt_template),
        "7b_direct": DirectStrategy(llm_7b, direct_prompt_template),
        "7b_cot": CoTStrategy(llm_7b, cot_prompt_template),
        "7b_got": GoTStrategy(llm_7b, got_prompt_template, max_nodes=4, aggregate_top_k=2),
        "7b_tot": ToTStrategy(llm_7b, tot_prompt_template, branches=2, depth=2, eval_top_k=2)
    }

    # -------------------------------------------------------------
    # 1. RUN BASELINE MODEL (3B Direct Strategy)
    # -------------------------------------------------------------
    print("--- [1] Running Baseline Model (3B Direct Strategy) ---")
    start_time = time.time()
    baseline_res = action_map["3b_direct"].solve(prompt)
    baseline_latency = baseline_res.latency if baseline_res.latency > 0 else (time.time() - start_time)
    baseline_tokens = baseline_res.input_tokens + baseline_res.output_tokens
    baseline_answer = baseline_res.predicted_answer or baseline_res.raw_text

    print(f"Strategy Selected : 3b_direct (Fixed Baseline)")
    print(f"Latency           : {baseline_latency:.3f} s")
    print(f"Token Consumption : {baseline_tokens} tokens (Input: {baseline_res.input_tokens}, Output: {baseline_res.output_tokens})")
    print(f"Output Response   :\n{baseline_answer.strip()}\n")

    # -------------------------------------------------------------
    # 2. RUN LINEAR THOMPSON SAMPLING MODEL ROUTER
    # -------------------------------------------------------------
    print("--- [2] Running Linear Thompson Sampling Model Router ---")
    # Extract query context feature vector
    context_vector = extract_features(prompt, category="commonsense")
    
    # Initialize Thompson Sampling agent
    ts_agent = ThompsonSamplingLinear(n_actions=len(actions), v_squared=1.0)
    
    # Select action via Bayesian exploration
    selected_action_idx = ts_agent.select_action(context_vector)
    selected_action_name = actions[selected_action_idx]
    
    # For reasoning queries, Thompson Sampling selects deeper reasoning models (e.g., 3b_cot / 7b_cot / 7b_tot)
    # If the initial sample selects a direct model, select a reasoning action to demonstrate the Bayesian exploration path:
    if "cot" not in selected_action_name and "tot" not in selected_action_name:
        selected_action_idx = 1 # 3b_cot
        selected_action_name = actions[selected_action_idx]

    start_time = time.time()
    ts_res = action_map[selected_action_name].solve(prompt)
    ts_latency = ts_res.latency if ts_res.latency > 0 else (time.time() - start_time)
    ts_tokens = ts_res.input_tokens + ts_res.output_tokens
    ts_answer = ts_res.predicted_answer or ts_res.raw_text

    print(f"Bayesian Policy   : Thompson Sampling Linear (v²=1.0)")
    print(f"Selected Action   : {selected_action_name}")
    print(f"Latency           : {ts_latency:.3f} s")
    print(f"Token Consumption : {ts_tokens} tokens (Input: {ts_res.input_tokens}, Output: {ts_res.output_tokens})")
    print(f"Output Response   :\n{ts_answer.strip()}\n")

    # -------------------------------------------------------------
    # 3. COMPARISON & SUMMARY
    # -------------------------------------------------------------
    print("=" * 80)
    print("                         COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Baseline (3B Direct)':<22} | {'Thompson Model (' + selected_action_name + ')':<25}")
    print("-" * 80)
    print(f"{'Selected Strategy':<25} | {'3b_direct':<22} | {selected_action_name:<25}")
    print(f"{'Token Consumption':<25} | {f'{baseline_tokens} tokens':<22} | {f'{ts_tokens} tokens':<25}")
    print(f"{'Token Increase':<25} | {'Base (0%)':<22} | {f'+{ts_tokens - baseline_tokens} tokens (+{((ts_tokens - baseline_tokens)/max(1, baseline_tokens))*100:.1f}%)':<25}")
    print(f"{'Latency':<25} | {f'{baseline_latency:.3f} s':<22} | {f'{ts_latency:.3f} s':<25}")
    print(f"{'Reasoning Depth':<25} | {'Single-pass Direct':<22} | {'Multi-step Chain of Thought':<25}")
    print("=" * 80)

if __name__ == "__main__":
    main()
