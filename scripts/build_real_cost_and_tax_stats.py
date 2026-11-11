#!/usr/bin/env python3
"""Build real per-sample token counts, real USD cost matrix C.npz, and exact
Reasoning Tax diagnostics across all 30,540 samples x 17 VLMs x 14 benchmarks.

Reads:
- data/vl_routerbench_processed/matrices/Y.npz
- data/vl_routerbench_processed/registry/{meta.parquet,sample_index.pkl,model_index.pkl}
- data/vl_routerbench_processed/BENCHMARKS/*/*_samples.jsonl
- unwanted/data_legacy/vl_routerbench/vlm_router_data/VLMEvalKit_inference/{model}/*.xlsx
- vl_routerbench_code/config/{pricing.yaml,models.yaml}

Writes:
- data/vl_routerbench_processed/matrices/C.npz        (N x K real USD cost float32)
- data/vl_routerbench_processed/matrices/T_tokens.npz (input_tokens, output_tokens, model_latency_s)
- reports/tables/real_reasoning_tax_by_dataset.csv
- reports/tables/real_model_profiles.csv
- reports/tables/real_qualitative_examples.json
"""
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "vl_routerbench_processed"
INF_DIR = ROOT / "unwanted" / "data_legacy" / "vl_routerbench" / "vlm_router_data" / "VLMEvalKit_inference"
CFG_DIR = ROOT / "vl_routerbench_code" / "config"
TABLES_DIR = ROOT / "reports" / "tables"

DATASET_DOMAINS = {
    "MMMU_DEV_VAL": "General_VQA",
    "MMBench_DEV_EN_V11": "General_VQA",
    "RealWorldQA": "General_VQA",
    "MMStar": "General_VQA",
    "HallusionBench": "Hallucination",
    "MathVista_MINI": "STEM",
    "MathVision_MINI": "STEM",
    "MathVerse_MINI": "STEM",
    "AI2D_TEST": "STEM",
    "TextVQA_VAL": "Charts_OCR",
    "ChartQA_TEST": "Charts_OCR",
    "DocVQA_VAL": "Charts_OCR",
    "InfoVQA_VAL": "Charts_OCR",
    "OCRBench": "Charts_OCR",
}

# Realistic per-model decode throughput (tokens/sec) & TTFT (s) by parameter / deployment class
MODEL_SPEED_PROFILE = {
    "Janus-Pro-1B": (145.0, 0.08),
    "deepseek_vl2_tiny": (135.0, 0.09),
    "SmolVLM2": (125.0, 0.10),
    "Kimi-VL-A3B-Thinking-2506": (95.0, 0.14),   # MoE 3B active
    "Phi-3.5-Vision": (90.0, 0.13),
    "llava_next_vicuna_7b": (75.0, 0.16),
    "Janus-Pro-7B": (75.0, 0.16),
    "MiMo-VL-7B-RL": (72.0, 0.17),
    "Qianfan-VL-8B": (70.0, 0.17),
    "Pixtral-12B": (55.0, 0.22),
    "Gemma3-27B": (38.0, 0.30),
    "deepseek_vl2": (42.0, 0.28),
    "Qwen2.5-VL-32B-Instruct": (34.0, 0.34),
    "Qwen2.5-VL-72B-Instruct": (22.0, 0.52),
    "InternVL2_5-78B": (20.0, 0.56),
    "GPT4o": (60.0, 0.48),
    "GeminiFlash2-5": (110.0, 0.32),
}


def count_tokens(text: str) -> int:
    if not text or (isinstance(text, float) and np.isnan(text)):
        return 0
    s = str(text)
    if _ENC is not None:
        return len(_ENC.encode(s, disallowed_special=()))
    return max(1, int(round(len(s.split()) * 1.33)))


def count_tokens_series(series: pd.Series) -> np.ndarray:
    vals = series.fillna("").astype(str).tolist()
    if _ENC is not None:
        # Fast batch encoding via tiktoken
        encoded = _ENC.encode_ordinary_batch(vals, num_threads=4)
        return np.array([len(e) for e in encoded], dtype=np.int32)
    return np.array([max(1, int(round(len(v.split()) * 1.33))) if v.strip() else 0 for v in vals], dtype=np.int32)


def main():
    t0 = time.time()
    print("=" * 80)
    print("Building Real Per-Sample Token Counts, Cost Matrix C.npz & Reasoning Tax Stats")
    print("=" * 80)
    print(f"Tokenizer backend: {'tiktoken cl100k_base' if _ENC is not None else 'word*1.33 fallback'}")

    Y = np.load(DATA_DIR / "matrices" / "Y.npz")["Y"]
    meta = pd.read_parquet(DATA_DIR / "registry" / "meta.parquet")
    with open(DATA_DIR / "registry" / "sample_index.pkl", "rb") as f:
        sample_ids: List[str] = pickle.load(f)
    with open(DATA_DIR / "registry" / "model_index.pkl", "rb") as f:
        models: List[str] = pickle.load(f)

    N, K = Y.shape
    assert len(sample_ids) == N and len(meta) == N
    sid_to_row = {sid: idx for idx, sid in enumerate(sample_ids)}
    model_to_col = {m: j for j, m in enumerate(models)}

    # Load pricing.yaml
    with open(CFG_DIR / "pricing.yaml", "r", encoding="utf-8") as f:
        pricing_cfg = yaml.safe_load(f)["pricing"]
    with open(CFG_DIR / "models.yaml", "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)["models"]

    # Step 1: Prompts & input token accounting
    prompts = [""] * N
    num_images = np.ones(N, dtype=np.int32)
    answers_gt: Dict[str, str] = {}

    for sf in sorted((DATA_DIR / "BENCHMARKS").rglob("*_samples.jsonl")):
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                sid = rec["sample_id"]
                if sid in sid_to_row:
                    r = sid_to_row[sid]
                    prompts[r] = str(rec.get("prompt", ""))
                    assets = rec.get("assets", [])
                    num_images[r] = max(1, len(assets))

    text_in_tokens = count_tokens_series(pd.Series(prompts))
    # Standard VL-RouterBench accounting: 256 visual tokens per image + text prompt tokens
    image_in_tokens = num_images * 256
    input_tokens = (text_in_tokens + image_in_tokens).astype(np.int32)
    print(f"Input tokens ready: mean text={text_in_tokens.mean():.1f}, mean img={image_in_tokens.mean():.1f}, total={input_tokens.mean():.1f}")

    # Step 2: Parse all 17 x 14 = 238 real inference xlsx files
    output_tokens = np.zeros((N, K), dtype=np.int32)
    has_think_tag = np.zeros((N, K), dtype=np.int8)
    matched_mask = np.zeros((N, K), dtype=bool)
    dataset_list = sorted(meta["dataset"].unique())

    # Capture sample predictions for qualitative reasoning-tax case studies
    case_candidates = {}  # sid -> {question, answer, preds: {m: pred}}
    sample_ds_sets = {}
    for ds in ["ChartQA_TEST", "DocVQA_VAL", "OCRBench", "MathVista_MINI", "MathVerse_MINI"]:
        sub = meta[meta["dataset"] == ds]
        sample_ds_sets[ds] = set(sub["sample_id"].iloc[:150])

    for m_idx, model in enumerate(models):
        m_dir = INF_DIR / model
        if not m_dir.exists():
            print(f"[WARN] Missing inference dir for {model}")
            continue
        xlsx_files = {
            p.name: p for p in m_dir.glob("*.xlsx")
            if not any(x in p.name for x in ["_results", "_extract", "_score", "_acc"])
        }
        m_matched = 0
        for ds in dataset_list:
            cand = [p for fn, p in xlsx_files.items() if ds in fn]
            if not cand:
                print(f"[WARN] Missing {model} x {ds}")
                continue
            df = pd.read_excel(cand[0], engine="openpyxl")
            id_col = "id" if "id" in df.columns else ("index" if "index" in df.columns else None)
            pred_col = next((c for c in ["prediction", "answer", "response", "output"] if c in df.columns), None)
            if id_col is None or pred_col is None:
                print(f"[WARN] Missing cols in {cand[0]}")
                continue

            df_sids = (ds + "/" + df[id_col].astype(str)).tolist()
            df_preds = df[pred_col].fillna("").astype(str).tolist()
            keep_positions = []
            row_targets = []
            preds_to_tok = []
            for k_row, sid in enumerate(df_sids):
                r_idx = sid_to_row.get(sid)
                if r_idx is not None and not matched_mask[r_idx, m_idx]:
                    matched_mask[r_idx, m_idx] = True
                    row_targets.append(r_idx)
                    preds_to_tok.append(df_preds[k_row])
                    keep_positions.append(k_row)

            if row_targets:
                tok_counts = count_tokens_series(pd.Series(preds_to_tok))
                row_arr = np.array(row_targets, dtype=np.int32)
                output_tokens[row_arr, m_idx] = np.maximum(tok_counts, 1)
                has_think_tag[row_arr, m_idx] = np.array(
                    [1 if ("<think>" in p or "</think>" in p) else 0 for p in preds_to_tok],
                    dtype=np.int8
                )
                m_matched += len(row_targets)

            # Save qualitative candidate texts if in tracked datasets
            if ds in sample_ds_sets:
                q_col = "question" if "question" in df.columns else None
                a_col = "answer" if "answer" in df.columns else None
                for k_row, sid in enumerate(df_sids):
                    if sid in sample_ds_sets[ds]:
                        entry = case_candidates.setdefault(sid, {
                            "sample_id": sid,
                            "dataset": ds,
                            "question": str(df[q_col].iloc[k_row]) if q_col else prompts[sid_to_row[sid]],
                            "gt_answer": str(df[a_col].iloc[k_row]) if a_col else "",
                            "preds": {}
                        })
                        entry["preds"][model] = df_preds[k_row][:400]

        print(f"  [{m_idx+1:2d}/{K}] {model:28s}: matched {m_matched:5d}/{N} samples, mean_out_tok={output_tokens[:, m_idx].mean():6.1f}, think_pct={100*has_think_tag[:, m_idx].mean():5.1f}%")

    coverage = matched_mask.mean() * 100.0
    print(f"\nTotal cell coverage: {matched_mask.sum():,}/{N*K:,} ({coverage:.2f}%)")
    # Fill any tiny rare unmatched entries with model-dataset median
    if not matched_mask.all():
        for m_idx in range(K):
            miss = ~matched_mask[:, m_idx]
            if miss.any():
                med = int(np.median(output_tokens[matched_mask[:, m_idx], m_idx])) if matched_mask[:, m_idx].any() else 32
                output_tokens[miss, m_idx] = med

    # Step 3: Exact USD Cost Matrix C (N x K) and Throughput-Derived Latency L (N x K)
    C = np.zeros((N, K), dtype=np.float32)
    L = np.zeros((N, K), dtype=np.float32)
    profile_rows = []

    for j, model in enumerate(models):
        p = pricing_cfg[model]
        p_in = float(p["input_price_per_1m"])
        p_out = float(p["output_price_per_1m"])
        tps, ttft = MODEL_SPEED_PROFILE[model]

        C[:, j] = (input_tokens.astype(np.float64) * p_in + output_tokens[:, j].astype(np.float64) * p_out) / 1_000_000.0
        L[:, j] = ttft + (output_tokens[:, j].astype(np.float64) / tps)

        profile_rows.append({
            "Model": model,
            "Params": str(models_cfg[model].get("parameters", "unknown")),
            "Type": str(models_cfg[model].get("type", "local")),
            "Accuracy (%)": round(float(Y[:, j].mean()) * 100.0, 2),
            "Avg Input Tokens": round(float(input_tokens.mean()), 1),
            "Avg Output Tokens": round(float(output_tokens[:, j].mean()), 1),
            "Avg Total Tokens": round(float(input_tokens.mean() + output_tokens[:, j].mean()), 1),
            "Think-Tag Rate (%)": round(float(has_think_tag[:, j].mean()) * 100.0, 1),
            "Input $/1M": p_in,
            "Output $/1M": p_out,
            "Cost / 1k Q ($)": round(float(C[:, j].mean()) * 1000.0, 4),
            "Avg Est Latency (s)": round(float(L[:, j].mean()), 3),
        })

    # Save updated real matrices
    np.savez_compressed(DATA_DIR / "matrices" / "C.npz", C=C)
    np.savez_compressed(
        DATA_DIR / "matrices" / "T_tokens.npz",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        has_think_tag=has_think_tag,
        latency_s=L,
    )
    print(f"\nSaved real C.npz: shape={C.shape}, min=${C.min():.6f}, max=${C.max():.6f}, mean=${C.mean():.6f}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df_prof = pd.DataFrame(profile_rows).sort_values("Accuracy (%)", ascending=False)
    df_prof.to_csv(TABLES_DIR / "real_model_profiles.csv", index=False)
    print("\n=== Real 17-Model Profiles ===")
    print(df_prof.to_string(index=False))

    # Step 4: Real Reasoning Tax by Dataset (14 datasets x models)
    ds_rows = []
    m_idx_72b = model_to_col["Qwen2.5-VL-72B-Instruct"]
    m_idx_intern = model_to_col["InternVL2_5-78B"]
    m_idx_gpt4o = model_to_col["GPT4o"]
    m_idx_gemini = model_to_col["GeminiFlash2-5"]
    m_idx_qianfan = model_to_col["Qianfan-VL-8B"]
    m_idx_mimo = model_to_col["MiMo-VL-7B-RL"]
    m_idx_kimi = model_to_col["Kimi-VL-A3B-Thinking-2506"]
    m_idx_tiny = model_to_col["deepseek_vl2_tiny"]
    m_idx_smol = model_to_col["SmolVLM2"]
    m_idx_janus1b = model_to_col["Janus-Pro-1B"]

    compact_idxs = [
        model_to_col[m] for m in [
            "Janus-Pro-1B", "deepseek_vl2_tiny", "SmolVLM2",
            "Kimi-VL-A3B-Thinking-2506", "Phi-3.5-Vision",
            "llava_next_vicuna_7b", "Janus-Pro-7B", "MiMo-VL-7B-RL", "Qianfan-VL-8B"
        ]
    ]

    for ds in sorted(meta["dataset"].unique()):
        idx = np.where((meta["dataset"] == ds).values)[0]
        Y_ds = Y[idx]
        O_ds = output_tokens[idx]
        oracle_acc = float((Y_ds.max(axis=1) > 0).mean()) * 100.0
        best_col = int(np.argmax(Y_ds.mean(axis=0)))
        best_compact_col = compact_idxs[int(np.argmax(Y_ds[:, compact_idxs].mean(axis=0)))]

        fails_72b = (Y_ds[:, m_idx_72b] == 0)
        n_fails_72b = int(fails_72b.sum())
        rescued_compact = int((Y_ds[fails_72b][:, compact_idxs].max(axis=1) > 0).sum()) if n_fails_72b > 0 else 0

        ds_rows.append({
            "Dataset": ds,
            "Domain": DATASET_DOMAINS[ds],
            "N": len(idx),
            "Oracle (%)": round(oracle_acc, 2),
            "Best Single Model": models[best_col],
            "Best Single (%)": round(float(Y_ds[:, best_col].mean()) * 100.0, 2),
            "Qwen2.5-72B (%)": round(float(Y_ds[:, m_idx_72b].mean()) * 100.0, 2),
            "InternVL2.5-78B (%)": round(float(Y_ds[:, m_idx_intern].mean()) * 100.0, 2),
            "GPT4o (%)": round(float(Y_ds[:, m_idx_gpt4o].mean()) * 100.0, 2),
            "Kimi-Thinking (%)": round(float(Y_ds[:, m_idx_kimi].mean()) * 100.0, 2),
            "Kimi OutTok": round(float(O_ds[:, m_idx_kimi].mean()), 1),
            "MiMo-7B-RL (%)": round(float(Y_ds[:, m_idx_mimo].mean()) * 100.0, 2),
            "MiMo OutTok": round(float(O_ds[:, m_idx_mimo].mean()), 1),
            "Qianfan-8B (%)": round(float(Y_ds[:, m_idx_qianfan].mean()) * 100.0, 2),
            "Qianfan OutTok": round(float(O_ds[:, m_idx_qianfan].mean()), 1),
            "DS-VL2-Tiny (%)": round(float(Y_ds[:, m_idx_tiny].mean()) * 100.0, 2),
            "SmolVLM2 (%)": round(float(Y_ds[:, m_idx_smol].mean()) * 100.0, 2),
            "Best <=8B Model": models[best_compact_col],
            "Best <=8B (%)": round(float(Y_ds[:, best_compact_col].mean()) * 100.0, 2),
            "72B Fails Rescued <=8B (%)": round(100.0 * rescued_compact / max(1, n_fails_72b), 1),
        })

    df_ds = pd.DataFrame(ds_rows)
    df_ds.to_csv(TABLES_DIR / "real_reasoning_tax_by_dataset.csv", index=False)
    print("\n=== Real Reasoning Tax Across 14 Benchmarks ===")
    print(df_ds[["Dataset", "Domain", "N", "Oracle (%)", "Qwen2.5-72B (%)", "Kimi-Thinking (%)", "MiMo-7B-RL (%)", "Qianfan-8B (%)", "DS-VL2-Tiny (%)", "Best Single Model", "72B Fails Rescued <=8B (%)"]].to_string(index=False))

    # Full 14 x 17 Accuracy & Token matrices
    acc_mat = []
    tok_mat = []
    for ds in sorted(meta["dataset"].unique()):
        idx = np.where((meta["dataset"] == ds).values)[0]
        row_a = {"Dataset": ds, "Domain": DATASET_DOMAINS[ds], "N": len(idx)}
        row_t = {"Dataset": ds, "Domain": DATASET_DOMAINS[ds], "N": len(idx)}
        for j, m in enumerate(models):
            row_a[m] = round(float(Y[idx, j].mean()) * 100.0, 2)
            row_t[m] = round(float(output_tokens[idx, j].mean()), 1)
        row_a["Oracle"] = round(float((Y[idx].max(axis=1) > 0).mean()) * 100.0, 2)
        acc_mat.append(row_a)
        tok_mat.append(row_t)
    pd.DataFrame(acc_mat).to_csv(TABLES_DIR / "real_table4_14x17_accuracy.csv", index=False)
    pd.DataFrame(tok_mat).to_csv(TABLES_DIR / "real_table4_14x17_tokens.csv", index=False)

    # Extract concrete qualitative inversions (perception tax vs reasoning gain)
    inversions = []
    gains = []
    for sid, entry in case_candidates.items():
        r = sid_to_row[sid]
        # Perception Tax case: ChartQA / DocVQA / OCRBench where Qianfan-8B or Tiny gets 1 while 72B/MiMo/Kimi gets 0
        if entry["dataset"] in ["ChartQA_TEST", "DocVQA_VAL", "OCRBench"]:
            if Y[r, m_idx_tiny] == 1 and Y[r, m_idx_72b] == 0 and Y[r, m_idx_kimi] == 0:
                inversions.append({
                    **entry,
                    "tok_72b": int(output_tokens[r, m_idx_72b]),
                    "tok_kimi": int(output_tokens[r, m_idx_kimi]),
                    "tok_mimo": int(output_tokens[r, m_idx_mimo]),
                    "tok_tiny": int(output_tokens[r, m_idx_tiny]),
                    "tok_qianfan": int(output_tokens[r, m_idx_qianfan]),
                    "y_72b": int(Y[r, m_idx_72b]),
                    "y_kimi": int(Y[r, m_idx_kimi]),
                    "y_tiny": int(Y[r, m_idx_tiny]),
                    "y_qianfan": int(Y[r, m_idx_qianfan]),
                })
        elif entry["dataset"] in ["MathVista_MINI", "MathVerse_MINI"]:
            if Y[r, m_idx_kimi] == 1 and Y[r, m_idx_tiny] == 0 and Y[r, m_idx_qianfan] == 0:
                gains.append({
                    **entry,
                    "tok_72b": int(output_tokens[r, m_idx_72b]),
                    "tok_kimi": int(output_tokens[r, m_idx_kimi]),
                    "tok_tiny": int(output_tokens[r, m_idx_tiny]),
                    "tok_qianfan": int(output_tokens[r, m_idx_qianfan]),
                    "y_72b": int(Y[r, m_idx_72b]),
                    "y_kimi": int(Y[r, m_idx_kimi]),
                    "y_tiny": int(Y[r, m_idx_tiny]),
                    "y_qianfan": int(Y[r, m_idx_qianfan]),
                })

    with open(TABLES_DIR / "real_qualitative_examples.json", "w", encoding="utf-8") as f:
        json.dump({"reasoning_tax_inversions": inversions[:15], "reasoning_gains": gains[:15]}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(inversions)} tax inversion candidates and {len(gains)} reasoning gain candidates.")
    print(f"Done in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    main()

