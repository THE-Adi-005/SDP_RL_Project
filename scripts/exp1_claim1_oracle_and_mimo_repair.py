#!/usr/bin/env python3
"""Claim 1 & Claim 2 Audit:
1. Chance-corrected Oracle control (global marginal shuffle & dataset-stratified shuffle).
2. Non-MCQ vs MCQ rescue rates & corrected Dense (<=8B) vs MoE-Active (16B total) definitions.
3. Re-extraction / repair of MiMo-VL-7B-RL (and SmolVLM2 InfoVQA check) from raw transcripts,
   diagnosing exact upstream VLMEvalKit parsing + int(ANLS) truncation bugs, and evaluating both
   raw Y and repaired Y_repaired.
4. Split-consistent reporting across Full (30,540), Test (6,108), and In-Dist (2,523).
"""
import ast
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "vl_routerbench_processed"
INF_DIR = ROOT / "unwanted" / "data_legacy" / "vl_routerbench" / "vlm_router_data" / "VLMEvalKit_inference"
EVAL_DIR = ROOT / "unwanted" / "data_legacy" / "vl_routerbench" / "vlm_router_data" / "VLMEvalKit_evaluation"
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MCQ_DATASETS = {"AI2D_TEST", "MMBench_DEV_EN_V11", "MMMU_DEV_VAL", "MMStar", "HallusionBench"}
NON_MCQ_DATASETS = {
    "ChartQA_TEST", "DocVQA_VAL", "InfoVQA_VAL", "OCRBench", "TextVQA_VAL",
    "RealWorldQA", "MathVerse_MINI", "MathVision_MINI", "MathVista_MINI",
}

DENSE_COMPACT_MODELS = [
    "Janus-Pro-1B", "deepseek_vl2_tiny", "SmolVLM2", "Phi-3.5-Vision",
    "llava_next_vicuna_7b", "Janus-Pro-7B", "MiMo-VL-7B-RL", "Qianfan-VL-8B",
]
LOW_ACTIVE_MOE_AND_DENSE = DENSE_COMPACT_MODELS + ["Kimi-VL-A3B-Thinking-2506"]


def strip_think_and_extract(text: str) -> str:
    """Robust parser stripping <think>...</think> and unclosed reasoning preambles."""
    if not isinstance(text, str):
        text = str(text)
    # 1. Remove closed <think>...</think> blocks
    if "</think>" in text:
        after = text.split("</think>")[-1].strip()
        if after:
            text = after
    else:
        # Unclosed <think>: salvage boxed / explicit final answer / last short line if present
        m_box = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if m_box:
            text = m_box[-1].strip()
        else:
            m_ans = re.findall(r"(?:final answer is|answer is|the answer is|therefore,?\s*answer:?)\s*[:\-]?\s*([^\n\.]+)", text, flags=re.I)
            if m_ans:
                text = m_ans[-1].strip()
            else:
                text = text.replace("<think>", "").strip()
                # If multi-line reasoning without answer tag, take last short line
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if lines:
                    text = lines[-1]
    text = text.strip().strip("\"'")
    # Remove trailing period for short answers unless decimal
    if text.endswith(".") and not re.match(r".*\d\.$", text):
        text = text[:-1].strip()
    return text


def normalize_str(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[\s]+", " ", s)
    s = s.strip(" .,!?;:\"'`")
    return s


def levenshtein_anls(pred: str, gts: List[str], threshold: float = 0.5) -> float:
    """Standard ANLS metric used in DocVQA / InfoVQA."""
    p = normalize_str(pred)
    if not p:
        return 0.0
    best = 0.0
    for g in gts:
        gn = normalize_str(g)
        if not gn:
            continue
        if p == gn:
            return 1.0
        # Edit distance
        n, m = len(p), len(gn)
        if n == 0 or m == 0:
            continue
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                cur = dp[j]
                if p[i - 1] == gn[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = cur
        nl = dp[m] / max(n, m)
        sim = 1.0 - nl if nl < threshold else 0.0
        if sim > best:
            best = sim
    return best


def chartqa_relaxed_correct(pred: str, gt: str) -> int:
    """VLMEvalKit ChartQA relaxed accuracy (exact string match OR +-5% numeric tolerance)."""
    pn = normalize_str(pred).replace(",", "").rstrip("%")
    gn = normalize_str(gt).replace(",", "").rstrip("%")
    if pn == gn and pn != "":
        return 1
    try:
        pf = float(pn)
        gf = float(gn)
        if gf == 0.0:
            return int(abs(pf) <= 1e-6)
        return int(abs(pf - gf) / abs(gf) <= 0.05)
    except Exception:
        return 0


def parse_gt_list(raw_ans: str) -> List[str]:
    s = str(raw_ans).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            vals = ast.literal_eval(s)
            if isinstance(vals, list):
                return [str(v) for v in vals]
        except Exception:
            pass
    return [s]


def textvqa_vqa_acc(pred: str, gts: List[str]) -> float:
    """Standard TextVQA VQA accuracy: min(count(gt == pred)/3, 1)."""
    # Strip trailing articles/punct lightly matching VLMEvalKit TextVQA normalization
    pn = normalize_str(pred)
    matches = sum(1 for g in gts if normalize_str(g) == pn)
    return min(matches / 3.0, 1.0)


def compute_chance_oracle(Y: np.ndarray, meta: pd.DataFrame, models: List[str], n_perm: int = 30, seed: int = 42):
    rng = np.random.RandomState(seed)
    N, K = Y.shape
    m72_idx = models.index("Qwen2.5-VL-72B-Instruct")
    best_single_acc = float(Y.mean(axis=0).max()) * 100.0
    qwen72_acc = float(Y[:, m72_idx].mean()) * 100.0
    real_oracle = float((Y.max(axis=1) > 0).mean()) * 100.0

    # 1. Global marginal-matched permutation
    glob_oracles = []
    # 2. Within-dataset stratified permutation (preserving each arm's dataset-specific accuracy!)
    ds_groups = [np.where((meta["dataset"] == ds).values)[0] for ds in meta["dataset"].unique()]
    strat_oracles = []

    for _ in range(n_perm):
        Y_glob = np.empty_like(Y)
        for j in range(K):
            Y_glob[:, j] = rng.permutation(Y[:, j])
        glob_oracles.append(float((Y_glob.max(axis=1) > 0).mean()) * 100.0)

        Y_strat = Y.copy()
        for idxs in ds_groups:
            for j in range(K):
                Y_strat[idxs, j] = Y_strat[rng.permutation(idxs), j]
        strat_oracles.append(float((Y_strat.max(axis=1) > 0).mean()) * 100.0)

    return {
        "N": int(N),
        "best_single_acc": round(best_single_acc, 2),
        "qwen72_acc": round(qwen72_acc, 2),
        "real_oracle": round(real_oracle, 2),
        "raw_gap_vs_best": round(real_oracle - best_single_acc, 2),
        "chance_oracle_global": round(float(np.mean(glob_oracles)), 2),
        "chance_oracle_global_std": round(float(np.std(glob_oracles)), 2),
        "chance_oracle_stratified": round(float(np.mean(strat_oracles)), 2),
        "chance_oracle_stratified_std": round(float(np.std(strat_oracles)), 2),
        "excess_gap_over_strat_chance": round(real_oracle - float(np.mean(strat_oracles)), 2),
    }


def compute_rescue_breakdown(Y: np.ndarray, meta: pd.DataFrame, models: List[str], label: str) -> List[Dict]:
    idx_72b = models.index("Qwen2.5-VL-72B-Instruct")
    dense_idxs = [models.index(m) for m in DENSE_COMPACT_MODELS]
    active_idxs = [models.index(m) for m in LOW_ACTIVE_MOE_AND_DENSE]

    rows = []
    masks = {
        "All 14 Benchmarks": np.ones(len(meta), dtype=bool),
        "Non-MCQ (Open-Ended / OCR / Math)": meta["dataset"].isin(NON_MCQ_DATASETS).values,
        "MCQ Benchmarks Only": meta["dataset"].isin(MCQ_DATASETS).values,
    }
    for subset_name, m in masks.items():
        Y_sub = Y[m]
        fails_72 = (Y_sub[:, idx_72b] == 0)
        n_fails = int(fails_72.sum())
        res_dense = int((Y_sub[fails_72][:, dense_idxs].max(axis=1) > 0).sum()) if n_fails > 0 else 0
        res_active = int((Y_sub[fails_72][:, active_idxs].max(axis=1) > 0).sum()) if n_fails > 0 else 0
        res_any = int((Y_sub[fails_72].max(axis=1) > 0).sum()) if n_fails > 0 else 0
        rows.append({
            "Matrix Version": label,
            "Subset": subset_name,
            "Queries": int(m.sum()),
            "72B Acc (%)": round(float(Y_sub[:, idx_72b].mean()) * 100.0, 2),
            "72B Failures": n_fails,
            "Dense <=8B Rescued": res_dense,
            "Dense <=8B Rescue (%)": round(100.0 * res_dense / max(1, n_fails), 1),
            "Dense<=8B + 3B-MoE Rescued": res_active,
            "Dense<=8B + 3B-MoE Rescue (%)": round(100.0 * res_active / max(1, n_fails), 1),
            "Any Other Model Rescue (%)": round(100.0 * res_any / max(1, n_fails), 1),
        })
    return rows


def main():
    print("=" * 80)
    print("CLAIM 1 & 2 AUDIT: Chance-Corrected Oracle, Non-MCQ Rescue, MiMo Parser Repair")
    print("=" * 80)

    Y_raw = np.load(DATA_DIR / "matrices" / "Y.npz")["Y"].astype(np.int8)
    meta = pd.read_parquet(DATA_DIR / "registry" / "meta.parquet")
    with open(DATA_DIR / "registry" / "sample_index.pkl", "rb") as f:
        sample_ids: List[str] = pickle.load(f)
    with open(DATA_DIR / "registry" / "model_index.pkl", "rb") as f:
        models: List[str] = pickle.load(f)
    sid_to_row = {sid: idx for idx, sid in enumerate(sample_ids)}

    # 1. Chance-corrected oracle on Full (30,540), Test (6,108), and Non-MCQ (21,708)
    test_mask = (meta["split"] == "test").values
    nonmcq_mask = meta["dataset"].isin(NON_MCQ_DATASETS).values
    in_mask = meta["dataset"].isin(["MMMU_DEV_VAL", "MMBench_DEV_EN_V11", "RealWorldQA", "MMStar",
                                     "MathVista_MINI", "MathVision_MINI", "MathVerse_MINI", "AI2D_TEST"]).values

    oracle_controls = []
    for sname, m in [
        ("Full VL-RouterBench (Raw Y)", np.ones(len(meta), dtype=bool)),
        ("Held-Out Test Split (Raw Y)", test_mask),
        ("Non-MCQ Benchmarks Only (Raw Y)", nonmcq_mask),
        ("In-Dist Test (General+STEM, Raw Y)", test_mask & in_mask),
    ]:
        res = compute_chance_oracle(Y_raw[m], meta[m].reset_index(drop=True), models, n_perm=30, seed=42)
        res["Split / Subset"] = sname
        oracle_controls.append(res)

    # 2. Repair MiMo-VL-7B-RL on the 4 OCR benchmarks (ChartQA, DocVQA, InfoVQA, TextVQA)
    #    Compute BOTH:
    #    (a) Exact (int(sc) == 1) matching upstream VL-RouterBench's int(is_correct) convention (apples-to-apples with Y_raw)
    #    (b) Standard (>=0.5 ANLS / >=0.5 VQA) across ALL 17 models cleanly from VLMEvalKit_evaluation/*.xlsx (fixing SmolVLM2/Janus/etc.)
    Y_mimo_exact = Y_raw.copy()
    Y_std_all17 = Y_raw.copy()
    mimo_col = models.index("MiMo-VL-7B-RL")

    # Apply standard >= 0.5 binarization consistently across all 17 models on DocVQA/InfoVQA/TextVQA first
    for ds in ["DocVQA_VAL", "InfoVQA_VAL", "TextVQA_VAL"]:
        ds_mask = (meta["dataset"] == ds).values
        for j, mod in enumerate(models):
            ev = EVAL_DIR / mod / f"{mod}_{ds}_results.xlsx"
            if not ev.exists():
                continue
            df_ev = pd.read_excel(ev)
            for _, row in df_ev.iterrows():
                sid = f"{ds}/{row['index']}"
                r_idx = sid_to_row.get(sid)
                if r_idx is not None:
                    sc_val = float(row.get("is_correct", 0.0))
                    Y_std_all17[r_idx, j] = int(sc_val >= 0.5)

    repair_rows = []
    for ds in ["ChartQA_TEST", "DocVQA_VAL", "InfoVQA_VAL", "TextVQA_VAL"]:
        ev_file = EVAL_DIR / "MiMo-VL-7B-RL" / f"MiMo-VL-7B-RL_{ds}_results.xlsx"
        df = pd.read_excel(ev_file)
        ds_mask = (meta["dataset"] == ds).values
        raw_acc = float(Y_raw[ds_mask, mimo_col].mean()) * 100.0

        unclosed_count = 0
        repaired_scores_continuous = []
        matched = 0
        for _, row in df.iterrows():
            sid = f"{ds}/{row['index']}"
            r_idx = sid_to_row.get(sid)
            if r_idx is None:
                continue
            matched += 1
            raw_pred = str(row.get("prediction", ""))
            if "<think>" in raw_pred and "</think>" not in raw_pred:
                unclosed_count += 1
            cleaned = strip_think_and_extract(raw_pred)
            gts = parse_gt_list(row.get("answer", ""))

            if ds == "ChartQA_TEST":
                sc = float(chartqa_relaxed_correct(cleaned, gts[0] if gts else ""))
            elif ds in ("DocVQA_VAL", "InfoVQA_VAL"):
                sc = float(levenshtein_anls(cleaned, gts, threshold=0.5))
            else:  # TextVQA_VAL
                sc = float(textvqa_vqa_acc(cleaned, gts))

            Y_mimo_exact[r_idx, mimo_col] = int(sc >= 1.0 - 1e-9)
            Y_std_all17[r_idx, mimo_col] = int(sc >= 0.5)
            repaired_scores_continuous.append(sc)

        rep_exact_acc = float(Y_mimo_exact[ds_mask, mimo_col].mean()) * 100.0
        rep_bin_acc = float(Y_std_all17[ds_mask, mimo_col].mean()) * 100.0
        rep_cont_acc = float(np.mean(repaired_scores_continuous)) * 100.0 if repaired_scores_continuous else 0.0
        repair_rows.append({
            "Model": "MiMo-VL-7B-RL",
            "Dataset": ds,
            "N": int(ds_mask.sum()),
            "Raw Y.npz Acc (%)": round(raw_acc, 2),
            "Upstream Failure Cause": "Unstripped <think> preamble",
            "Unclosed <think> (%)": round(100.0 * unclosed_count / max(1, matched), 1),
            "Repaired Exact (==1.0, %)": round(rep_exact_acc, 2),
            "Repaired Binary (>=0.5, %)": round(rep_bin_acc, 2),
            "Repaired Continuous ANLS/VQA (%)": round(rep_cont_acc, 2),
        })

    # Document SmolVLM2 & Janus-Pro InfoVQA/DocVQA int(ANLS) truncation transparently
    for mod_check, ds_check in [("SmolVLM2", "InfoVQA_VAL"), ("Janus-Pro-1B", "DocVQA_VAL"), ("Janus-Pro-7B", "DocVQA_VAL")]:
        col_c = models.index(mod_check)
        m_c = (meta["dataset"] == ds_check).values
        df_c = pd.read_excel(EVAL_DIR / mod_check / f"{mod_check}_{ds_check}_results.xlsx")
        repair_rows.append({
            "Model": mod_check,
            "Dataset": ds_check,
            "N": int(m_c.sum()),
            "Raw Y.npz Acc (%)": round(float(Y_raw[m_c, col_c].mean()) * 100.0, 2),
            "Upstream Failure Cause": "Upstream int(ANLS < 1.0) truncation in Y.npz",
            "Unclosed <think> (%)": 0.0,
            "Repaired Exact (==1.0, %)": round(float(Y_raw[m_c, col_c].mean()) * 100.0, 2),
            "Repaired Binary (>=0.5, %)": round(float(Y_std_all17[m_c, col_c].mean()) * 100.0, 2),
            "Repaired Continuous ANLS/VQA (%)": round(float(df_c["is_correct"].astype(float).mean()) * 100.0, 2),
        })

    # Save repaired matrices separately (keep raw Y.npz untouched as official upstream baseline!)
    np.savez_compressed(DATA_DIR / "matrices" / "Y_repaired.npz", Y=Y_mimo_exact)
    np.savez_compressed(DATA_DIR / "matrices" / "Y_repaired_std05.npz", Y=Y_std_all17)

    # Also compute Chance-Corrected Oracle on Repaired Exact Y & Standard >=0.5 Y
    res_rep_exact = compute_chance_oracle(Y_mimo_exact, meta, models, n_perm=30, seed=42)
    res_rep_exact["Split / Subset"] = "Full VL-RouterBench (MiMo Repaired Exact)"
    oracle_controls.append(res_rep_exact)

    res_rep_std = compute_chance_oracle(Y_std_all17, meta, models, n_perm=30, seed=42)
    res_rep_std["Split / Subset"] = "Full VL-RouterBench (All-17 Std >=0.5)"
    oracle_controls.append(res_rep_std)

    df_ora = pd.DataFrame(oracle_controls)[[
        "Split / Subset", "N", "qwen72_acc", "best_single_acc", "real_oracle",
        "raw_gap_vs_best", "chance_oracle_global", "chance_oracle_stratified", "excess_gap_over_strat_chance"
    ]]
    df_ora.to_csv(TABLES_DIR / "exp1_chance_corrected_oracle.csv", index=False)

    df_rep = pd.DataFrame(repair_rows)
    df_rep.to_csv(TABLES_DIR / "exp1_mimo_parser_repair.csv", index=False)

    rescue_rows = compute_rescue_breakdown(Y_raw, meta, models, "Raw Upstream Y") + \
                  compute_rescue_breakdown(Y_mimo_exact, meta, models, "MiMo Repaired (Exact)") + \
                  compute_rescue_breakdown(Y_std_all17, meta, models, "All-17 Repaired (Std >=0.5)")
    df_rescue = pd.DataFrame(rescue_rows)
    df_rescue.to_csv(TABLES_DIR / "exp1_rescue_nonmcq_and_dense_compact.csv", index=False)

    print("\n=== Chance-Corrected Oracle Control ===")
    print(df_ora.to_string(index=False))
    print("\n=== Upstream Parser Failure Diagnosis & Repair ===")
    print(df_rep.to_string(index=False))
    print("\n=== Compact Rescue Rate: Dense (<=8B) vs +MoE-3B-Active across MCQ & Non-MCQ ===")
    print(df_rescue.to_string(index=False))
    print(f"\nSaved global MiMo raw={100*Y_raw[:, mimo_col].mean():.2f}% -> exact={100*Y_mimo_exact[:, mimo_col].mean():.2f}% -> std>=0.5={100*Y_std_all17[:, mimo_col].mean():.2f}%")


if __name__ == "__main__":
    main()
