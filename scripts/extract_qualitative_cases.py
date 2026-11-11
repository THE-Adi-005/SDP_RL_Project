#!/usr/bin/env python3
"""Find crystal-clear real qualitative case studies across all 30,540 samples and save their real images."""
import base64
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "vl_routerbench_processed"
TSV_DIR = ROOT / "unwanted" / "data_legacy" / "vl_routerbench" / "vlm_router_data" / "TSV_images"
INF_DIR = ROOT / "unwanted" / "data_legacy" / "vl_routerbench" / "vlm_router_data" / "VLMEvalKit_inference"
OUT_DIR = ROOT / "reports" / "tables"
IMG_OUT = ROOT / "reports" / "figures" / "qualitative_raw"


def main():
    Y = np.load(DATA_DIR / "matrices" / "Y.npz")["Y"]
    T = np.load(DATA_DIR / "matrices" / "T_tokens.npz")["output_tokens"]
    meta = pd.read_parquet(DATA_DIR / "registry" / "meta.parquet")
    with open(DATA_DIR / "registry" / "sample_index.pkl", "rb") as f:
        sample_ids = pickle.load(f)
    with open(DATA_DIR / "registry" / "model_index.pkl", "rb") as f:
        models = pickle.load(f)

    sid_to_row = {s: i for i, s in enumerate(sample_ids)}
    i_tiny = models.index("deepseek_vl2_tiny")
    i_qian = models.index("Qianfan-VL-8B")
    i_kimi = models.index("Kimi-VL-A3B-Thinking-2506")
    i_mimo = models.index("MiMo-VL-7B-RL")
    i_72b = models.index("Qwen2.5-VL-72B-Instruct")

    # 1. Case A candidates: ChartQA / OCRBench (where MiMo IS evaluated on OCRBench, Kimi evaluated on both)
    # Specifically prefer OCRBench/ChartQA where Qianfan-8B==1 & Tiny==1 & Kimi==0 (tok>=20) & 72B==0
    tax_rows = []
    for ds in ["ChartQA_TEST", "OCRBench"]:
        idx = np.where((meta["dataset"] == ds).values)[0]
        for r in idx:
            if ds == "OCRBench":
                if Y[r, i_qian] == 1 and Y[r, i_kimi] == 0 and Y[r, i_mimo] == 0 and T[r, i_mimo] >= 60 and T[r, i_qian] <= 10:
                    tax_rows.append((int(T[r, i_mimo] + T[r, i_kimi]), r, ds))
            else:
                if Y[r, i_tiny] == 1 and Y[r, i_qian] == 1 and Y[r, i_kimi] == 0 and Y[r, i_72b] == 0 and T[r, i_kimi] >= 40:
                    tax_rows.append((int(T[r, i_kimi]), r, ds))
    tax_rows.sort(reverse=True)

    # 2. Case B candidates: MathVista_MINI where Kimi==1 & MiMo==1 & 72B==1 (or ==0) & Qianfan==0 & Tiny==0 with nontrivial question
    gain_rows = []
    idx_mv = np.where((meta["dataset"] == "MathVista_MINI").values)[0]
    for r in idx_mv:
        if Y[r, i_kimi] == 1 and Y[r, i_mimo] == 1 and Y[r, i_qian] == 0 and Y[r, i_tiny] == 0 and Y[r, i_72b] == 0:
            if 40 <= T[r, i_kimi] <= 300:
                gain_rows.append((int(T[r, i_kimi]), r, "MathVista_MINI"))
    gain_rows.sort(reverse=True)

    # Load prompts/assets
    sid_meta = {}
    for sf in sorted((DATA_DIR / "BENCHMARKS").rglob("*_samples.jsonl")):
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                sid_meta[rec["sample_id"]] = rec

    selected = []
    for label, cands in [("tax_inversion", tax_rows[:8]), ("reasoning_gain", gain_rows[:8])]:
        for score, r, ds in cands:
            sid = sample_ids[r]
            rec = sid_meta[sid]
            selected.append((label, r, ds, sid, rec))

    # Fetch inference predictions for these selected rows
    by_ds = {}
    for label, r, ds, sid, rec in selected:
        by_ds.setdefault(ds, []).append((label, r, sid, rec))

    results = []
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    for ds, items in by_ds.items():
        tsv_df = pd.read_csv(TSV_DIR / f"{ds}.tsv", sep="\t")
        # Load inference xlsx for key models
        inf_dfs = {}
        for m in ["deepseek_vl2_tiny", "Qianfan-VL-8B", "Kimi-VL-A3B-Thinking-2506", "MiMo-VL-7B-RL", "Qwen2.5-VL-72B-Instruct"]:
            p = next((INF_DIR / m).glob(f"*{ds}.xlsx"))
            df_m = pd.read_excel(p, engine="openpyxl")
            id_col = "id" if "id" in df_m.columns else "index"
            df_m["_sid"] = ds + "/" + df_m[id_col].astype(str)
            inf_dfs[m] = df_m.set_index("_sid")["prediction"].to_dict()

        for idx_item, (label, r, sid, rec) in enumerate(items):
            pos = int(rec["assets"][0]["index"])
            b64 = str(tsv_df["image"].iloc[pos])
            if len(b64) <= 64 and "index" in tsv_df.columns:
                ref_map = dict(zip(tsv_df["index"].astype(str), tsv_df["image"].astype(str)))
                b64 = ref_map.get(str(tsv_df["index"].iloc[pos]), b64)
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            img_file = IMG_OUT / f"{label}_{ds}_{sid.split('/')[1]}.png"
            img.save(img_file)

            q_text = str(tsv_df["question"].iloc[pos]) if "question" in tsv_df.columns else rec["prompt"]
            gt_ans = str(tsv_df["answer"].iloc[pos]) if "answer" in tsv_df.columns else ""
            preds = {m: str(inf_dfs[m].get(sid, ""))[:350] for m in inf_dfs}
            res_entry = {
                "case_type": label,
                "sample_id": sid,
                "dataset": ds,
                "image_path": str(img_file),
                "image_size": list(img.size),
                "question": q_text,
                "gt_answer": gt_ans,
                "y": {
                    "tiny_1.5b": int(Y[r, i_tiny]),
                    "qianfan_8b": int(Y[r, i_qian]),
                    "kimi_thinking": int(Y[r, i_kimi]),
                    "mimo_7b_rl": int(Y[r, i_mimo]),
                    "qwen_72b": int(Y[r, i_72b]),
                },
                "out_tokens": {
                    "tiny_1.5b": int(T[r, i_tiny]),
                    "qianfan_8b": int(T[r, i_qian]),
                    "kimi_thinking": int(T[r, i_kimi]),
                    "mimo_7b_rl": int(T[r, i_mimo]),
                    "qwen_72b": int(T[r, i_72b]),
                },
                "preds": preds,
            }
            results.append(res_entry)

    out_json = OUT_DIR / "real_selected_qualitative_cases.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} qualitative cases with decoded PNG images to {out_json}")
    for res in results:
        if res["case_type"] == "tax_inversion":
            print("[TAX ]", res["sample_id"], "| Q:", res["question"][:70], "| GT:", res["gt_answer"], "| Y:", res["y"], "| Tok:", res["out_tokens"])
        else:
            print("[GAIN]", res["sample_id"], "| Q:", res["question"][:70], "| GT:", res["gt_answer"], "| Y:", res["y"], "| Tok:", res["out_tokens"])


if __name__ == "__main__":
    main()

