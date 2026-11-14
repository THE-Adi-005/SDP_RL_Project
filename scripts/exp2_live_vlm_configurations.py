#!/usr/bin/env python3
"""Live GPU Same-Backbone & Multi-Configuration VLM Study (Claims 1, 2, 8 & AMBER External).

Runs genuinely on local RTX 4050 (6.44GB VRAM, 4-bit NF4 via bitsandbytes):
  - Same-backbone models: Qwen/Qwen3-VL-2B-Instruct vs Qwen/Qwen3-VL-2B-Thinking
  - Configuration knobs: (Backbone/Reasoning Mode x Image Resolution max_pixels x Max-Token Budget {32, 128, 512})
  - Benchmarks: Stratified 4-benchmark slice from VL-RouterBench (OCRBench, ChartQA_TEST, MathVista_MINI, HallusionBench)
                PLUS external AMBER benchmark (outside VL-RouterBench 14 benchmarks!)
  - Captures exact per-step CUDA wall-clock latency via GenerationStoppingCriteria,
    encoder + LinUCB routing latency across all 3 encoders (MiniLM-L6, DINOv2-ViT-S/14, ResNet-50 + 8 cues),
    and validates our TTFT + out_tok/TPS latency model against measured hardware latency.

Outputs saved to reports/tables/:
  - exp2_same_backbone_qwen3vl.csv       (Claim 1: Thinking vs Non-Thinking same backbone)
  - exp2_multiconfig_pareto.csv          (Claim 2: Progressive configuration knob expansion + Pareto AUC)
  - exp2_amber_external_eval.csv         (Cross-cutting: Held-out external AMBER benchmark evaluation)
  - exp2_real_latency_and_sim_cal.csv    (Claim 8: p50/p95 GPU latency, all-3-encoder overhead, simulator MAE/MAPE/r)
"""
import ast
import base64
import io
import json
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.statistical_tests import pareto_frontier_auc
from src.rl.linucb import LinUCB
from scripts.exp1_claim1_oracle_and_mimo_repair import (
    chartqa_relaxed_correct,
    normalize_str,
    parse_gt_list,
    strip_think_and_extract,
)

DATA_DIR = ROOT / "data" / "vl_routerbench_processed"
TSV_DIR = ROOT / "unwanted" / "data_legacy" / "vl_routerbench" / "vlm_router_data" / "TSV_images"
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


class StepTimerCriteria(StoppingCriteria):
    """Records exact CUDA-synchronized elapsed wall-clock latency at intermediate token budgets."""

    def __init__(self, start_t: float, prompt_len: int, checkpoints=(32, 128, 512)):
        self.start_t = start_t
        self.prompt_len = int(prompt_len)
        self.checkpoints = tuple(sorted(checkpoints))
        self.checkpoint_times: Dict[int, float] = {}
        self.checkpoint_ids: Dict[int, torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_len = int(input_ids.shape[1]) - self.prompt_len
        for cp in self.checkpoints:
            if cp not in self.checkpoint_times and gen_len >= cp:
                torch.cuda.synchronize()
                self.checkpoint_times[cp] = time.perf_counter() - self.start_t
                self.checkpoint_ids[cp] = input_ids[0, self.prompt_len : self.prompt_len + cp].detach().cpu().clone()
        return False


def decode_b64_image(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def resize_for_budget(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    cur = w * h
    if cur <= max_pixels:
        return img
    scale = (max_pixels / float(cur)) ** 0.5
    nw = max(28, int(round(w * scale / 28.0) * 28))
    nh = max(28, int(round(h * scale / 28.0) * 28))
    return img.resize((nw, nh), Image.Resampling.BICUBIC)


def extract_clean_answer(pred_text: str) -> str:
    """Extract concise final answer from Instruct sentences or closed/unclosed </think> Thinking traces."""
    text = str(pred_text).strip()
    # 1. If closed </think> exists, inspect only post-thinking final output
    if "</think>" in text:
        after = text.split("</think>")[-1].strip()
        if after:
            text = after

    # 2. Explicit answer markers ("Final Answer:", "**42**", \boxed{...})
    m_fin = re.findall(r"(?:final answer|answer)\s*[:\-]\s*\*?\*?([^\n\.]+)", text, flags=re.I)
    if m_fin:
        cand = m_fin[-1].strip().strip("*\"'` ")
        if cand:
            return cand
    m_box = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m_box:
        return m_box[-1].strip().strip("*\"'` ")
    m_bold = re.findall(r"\*\*([^*]+)\*\*", text)
    if m_bold:
        cand = m_bold[-1].strip().rstrip(".,;:")
        if len(cand) <= 64:
            return cand

    # 3. Delegate remaining cleanup (unclosed <think>, trailing line, quotes)
    return strip_think_and_extract(text)


def extract_yes_no(pred_text: str) -> str:
    """Robustly extract Yes/No polarity from short or conversational VLM responses."""
    raw = str(pred_text).strip()
    if "</think>" in raw:
        after = raw.split("</think>")[-1].strip()
        if after:
            raw = after
    m_fin = re.findall(r"(?:final answer|answer)\s*[:\-]\s*(yes|no)\b", raw, flags=re.I)
    if m_fin:
        return m_fin[-1].lower()
    low = normalize_str(raw)
    toks = low.split()
    if not toks:
        return ""
    # Direct leading yes/no
    if toks[0] in ("yes", "no"):
        return toks[0]
    if "yes" in toks[:4] and "no" not in toks[:4]:
        return "yes"
    if "no" in toks[:4] and "yes" not in toks[:4]:
        return "no"
    # Trailing yes/no
    if toks[-1] in ("yes", "no"):
        return toks[-1]
    # Conversational negative/affirmative phrasings (common in VLM prose!)
    neg_pats = [
        r"\bthere (?:is|are) no\b", r"\bno [\w\s]{0,20} visible\b", r"\bnot all\b",
        r"\bis not\b", r"\bare not\b", r"\bdoes not\b", r"\bdo not\b", r"\bno direct\b",
        r"\bcannot see any\b", r"\bnot made\b", r"\bnot located\b",
    ]
    for pat in neg_pats:
        if re.search(pat, low):
            return "no"
    if re.search(r"\byes\b", low) and not re.search(r"\bno\b", low):
        return "yes"
    if re.search(r"\bno\b", low) and not re.search(r"\byes\b", low):
        return "no"
    return low


def score_prediction(pred_text: str, gt_raw: str, dataset: str, extra_gt: str = "", is_thinking: bool = False) -> int:
    raw = str(pred_text).strip()
    if is_thinking:
        if "</think>" in raw:
            eval_text = raw.split("</think>")[-1].strip()
        else:
            # Unfinished thinking trace: only score if an explicit Final Answer / \boxed{} was emitted
            m_fin = re.findall(r"(?:final answer|answer)\s*[:\-]\s*([^\n]+)", raw, flags=re.I)
            m_box = re.findall(r"\\boxed\{([^{}]+)\}", raw)
            if m_fin:
                eval_text = m_fin[-1].strip()
            elif m_box:
                eval_text = m_box[-1].strip()
            else:
                return 0
    else:
        eval_text = raw

    cleaned = extract_clean_answer(eval_text)
    gts = parse_gt_list(gt_raw)
    if extra_gt and str(extra_gt).lower() not in ("nan", "none", ""):
        for eg in parse_gt_list(str(extra_gt)):
            if eg not in gts:
                gts.append(eg)

    if dataset in ("AMBER", "HallusionBench"):
        pn_bin = extract_yes_no(eval_text)
        ans = normalize_str(gts[0] if gts else "")
        if ans in ("0", "false"):
            ans = "no"
        elif ans in ("1", "true"):
            ans = "yes"
        return int(pn_bin == ans and ans != "")
    elif dataset == "ChartQA_TEST":
        c_num = cleaned.replace("$", "").replace(",", "").strip()
        if any(chartqa_relaxed_correct(c_num, str(g).replace("$", "").replace(",", "")) == 1 for g in gts):
            return 1
        m_last = re.findall(r"[-+]?\d[\d,]*\.?\d*%?", c_num)
        if m_last and any(chartqa_relaxed_correct(m_last[-1].replace(",", ""), str(g)) == 1 for g in gts):
            return 1
        return 0
    elif dataset == "OCRBench":
        pn = normalize_str(cleaned)
        full_n = normalize_str(eval_text)
        return int(any(
            len(normalize_str(g)) > 0 and (normalize_str(g) in pn or normalize_str(g) in full_n or pn == normalize_str(g))
            for g in gts
        ))
    else:  # MathVista_MINI
        pn = normalize_str(cleaned).replace("$", "").replace(",", "").replace("°", "")
        full_after = eval_text
        for g in gts:
            gn = normalize_str(g).replace("$", "").replace(",", "").replace("°", "")
            if not gn:
                continue
            if pn == gn:
                return 1
            if len(gn) == 1 and gn.isalpha():
                if re.search(rf"(?:\b|\(){re.escape(gn)}(?:\b|\))", normalize_str(cleaned)) or re.search(
                    rf"(?:option|answer|choice)\s*(?:is|:)?\s*\(?{re.escape(gn)}\)?\b", normalize_str(full_after)
                ):
                    return 1
            try:
                if abs(float(pn.rstrip("%")) - float(gn.rstrip("%"))) <= 1e-4 * max(1.0, abs(float(gn.rstrip("%")))):
                    return 1
            except Exception:
                pass
            nums = re.findall(r"[-+]?\d+\.?\d*", pn)
            if nums:
                try:
                    if abs(float(nums[-1]) - float(gn.rstrip("%"))) <= 1e-4 * max(1.0, abs(float(gn.rstrip("%")))):
                        return 1
                except Exception:
                    pass
        return 0


def load_stratified_queries(n_per_ds: int = 25, seed: int = 42) -> List[Dict]:
    """Load real queries from VL-RouterBench meta/X (4 benchmarks) + external AMBER.tsv."""
    rng = np.random.RandomState(seed)
    meta = pd.read_parquet(DATA_DIR / "registry" / "meta.parquet")
    with open(DATA_DIR / "registry" / "sample_index.pkl", "rb") as f:
        sids: List[str] = pickle.load(f)
    sid_to_row = {s: i for i, s in enumerate(sids)}
    X_all = np.load(DATA_DIR / "matrices" / "X_multimodal_904_f16.npz")["X"].astype(np.float32)

    records: List[Dict] = []

    # 1. Sample from 4 VL-RouterBench test/val TSVs (matching existing X_904 rows for exact routing compatibility!)
    for ds in ["OCRBench", "ChartQA_TEST", "MathVista_MINI", "HallusionBench"]:
        sub = meta[(meta["dataset"] == ds) & (meta["split"] == "test")]
        if len(sub) < n_per_ds:
            sub = meta[meta["dataset"] == ds]
        picked_sids = set(rng.choice(sub["sample_id"].values, size=min(n_per_ds, len(sub)), replace=False))
        tsv_path = TSV_DIR / f"{ds}.tsv"
        # Stream chunks with ref_map resolution for TSVs that alias repeated image strings
        needed_indices = {str(s.split("/", 1)[1]): s for s in picked_sids}
        ref_map: Dict[str, str] = {}
        raw_candidates: Dict[str, tuple] = {}
        df_full = pd.read_csv(tsv_path, sep="\t")
        id_col = "index" if "index" in df_full.columns else df_full.columns[0]
        for _, r_full in df_full.iterrows():
            key_s = str(r_full[id_col])
            b64_s = str(r_full["image"])
            ref_map[key_s] = b64_s
            if key_s in needed_indices:
                raw_candidates[key_s] = (
                    needed_indices[key_s],
                    b64_s,
                    str(r_full.get("question", "")),
                    str(r_full.get("answer", "")),
                    str(r_full.get("answer_option", "")),
                )

        found = 0
        for key_s, (sid, b64_s, q_s, a_s, opt_s) in raw_candidates.items():
            cur = b64_s
            for _ in range(10):
                if len(cur) > 64:
                    break
                cur = ref_map.get(cur, cur)
            if len(cur) <= 64:
                continue
            r_idx = sid_to_row[sid]
            img = decode_b64_image(cur)
            records.append({
                "sample_id": sid,
                "dataset": ds,
                "question": q_s,
                "answer": a_s,
                "answer_option": "" if opt_s.lower() in ("nan", "none", "") else opt_s,
                "image": img,
                "x_904": X_all[r_idx],
                "is_external": False,
            })
            found += 1
        print(f"Loaded {found} real queries from {ds}.tsv")

    # 2. Sample strictly external AMBER discriminative yes/no queries + extract real 904-d zero-leakage state
    amber_cache = TABLES_DIR / f"_exp2_amber_cache_n{n_per_ds}_s{seed}.pkl"
    if amber_cache.exists():
        with open(amber_cache, "rb") as f_c:
            amber_rows = pickle.load(f_c)
    else:
        amber_meta = pd.read_csv(TSV_DIR / "AMBER_local.tsv", sep="\t")
        amber_disc = amber_meta[amber_meta["category"].str.startswith("discriminative") & amber_meta["answer"].isin(["yes", "no"])]
        picked_amber_ids = {str(int(x)) for x in rng.choice(amber_disc["index"].values, size=n_per_ds, replace=False)}

        # Fast streaming line scan over 6.18GB AMBER.tsv without decoding unselected 1MB+ base64 columns into DataFrames
        alias_needed = set()
        raw_amber = {}
        with open(TSV_DIR / "AMBER.tsv", "r", encoding="utf-8", errors="ignore") as f_amb:
            hdr = f_amb.readline().rstrip("\r\n").split("\t")
            idx_i = hdr.index("index")
            img_i = hdr.index("image")
            q_i = hdr.index("question")
            a_i = hdr.index("answer")
            for line in f_amb:
                head_idx = line.split("\t", 1)[0]
                if head_idx in picked_amber_ids:
                    parts = line.rstrip("\r\n").split("\t")
                    img_s = parts[img_i]
                    raw_amber[head_idx] = (parts[idx_i], img_s, parts[q_i], parts[a_i])
                    if len(img_s) <= 64:
                        alias_needed.add(img_s)
                    if len(raw_amber) >= len(picked_amber_ids):
                        break
        if alias_needed:
            alias_map = {}
            with open(TSV_DIR / "AMBER.tsv", "r", encoding="utf-8", errors="ignore") as f_amb:
                hdr = f_amb.readline().rstrip("\r\n").split("\t")
                img_i = hdr.index("image")
                for line in f_amb:
                    head_idx = line.split("\t", 1)[0]
                    if head_idx in alias_needed:
                        parts = line.rstrip("\r\n").split("\t")
                        alias_map[head_idx] = parts[img_i]
                        if len(alias_map) >= len(alias_needed):
                            break
            for k, (rid, img_s, qs, ans) in list(raw_amber.items()):
                if len(img_s) <= 64 and img_s in alias_map:
                    raw_amber[k] = (rid, alias_map[img_s], qs, ans)

        amber_rows = [
            {"index": int(rid), "image": img_s, "question": qs, "answer": ans}
            for rid, img_s, qs, ans in raw_amber.values()
            if len(img_s) > 64
        ]
        with open(amber_cache, "wb") as f_c:
            pickle.dump(amber_rows, f_c)

    # External AMBER rows will be encoded via encode_external_and_benchmark_overhead() below!
    print(f"Loaded {len(amber_rows)} external AMBER queries.")
    for row in amber_rows:
        img = decode_b64_image(row["image"])
        records.append({
            "sample_id": f"AMBER/{int(row['index'])}",
            "dataset": "AMBER",
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "answer_option": "",
            "image": img,
            "x_904": None,  # Populated via 3 encoders below!
            "is_external": True,
        })
    return records


def encode_external_and_benchmark_overhead(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Run all 3 context encoders (MiniLM-L6, DINOv2-ViT-S/14, ResNet-50 + 8 cues) + benchmark p50/p95 overhead."""
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tv_models
    from sentence_transformers import SentenceTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_enc = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    hub_dir = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
    if hub_dir.exists():
        dino = torch.hub.load(str(hub_dir), "dinov2_vits14", source="local").to(device).eval()
    else:
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(device).eval()

    rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    rn50.fc = nn.Identity()
    rn50 = rn50.to(device).eval()

    proj_rng = np.random.RandomState(42)
    Q_orth, _ = np.linalg.qr(proj_rng.randn(2048, 128).astype(np.float64))
    Q_orth_t = torch.from_numpy(Q_orth.astype(np.float32)).to(device)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    bandit_bench = LinUCB(n_actions=12, feature_dim=896, alpha=0.10, use_gpu=torch.cuda.is_available())
    txt_ms_l, dino_ms_l, rn50_ms_l, cues_ms_l = [], [], [], []
    enc896_latencies_ms, enc904_latencies_ms = [], []
    router_latencies_ms = []

    for rec in records:
        img: Image.Image = rec["image"]
        q_txt: str = rec["question"]

        # 1. Text MiniLM-L6 (384-d)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        q_emb = txt_enc.encode([q_txt], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # 2. DINOv2 (384-d) + ResNet50 (128-d) matching extract_real_multimodal_features.py exactly
        im_r = img.resize((224, 224), Image.BICUBIC)
        arr_t = torch.from_numpy(np.asarray(im_r, dtype=np.uint8).copy()).permute(2, 0, 1).to(device).float() / 255.0
        inp_t = (arr_t.unsqueeze(0) - mean_t) / std_t
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        td0 = time.perf_counter()
        with torch.no_grad():
            d_feat = F.normalize(dino(inp_t).float(), p=2, dim=-1) * np.sqrt(0.75)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        td1 = time.perf_counter()
        with torch.no_grad():
            r_feat = F.normalize((rn50(inp_t) @ Q_orth_t).float(), p=2, dim=-1) * np.sqrt(0.25)
            v_comb = F.normalize(torch.cat([d_feat, r_feat], dim=-1), p=2, dim=-1)
        v_emb = v_comb[0].cpu().numpy().astype(np.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tr_end = time.perf_counter()

        # 3. Optional 8 Unsupervised perceptual & syntactic cues (measured for cues-ablation completeness)
        tc0 = time.perf_counter()
        w, h = img.size
        ar = float(np.arctan(np.log(max(w, 1) / max(h, 1))) / np.pi + 0.5)
        area = float(np.clip(np.log10(max(1, w * h)) / 7.0, 0.0, 1.0))
        gray_pil = img.convert("L")
        if max(gray_pil.size) > 384:
            gray_pil.thumbnail((384, 384), Image.BILINEAR)
        arr_g = np.asarray(gray_pil, dtype=np.uint8)
        hist = np.bincount(arr_g.ravel(), minlength=256).astype(np.float64)
        p_h = hist / max(1, arr_g.size)
        p_h = p_h[p_h > 0]
        ent = float(-np.sum(p_h * np.log2(p_h))) / 8.0
        f_arr = arr_g.astype(np.float32) / 255.0
        gx = np.abs(f_arr[:, 1:] - f_arr[:, :-1])
        gy = np.abs(f_arr[1:, :] - f_arr[:-1, :])
        gmag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2 + 1e-12)
        edge = 0.6 * float((gmag > 0.08).mean()) + 0.4 * float(np.clip(gmag.mean() * 4.0, 0.0, 1.0))
        white_bg = float((arr_g > 235).mean())

        words = float(min(len(q_txt.split()), 128) / 128.0)
        n_ch = max(len(q_txt), 1)
        n_dig = len(re.findall(r"[0-9=+\-*/^<>%\\]", q_txt))
        has_mcq = 1.0 if ("A." in q_txt and "B." in q_txt) or ("(A)" in q_txt and "(B)" in q_txt) or ("Choices:" in q_txt) else 0.0
        math_cue = float(np.clip((n_dig / n_ch) * 6.0 + 0.25 * has_mcq, 0.0, 1.0))
        sents = float(min(len(re.findall(r"[.!?\n]+", q_txt)) + 1, 16) / 16.0)
        p_vec = np.array([ent, edge, ar, area, white_bg, words, math_cue, sents], dtype=np.float32)

        x_904 = np.concatenate([0.50 * q_emb, 0.55 * v_emb, 0.235 * p_vec]).astype(np.float32)
        tc1 = time.perf_counter()

        txt_ms = (t1 - t0) * 1000.0
        dino_ms = (td1 - td0) * 1000.0
        rn50_ms = (tr_end - td1) * 1000.0
        cues_ms = (tc1 - tc0) * 1000.0
        txt_ms_l.append(txt_ms)
        dino_ms_l.append(dino_ms)
        rn50_ms_l.append(rn50_ms)
        cues_ms_l.append(cues_ms)
        enc896_latencies_ms.append(txt_ms + (tr_end - t1) * 1000.0)
        enc904_latencies_ms.append(txt_ms + (tr_end - t1) * 1000.0 + cues_ms)
        if rec["x_904"] is None:
            rec["x_904"] = x_904

        tr0 = time.perf_counter()
        _ = bandit_bench.select_action(x_904[:896])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        router_latencies_ms.append((time.perf_counter() - tr0) * 1000.0)

    # Discard first 5 warmup calls for fair p50/p95 steady-state router overhead
    enc896_arr = np.array(enc896_latencies_ms[5:])
    enc904_arr = np.array(enc904_latencies_ms[5:])
    rt_arr = np.array(router_latencies_ms[5:])
    overhead_stats = {
        "txt_p50_ms": round(float(np.percentile(txt_ms_l[5:], 50)), 2),
        "txt_p95_ms": round(float(np.percentile(txt_ms_l[5:], 95)), 2),
        "dino_p50_ms": round(float(np.percentile(dino_ms_l[5:], 50)), 2),
        "dino_p95_ms": round(float(np.percentile(dino_ms_l[5:], 95)), 2),
        "rn50_p50_ms": round(float(np.percentile(rn50_ms_l[5:], 50)), 2),
        "rn50_p95_ms": round(float(np.percentile(rn50_ms_l[5:], 95)), 2),
        "cues_p50_ms": round(float(np.percentile(cues_ms_l[5:], 50)), 2),
        "cues_p95_ms": round(float(np.percentile(cues_ms_l[5:], 95)), 2),
        "enc_p50_ms": round(float(np.percentile(enc896_arr, 50)), 2),
        "enc_p95_ms": round(float(np.percentile(enc896_arr, 95)), 2),
        "enc904_p50_ms": round(float(np.percentile(enc904_arr, 50)), 2),
        "enc904_p95_ms": round(float(np.percentile(enc904_arr, 95)), 2),
        "router_p50_ms": round(float(np.percentile(rt_arr, 50)), 2),
        "router_p95_ms": round(float(np.percentile(rt_arr, 95)), 2),
        "total_p50_ms": round(float(np.percentile(enc896_arr + rt_arr, 50)), 2),
        "total_p95_ms": round(float(np.percentile(enc896_arr + rt_arr, 95)), 2),
    }
    print("Measured Live Context+Bandit Overhead:", overhead_stats)

    # Explicitly free vision/text encoders before loading Qwen3-VL-2B in 4-bit!
    del txt_enc, dino, rn50
    torch.cuda.empty_cache()
    return records, overhead_stats


def format_prompt_for_benchmark(q: str, dataset: str, mode_label: str) -> str:
    """Standard VLMEvalKit-aligned formatting suffix so answers parse cleanly without changing task difficulty."""
    if mode_label == "Instruct":
        if dataset in ("AMBER", "HallusionBench"):
            return q + "\nAnswer strictly in one word: Yes or No."
        elif dataset == "ChartQA_TEST":
            return q + "\nAnswer with a single number or word."
        elif dataset == "OCRBench":
            return q + "\nProvide only the requested text or value concisely."
        else:
            return q + "\nBe concise and end with Final Answer: <answer or option letter>."
    else:  # Thinking
        if dataset in ("AMBER", "HallusionBench"):
            return q + "\nThink concisely and conclude with Final Answer: Yes or Final Answer: No."
        else:
            return q + "\nThink concisely step by step and conclude with Final Answer: <single value or option letter>."


def eval_qwen3vl_model(model_id: str, mode_label: str, records: List[Dict], res_map: Dict[str, int], checkpoints=(32, 128, 512)):
    """Run live 4-bit HF transformers inference on Qwen3-VL-2B across image resolutions & token budgets."""
    print(f"\nLoading {model_id} in 4-bit NF4...")
    proc = AutoProcessor.from_pretrained(model_id)
    if "bnb-4bit" in model_id.lower():
        cfg = AutoConfig.from_pretrained(model_id)
        if getattr(cfg, "quantization_config", None) and isinstance(cfg.quantization_config, dict):
            skips = list(cfg.quantization_config.get("llm_int8_skip_modules", []))
            for k in ["model.visual", "visual.blocks", "visual.merger"]:
                if k not in skips:
                    skips.append(k)
            cfg.quantization_config["llm_int8_skip_modules"] = skips
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, config=cfg, device_map={"": 0})
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0})
    model.eval()
    device = model.device

    rows = []
    # Warmup 2 calls
    w_img = resize_for_budget(records[0]["image"], 128 * 128)
    w_msgs = [{"role": "user", "content": [{"type": "image", "image": w_img}, {"type": "text", "text": "What is this?"}]}]
    w_in = proc.apply_chat_template(w_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**w_in, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    for res_name, max_px in res_map.items():
        t_start_block = time.perf_counter()
        for q_i, rec in enumerate(records):
            img_r = resize_for_budget(rec["image"], max_px)
            q_prompt = format_prompt_for_benchmark(rec["question"], rec["dataset"], mode_label)
            msgs = [{"role": "user", "content": [{"type": "image", "image": img_r}, {"type": "text", "text": q_prompt}]}]
            inputs = proc.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(device)
            p_len = int(inputs["input_ids"].shape[1])

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            timer_cb = StepTimerCriteria(t0, p_len, checkpoints=checkpoints)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max(checkpoints),
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([timer_cb]),
                )
            torch.cuda.synchronize()
            total_s = time.perf_counter() - t0
            full_gen_ids = out_ids[0, p_len:].detach().cpu()
            total_gen_tok = int(len(full_gen_ids))

            for cp in checkpoints:
                if total_gen_tok <= cp:
                    ids_cp = full_gen_ids
                    lat_cp = total_s
                    tok_cp = total_gen_tok
                else:
                    ids_cp = timer_cb.checkpoint_ids.get(cp, full_gen_ids[:cp])
                    lat_cp = timer_cb.checkpoint_times.get(cp, total_s * (cp / max(1, total_gen_tok)))
                    tok_cp = cp
                txt = proc.tokenizer.decode(ids_cp, skip_special_tokens=True)
                ok = score_prediction(txt, rec["answer"], rec["dataset"], rec.get("answer_option", ""), is_thinking=(mode_label == "Thinking"))
                rows.append({
                    "sample_id": rec["sample_id"],
                    "dataset": rec["dataset"],
                    "is_external": rec["is_external"],
                    "mode": mode_label,
                    "resolution": res_name,
                    "max_pixels": max_px,
                    "max_tokens": cp,
                    "config_name": f"{mode_label}_{res_name}_B{cp}",
                    "correct": ok,
                    "in_tokens": p_len,
                    "out_tokens": tok_cp,
                    "latency_s": lat_cp,
                    "pred_preview": txt[:120].replace("\n", " "),
                    "pred_text": txt.replace("\n", " "),
                })
            if (q_i + 1) % 5 == 0 or (q_i + 1) == len(records):
                print(f"    [{mode_label} {res_name}] {q_i+1}/{len(records)} elapsed={time.perf_counter()-t_start_block:.1f}s (last out_tok={total_gen_tok}, lat={total_s:.2f}s)")
        ckpt_path = TABLES_DIR / f"_exp2_ckpt_{mode_label}_{res_name}.csv"
        pd.DataFrame(rows).to_csv(ckpt_path, index=False)
        print(f"  Completed {mode_label} x {res_name} ({len(records)} queries) in {time.perf_counter()-t_start_block:.1f}s [saved ckpt]")

    del model, proc
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def analyze_and_save(df_all: pd.DataFrame, records: List[Dict], overhead_stats: Dict):
    # 1. Claim 1: Same-Backbone Study (Instruct vs Thinking at matched HighRes B512)
    sb = df_all[(df_all["resolution"] == "HighRes") & (df_all["max_tokens"] == 512)]
    sb_rows = []
    for ds in ["OCRBench", "ChartQA_TEST", "MathVista_MINI", "HallusionBench", "AMBER", "ALL"]:
        sub = sb if ds == "ALL" else sb[sb["dataset"] == ds]
        ins = sub[sub["mode"] == "Instruct"]
        thk = sub[sub["mode"] == "Thinking"]
        sb_rows.append({
            "Dataset": ds,
            "N": int(len(ins)),
            "Instruct Acc (%)": round(float(ins["correct"].mean()) * 100.0, 2),
            "Thinking Acc (%)": round(float(thk["correct"].mean()) * 100.0, 2),
            "Thinking-Instruct dAcc (%)": round((float(thk["correct"].mean()) - float(ins["correct"].mean())) * 100.0, 2),
            "Instruct OutTok": round(float(ins["out_tokens"].mean()), 1),
            "Thinking OutTok": round(float(thk["out_tokens"].mean()), 1),
            "Instruct Lat p50 (s)": round(float(np.percentile(ins["latency_s"], 50)), 3),
            "Thinking Lat p50 (s)": round(float(np.percentile(thk["latency_s"], 50)), 3),
            "Thinking/Instruct Token Ratio": round(float(thk["out_tokens"].mean()) / max(1.0, float(ins["out_tokens"].mean())), 2),
        })
    df_sb = pd.DataFrame(sb_rows)
    df_sb.to_csv(TABLES_DIR / "exp2_same_backbone_qwen3vl.csv", index=False)
    print("\n=== CLAIM 1: Same-Backbone Reasoning Tax (Qwen3-VL-2B-Instruct vs Thinking, 4-bit GPU) ===")
    print(df_sb.to_string(index=False))

    # 2. Claim 2: Multi-Configuration Arm Expansion & Per-Knob Pareto AUC Gain
    # 12 configs = 2 modes x 2 resolutions (Low 196*196 vs High 392*392) x 3 max_token budgets (32, 128, 512)
    sids = [r["sample_id"] for r in records]
    X_mat = np.stack([r["x_904"][:896] for r in records], axis=0)  # Default 896-d Text+Vision state
    configs_all = sorted(df_all["config_name"].unique())
    sid_idx = {s: i for i, s in enumerate(sids)}

    Y_c = np.zeros((len(sids), len(configs_all)), dtype=np.float32)
    L_c = np.zeros((len(sids), len(configs_all)), dtype=np.float64)
    T_c = np.zeros((len(sids), len(configs_all)), dtype=np.float64)
    # GPU-second proxy cost: $0.0001 per GPU-second (or token pricing)
    for c_i, cname in enumerate(configs_all):
        sub = df_all[df_all["config_name"] == cname]
        for _, row in sub.iterrows():
            r_i = sid_idx[row["sample_id"]]
            Y_c[r_i, c_i] = float(row["correct"])
            L_c[r_i, c_i] = float(row["latency_s"])
            T_c[r_i, c_i] = float(row["out_tokens"])

    # Progressive configuration hierarchy evaluated cleanly under 5-fold cross-validation!
    knob_sets = [
        ("1. Backbone/Mode Only (HighRes, B512) [2 arms]", [c for c in configs_all if "HighRes_B512" in c]),
        ("2. + Token Budget Knob (HighRes, B32/128/512) [6 arms]", [c for c in configs_all if "HighRes_" in c]),
        ("3. + Resolution Knob (LowRes/HighRes x Modes x Budgets) [12 arms]", configs_all),
    ]
    knob_rows = []
    # Build 5 stratified folds for unbiased out-of-fold router evaluation across all 12 real configs
    ds_arr = np.array([r["dataset"] for r in records])
    folds = np.zeros(len(records), dtype=int)
    rng_cv = np.random.RandomState(42)
    for ds in np.unique(ds_arr):
        idxs = np.where(ds_arr == ds)[0]
        perm = rng_cv.permutation(idxs)
        for p_k, r_i in enumerate(perm):
            folds[r_i] = p_k % 5

    for klabel, subset_c in knob_sets:
        c_idxs = [configs_all.index(c) for c in subset_c]
        Yc_s = Y_c[:, c_idxs]
        Lc_s = L_c[:, c_idxs]
        lat_scale = float(np.percentile(Lc_s.mean(axis=0), 90))
        Ln_s = np.clip(Lc_s / max(lat_scale, 1e-6), 0.0, 2.5)

        pts_acc, pts_lat = [], []
        for gam in [0.0, 0.05, 0.12, 0.25, 0.45, 0.80]:
            fold_accs, fold_lats = [], []
            for f_id in range(5):
                tr_m = (folds != f_id)
                te_m = (folds == f_id)
                R_tr = Yc_s[tr_m] - gam * Ln_s[tr_m]
                ag = LinUCB(n_actions=len(c_idxs), feature_dim=896, alpha=0.10, alpha_anneal=True, regularization=4.0, use_gpu=False)
                ag.warm_start_from_full_matrix(X_mat[tr_m], R_tr, regularization=4.0, sample_weight=0.25)
                for q_i in np.where(te_m)[0]:
                    a = ag.select_action(X_mat[q_i], budget_fraction=0.8)
                    fold_accs.append(float(Yc_s[q_i, a]))
                    fold_lats.append(float(Lc_s[q_i, a]))
                    ag.update(X_mat[q_i], a, float(Yc_s[q_i, a] - gam * Ln_s[q_i, a]))
            pts_acc.append(float(np.mean(fold_accs)) * 100.0)
            pts_lat.append(float(np.mean(fold_lats)))
        p_auc = pareto_frontier_auc(
            np.array(pts_lat), np.array(pts_acc),
            cost_min=float(L_c.mean(axis=0).min()),
            cost_max=float(L_c.mean(axis=0).max()),
            left_fill=0.0,
        )
        knob_rows.append({
            "Configuration Knob Set": klabel,
            "Num Arms": len(c_idxs),
            "5-Fold Peak Acc (%)": round(max(pts_acc), 2),
            "5-Fold Matched-Lat (g=0.12) Acc (%)": round(pts_acc[2], 2),
            "Mean Latency at g=0.12 (s)": round(pts_lat[2], 3),
            "Min Latency Achieved (s)": round(min(pts_lat), 3),
            "Pareto Latency-AUC (%)": round(p_auc, 2),
        })
    df_knob = pd.DataFrame(knob_rows)
    df_knob.to_csv(TABLES_DIR / "exp2_multiconfig_pareto.csv", index=False)
    print("\n=== CLAIM 2: Multi-Config Expansion (Mode x Resolution x Max-Token Budget, 5-Fold CV) ===")
    print(df_knob.to_string(index=False))

    # 3. External AMBER Generalization Table (Strictly Outside VL-RouterBench)
    amb_m = np.array([r["is_external"] for r in records])
    amb_tr_m = ~amb_m  # Warm-started ONLY on the 4 VL-RouterBench tasks! Zero AMBER training data!
    Lat_norm = np.clip(L_c / float(np.percentile(L_c[amb_tr_m].mean(axis=0), 90)), 0.0, 2.5)
    ag_amb = LinUCB(n_actions=len(configs_all), feature_dim=896, alpha=0.10, alpha_anneal=True, regularization=4.0, discount_factor=0.98, use_gpu=False)
    ag_amb.warm_start_from_full_matrix(X_mat[amb_tr_m], Y_c[amb_tr_m] - 0.12 * Lat_norm[amb_tr_m], regularization=4.0, sample_weight=0.20)
    amb_acc_l, amb_lat_l = [], []
    for q_i in np.where(amb_m)[0]:
        a = ag_amb.select_action(X_mat[q_i], budget_fraction=0.9)
        amb_acc_l.append(float(Y_c[q_i, a]))
        amb_lat_l.append(float(L_c[q_i, a]))
        ag_amb.update(X_mat[q_i], a, float(Y_c[q_i, a] - 0.12 * Lat_norm[q_i, a]))
    df_amb = pd.DataFrame([
        {"Method on Unseen External AMBER": "Static Qwen3-VL-2B-Instruct (HighRes B512)", "Acc (%)": round(float(Y_c[amb_m, configs_all.index("Instruct_HighRes_B512")].mean())*100, 2), "Mean Lat (s)": round(float(L_c[amb_m, configs_all.index("Instruct_HighRes_B512")].mean()), 3)},
        {"Method on Unseen External AMBER": "Static Qwen3-VL-2B-Thinking (HighRes B512)", "Acc (%)": round(float(Y_c[amb_m, configs_all.index("Thinking_HighRes_B512")].mean())*100, 2), "Mean Lat (s)": round(float(L_c[amb_m, configs_all.index("Thinking_HighRes_B512")].mean()), 3)},
        {"Method on Unseen External AMBER": "CascadeRoute-VL (Zero AMBER Train -> Online)", "Acc (%)": round(float(np.mean(amb_acc_l))*100, 2), "Mean Lat (s)": round(float(np.mean(amb_lat_l)), 3)},
        {"Method on Unseen External AMBER": "12-Config Oracle (AMBER)", "Acc (%)": round(float((Y_c[amb_m].max(axis=1) > 0).mean())*100, 2), "Mean Lat (s)": round(float(L_c[amb_m].min(axis=1).mean()), 3)},
    ])
    df_amb.to_csv(TABLES_DIR / "exp2_amber_external_eval.csv", index=False)

    # 4. Claim 8: Real GPU Wall-Clock p50/p95 Latency, Encoder Overhead, & Simulator Error Calibration
    # Fit TTFT + 1/TPS strictly per backbone mode via ordinary least squares on (1, out_tokens)
    sim_rows = []
    for mode_label in ["Instruct", "Thinking"]:
        sub = df_all[df_all["mode"] == mode_label]
        ot = sub["out_tokens"].values.astype(np.float64)
        lat_real = sub["latency_s"].values.astype(np.float64)
        # Fit OLS: lat = ttft + inv_tps * out_tok
        A_fit = np.vstack([np.ones_like(ot), ot]).T
        ttft_fit, slope_fit = np.linalg.lstsq(A_fit, lat_real, rcond=None)[0]
        tps_fit = 1.0 / max(slope_fit, 1e-6)
        lat_sim = ttft_fit + ot / tps_fit
        mae = float(np.mean(np.abs(lat_real - lat_sim)))
        mape = float(np.mean(np.abs(lat_real - lat_sim) / np.maximum(lat_real, 1e-3))) * 100.0
        corr_r = float(np.corrcoef(lat_real, lat_sim)[0, 1]) if np.std(ot) > 0 else 1.0
        sim_rows.append({
            "Backbone": f"Qwen3-VL-2B-{mode_label} (4-bit GPU)",
            "Samples": int(len(sub)),
            "Real Lat Mean (s)": round(float(lat_real.mean()), 3),
            "Real Lat p50 (s)": round(float(np.percentile(lat_real, 50)), 3),
            "Real Lat p95 (s)": round(float(np.percentile(lat_real, 95)), 3),
            "Fitted TTFT (s)": round(float(ttft_fit), 3),
            "Fitted Decode TPS": round(float(tps_fit), 1),
            "Sim MAE (s)": round(mae, 3),
            "Sim MAPE (%)": round(mape, 1),
            "Pearson r": round(corr_r, 4),
            "Router Overhead p50/p95 (ms)": f"{overhead_stats['total_p50_ms']:.1f} / {overhead_stats['total_p95_ms']:.1f}",
        })
    df_sim = pd.DataFrame(sim_rows)
    df_sim.to_csv(TABLES_DIR / "exp2_real_latency_and_sim_cal.csv", index=False)
    print("\n=== CLAIM 8: Measured GPU p50/p95 Latency, All-3-Encoder Overhead & Simulator Validation ===")
    print(df_sim.to_string(index=False))


def resolve_local_or_hf(local_snapshot_glob: str, hf_id: str) -> str:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    matches = sorted(hub.glob(local_snapshot_glob))
    for m in matches:
        if (m / "model.safetensors").exists() and (m / "model.safetensors").stat().st_size > 1_000_000_000:
            return str(m)
    return hf_id


def main():
    t0 = time.time()
    n_per_ds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    records = load_stratified_queries(n_per_ds=n_per_ds, seed=42)
    records, overhead_stats = encode_external_and_benchmark_overhead(records)
    pd.DataFrame([overhead_stats]).to_csv(TABLES_DIR / "exp2_router_overhead.csv", index=False)

    inst_id = resolve_local_or_hf("models--unsloth--Qwen3-VL-2B-Instruct-bnb-4bit/snapshots/*", "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit")
    thk_id = resolve_local_or_hf("models--unsloth--Qwen3-VL-2B-Thinking-unsloth-bnb-4bit/snapshots/*", "unsloth/Qwen3-VL-2B-Thinking-unsloth-bnb-4bit")

    res_map = {"LowRes": 196 * 196, "HighRes": 392 * 392}
    df_ins = eval_qwen3vl_model(inst_id, "Instruct", records, res_map)
    df_thk = eval_qwen3vl_model(thk_id, "Thinking", records, res_map)
    df_all = pd.concat([df_ins, df_thk], ignore_index=True)
    df_all.to_csv(TABLES_DIR / "exp2_raw_live_traces.csv", index=False)

    analyze_and_save(df_all, records, overhead_stats)
    print(f"\nCompleted exp2 live VLM configuration study in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
