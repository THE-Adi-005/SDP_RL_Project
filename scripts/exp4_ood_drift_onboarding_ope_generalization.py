#!/usr/bin/env python3
"""Comprehensive Empirical Validation for Claims 6, 7 & Cross-Cutting Fixes.

Outputs saved to reports/tables/:
  - exp4_claim6_ood_unified_w0.csv          (Discounted & Sliding-Window LinUCB w/ unified w0=0.20 + Cost/Latency vs 72B)
  - exp4_claim6_continuous_drift.csv        (Gradual mixture drift, repeated switching, & 3x mid-stream price shock)
  - exp4_claim6_hot_onboarding.csv          (Test-time 17th-model onboarding: adoption curves & accuracy gain vs frozen MLP)
  - exp4_claim7_ope_audit.csv               (Online Adaptive Policy OPE + Stationary Snapshot OPE + multi-seed coverage audit)
  - exp4_cross_lodo_benchmarks.csv          (Leave-One-Dataset-Out held-out benchmark generalization)
  - exp4_cross_pool_transfer.csv            (Disjoint Pool A -> Pool B unseen model pool transfer)
  - exp4_cross_partial_and_noisy_fb.csv     (Partial feedback 5%..100% & Noisy LLM-Judge 0%..20% flip rate)
"""
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.statistical_tests import paired_bootstrap_ci
from src.rl.arms_router import ARMSRouter
from src.rl.linucb import LinUCB
from src.rl.offline_router import OfflineCostAwareRouter
from scripts.adapt_real_vlrouterbench import (
    DATASET_DOMAINS,
    build_model_profiles_dict,
    mlp_predict_probs,
    train_mlp_router,
)

DATA_DIR = ROOT / "data" / "vl_routerbench_processed"
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]


def run_linucb_variant(
    X_w, R_w, X_s, Y_s, C_s, Cn_s, L_s,
    gamma: float, w0: float = 0.20, alpha0: float = 0.10, alpha_min: float = 0.01,
    reg_lambda: float = 4.0, discount_factor: float = 1.0, sliding_window: int = None,
    use_warm: bool = True, seed: int = 42, order: np.ndarray = None,
):
    N, K = Y_s.shape
    d = X_s.shape[1]
    use_gpu = torch.cuda.is_available()
    agent = LinUCB(
        n_actions=K, feature_dim=d, alpha=alpha0, alpha_min=alpha_min,
        alpha_anneal=True, regularization=reg_lambda,
        discount_factor=discount_factor, sliding_window=sliding_window, use_gpu=use_gpu
    )
    if use_warm:
        agent.warm_start_from_full_matrix(X_w, R_w, regularization=reg_lambda, sample_weight=w0)
    if order is None:
        order = np.random.RandomState(seed).permutation(N)

    corr = np.zeros(N, dtype=np.float64)
    cost = np.zeros(N, dtype=np.float64)
    lat = np.zeros(N, dtype=np.float64)
    acts = np.zeros(N, dtype=np.int64)
    Xg = torch.from_numpy(X_s.astype(np.float32)).cuda() if use_gpu else None

    B_max = max(float(C_s.mean() * 1.5), 0.00025) * N
    spent = 0.0
    for q in order:
        bf = max(0.0, min(1.0, (B_max - spent) / max(B_max, 1e-12)))
        st = Xg[q] if use_gpu else X_s[q]
        a = agent.select_action(st, budget_fraction=bf)
        spent += float(C_s[q, a])
        corr[q] = float(Y_s[q, a])
        cost[q] = float(C_s[q, a])
        lat[q] = float(L_s[q, a])
        acts[q] = a
        agent.update(st, a, float(Y_s[q, a] - gamma * Cn_s[q, a]))

    return {"correct": corr, "cost": cost, "latency": lat, "actions": acts, "order": order}


def run_claim6_ood_and_drift(Y, C, C_norm, L, X, meta, models):
    print("\n" + "=" * 80)
    print("CLAIM 6: Unified w0=0.20 Non-Stationary OOD, Continuous Drift, & Hot Onboarding")
    print("=" * 80)

    is_in = meta["dataset"].apply(lambda ds: DATASET_DOMAINS[ds] in ["General_VQA", "STEM"]).values
    in_tr = np.where(is_in & (meta["split"] == "train").values)[0]
    in_te = np.where(is_in & (meta["split"] == "test").values)[0]
    ood_idx = np.where(~is_in)[0]

    gamma = 0.08
    R_in = Y[in_tr] - gamma * C_norm[in_tr]
    R_in_lowc = Y[in_tr] - 0.02 * C_norm[in_tr]
    arm_norm_cost = C_norm[in_tr].mean(axis=0)
    K = len(models)

    # Fit Offline models on In-Domain train
    off_lin = OfflineCostAwareRouter(n_actions=K, lambda_weight=0.0, reg_lambda=15.0)
    off_lin.fit_from_matrices(X[in_tr], Y[in_tr], C_norm[in_tr])
    prof = build_model_profiles_dict(Y[in_tr], C[in_tr], models)
    arms_r = ARMSRouter(n_actions=K, profiles=prof, lambda_weight=0.0)
    arms_r.fit_from_matrices(X[in_tr], Y[in_tr], C[in_tr], reg_lambda=15.0)
    val_sp = int(0.9 * len(in_tr))
    mlp_in, _, _ = train_mlp_router(X[in_tr[:val_sp]], Y[in_tr[:val_sp]], X[in_tr[val_sp:]], Y[in_tr[val_sp:]], epochs=26, seed=42)

    p_lin_oo = np.clip((off_lin.scaler.transform(X[ood_idx]) @ off_lin.W) + off_lin.b[None, :], 0.0, 1.0)
    a_lin_oo = np.argmax(p_lin_oo - gamma * arm_norm_cost[None, :], axis=1)
    p_mlp_oo = mlp_predict_probs(mlp_in, X[ood_idx])
    a_mlp_oo = np.argmax(p_mlp_oo - gamma * arm_norm_cost[None, :], axis=1)
    xs_oo = arms_r.scaler.transform(X[ood_idx])
    p_arm_oo = 0.65 * ((xs_oo @ arms_r._W_align) @ arms_r._P_mat.T + arms_r._b_arm[None, :]) + \
               0.35 * ((xs_oo @ arms_r._W_full) + arms_r._b_arm[None, :])
    a_arm_oo = np.argmax(p_arm_oo - gamma * arm_norm_cost[None, :], axis=1)

    idx_72b = models.index("Qwen2.5-VL-72B-Instruct")
    qwen_oo_acc = float(Y[ood_idx, idx_72b].mean()) * 100.0
    qwen_oo_cst = float(C[ood_idx, idx_72b].mean()) * 1000.0
    qwen_oo_lat = float(L[ood_idx, idx_72b].mean())

    ood_rows = [
        {
            "Method": "Qwen2.5-VL-72B (Static Frontier)", "w_0": "-", "Variant": "Static",
            "OOD Acc (%)": round(qwen_oo_acc, 2), "Acc Std": 0.0,
            "OOD Cost/1k ($)": round(qwen_oo_cst, 4), "OOD Lat (s)": round(qwen_oo_lat, 3),
            "USD Saved vs 72B": "0.0%", "Speedup": "1.00x",
        },
        {
            "Method": "MLP-Router (Frozen Offline)", "w_0": "-", "Variant": "Offline",
            "OOD Acc (%)": round(float(Y[ood_idx, a_mlp_oo].mean()) * 100.0, 2), "Acc Std": 0.0,
            "OOD Cost/1k ($)": round(float(C[ood_idx, a_mlp_oo].mean()) * 1000.0, 4),
            "OOD Lat (s)": round(float(L[ood_idx, a_mlp_oo].mean()), 3),
            "USD Saved vs 72B": f"{(1 - float(C[ood_idx, a_mlp_oo].mean())*1000/qwen_oo_cst)*100:.1f}%",
            "Speedup": f"{qwen_oo_lat/float(L[ood_idx, a_mlp_oo].mean()):.2f}x",
        },
        {
            "Method": "ARMS-Router (Frozen Offline)", "w_0": "-", "Variant": "Offline",
            "OOD Acc (%)": round(float(Y[ood_idx, a_arm_oo].mean()) * 100.0, 2), "Acc Std": 0.0,
            "OOD Cost/1k ($)": round(float(C[ood_idx, a_arm_oo].mean()) * 1000.0, 4),
            "OOD Lat (s)": round(float(L[ood_idx, a_arm_oo].mean()), 3),
            "USD Saved vs 72B": f"{(1 - float(C[ood_idx, a_arm_oo].mean())*1000/qwen_oo_cst)*100:.1f}%",
            "Speedup": f"{qwen_oo_lat/float(L[ood_idx, a_arm_oo].mean()):.2f}x",
        },
    ]

    # Compare Stationary vs Discounted vs Sliding-Window ALL with unified w_0=0.20!
    specs = [
        ("Cold LinUCB (Online Zero-Prior)", 0.0, False, 0.35, 1.0, None, gamma, R_in, 904),
        ("CascadeRoute-VL Standard (g=0.08, 904d)", 0.20, True, 0.10, 1.0, None, gamma, R_in, 904),
        ("Discounted CascadeRoute (beta=0.996, g=0.08, 904d)", 0.20, True, 0.10, 0.996, None, gamma, R_in, 904),
        ("Sliding-Window CascadeRoute (W=400, g=0.08, 904d)", 0.20, True, 0.10, 1.0, 400, gamma, R_in, 904),
        ("Discounted CascadeRoute (beta=0.996, g=0.02, 904d)", 0.20, True, 0.10, 0.996, None, 0.02, R_in_lowc, 904),
        ("CascadeRoute-VL Standard (g=0.08, 896d cues-cut)", 0.20, True, 0.10, 1.0, None, gamma, R_in, 896),
        ("Discounted CascadeRoute (beta=0.996, g=0.08, 896d cues-cut)", 0.20, True, 0.10, 0.996, None, gamma, R_in, 896),
        ("Discounted CascadeRoute (beta=0.996, g=0.02, 896d cues-cut)", 0.20, True, 0.10, 0.996, None, 0.02, R_in_lowc, 896),
    ]
    for mname, w0, uw, a0, beta, sw, g_v, R_v, dim_use in specs:
        X_use = X[:, :dim_use]
        accs, csts, lats = [], [], []
        for sd in SEEDS:
            res = run_linucb_variant(
                X_use[in_tr], R_v, X_use[ood_idx], Y[ood_idx], C[ood_idx], C_norm[ood_idx], L[ood_idx],
                gamma=g_v, w0=w0, alpha0=a0, discount_factor=beta, sliding_window=sw, use_warm=uw, seed=sd
            )
            accs.append(float(res["correct"].mean()) * 100.0)
            csts.append(float(res["cost"].mean()) * 1000.0)
            lats.append(float(res["latency"].mean()))
        cm = float(np.mean(csts))
        lm = float(np.mean(lats))
        ood_rows.append({
            "Method": mname, "w_0": f"{w0:.2f}" if uw else "0.00", "Variant": "D-LinUCB" if beta < 1.0 else ("SW-LinUCB" if sw else "LinUCB"),
            "OOD Acc (%)": round(float(np.mean(accs)), 2), "Acc Std": round(float(np.std(accs)), 2),
            "OOD Cost/1k ($)": round(cm, 4), "OOD Lat (s)": round(lm, 3),
            "USD Saved vs 72B": f"{(1.0 - cm / qwen_oo_cst)*100:.1f}%",
            "Speedup": f"{qwen_oo_lat / lm:.2f}x",
        })
    df_oo = pd.DataFrame(ood_rows)
    df_oo.to_csv(TABLES_DIR / "exp4_claim6_ood_unified_w0.csv", index=False)
    print(df_oo.to_string(index=False))

    # --- Continuous Drift Streams (Gradual Drift, Repeated Switches, 3x Mid-Stream Price Shock) ---
    in_te_all = np.where(is_in & (meta["split"] != "train").values)[0]
    oo_all = ood_idx
    rng_dr = np.random.RandomState(42)

    # Stream 1: Gradual Mixture Drift (N=10,000, p_ood goes linearly 0 -> 1)
    T_dr = 10000
    grad_idx = []
    for t in range(T_dr):
        p_oo = t / float(T_dr - 1)
        pool = oo_all if rng_dr.rand() < p_oo else in_te_all
        grad_idx.append(int(rng_dr.choice(pool)))
    grad_idx = np.array(grad_idx)

    # Stream 2: Repeated Switching Regimes (STEM 2k -> OCR 2k -> General 2k -> OCR 2k -> STEM 2k)
    dom_arr = meta["dataset"].map(DATASET_DOMAINS).values
    non_tr = np.where((meta["split"] != "train").values)[0]
    stem_pool = non_tr[dom_arr[non_tr] == "STEM"]
    gen_pool = non_tr[dom_arr[non_tr] == "General_VQA"]
    ocr_pool = np.where(dom_arr == "Charts_OCR")[0]
    sw_idx = np.concatenate([
        rng_dr.choice(stem_pool, 2000),
        rng_dr.choice(ocr_pool, 2000),
        rng_dr.choice(gen_pool, 2000),
        rng_dr.choice(ocr_pool, 2000),
        rng_dr.choice(stem_pool, 2000),
    ])

    drift_rows = []
    for sname, s_idx in [("1. Gradual Mixture Drift (10k)", grad_idx), ("2. Repeated Regime Switches (10k)", sw_idx)]:
        # Evaluate Frozen MLP, Standard LinUCB, Discounted LinUCB (0.996), Sliding-Window (400)
        p_m = mlp_predict_probs(mlp_in, X[s_idx])
        a_m = np.argmax(p_m - gamma * arm_norm_cost[None, :], axis=1)
        drift_rows.append({
            "Stream": sname, "Method": "MLP-Router (Frozen)",
            "Acc (%)": round(float(Y[s_idx, a_m].mean()) * 100.0, 2),
            "Cost/1k ($)": round(float(C[s_idx, a_m].mean()) * 1000.0, 4),
            "Avg Reward": round(float((Y[s_idx, a_m] - gamma * C_norm[s_idx, a_m]).mean()), 4),
        })
        seq_order = np.arange(len(s_idx))  # Preserve temporal drift ordering!
        for vname, beta, sw in [
            ("LinUCB (w0=0.20, stationary)", 1.0, None),
            ("Discounted LinUCB (beta=0.996)", 0.996, None),
            ("Sliding-Window LinUCB (W=400)", 1.0, 400),
        ]:
            rv = run_linucb_variant(
                X[in_tr], R_in, X[s_idx], Y[s_idx], C[s_idx], C_norm[s_idx], L[s_idx],
                gamma=gamma, w0=0.20, alpha0=0.10, discount_factor=beta, sliding_window=sw,
                use_warm=True, order=seq_order
            )
            rew_v = rv["correct"] - gamma * np.clip(rv["cost"] / float(np.percentile(C[in_tr].mean(axis=0), 90)), 0.0, 2.5)
            drift_rows.append({
                "Stream": sname, "Method": vname,
                "Acc (%)": round(float(rv["correct"].mean()) * 100.0, 2),
                "Cost/1k ($)": round(float(rv["cost"].mean()) * 1000.0, 4),
                "Avg Reward": round(float(rew_v.mean()), 4),
            })

    # Stream 3: Mid-Stream 3x Price Shock on Qwen72B + Gemini2.5 at t=2000 of Held-Out Test (6,108 Q)
    tr_idx = np.where((meta["split"] == "train").values)[0]
    te_idx = np.where((meta["split"] == "test").values)[0]
    c_sc_full = float(np.percentile(C[tr_idx].mean(axis=0), 90))
    Cn_full = np.clip(C / c_sc_full, 0.0, 2.5)
    R_full_tr = Y[tr_idx] - gamma * Cn_full[tr_idx]

    C_shock = C[te_idx].copy()
    idx_gem = models.index("GeminiFlash2-5")
    shock_t = 2000
    order_te = np.random.RandomState(42).permutation(len(te_idx))
    post_q = order_te[shock_t:]
    C_shock[post_q][:, [idx_72b, idx_gem]] *= 3.0  # Price spike applied after t=2000
    Cn_shock = np.clip(C_shock / c_sc_full, 0.0, 7.5)  # Unclipped spike in post-shock penalty!

    # Frozen MLP trained on pre-shock costs
    mlp_full, _, _ = train_mlp_router(X[tr_idx], Y[tr_idx], X[np.where((meta["split"]=="dev").values)[0]], Y[np.where((meta["split"]=="dev").values)[0]], epochs=26, seed=42)
    am_shock = np.argmax(mlp_predict_probs(mlp_full, X[te_idx]) - gamma * Cn_full[tr_idx].mean(axis=0)[None, :], axis=1)
    post_exp_mlp = float(np.isin(am_shock[post_q], [idx_72b, idx_gem]).mean()) * 100.0
    drift_rows.append({
        "Stream": "3. 3x Price Shock (t>2000 post-shock)", "Method": f"MLP-Router (Frozen, {post_exp_mlp:.1f}% expensive picks)",
        "Acc (%)": round(float(Y[te_idx][post_q, am_shock[post_q]].mean()) * 100.0, 2),
        "Cost/1k ($)": round(float(C_shock[post_q, am_shock[post_q]].mean()) * 1000.0, 4),
        "Avg Reward": round(float((Y[te_idx][post_q, am_shock[post_q]] - gamma * Cn_shock[post_q, am_shock[post_q]]).mean()), 4),
    })

    for vname, beta, sw in [("LinUCB (stationary)", 1.0, None), ("Discounted LinUCB (beta=0.996)", 0.996, None)]:
        rv = run_linucb_variant(
            X[tr_idx], R_full_tr, X[te_idx], Y[te_idx], C_shock, Cn_shock, L[te_idx],
            gamma=gamma, w0=0.20, alpha0=0.10, discount_factor=beta, sliding_window=sw,
            use_warm=True, order=order_te
        )
        post_acts = rv["actions"][post_q]
        post_exp = float(np.isin(post_acts, [idx_72b, idx_gem]).mean()) * 100.0
        post_rew = float((rv["correct"][post_q] - gamma * Cn_shock[post_q, post_acts]).mean())
        drift_rows.append({
            "Stream": "3. 3x Price Shock (t>2000 post-shock)",
            "Method": f"{vname} ({post_exp:.1f}% expensive picks)",
            "Acc (%)": round(float(rv["correct"][post_q].mean()) * 100.0, 2),
            "Cost/1k ($)": round(float(rv["cost"][post_q].mean()) * 1000.0, 4),
            "Avg Reward": round(post_rew, 4),
        })

    df_dr = pd.DataFrame(drift_rows)
    df_dr.to_csv(TABLES_DIR / "exp4_claim6_continuous_drift.csv", index=False)
    print("\n--- Continuous Drift & Price Shock ---")
    print(df_dr.to_string(index=False))

    # --- Hot New-Model Onboarding at Test Time (hide Kimi-VL-A3B-Thinking until t=500!) ---
    idx_kimi = models.index("Kimi-VL-A3B-Thinking-2506")
    # Run Discounted LinUCB where Kimi is masked out for t < 500 and onboarded at t = 500
    use_gpu = torch.cuda.is_available()
    d = X.shape[1]
    order_hb = np.random.RandomState(42).permutation(len(te_idx))
    onboard_t = 500

    # Zero out Kimi in offline warm-start matrix (16-arm offline history; Kimi is truly new at t=500!)
    R_tr_16 = R_full_tr.copy()
    R_tr_16[:, idx_kimi] = -5.0  # Unseen in offline warm-start
    ag_on = LinUCB(n_actions=K, feature_dim=d, alpha=0.35, alpha_anneal=False, regularization=4.0, discount_factor=0.998, use_gpu=use_gpu)
    ag_on.warm_start_from_full_matrix(X[tr_idx], R_tr_16, regularization=4.0, sample_weight=0.20)
    # Give new model weak initial uninformative lambda*I prior and mask before t=500
    Xg_te = torch.from_numpy(X[te_idx].astype(np.float32)).cuda() if use_gpu else None

    mean_arm_theta = np.mean(ag_on._theta_stack[[j for j in range(K) if j != idx_kimi]], axis=0)
    acts_hb, corr_hb = np.zeros(len(te_idx), dtype=np.int64), np.zeros(len(te_idx), dtype=np.float64)
    step_acts = []
    for step_i, q in enumerate(order_hb):
        if step_i == onboard_t:
            ag_on.hot_onboard_arm(idx_kimi, mean_prior_theta=mean_arm_theta, prior_strength=1.0, init_reg=0.5)
        st = Xg_te[q] if use_gpu else X[te_idx][q]
        _, _, ucb = ag_on.score_arms(st)
        if step_i < onboard_t:
            ucb[idx_kimi] = -1e9  # 16-arm pool before onboarding
        a = int(np.argmax(ucb))
        acts_hb[q] = a
        corr_hb[q] = Y[te_idx][q, a]
        step_acts.append((step_i, a, float(Y[te_idx][q, a])))
        ag_on.update(st, a, float(Y[te_idx][q, a] - gamma * Cn_full[te_idx][q, a]))

    # Summarize adoption in post-onboarding windows
    hb_rows = []
    first_pull = next((x[0] - onboard_t for x in step_acts[onboard_t:] if x[1] == idx_kimi), -1)
    for lo, hi in [(0, 500), (500, 550), (550, 700), (700, 1500), (1500, 3500), (3500, len(te_idx))]:
        seg = step_acts[lo:hi]
        adopt_pct = 100.0 * np.mean([1.0 if x[1] == idx_kimi else 0.0 for x in seg])
        acc_pct = 100.0 * np.mean([x[2] for x in seg])
        hb_rows.append({
            "Window (Queries)": f"[{lo}, {hi})",
            "Pool Size": 16 if hi <= 500 else 17,
            "CascadeRoute Kimi Adoption (%)": round(adopt_pct, 2),
            "First Kimi Pick (queries after t=500)": first_pull if lo == 500 else "-",
            "CascadeRoute Acc (%)": round(acc_pct, 2),
            "Frozen Offline Router Adoption (%)": 0.0,
        })
    df_hb = pd.DataFrame(hb_rows)
    df_hb.to_csv(TABLES_DIR / "exp4_claim6_hot_onboarding.csv", index=False)
    print("\n--- Hot Test-Time Model Onboarding (17th Arm Kimi-Thinking Inserted at t=500) ---")
    print(df_hb.to_string(index=False))


def run_claim7_ope_audit(Y, C, C_norm, X, meta, models):
    print("\n" + "=" * 80)
    print("CLAIM 7: Counterfactual OPE (Online Adaptive Trajectory + 5-Seed Snapshot Coverage)")
    print("=" * 80)

    tr_idx = np.where((meta["split"] == "train").values)[0]
    dev_idx = np.where((meta["split"] == "dev").values)[0]
    te_idx = np.where((meta["split"] == "test").values)[0]
    K = len(models)
    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_te, Y_te = X[te_idx], Y[te_idx]
    N = len(te_idx)

    # Logger: uniform mixed epsilon=0.60 over offline prior so rare arms like 72B have sufficient propensity support
    off_log = OfflineCostAwareRouter(n_actions=K, lambda_weight=0.10, reg_lambda=25.0)
    off_log.fit_from_matrices(X_tr, Y_tr, C_norm[tr_idx])
    eta_hat_te = np.clip((off_log.scaler.transform(X_te) @ off_log.W) + off_log.b[None, :], 0.0, 1.0)
    base_acts = off_log.predict_batch(X_te)

    # Evaluate Online Adaptive CascadeRoute-VL policy alongside stationary snapshots across 5 logger seeds
    R_tr = Y_tr - 0.08 * C_norm[tr_idx]
    cr_adapt = LinUCB(n_actions=K, feature_dim=X.shape[1], alpha=0.10, alpha_anneal=True, regularization=4.0, use_gpu=torch.cuda.is_available())
    cr_adapt.warm_start_from_full_matrix(X_tr, R_tr, regularization=4.0, sample_weight=0.20)
    adapt_acts = np.zeros(N, dtype=np.int32)
    order = np.random.RandomState(42).permutation(N)
    for step_i, q in enumerate(order):
        bf = max(0.0, 1.0 - step_i / N)
        a = cr_adapt.select_action(X_te[q], budget_fraction=bf)
        adapt_acts[q] = a
        cr_adapt.update(X_te[q], a, float(Y_te[q, a] - 0.08 * C_norm[te_idx][q, a]))

    # Frozen warm-start snapshot
    cr_snap = LinUCB(n_actions=K, feature_dim=X.shape[1], alpha=0.0, regularization=4.0, use_gpu=torch.cuda.is_available())
    cr_snap.warm_start_from_full_matrix(X_tr, R_tr, regularization=4.0, sample_weight=0.20)
    snap_acts = np.array([cr_snap.select_action(X_te[i]) for i in range(N)], dtype=np.int32)

    idx_72b = models.index("Qwen2.5-VL-72B-Instruct")
    idx_qian = models.index("Qianfan-VL-8B")
    idx_tiny = models.index("deepseek_vl2_tiny")
    mlp_m, _, _ = train_mlp_router(X_tr, Y_tr, X[dev_idx], Y[dev_idx], epochs=24, seed=42)
    mlp_acts = np.argmax(mlp_predict_probs(mlp_m, X_te) - 0.08 * C_norm[tr_idx].mean(axis=0)[None, :], axis=1)

    policies = [
        ("Always-DS-Tiny (Static)", np.full(N, idx_tiny, dtype=np.int32)),
        ("Always-Qianfan-8B (Static)", np.full(N, idx_qian, dtype=np.int32)),
        ("Always-Qwen-72B (Static)", np.full(N, idx_72b, dtype=np.int32)),
        ("Offline-Linear", base_acts.astype(np.int32)),
        ("MLP-Router", mlp_acts.astype(np.int32)),
        ("CascadeRoute-VL (Warm-Start Snapshot)", snap_acts),
        ("CascadeRoute-VL (Online Adaptive Trajectory)", adapt_acts),
    ]

    # Single canonical balanced logger (eps=0.75) + 20-seed CI coverage check!
    eps = 0.75
    ope_rows = []
    boot_rng = np.random.RandomState(123)
    for pname, pi_acts in policies:
        v_true = float(Y_te[np.arange(N), pi_acts].mean())
        cover_dr_count = 0
        mean_dr, mean_snips, mean_ips, mean_ess = [], [], [], []
        ci0_snips = ci0_dr = ci0_ips = ""
        for ls in range(10):
            rng_l = np.random.RandomState(100 + ls)
            log_a = np.zeros(N, dtype=np.int32)
            log_p = np.zeros(N, dtype=np.float64)
            log_r = np.zeros(N, dtype=np.float64)
            for i in range(N):
                pr = np.full(K, eps / K)
                pr[int(base_acts[i])] += (1.0 - eps)
                a = int(rng_l.choice(K, p=pr))
                log_a[i], log_p[i], log_r[i] = a, pr[a], float(Y_te[i, a])

            w = (pi_acts == log_a).astype(np.float64) / log_p
            v_ips = float(np.mean(w * log_r))
            v_sn = float(np.sum(w * log_r) / max(np.sum(w), 1e-12))
            dr_t = eta_hat_te[np.arange(N), pi_acts] + w * (log_r - eta_hat_te[np.arange(N), log_a])
            v_dr = float(np.mean(dr_t))
            ess = float((np.sum(w) ** 2) / max(np.sum(w ** 2), 1e-12))

            b_dr = [np.mean(dr_t[boot_rng.randint(0, N, N)]) for _ in range(400)]
            lo_dr, hi_dr = float(np.percentile(b_dr, 2.5)), float(np.percentile(b_dr, 97.5))
            if lo_dr <= v_true <= hi_dr:
                cover_dr_count += 1
            mean_ips.append(v_ips); mean_snips.append(v_sn); mean_dr.append(v_dr); mean_ess.append(ess)
            if ls == 0:
                b_sn = [np.sum((w * log_r)[bi := boot_rng.randint(0, N, N)]) / max(np.sum(w[bi]), 1e-12) for _ in range(500)]
                b_ip = [np.mean((w * log_r)[boot_rng.randint(0, N, N)]) for _ in range(500)]
                ci0_ips = f"{v_ips:.4f} [{np.percentile(b_ip, 2.5):.4f}, {np.percentile(b_ip, 97.5):.4f}]"
                ci0_snips = f"{v_sn:.4f} [{np.percentile(b_sn, 2.5):.4f}, {np.percentile(b_sn, 97.5):.4f}]"
                ci0_dr = f"{v_dr:.4f} [{lo_dr:.4f}, {hi_dr:.4f}]"

        ope_rows.append({
            "Policy": pname,
            "True Acc": round(v_true, 4),
            "IPS (Seed0 CI)": ci0_ips,
            "SNIPS (Seed0 CI)": ci0_snips,
            "DR (Seed0 CI)": ci0_dr,
            "Mean DR (10 Loggers)": round(float(np.mean(mean_dr)), 4),
            "10-Logger DR 95% CI Cov": f"{cover_dr_count}/10",
            "Mean ESS": round(float(np.mean(mean_ess)), 1),
        })
    df_ope = pd.DataFrame(ope_rows)
    df_ope.to_csv(TABLES_DIR / "exp4_claim7_ope_audit.csv", index=False)
    print(df_ope.to_string(index=False))


def run_crosscutting_generalization(Y, C, C_norm, L, X, meta, models):
    print("\n" + "=" * 80)
    print("CROSS-CUTTING: LODO Held-Out Benchmarks, Disjoint Pool A->B, Partial & Noisy Feedback")
    print("=" * 80)

    gamma = 0.08
    use_gpu = torch.cuda.is_available()
    K, d = len(models), X.shape[1]

    # 1. Leave-One-Dataset-Out (LODO)
    lodo_rows = []
    for hold_ds in ["InfoVQA_VAL", "MathVision_MINI", "HallusionBench"]:
        tr_m = (meta["dataset"] != hold_ds).values & (meta["split"] == "train").values
        dv_m = (meta["dataset"] != hold_ds).values & (meta["split"] == "dev").values
        te_m = (meta["dataset"] == hold_ds).values

        Xt, Yt = X[tr_m], Y[tr_m]
        Ct, Cnt = C[tr_m], C_norm[tr_m]
        Xe, Ye, Ce, Cne = X[te_m], Y[te_m], C[te_m], C_norm[te_m]
        Rt = Yt - gamma * Cnt
        anc = Cnt.mean(axis=0)

        mlp_lo, _, _ = train_mlp_router(Xt, Yt, X[dv_m], Y[dv_m], epochs=22, seed=42)
        a_mlp = np.argmax(mlp_predict_probs(mlp_lo, Xe) - gamma * anc[None, :], axis=1)
        idx_72 = models.index("Qwen2.5-VL-72B-Instruct")

        r_cr = run_linucb_variant(
            Xt, Rt, Xe, Ye, Ce, Cne, L[te_m], gamma=gamma, w0=0.20, alpha0=0.10,
            discount_factor=0.996, use_warm=True, seed=42
        )
        lodo_rows.append({
            "Held-Out Dataset": hold_ds, "N": int(te_m.sum()),
            "Qwen2.5-VL-72B (%)": round(float(Ye[:, idx_72].mean()) * 100.0, 2),
            "Frozen MLP (%)": round(float(Ye[np.arange(len(Xe)), a_mlp].mean()) * 100.0, 2),
            "MLP Cost/1k ($)": round(float(Ce[np.arange(len(Xe)), a_mlp].mean()) * 1000.0, 4),
            "CascadeRoute-VL D-LinUCB (%)": round(float(r_cr["correct"].mean()) * 100.0, 2),
            "CR Cost/1k ($)": round(float(r_cr["cost"].mean()) * 1000.0, 4),
        })
    df_lodo = pd.DataFrame(lodo_rows)
    df_lodo.to_csv(TABLES_DIR / "exp4_cross_lodo_benchmarks.csv", index=False)
    print("\n--- Leave-One-Dataset-Out (LODO) Generalization ---")
    print(df_lodo.to_string(index=False))

    # 2. Disjoint Cross-Pool Transfer (Pool A: 9 models -> Pool B: 8 unseen models)
    # Build balanced Pool A & Pool B across scale tiers
    pool_a_names = [
        "Janus-Pro-1B", "SmolVLM2", "Phi-3.5-Vision", "Janus-Pro-7B", "Qianfan-VL-8B",
        "Gemma3-27B", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct", "GPT4o"
    ]
    pool_b_names = [m for m in models if m not in pool_a_names]  # 8 unseen models including DeepSeek-Tiny, Kimi-3B-Think, MiMo-7B-RL, DeepSeek-27B, InternVL2.5-78B, GeminiFlash2.5!
    pool_b_idx = [models.index(m) for m in pool_b_names]

    tr_idx = np.where((meta["split"] == "train").values)[0]
    te_idx = np.where((meta["split"] == "test").values)[0]
    # Router pre-learns query difficulty / shared linear utility direction w_bar from Pool A ONLY (zero Pool B outcomes seen!)
    R_A = Y[tr_idx][:, [models.index(m) for m in pool_a_names]] - gamma * C_norm[tr_idx][:, [models.index(m) for m in pool_a_names]]
    mean_yA = R_A.mean(axis=1)
    # Fit single shared difficulty vector theta_shared on Pool A
    XtX_tr = X[tr_idx].astype(np.float64).T @ X[tr_idx].astype(np.float64) + 10.0 * np.eye(d)
    theta_shared_A = np.linalg.solve(XtX_tr, X[tr_idx].astype(np.float64).T @ mean_yA)

    # Deploy onto completely unseen Pool B (8 arms): Cold vs Prior-Transferred from Pool A
    Yb_te, Cb_te, Cnb_te, Lb_te = Y[te_idx][:, pool_b_idx], C[te_idx][:, pool_b_idx], C_norm[te_idx][:, pool_b_idx], L[te_idx][:, pool_b_idx]
    order_te = np.random.RandomState(42).permutation(len(te_idx))

    ag_cold_b = LinUCB(n_actions=len(pool_b_idx), feature_dim=d, alpha=0.25, regularization=4.0, use_gpu=use_gpu)
    ag_xfer_b = LinUCB(n_actions=len(pool_b_idx), feature_dim=d, alpha=0.20, regularization=4.0, discount_factor=0.998, use_gpu=use_gpu)
    for b_arm in range(len(pool_b_idx)):
        ag_xfer_b.hot_onboard_arm(b_arm, mean_prior_theta=theta_shared_A, prior_strength=1.0)

    c_cold, s_cold, c_xfer, s_xfer = [], [], [], []
    for q in order_te:
        st = X[te_idx][q]
        ac = ag_cold_b.select_action(st)
        ax = ag_xfer_b.select_action(st)
        ag_cold_b.update(st, ac, float(Yb_te[q, ac] - gamma * Cnb_te[q, ac]))
        ag_xfer_b.update(st, ax, float(Yb_te[q, ax] - gamma * Cnb_te[q, ax]))
        c_cold.append(Yb_te[q, ac]); s_cold.append(Cb_te[q, ac] * 1000.0)
        c_xfer.append(Yb_te[q, ax]); s_xfer.append(Cb_te[q, ax] * 1000.0)

    best_static_B = float(Yb_te.mean(axis=0).max()) * 100.0
    oracle_B = float((Yb_te.max(axis=1) > 0).mean()) * 100.0
    df_pool = pd.DataFrame([
        {"Method on Unseen Pool B (8 Arms)": "Best Single Static in Pool B (InternVL2.5-78B)", "Test Acc (%)": round(best_static_B, 2), "Cost/1k ($)": round(float(Cb_te[:, Yb_te.mean(axis=0).argmax()].mean())*1000, 4)},
        {"Method on Unseen Pool B (8 Arms)": "Cold Online LinUCB on Pool B", "Test Acc (%)": round(float(np.mean(c_cold))*100, 2), "Cost/1k ($)": round(float(np.mean(s_cold)), 4)},
        {"Method on Unseen Pool B (8 Arms)": "Pool A -> Pool B Prior Transfer + Online", "Test Acc (%)": round(float(np.mean(c_xfer))*100, 2), "Cost/1k ($)": round(float(np.mean(s_xfer)), 4)},
        {"Method on Unseen Pool B (8 Arms)": "Oracle Pool B (8 Arms)", "Test Acc (%)": round(oracle_B, 2), "Cost/1k ($)": round(float(Cb_te.min(axis=1).mean())*1000, 4)},
    ])
    df_pool.to_csv(TABLES_DIR / "exp4_cross_pool_transfer.csv", index=False)
    print("\n--- Disjoint Cross-Pool Transfer (Pool A 9 Arms -> Unseen Pool B 8 Arms) ---")
    print(df_pool.to_string(index=False))

    # 3. Partial-Feedback & Noisy LLM-Judge Feedback Experiments (i.i.d. Test & OOD)
    is_in = meta["dataset"].apply(lambda ds: DATASET_DOMAINS[ds] in ["General_VQA", "STEM"]).values
    in_tr = np.where(is_in & (meta["split"] == "train").values)[0]
    ood_idx = np.where(~is_in)[0]
    R_id_tr = Y[tr_idx] - gamma * C_norm[tr_idx]
    R_oo_tr = Y[in_tr] - gamma * C_norm[in_tr]

    fb_rows = []
    # (a) Partial feedback fraction sweep on Cold, Warm i.i.d., and OOD
    for fb_rate in [0.05, 0.10, 0.25, 0.50, 1.00]:
        for reg_name, Xw, Rw, Xs, Ys, Cs, Cns, w0, uw, beta in [
            ("Cold LinUCB (i.i.d. Test)", X[tr_idx], R_id_tr, X[te_idx], Y[te_idx], C[te_idx], C_norm[te_idx], 0.0, False, 1.0),
            ("CascadeRoute-VL (Warm i.i.d.)", X[tr_idx], R_id_tr, X[te_idx], Y[te_idx], C[te_idx], C_norm[te_idx], 0.20, True, 1.0),
            ("CascadeRoute-VL D-LinUCB (OOD)", X[in_tr], R_oo_tr, X[ood_idx], Y[ood_idx], C[ood_idx], C_norm[ood_idx], 0.20, True, 0.996),
        ]:
            rng_f = np.random.RandomState(42)
            ag = LinUCB(n_actions=K, feature_dim=d, alpha=0.10 if uw else 0.35, alpha_anneal=uw, regularization=4.0 if uw else 2.0, discount_factor=beta, use_gpu=use_gpu)
            if uw:
                ag.warm_start_from_full_matrix(Xw, Rw, regularization=4.0, sample_weight=w0)
            Xg_s = torch.from_numpy(Xs.astype(np.float32)).cuda() if use_gpu else None
            cv, sv = [], []
            for q in rng_f.permutation(len(Xs)):
                st = Xg_s[q] if use_gpu else Xs[q]
                a = ag.select_action(st)
                cv.append(Ys[q, a]); sv.append(Cs[q, a] * 1000.0)
                if rng_f.rand() < fb_rate:
                    ag.update(st, a, float(Ys[q, a] - gamma * Cns[q, a]))
            fb_rows.append({
                "Experiment": "Partial Feedback Rate",
                "Regime": reg_name, "Level": f"{int(fb_rate*100)}%",
                "Acc (%)": round(float(np.mean(cv)) * 100.0, 2),
                "Cost/1k ($)": round(float(np.mean(sv)), 4),
            })

    # (b) Noisy LLM-Judge feedback flip rate eta in {0%, 5%, 10%, 15%, 20%}
    for flip_p in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for reg_name, Xw, Rw, Xs, Ys, Cs, Cns, w0, uw, beta in [
            ("CascadeRoute-VL (Warm i.i.d.)", X[tr_idx], R_id_tr, X[te_idx], Y[te_idx], C[te_idx], C_norm[te_idx], 0.20, True, 1.0),
            ("CascadeRoute-VL D-LinUCB (OOD)", X[in_tr], R_oo_tr, X[ood_idx], Y[ood_idx], C[ood_idx], C_norm[ood_idx], 0.20, True, 0.996),
        ]:
            rng_f = np.random.RandomState(77)
            ag = LinUCB(n_actions=K, feature_dim=d, alpha=0.10, alpha_anneal=True, regularization=4.0, discount_factor=beta, use_gpu=use_gpu)
            ag.warm_start_from_full_matrix(Xw, Rw, regularization=4.0, sample_weight=w0)
            Xg_s = torch.from_numpy(Xs.astype(np.float32)).cuda() if use_gpu else None
            cv, sv = [], []
            for q in rng_f.permutation(len(Xs)):
                st = Xg_s[q] if use_gpu else Xs[q]
                a = ag.select_action(st)
                y_obs = float(Ys[q, a])
                if rng_f.rand() < flip_p:
                    y_obs = 1.0 - y_obs
                ag.update(st, a, float(y_obs - gamma * Cns[q, a]))
                cv.append(Ys[q, a]); sv.append(Cs[q, a] * 1000.0)
            fb_rows.append({
                "Experiment": "Noisy LLM-Judge Flip",
                "Regime": reg_name, "Level": f"{int(flip_p*100)}%",
                "Acc (%)": round(float(np.mean(cv)) * 100.0, 2),
                "Cost/1k ($)": round(float(np.mean(sv)), 4),
            })

    df_fb = pd.DataFrame(fb_rows)
    df_fb.to_csv(TABLES_DIR / "exp4_cross_partial_and_noisy_fb.csv", index=False)
    print("\n--- Partial Feedback & Noisy LLM-Judge Robustness ---")
    print(df_fb.to_string(index=False))


def main():
    t0 = time.time()
    Y = np.load(DATA_DIR / "matrices" / "Y.npz")["Y"].astype(np.float32)
    C = np.load(DATA_DIR / "matrices" / "C.npz")["C"].astype(np.float64)
    T_npz = np.load(DATA_DIR / "matrices" / "T_tokens.npz")
    L = T_npz["latency_s"].astype(np.float64)
    feat_npz = np.load(DATA_DIR / "matrices" / "X_multimodal_904_f16.npz")
    X = feat_npz["X"].astype(np.float32)
    meta = pd.read_parquet(DATA_DIR / "registry" / "meta.parquet")
    with open(DATA_DIR / "registry" / "model_index.pkl", "rb") as f:
        models: List[str] = pickle.load(f)

    tr_idx = np.where((meta["split"] == "train").values)[0]
    c_scale = float(np.percentile(C[tr_idx].mean(axis=0), 90))
    C_norm = np.clip(C / c_scale, 0.0, 2.5)

    if "--hot-only" in sys.argv:
        # Fast standalone rerun for hot onboarding only
        tr_idx = np.where((meta["split"] == "train").values)[0]
        te_idx = np.where((meta["split"] == "test").values)[0]
        gamma = 0.08
        R_full_tr = Y[tr_idx] - gamma * C_norm[tr_idx]
        c_sc_full = float(np.percentile(C[tr_idx].mean(axis=0), 90))
        Cn_full = np.clip(C / c_sc_full, 0.0, 2.5)
        K, d = len(models), X.shape[1]
        idx_kimi = models.index("Kimi-VL-A3B-Thinking-2506")
        use_gpu = torch.cuda.is_available()
        order_hb = np.random.RandomState(42).permutation(len(te_idx))
        onboard_t = 500
        R_tr_16 = R_full_tr.copy(); R_tr_16[:, idx_kimi] = -5.0
        ag_on = LinUCB(n_actions=K, feature_dim=d, alpha=0.35, alpha_anneal=False, regularization=4.0, discount_factor=0.998, use_gpu=use_gpu)
        ag_on.warm_start_from_full_matrix(X[tr_idx], R_tr_16, regularization=4.0, sample_weight=0.20)
        Xg_te = torch.from_numpy(X[te_idx].astype(np.float32)).cuda() if use_gpu else None
        mean_arm_theta = np.mean(ag_on._theta_stack[[j for j in range(K) if j != idx_kimi]], axis=0)
        step_acts = []
        for step_i, q in enumerate(order_hb):
            if step_i == onboard_t:
                ag_on.hot_onboard_arm(idx_kimi, mean_prior_theta=mean_arm_theta, prior_strength=1.0, init_reg=0.5)
            st = Xg_te[q] if use_gpu else X[te_idx][q]
            _, _, ucb = ag_on.score_arms(st)
            if step_i < onboard_t:
                ucb[idx_kimi] = -1e9
            a = int(np.argmax(ucb))
            step_acts.append((step_i, a, float(Y[te_idx][q, a])))
            ag_on.update(st, a, float(Y[te_idx][q, a] - gamma * Cn_full[te_idx][q, a]))
        first_pull = next((x[0] - onboard_t for x in step_acts[onboard_t:] if x[1] == idx_kimi), -1)
        hb_rows = []
        for lo, hi in [(0, 500), (500, 550), (550, 700), (700, 1500), (1500, 3500), (3500, len(te_idx))]:
            seg = step_acts[lo:hi]
            hb_rows.append({
                "Window (Queries)": f"[{lo}, {hi})",
                "Pool Size": 16 if hi <= 500 else 17,
                "CascadeRoute Kimi Adoption (%)": round(100.0 * np.mean([1.0 if x[1] == idx_kimi else 0.0 for x in seg]), 2),
                "First Kimi Pick (queries after t=500)": first_pull if lo == 500 else "-",
                "CascadeRoute Acc (%)": round(100.0 * np.mean([x[2] for x in seg]), 2),
                "Frozen Offline Router Adoption (%)": 0.0,
            })
        df_hb = pd.DataFrame(hb_rows)
        df_hb.to_csv(TABLES_DIR / "exp4_claim6_hot_onboarding.csv", index=False)
        print(df_hb.to_string(index=False))
        return

    run_claim6_ood_and_drift(Y, C, C_norm, L, X, meta, models)
    run_claim7_ope_audit(Y, C, C_norm, X, meta, models)
    run_crosscutting_generalization(Y, C, C_norm, L, X, meta, models)
    print(f"\nCompleted exp4 (Claims 6, 7, Cross-Cutting) in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
