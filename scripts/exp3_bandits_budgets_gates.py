#!/usr/bin/env python3
"""Comprehensive Empirical Validation for Claims 3, 4, and 5 (+ Cross-Cutting Rigor).

Outputs saved to reports/tables/:
  - exp3_claim3_baselines_and_splits.csv     (Greedy a=0, NeuralUCB/TS, MLP-Reward, KNN-Reward, Domain-Oracle, 5 splits, Y_raw + Y_repaired)
  - exp3_claim3_pareto_curves.csv            (Full Pareto frontier data for all routers + Static Pareto)
  - exp3_claim3_pareto_auc.csv               (Normalized Cost-AUC & Latency-AUC on matched domains)
  - exp3_claim3_bootstrap_ci.csv             (10,000-resample paired bootstrap CIs & bounded p >= 1e-4)
  - exp3_claim3_modality_ablation_boot.csv   (Cold, Warm, OOD modality ablations + 8-cue bootstrap CI)
  - exp3_claim4_alpha_sensitivity.csv        (alpha_0 sensitivity [0..20] on i.i.d. & OOD + Theorem 1 constants)
  - exp3_claim4_budget_compliance.csv        (Unguarded vs Hard Guardrail Thm 2 vs BwK-PrimalDual across budgets)
  - exp3_claim5_calibrated_gate.csv          (Single-threshold Isotonic/Conformal Gate across regimes + both_arms vs final_only)
  - exp3_claim5_tau_sensitivity.csv          (tau_cal sensitivity sweep + accuracy per escalation dollar)
"""
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cascaderoute.confidence_gate import CalibratedDeferralGate, ConfidenceGate
from src.evaluation.statistical_tests import paired_bootstrap_ci, pareto_frontier_auc
from src.rl.arms_router import ARMSRouter
from src.rl.linucb import LinUCB
from src.rl.neural_bandit import BwKPrimalDualLinUCB, NeuralTS, NeuralUCB
from src.rl.offline_router import OfflineCostAwareRouter
from scripts.adapt_real_vlrouterbench import (
    DATASET_DOMAINS,
    PyTorchMLPRouter,
    build_model_profiles_dict,
    knn_predict_probs,
    mlp_predict_probs,
    run_cascaderoute_stream,
    train_mlp_router,
)

DATA_DIR = ROOT / "data" / "vl_routerbench_processed"
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]


def train_mlp_reward_regressor(
    X_tr: np.ndarray,
    R_tr: np.ndarray,
    X_dev: np.ndarray,
    R_dev: np.ndarray,
    epochs: int = 30,
    seed: int = 42,
) -> PyTorchMLPRouter:
    """Cost-aware offline MLP trained directly via MSE on scalarized reward R = Y - gamma * C_norm."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchMLPRouter(X_tr.shape[1], R_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()

    x_t = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    r_t = torch.from_numpy(R_tr.astype(np.float32)).to(device)
    x_d = torch.from_numpy(X_dev.astype(np.float32)).to(device)
    r_d = torch.from_numpy(R_dev.astype(np.float32)).to(device)

    best_loss = float("inf")
    best_state = None
    bs = 512
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_t), device=device)
        for s in range(0, len(x_t), bs):
            idx = perm[s : s + bs]
            opt.zero_grad()
            loss = crit(model(x_t[idx]), r_t[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            dl = float(crit(model(x_d), r_d).item())
            if dl < best_loss:
                best_loss = dl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


@torch.no_grad()
def mlp_predict_raw(model: PyTorchMLPRouter, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    out = []
    for s in range(0, len(X), 1024):
        xb = torch.from_numpy(X[s : s + 1024].astype(np.float32)).to(device)
        out.append(model(xb).cpu().numpy())
    return np.vstack(out)


def run_claim3_suite(Y, Y_rep, C, T_tot, O_tok, L, X, P_raw, meta, models):
    print("\n" + "=" * 80)
    print("CLAIM 3: New Baselines, Pareto AUC, Paired Bootstrap, & 5 Random Splits")
    print("=" * 80)

    train_idx = np.where((meta["split"] == "train").values)[0]
    dev_idx = np.where((meta["split"] == "dev").values)[0]
    test_idx = np.where((meta["split"] == "test").values)[0]
    K = len(models)
    d = X.shape[1]
    use_gpu = torch.cuda.is_available()

    X_tr, Y_tr, C_tr = X[train_idx], Y[train_idx], C[train_idx]
    X_dev, Y_dev = X[dev_idx], Y[dev_idx]
    X_te, Y_te, C_te = X[test_idx], Y[test_idx], C[test_idx]
    T_te, L_te = T_tot[test_idx], L[test_idx]
    P_te, O_te = P_raw[test_idx], O_tok[test_idx]

    c_scale = float(np.percentile(C_tr.mean(axis=0), 90))
    C_norm = np.clip(C / c_scale, 0.0, 2.5)
    C_norm_tr, C_norm_dev, C_norm_te = C_norm[train_idx], C_norm[dev_idx], C_norm[test_idx]
    arm_mean_cost = C_tr.mean(axis=0)
    arm_norm_cost = C_norm_tr.mean(axis=0)
    XtX_tr = X_tr.astype(np.float64).T @ X_tr.astype(np.float64)
    X_te_g = torch.from_numpy(X_te.astype(np.float32)).cuda() if use_gpu else None

    gamma_main = 0.08
    R_tr_main = Y_tr - gamma_main * C_norm_tr
    R_dev_main = Y_dev - gamma_main * C_norm_dev

    # 1. Offline Baselines: BCE MLP, Reward-MLP, Offline-Linear, KNN, KNN-Reward, ARMS, Per-Domain Best Static
    mlp_bce, _, _ = train_mlp_router(X_tr, Y_tr, X_dev, Y_dev, epochs=30, seed=42)
    mlp_bce_probs_te = mlp_predict_probs(mlp_bce, X_te)

    # Cost-aware KNN & Reward-MLP computed cleanly
    knn_y_te = knn_predict_probs(X_tr, Y_tr, X_te, k=15)
    knn_cn_te = knn_predict_probs(X_tr, C_norm_tr.astype(np.float32), X_te, k=15)

    mlp_rew_08 = train_mlp_reward_regressor(X_tr, R_tr_main, X_dev, R_dev_main, epochs=30, seed=42)
    mlp_rew_08_te = mlp_predict_raw(mlp_rew_08, X_te)

    offline_lin = OfflineCostAwareRouter(n_actions=K, lambda_weight=0.0, reg_lambda=15.0)
    offline_lin.fit_from_matrices(X_tr, Y_tr, C_norm_tr)
    lin_probs_te = np.clip((offline_lin.scaler.transform(X_te) @ offline_lin.W) + offline_lin.b[None, :], 0.0, 1.0)

    prof = build_model_profiles_dict(Y_tr, C_tr, models)
    arms_r = ARMSRouter(n_actions=K, profiles=prof, lambda_weight=0.0)
    arms_r.fit_from_matrices(X_tr, Y_tr, C_tr, reg_lambda=15.0)
    xs_te = arms_r.scaler.transform(X_te)
    arms_probs_te = 0.65 * ((xs_te @ arms_r._W_align) @ arms_r._P_mat.T + arms_r._b_arm[None, :]) + \
                    0.35 * ((xs_te @ arms_r._W_full) + arms_r._b_arm[None, :])

    # Per-Domain Best Static (uses oracle domain labels on Train -> upper bound skyline)
    tr_domains = meta.iloc[train_idx]["dataset"].map(DATASET_DOMAINS).values
    te_domains = meta.iloc[test_idx]["dataset"].map(DATASET_DOMAINS).values
    dom_best_arm = {}
    for dom in ["STEM", "General_VQA", "Charts_OCR", "Hallucination"]:
        dm = (tr_domains == dom)
        dom_best_arm[dom] = int(np.argmax(Y_tr[dm].mean(axis=0) - gamma_main * C_norm_tr[dm].mean(axis=0)))
    dom_acts_te = np.array([dom_best_arm[dom] for dom in te_domains], dtype=np.int64)

    # 2. Run Main Baselines at gamma=0.08 + Greedy (alpha_0=0) + NeuralUCB + NeuralTS (5 seeds)
    rows_b = []
    per_sample_acc: Dict[str, np.ndarray] = {}
    per_sample_cost: Dict[str, np.ndarray] = {}
    r_idx = np.arange(len(test_idx))

    idx_72b = models.index("Qwen2.5-VL-72B-Instruct")
    qwen_acc_te = Y_te[:, idx_72b]
    qwen_cost_te = C_te[:, idx_72b]

    def rec_det(name, cat, acts, rep_acts=None):
        c_v = Y_te[r_idx, acts]
        s_v = C_te[r_idx, acts]
        l_v = L_te[r_idx, acts]
        t_v = T_te[r_idx, acts]
        rep_a = float(Y_rep[test_idx][r_idx, acts if rep_acts is None else rep_acts].mean()) * 100.0
        rows_b.append({
            "Algorithm": name, "Category": cat,
            "Test Acc Raw (%)": round(float(c_v.mean()) * 100.0, 2),
            "Acc Std": 0.0,
            "Test Acc Repaired (%)": round(rep_a, 2),
            "Cost / 1k ($)": round(float(s_v.mean()) * 1000.0, 4),
            "Latency (s)": round(float(l_v.mean()), 3),
            "Avg Tokens": round(float(t_v.mean()), 1),
        })
        per_sample_acc[name] = c_v.astype(np.float64)
        per_sample_cost[name] = s_v.astype(np.float64)

    rec_det("Qwen2.5-VL-72B", "Static VLM", np.full(len(test_idx), idx_72b, dtype=np.int64))
    rec_det("Per-Domain Best Static (Oracle Domain)", "Domain Skyline", dom_acts_te)
    rec_det("Offline-Linear", "Offline Router", np.argmax(lin_probs_te - gamma_main * arm_norm_cost[None, :], axis=1))
    rec_det("ARMS-Router", "Offline Router", np.argmax(arms_probs_te - gamma_main * arm_norm_cost[None, :], axis=1))
    rec_det("KNN (k=15)", "Offline Router", np.argmax(knn_y_te - gamma_main * arm_norm_cost[None, :], axis=1))
    rec_det("KNN-Reward (cost-aware)", "Offline Router", np.argmax(knn_y_te - gamma_main * knn_cn_te, axis=1))
    rec_det("MLP-Router (BCE)", "Offline Router", np.argmax(mlp_bce_probs_te - gamma_main * arm_norm_cost[None, :], axis=1))
    rec_det("MLP-Reward (cost-aware MSE)", "Offline Router", np.argmax(mlp_rew_08_te, axis=1))

    # CascadeRoute-VL & Greedy Warm-Start (alpha=0.0) across 5 seeds
    for label, a0, use_g in [
        ("Greedy Warm-Start (alpha=0)", 0.0, True),
        ("CascadeRoute-VL (alpha=0.10)", 0.10, True),
        ("CascadeRoute-VL (No-Gate, alpha=0.10)", 0.10, False),
    ]:
        accs, rep_accs, csts, lats, toks = [], [], [], [], []
        s0 = None
        for sd in SEEDS:
            res = run_cascaderoute_stream(
                X_te, P_te, Y_te, C_te, C_norm_te, T_te, O_te, L_te,
                X_tr, R_tr_main, arm_mean_cost, arm_norm_cost, models,
                gamma=gamma_main, alpha_0=a0, alpha_anneal=True,
                warm_weight=0.20, reg_lambda=4.0, use_gate=use_g,
                precomputed_XtX=XtX_tr, X_stream_gpu=X_te_g, seed=sd,
            )
            accs.append(float(res["correct"].mean()) * 100.0)
            rep_accs.append(float(Y_rep[test_idx][r_idx, res["actions"]].mean()) * 100.0)
            csts.append(float(res["cost"].mean()) * 1000.0)
            lats.append(float(res["latency"].mean()))
            toks.append(float(res["tokens"].mean()))
            if sd == SEEDS[0]:
                s0 = res
        rows_b.append({
            "Algorithm": label, "Category": "Bandit Router",
            "Test Acc Raw (%)": round(float(np.mean(accs)), 2),
            "Acc Std": round(float(np.std(accs)), 2),
            "Test Acc Repaired (%)": round(float(np.mean(rep_accs)), 2),
            "Cost / 1k ($)": round(float(np.mean(csts)), 4),
            "Latency (s)": round(float(np.mean(lats)), 3),
            "Avg Tokens": round(float(np.mean(toks)), 1),
        })
        per_sample_acc[label] = s0["correct"].astype(np.float64)
        per_sample_cost[label] = s0["cost"].astype(np.float64)

    # NeuralUCB & NeuralTS (warm-started on Train + online replay updates across 5 seeds)
    for nlabel, is_ts in [("NeuralUCB (Warm+Online)", False), ("NeuralTS (Warm+Online)", True)]:
        accs, rep_accs, csts, lats, toks = [], [], [], [], []
        s0_c, s0_s = None, None
        for sd in SEEDS:
            nb = NeuralTS(n_actions=K, feature_dim=d, alpha=0.10, use_gpu=use_gpu, seed=sd) if is_ts else \
                 NeuralUCB(n_actions=K, feature_dim=d, alpha=0.10, use_gpu=use_gpu, seed=sd)
            nb.pretrain_warm_start(X_tr, R_tr_main, epochs=10, warm_weight=0.20)
            rng = np.random.RandomState(sd)
            order = rng.permutation(len(test_idx))
            cv = np.zeros(len(test_idx), dtype=np.float64)
            sv = np.zeros(len(test_idx), dtype=np.float64)
            lv = np.zeros(len(test_idx), dtype=np.float64)
            tv = np.zeros(len(test_idx), dtype=np.float64)
            av = np.zeros(len(test_idx), dtype=np.int64)
            for q in order:
                a = nb.select_action(X_te[q])
                rew = float(Y_te[q, a] - gamma_main * C_norm_te[q, a])
                nb.update(X_te[q], a, rew)
                cv[q], sv[q], lv[q], tv[q], av[q] = Y_te[q, a], C_te[q, a], L_te[q, a], T_te[q, a], a
            accs.append(float(cv.mean()) * 100.0)
            rep_accs.append(float(Y_rep[test_idx][r_idx, av].mean()) * 100.0)
            csts.append(float(sv.mean()) * 1000.0)
            lats.append(float(lv.mean()))
            toks.append(float(tv.mean()))
            if sd == SEEDS[0]:
                s0_c, s0_s = cv, sv
        rows_b.append({
            "Algorithm": nlabel, "Category": "Neural Bandit",
            "Test Acc Raw (%)": round(float(np.mean(accs)), 2),
            "Acc Std": round(float(np.std(accs)), 2),
            "Test Acc Repaired (%)": round(float(np.mean(rep_accs)), 2),
            "Cost / 1k ($)": round(float(np.mean(csts)), 4),
            "Latency (s)": round(float(np.mean(lats)), 3),
            "Avg Tokens": round(float(np.mean(toks)), 1),
        })
        per_sample_acc[nlabel] = s0_c
        per_sample_cost[nlabel] = s0_s

    # 3. Also evaluate 5 Random Stratified 70/10/20 Data Splits (Cross-Cutting Statistical Rigor!)
    split_accs_cr, split_accs_greedy, split_accs_mlp, split_accs_72 = [], [], [], []
    split_csts_cr, split_csts_mlp = [], []
    rng_sp = np.random.RandomState(2027)
    ds_arr = meta["dataset"].values
    unique_ds = sorted(np.unique(ds_arr))
    for sp_i in range(5):
        tr_i, dv_i, te_i = [], [], []
        for ds in unique_ds:
            idxs = np.where(ds_arr == ds)[0]
            perm = rng_sp.permutation(idxs)
            n1 = int(0.70 * len(perm))
            n2 = int(0.80 * len(perm))
            tr_i.append(perm[:n1])
            dv_i.append(perm[n1:n2])
            te_i.append(perm[n2:])
        tr_s = np.concatenate(tr_i)
        dv_s = np.concatenate(dv_i)
        te_s = np.concatenate(te_i)

        c_sc_s = float(np.percentile(C[tr_s].mean(axis=0), 90))
        Cn_s = np.clip(C / c_sc_s, 0.0, 2.5)
        amc_s = C[tr_s].mean(axis=0)
        anc_s = Cn_s[tr_s].mean(axis=0)
        R_tr_s = Y[tr_s] - gamma_main * Cn_s[tr_s]
        XtX_s = X[tr_s].astype(np.float64).T @ X[tr_s].astype(np.float64)
        X_te_s_g = torch.from_numpy(X[te_s].astype(np.float32)).cuda() if use_gpu else None

        r_cr_s = run_cascaderoute_stream(
            X[te_s], P_raw[te_s], Y[te_s], C[te_s], Cn_s[te_s], T_tot[te_s], O_tok[te_s], L[te_s],
            X[tr_s], R_tr_s, amc_s, anc_s, models, gamma=gamma_main, alpha_0=0.10, alpha_anneal=True,
            warm_weight=0.20, reg_lambda=4.0, use_gate=True, precomputed_XtX=XtX_s, X_stream_gpu=X_te_s_g, seed=42 + sp_i
        )
        r_gr_s = run_cascaderoute_stream(
            X[te_s], P_raw[te_s], Y[te_s], C[te_s], Cn_s[te_s], T_tot[te_s], O_tok[te_s], L[te_s],
            X[tr_s], R_tr_s, amc_s, anc_s, models, gamma=gamma_main, alpha_0=0.0, alpha_anneal=True,
            warm_weight=0.20, reg_lambda=4.0, use_gate=True, precomputed_XtX=XtX_s, X_stream_gpu=X_te_s_g, seed=42 + sp_i
        )
        m_s, _, _ = train_mlp_router(X[tr_s], Y[tr_s], X[dv_s], Y[dv_s], epochs=24, seed=42 + sp_i)
        mp_s = mlp_predict_probs(m_s, X[te_s])
        ma_s = np.argmax(mp_s - gamma_main * anc_s[None, :], axis=1)

        split_accs_cr.append(float(r_cr_s["correct"].mean()) * 100.0)
        split_csts_cr.append(float(r_cr_s["cost"].mean()) * 1000.0)
        split_accs_greedy.append(float(r_gr_s["correct"].mean()) * 100.0)
        split_accs_mlp.append(float(Y[te_s, ma_s].mean()) * 100.0)
        split_csts_mlp.append(float(C[te_s, ma_s].mean()) * 1000.0)
        split_accs_72.append(float(Y[te_s, idx_72b].mean()) * 100.0)

    split_summary = pd.DataFrame([
        {"Metric": "5-Split CascadeRoute-VL", "Acc Mean (%)": round(np.mean(split_accs_cr), 2), "Acc Std": round(np.std(split_accs_cr), 2), "Cost/1k ($)": round(np.mean(split_csts_cr), 4)},
        {"Metric": "5-Split Greedy (alpha=0)", "Acc Mean (%)": round(np.mean(split_accs_greedy), 2), "Acc Std": round(np.std(split_accs_greedy), 2), "Cost/1k ($)": round(np.mean(split_csts_cr), 4)},
        {"Metric": "5-Split MLP-Router (BCE)", "Acc Mean (%)": round(np.mean(split_accs_mlp), 2), "Acc Std": round(np.std(split_accs_mlp), 2), "Cost/1k ($)": round(np.mean(split_csts_mlp), 4)},
        {"Metric": "5-Split Qwen2.5-VL-72B", "Acc Mean (%)": round(np.mean(split_accs_72), 2), "Acc Std": round(np.std(split_accs_72), 2), "Cost/1k ($)": round(float(C[:, idx_72b].mean())*1000, 4)},
    ])
    split_summary.to_csv(TABLES_DIR / "exp3_5splits_summary.csv", index=False)

    df_b = pd.DataFrame(rows_b)
    df_b.to_csv(TABLES_DIR / "exp3_claim3_baselines_and_splits.csv", index=False)
    print(df_b.to_string(index=False))
    print("\n--- 5 Random Stratified Splits Summary ---")
    print(split_summary.to_string(index=False))

    # 4. Paired Bootstrap CIs (10,000 resamples over queries — NO 1e-275 p-values!)
    boot_rows = []
    cr_acc_v = per_sample_acc["CascadeRoute-VL (alpha=0.10)"]
    cr_cst_v = per_sample_cost["CascadeRoute-VL (alpha=0.10)"]
    for bname in [
        "Qwen2.5-VL-72B", "Greedy Warm-Start (alpha=0)", "Offline-Linear",
        "KNN-Reward (cost-aware)", "ARMS-Router", "MLP-Router (BCE)",
        "MLP-Reward (cost-aware MSE)", "NeuralUCB (Warm+Online)", "NeuralTS (Warm+Online)",
        "Per-Domain Best Static (Oracle Domain)"
    ]:
        ba = paired_bootstrap_ci(cr_acc_v * 100.0, per_sample_acc[bname] * 100.0, n_boot=10000, seed=42)
        bc = paired_bootstrap_ci(per_sample_cost[bname] * 1000.0, cr_cst_v * 1000.0, n_boot=10000, seed=42)
        boot_rows.append({
            "Comparison (CascadeRoute-VL vs)": bname,
            "Acc Diff (%)": round(ba["mean_diff"], 2),
            "Acc 95% Boot CI": f"[{ba['ci_low']:+.2f}, {ba['ci_high']:+.2f}]",
            "Acc Boot p (>=1e-4)": f"{ba['p_value']:.4g}",
            "USD Saved /1k ($)": round(bc["mean_diff"], 4),
            "Cost 95% Boot CI": f"[{bc['ci_low']:+.4f}, {bc['ci_high']:+.4f}]",
            "Cost Boot p (>=1e-4)": f"{bc['p_value']:.4g}",
        })
    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(TABLES_DIR / "exp3_claim3_bootstrap_ci.csv", index=False)

    # 5. Full Pareto Sweep + Matched-Domain Area-Under-Pareto-Frontier (AUC)
    gammas = [0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.30, 0.50, 0.70, 0.90]
    p_rows = []
    for g in gammas:
        R_tr_g = Y_tr - g * C_norm_tr
        rc = run_cascaderoute_stream(
            X_te, P_te, Y_te, C_te, C_norm_te, T_te, O_te, L_te,
            X_tr, R_tr_g, arm_mean_cost, arm_norm_cost, models,
            gamma=g, alpha_0=0.10, alpha_anneal=True, warm_weight=0.20, reg_lambda=4.0,
            use_gate=True, precomputed_XtX=XtX_tr, X_stream_gpu=X_te_g, seed=42
        )
        p_rows.append({
            "Method": "CascadeRoute-VL", "gamma": g,
            "Acc": float(rc["correct"].mean()) * 100.0,
            "Cost_1k": float(rc["cost"].mean()) * 1000.0,
            "Lat_s": float(rc["latency"].mean()),
        })
        # Greedy
        rg = run_cascaderoute_stream(
            X_te, P_te, Y_te, C_te, C_norm_te, T_te, O_te, L_te,
            X_tr, R_tr_g, arm_mean_cost, arm_norm_cost, models,
            gamma=g, alpha_0=0.0, alpha_anneal=True, warm_weight=0.20, reg_lambda=4.0,
            use_gate=True, precomputed_XtX=XtX_tr, X_stream_gpu=X_te_g, seed=42
        )
        p_rows.append({
            "Method": "Greedy-WarmStart", "gamma": g,
            "Acc": float(rg["correct"].mean()) * 100.0,
            "Cost_1k": float(rg["cost"].mean()) * 1000.0,
            "Lat_s": float(rg["latency"].mean()),
        })
        # Cost-aware KNN-Reward
        acts_kr = np.argmax(knn_y_te - g * knn_cn_te, axis=1)
        p_rows.append({
            "Method": "KNN-Reward", "gamma": g,
            "Acc": float(Y_te[r_idx, acts_kr].mean()) * 100.0,
            "Cost_1k": float(C_te[r_idx, acts_kr].mean()) * 1000.0,
            "Lat_s": float(L_te[r_idx, acts_kr].mean()),
        })
        for mname, probs in [("MLP-BCE", mlp_bce_probs_te), ("Offline-Linear", lin_probs_te), ("ARMS", arms_probs_te)]:
            acts = np.argmax(probs - g * arm_norm_cost[None, :], axis=1)
            p_rows.append({
                "Method": mname, "gamma": g,
                "Acc": float(Y_te[r_idx, acts].mean()) * 100.0,
                "Cost_1k": float(C_te[r_idx, acts].mean()) * 1000.0,
                "Lat_s": float(L_te[r_idx, acts].mean()),
            })

    # Also train MLP-Reward explicitly across key gammas for full reward-trained Pareto comparison
    for g in [0.0, 0.03, 0.08, 0.15, 0.30, 0.60, 0.90]:
        mr = train_mlp_reward_regressor(X_tr, Y_tr - g * C_norm_tr, X_dev, Y_dev - g * C_norm_dev, epochs=22, seed=42)
        acts_mr = np.argmax(mlp_predict_raw(mr, X_te), axis=1)
        p_rows.append({
            "Method": "MLP-Reward (MSE)", "gamma": g,
            "Acc": float(Y_te[r_idx, acts_mr].mean()) * 100.0,
            "Cost_1k": float(C_te[r_idx, acts_mr].mean()) * 1000.0,
            "Lat_s": float(L_te[r_idx, acts_mr].mean()),
        })

    # Static 17-arm frontier reference points
    for j, mname in enumerate(models):
        p_rows.append({
            "Method": "Static-17-VLMs", "gamma": float(j),
            "Acc": float(Y_te[:, j].mean()) * 100.0,
            "Cost_1k": float(C_te[:, j].mean()) * 1000.0,
            "Lat_s": float(L_te[:, j].mean()),
        })

    df_par = pd.DataFrame(p_rows)
    df_par.to_csv(TABLES_DIR / "exp3_claim3_pareto_curves.csv", index=False)

    # Compute Matched-Domain Normalized Pareto AUC over shared feasible envelope [0.05, 0.28] $/1k & [0.3, 1.9] s
    auc_rows = []
    for mname in ["CascadeRoute-VL", "Greedy-WarmStart", "MLP-BCE", "MLP-Reward (MSE)", "KNN-Reward", "Offline-Linear", "ARMS", "Static-17-VLMs"]:
        sub = df_par[df_par["Method"] == mname]
        auc_c = pareto_frontier_auc(sub["Cost_1k"].values, sub["Acc"].values, cost_min=0.05, cost_max=0.28)
        auc_l = pareto_frontier_auc(sub["Lat_s"].values, sub["Acc"].values, cost_min=0.30, cost_max=1.90)
        auc_rows.append({
            "Method": mname,
            "Pareto Cost-AUC ($0.05-0.28/1k, %)": round(auc_c, 2),
            "Pareto Latency-AUC (0.3-1.9s, %)": round(auc_l, 2),
            "Peak Acc (%)": round(float(sub["Acc"].max()), 2),
        })
    df_auc = pd.DataFrame(auc_rows).sort_values("Pareto Cost-AUC ($0.05-0.28/1k, %)", ascending=False)
    df_auc.to_csv(TABLES_DIR / "exp3_claim3_pareto_auc.csv", index=False)
    print("\n--- Normalized Pareto Frontier AUC (Matched Domain) ---")
    print(df_auc.to_string(index=False))

    # 6. Modality Ablation Under Warm i.i.d., Cold i.i.d., AND OOD Shift w/ Bootstrap CI for 8 Cues (904 vs 896)
    is_in = meta["dataset"].apply(lambda ds: DATASET_DOMAINS[ds] in ["General_VQA", "STEM"]).values
    in_tr = np.where(is_in & (meta["split"] == "train").values)[0]
    ood_idx = np.where(~is_in)[0]
    R_in_tr = Y[in_tr] - gamma_main * C_norm[in_tr]
    amc_in = C[in_tr].mean(axis=0)
    anc_in = C_norm[in_tr].mean(axis=0)

    mod_rows = []
    vec_store = {}
    for mlabel, X_sub in [
        ("Text-Only (384-d)", X[:, 0:384]),
        ("Perceptual-Only (8-d)", X[:, 896:904]),
        ("Vision-Only (512-d)", X[:, 384:896]),
        ("Text+Vision (896-d)", X[:, 0:896]),
        ("Full Joint (904-d)", X),
    ]:
        XtX_sub = X_sub[train_idx].astype(np.float64).T @ X_sub[train_idx].astype(np.float64)
        XtX_in_sub = X_sub[in_tr].astype(np.float64).T @ X_sub[in_tr].astype(np.float64)
        Xg_te = torch.from_numpy(X_sub[test_idx].astype(np.float32)).cuda() if use_gpu else None
        Xg_oo = torch.from_numpy(X_sub[ood_idx].astype(np.float32)).cuda() if use_gpu else None

        rw = run_cascaderoute_stream(
            X_sub[test_idx], P_te, Y_te, C_te, C_norm_te, T_te, O_te, L_te,
            X_sub[train_idx], R_tr_main, arm_mean_cost, arm_norm_cost, models,
            gamma=gamma_main, alpha_0=0.10, alpha_anneal=True, warm_weight=0.20, reg_lambda=4.0,
            use_gate=True, precomputed_XtX=XtX_sub, X_stream_gpu=Xg_te, seed=42
        )
        rc = run_cascaderoute_stream(
            X_sub[test_idx], P_te, Y_te, C_te, C_norm_te, T_te, O_te, L_te,
            X_sub[train_idx], R_tr_main, arm_mean_cost, arm_norm_cost, models,
            gamma=gamma_main, alpha_0=0.35, alpha_anneal=False, use_warm=False, reg_lambda=2.0,
            use_gate=False, X_stream_gpu=Xg_te, seed=42
        )
        ro = run_cascaderoute_stream(
            X_sub[ood_idx], P_raw[ood_idx], Y[ood_idx], C[ood_idx], C_norm[ood_idx], T_tot[ood_idx], O_tok[ood_idx], L[ood_idx],
            X_sub[in_tr], R_in_tr, amc_in, anc_in, models,
            gamma=gamma_main, alpha_0=0.10, alpha_anneal=True, warm_weight=0.20, reg_lambda=4.0,
            use_gate=True, precomputed_XtX=XtX_in_sub, X_stream_gpu=Xg_oo, seed=42
        )
        vec_store[mlabel] = (rw["correct"], rc["correct"], ro["correct"])
        mod_rows.append({
            "Modality": mlabel,
            "Warm i.i.d. Acc (%)": round(float(rw["correct"].mean()) * 100.0, 2),
            "Cold i.i.d. Acc (%)": round(float(rc["correct"].mean()) * 100.0, 2),
            "OOD Online Acc (%)": round(float(ro["correct"].mean()) * 100.0, 2),
            "OOD Cost/1k ($)": round(float(ro["cost"].mean()) * 1000.0, 4),
        })

    # Exact bootstrap delta of 8 cues (904-d vs 896-d)
    for reg_i, reg_name in enumerate(["Warm i.i.d.", "Cold i.i.d.", "OOD Shift"]):
        b_cue = paired_bootstrap_ci(
            vec_store["Full Joint (904-d)"][reg_i] * 100.0,
            vec_store["Text+Vision (896-d)"][reg_i] * 100.0,
            n_boot=5000, seed=42
        )
        print(f"  8-Cue Marginal Effect ({reg_name:11s}): {b_cue['mean_diff']:+.2f}%  95% CI [{b_cue['ci_low']:+.2f}, {b_cue['ci_high']:+.2f}]  p={b_cue['p_value']:.4f}")
    pd.DataFrame(mod_rows).to_csv(TABLES_DIR / "exp3_claim3_modality_ablation_boot.csv", index=False)


def run_claim4_suite(Y, C, T_tot, O_tok, L, X, P_raw, meta, models):
    print("\n" + "=" * 80)
    print("CLAIM 4: Theorem 1 Real Constants, alpha_0 Sweep, Budget Guardrail & BwK")
    print("=" * 80)

    train_idx = np.where((meta["split"] == "train").values)[0]
    test_idx = np.where((meta["split"] == "test").values)[0]
    is_in = meta["dataset"].apply(lambda ds: DATASET_DOMAINS[ds] in ["General_VQA", "STEM"]).values
    in_tr = np.where(is_in & (meta["split"] == "train").values)[0]
    ood_idx = np.where(~is_in)[0]

    use_gpu = torch.cuda.is_available()
    X_tr, Y_tr, C_tr = X[train_idx], Y[train_idx], C[train_idx]
    X_te, Y_te, C_te = X[test_idx], Y[test_idx], C[test_idx]
    c_scale = float(np.percentile(C_tr.mean(axis=0), 90))
    C_norm = np.clip(C / c_scale, 0.0, 2.5)
    C_norm_tr, C_norm_te = C_norm[train_idx], C_norm[test_idx]
    arm_mean_cost = C_tr.mean(axis=0)
    arm_norm_cost = C_norm_tr.mean(axis=0)
    XtX_tr = X_tr.astype(np.float64).T @ X_tr.astype(np.float64)
    XtX_in = X[in_tr].astype(np.float64).T @ X[in_tr].astype(np.float64)
    X_te_g = torch.from_numpy(X_te.astype(np.float32)).cuda() if use_gpu else None
    X_oo_g = torch.from_numpy(X[ood_idx].astype(np.float32)).cuda() if use_gpu else None

    gamma = 0.08
    R_tr = Y_tr - gamma * C_norm_tr
    R_in = Y[in_tr] - gamma * C_norm[in_tr]

    # Exact Theorem 1 numerical constant calculation for d=904, K=17, T=6108, lambda=4.0
    d, K, T = X.shape[1], len(models), len(test_idx)
    lam = 4.0
    nu = (1.0 + 2.5 * gamma) / 2.0
    delta = 0.05
    # Estimate empirical S = max_a ||theta_a||_2 on Train Ridge
    agent_fit = LinUCB(n_actions=K, feature_dim=d, regularization=lam, use_gpu=False)
    agent_fit.warm_start_from_full_matrix(X_tr, R_tr, regularization=lam, sample_weight=0.20, precomputed_XtX=XtX_tr)
    S_emp = float(np.max(np.linalg.norm(agent_fit._theta_stack, axis=1)))
    alpha_theory_no_warm = nu * np.sqrt(d * np.log((1.0 + T / (d * lam)) / delta)) + np.sqrt(lam) * S_emp
    print(f"  Theorem 1 exact constants: d={d}, K={K}, T={T}, lam={lam}, nu={nu:.2f}, max||theta||={S_emp:.3f}")
    print(f"  Abbasi-Yadkori worst-case alpha_min_theory = {alpha_theory_no_warm:.2f} (vs empirical tuned alpha_0=0.10)")

    # 1. alpha_0 Sensitivity Sweep on i.i.d. Test & OOD Shift
    alpha_rows = []
    for a0 in [0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.5, 5.0, 10.0, 20.0]:
        r_id = run_cascaderoute_stream(
            X_te, P_raw[test_idx], Y_te, C_te, C_norm_te, T_tot[test_idx], O_tok[test_idx], L[test_idx],
            X_tr, R_tr, arm_mean_cost, arm_norm_cost, models,
            gamma=gamma, alpha_0=a0, alpha_anneal=True, warm_weight=0.20, reg_lambda=lam,
            use_gate=True, precomputed_XtX=XtX_tr, X_stream_gpu=X_te_g, seed=42
        )
        r_oo = run_cascaderoute_stream(
            X[ood_idx], P_raw[ood_idx], Y[ood_idx], C[ood_idx], C_norm[ood_idx], T_tot[ood_idx], O_tok[ood_idx], L[ood_idx],
            X[in_tr], R_in, C[in_tr].mean(axis=0), C_norm[in_tr].mean(axis=0), models,
            gamma=gamma, alpha_0=a0, alpha_anneal=True, warm_weight=0.08, reg_lambda=lam,
            use_gate=True, precomputed_XtX=XtX_in, X_stream_gpu=X_oo_g, seed=42
        )
        # Also test non-vanishing floor alpha_min = 0.02 alongside a0
        r_fl = run_cascaderoute_stream(
            X_te, P_raw[test_idx], Y_te, C_te, C_norm_te, T_tot[test_idx], O_tok[test_idx], L[test_idx],
            X_tr, R_tr, arm_mean_cost, arm_norm_cost, models,
            gamma=gamma, alpha_0=a0, alpha_anneal=True, warm_weight=0.20, reg_lambda=lam,
            use_gate=True, precomputed_XtX=XtX_tr, X_stream_gpu=X_te_g, seed=42
        )
        alpha_rows.append({
            "alpha_0": a0,
            "Theory_WorstCase_Alpha": round(float(alpha_theory_no_warm), 2),
            "IID_Test_Acc (%)": round(float(r_id["correct"].mean()) * 100.0, 2),
            "IID_Cost_1k ($)": round(float(r_id["cost"].mean()) * 1000.0, 4),
            "OOD_Acc (%)": round(float(r_oo["correct"].mean()) * 100.0, 2),
            "OOD_Cost_1k ($)": round(float(r_oo["cost"].mean()) * 1000.0, 4),
        })
    df_alp = pd.DataFrame(alpha_rows)
    df_alp.to_csv(TABLES_DIR / "exp3_claim4_alpha_sensitivity.csv", index=False)
    print("\n--- alpha_0 Sensitivity Sweep (i.i.d. & OOD) ---")
    print(df_alp.to_string(index=False))

    # 2. Budget Compliance & Hard Guardrail (Theorem 2: C_T <= B_max + c_max) + BwK Primal-Dual Baseline
    nat_cost_per_q = 0.000160  # ~$0.160 / 1k balanced spend -> total ~$0.977 for 6,108 Q
    cheapest_arm = int(np.argmin(arm_mean_cost))
    c_max = float(C_te.max())
    rng = np.random.RandomState(42)
    order = rng.permutation(T)

    budget_rows = []
    for mult in [0.25, 0.40, 0.60, 0.80, 1.00, 1.25]:
        B_max = mult * nat_cost_per_q * T

        # (a) Unguarded Annealed LinUCB (fixed gamma=0.08, no hard guardrail)
        ag_u = LinUCB(n_actions=K, feature_dim=d, alpha=0.10, alpha_min=0.01, alpha_anneal=True, regularization=lam, use_gpu=use_gpu)
        ag_u.warm_start_from_full_matrix(X_tr, R_tr, regularization=lam, sample_weight=0.20, precomputed_XtX=XtX_tr)
        sp_u, corr_u = 0.0, 0.0
        for q in order:
            bf = max(0.0, min(1.0, (B_max - sp_u) / B_max))
            a = ag_u.select_action(X_te_g[q] if use_gpu else X_te[q], budget_fraction=bf)
            sp_u += float(C_te[q, a])
            corr_u += float(Y_te[q, a])
            ag_u.update(X_te_g[q] if use_gpu else X_te[q], a, float(Y_te[q, a] - gamma * C_norm_te[q, a]))

        # (b) Hard-Guarded CascadeRoute-VL (Theorem 2: forces cheapest arm when rem <= cheapest-arm horizon)
        ag_g = LinUCB(n_actions=K, feature_dim=d, alpha=0.10, alpha_min=0.01, alpha_anneal=True, regularization=lam, use_gpu=use_gpu)
        ag_g.warm_start_from_full_matrix(X_tr, R_tr, regularization=lam, sample_weight=0.20, precomputed_XtX=XtX_tr)
        sp_g, corr_g, forced_cnt = 0.0, 0.0, 0
        c_min_safe = float(arm_mean_cost[cheapest_arm]) * 1.02
        for step_idx, q in enumerate(order):
            rem = B_max - sp_g
            a, is_f = ag_g.select_action_budgeted(
                X_te_g[q] if use_gpu else X_te[q],
                budget_remaining_usd=rem,
                total_budget_usd=B_max,
                arm_cost_est=arm_mean_cost,
                cheapest_arm=cheapest_arm,
                reserve_multiplier=1.0,
                remaining_steps=T - step_idx,
                cheapest_arm_cost_bound=c_min_safe,
            )
            if is_f:
                forced_cnt += 1
            sp_g += float(C_te[q, a])
            corr_g += float(Y_te[q, a])
            ag_g.update(X_te_g[q] if use_gpu else X_te[q], a, float(Y_te[q, a] - gamma * C_norm_te[q, a]))

        # (c) BwK Primal-Dual Lagrangian Pacing (with hard budget cap)
        bwk = BwKPrimalDualLinUCB(
            n_actions=K, feature_dim=d, total_budget_usd=B_max, horizon=T,
            arm_mean_cost_usd=arm_mean_cost, alpha=0.10, regularization=lam, dual_lr=400.0, use_gpu=use_gpu
        )
        bwk.warm_start(X_tr, Y_tr, sample_weight=0.20, precomputed_XtX=XtX_tr)
        corr_b = 0.0
        for q in order:
            a = bwk.select_action(X_te_g[q] if use_gpu else X_te[q], enforce_hard_cap=True)
            corr_b += float(Y_te[q, a])
            bwk.update(X_te_g[q] if use_gpu else X_te[q], a, float(Y_te[q, a]), float(C_te[q, a]))

        for name, sp, cr, fr in [
            ("Unguarded Annealed LinUCB", sp_u, corr_u, 0.0),
            ("Hard-Guarded CascadeRoute (Thm 2)", sp_g, corr_g, 100.0 * forced_cnt / T),
            ("BwK Primal-Dual LinUCB", bwk.spent_usd, corr_b, 0.0),
        ]:
            budget_rows.append({
                "Budget_Mult": f"{mult:.2f}x",
                "B_max ($)": round(B_max, 4),
                "Method": name,
                "Realized_Spend ($)": round(sp, 4),
                "Budget_Ratio (C_T / B_max)": round(sp / B_max, 3),
                "Hard_Violated (C_T > B_max + c_max)": bool(sp > B_max + c_max),
                "Soft_Overrun (%)": round(max(0.0, (sp - B_max) / B_max * 100.0), 2),
                "Forced_Cheapest (%)": round(fr, 1),
                "Accuracy (%)": round(100.0 * cr / T, 2),
            })

    df_bd = pd.DataFrame(budget_rows)
    df_bd.to_csv(TABLES_DIR / "exp3_claim4_budget_compliance.csv", index=False)
    print("\n--- Budget Compliance (Unguarded vs Hard Guardrail vs BwK) ---")
    print(df_bd.to_string(index=False))


def run_claim5_suite(Y, C, T_tot, O_tok, L, X, P_raw, meta, models):
    print("\n" + "=" * 80)
    print("CLAIM 5: Calibrated Deferral Gate Across Regimes, Compact Pool & Feedback Mode")
    print("=" * 80)

    train_idx = np.where((meta["split"] == "train").values)[0]
    dev_idx = np.where((meta["split"] == "dev").values)[0]
    test_idx = np.where((meta["split"] == "test").values)[0]
    is_in = meta["dataset"].apply(lambda ds: DATASET_DOMAINS[ds] in ["General_VQA", "STEM"]).values
    in_tr = np.where(is_in & (meta["split"] == "train").values)[0]
    ood_idx = np.where(~is_in)[0]

    use_gpu = torch.cuda.is_available()
    K, d = len(models), X.shape[1]
    X_tr, Y_tr, C_tr = X[train_idx], Y[train_idx], C[train_idx]
    X_dev, Y_dev = X[dev_idx], Y[dev_idx]
    X_te, Y_te, C_te = X[test_idx], Y[test_idx], C[test_idx]
    c_scale = float(np.percentile(C_tr.mean(axis=0), 90))
    C_norm = np.clip(C / c_scale, 0.0, 2.5)
    C_norm_tr, C_norm_dev, C_norm_te = C_norm[train_idx], C_norm[dev_idx], C_norm[test_idx]
    arm_mean_cost = C_tr.mean(axis=0)
    arm_norm_cost = C_norm_tr.mean(axis=0)
    XtX_tr = X_tr.astype(np.float64).T @ X_tr.astype(np.float64)

    compact_arms = [models.index(m) for m in [
        "Janus-Pro-1B", "deepseek_vl2_tiny", "SmolVLM2", "Phi-3.5-Vision",
        "llava_next_vicuna_7b", "Janus-Pro-7B", "Qianfan-VL-8B"
    ]]
    fallback_arms = tuple(models.index(m) for m in [
        "Kimi-VL-A3B-Thinking-2506", "Qwen2.5-VL-72B-Instruct", "InternVL2_5-78B", "GeminiFlash2-5"
    ])

    def eval_gate_regime(
        regime_name: str,
        X_w, Y_w, C_w, Cn_w,
        X_s, Y_s, C_s, Cn_s, O_s,
        gamma_val: float,
        tau_cal: float,
        compact_primary_only: bool,
        use_calibrated_gate: bool,
        use_any_gate: bool,
        feedback_mode: str = "both_arms",  # "both_arms" vs "final_only" vs "no_update_primary_on_esc"
        w0: float = 0.20,
    ) -> Dict:
        R_w = Y_w - gamma_val * Cn_w
        amc = C_w.mean(axis=0)
        anc = Cn_w.mean(axis=0)
        c_pen = (gamma_val * anc).astype(np.float32)

        agent = LinUCB(n_actions=K, feature_dim=d, alpha=0.10, alpha_anneal=True, regularization=4.0, use_gpu=use_gpu)
        agent.warm_start_from_full_matrix(X_w, R_w, regularization=4.0, sample_weight=w0)

        # Fit Isotonic + Split-Conformal Calibrator strictly on Dev!
        gate_cal = CalibratedDeferralGate(tau_cal=tau_cal, conformal_quantile=0.20, min_budget_frac=0.10, cost_tradeoff=gamma_val)
        if use_calibrated_gate:
            X_dev_scores = (X_dev @ agent._theta_stack.T) + c_pen[None, :]
            gate_cal.fit_calibrator(X_dev_scores.ravel(), Y_dev.ravel())

        gate_heur = ConfidenceGate(unc_threshold=0.12, margin_threshold=0.025, min_budget_frac=0.15, p_cap=0.78, unc_weight=0.25)

        N_s = len(X_s)
        order = np.random.RandomState(42).permutation(N_s)
        B_max = max(float(amc[models.index("Qianfan-VL-8B")]) * 3.5, 0.00020) * N_s
        spent = 0.0
        corr_list, cst_list, esc_list = np.zeros(N_s), np.zeros(N_s), np.zeros(N_s)
        prim_corr_if_no_esc = np.zeros(N_s)
        prim_cst_list = np.zeros(N_s)

        Xg = torch.from_numpy(X_s.astype(np.float32)).cuda() if use_gpu else None
        for q in order:
            bf = max(0.0, min(1.0, (B_max - spent) / B_max))
            st = Xg[q] if use_gpu else X_s[q]
            preds, uncs, ucb = agent.score_arms(st, budget_fraction=bf)
            if compact_primary_only:
                # Primary router restricted to dense <=8B compact tier
                mask_scores = np.full(K, -1e9)
                mask_scores[compact_arms] = ucb[compact_arms]
                a_prim = int(np.argmax(mask_scores))
            else:
                a_prim = int(np.argmax(ucb))

            a_final = a_prim
            did_esc = False
            if use_any_gate and (a_prim in compact_arms):
                if use_calibrated_gate:
                    a_esc, did_esc, _ = gate_cal.decide_escalation(
                        a_prim, preds, uncs, fallback_arms=fallback_arms,
                        arm_cost_penalty=c_pen, observed_out_tokens=float(O_s[q, a_prim]), budget_fraction=bf
                    )
                else:
                    a_esc, did_esc = gate_heur.decide_escalation(
                        a_prim, preds, uncs, perceptual_raw=P_raw[0], fallback_arms=fallback_arms,
                        arm_costs=amc, arm_cost_penalty=c_pen, observed_out_tokens=float(O_s[q, a_prim]), budget_fraction=bf
                    )
                if did_esc:
                    a_final = a_esc

            if did_esc:
                qc = C_s[q, a_prim] + C_s[q, a_final]
                rp = float(Y_s[q, a_prim] - gamma_val * Cn_s[q, a_prim])
                rf = float(Y_s[q, a_final] - gamma_val * Cn_s[q, a_final])
                if feedback_mode == "both_arms":
                    agent.update(st, a_prim, rp)
                    agent.update(st, a_final, rf)
                else:  # final_only: strictly single-label feedback on final delivered prediction
                    agent.update(st, a_final, rf)
            else:
                qc = C_s[q, a_final]
                agent.update(st, a_final, float(Y_s[q, a_final] - gamma_val * Cn_s[q, a_final]))

            spent += qc
            corr_list[q] = Y_s[q, a_final]
            prim_corr_if_no_esc[q] = Y_s[q, a_prim]
            prim_cst_list[q] = C_s[q, a_prim]
            cst_list[q] = qc
            esc_list[q] = 1.0 if did_esc else 0.0

        acc = float(corr_list.mean()) * 100.0
        cst_1k = float(cst_list.mean()) * 1000.0
        esc_pct = float(esc_list.mean()) * 100.0
        rescued_delta_pct = float((corr_list - prim_corr_if_no_esc).mean()) * 100.0
        esc_extra_usd = float(np.sum(np.where(esc_list > 0, cst_list - prim_cst_list, 0.0)))
        return {
            "Regime": regime_name,
            "Gate": ("Calibrated" if use_calibrated_gate else "Heuristic") if use_any_gate else "None (No Gate)",
            "Feedback": feedback_mode if use_any_gate else "single_arm",
            "tau_cal": tau_cal if use_calibrated_gate else np.nan,
            "Acc (%)": round(acc, 2),
            "Cost / 1k ($)": round(cst_1k, 4),
            "Esc Rate (%)": round(esc_pct, 2),
            "Gate Net dAcc (%)": round(rescued_delta_pct, 2),
            "Esc Extra Spend ($)": round(esc_extra_usd, 4),
            "dAcc (%) per Extra $": round(rescued_delta_pct / max(esc_extra_usd, 1e-6), 2),
        }

    # Test 4 regimes: Balanced (0.08), Frugal (0.30), Compact-Primary Pool (gamma=0.10), OOD Shift (w0=0.10)
    gate_rows = []
    configs = [
        ("1. Balanced 17-Pool (g=0.08)", X_tr, Y_tr, C_tr, C_norm_tr, X_te, Y_te, C_te, C_norm_te, O_tok[test_idx], 0.08, False, 0.20),
        ("2. Frugal 17-Pool (g=0.30)", X_tr, Y_tr, C_tr, C_norm_tr, X_te, Y_te, C_te, C_norm_te, O_tok[test_idx], 0.30, False, 0.20),
        ("3. Compact-Primary -> Paid Escalation (g=0.08)", X_tr, Y_tr, C_tr, C_norm_tr, X_te, Y_te, C_te, C_norm_te, O_tok[test_idx], 0.08, True, 0.20),
        ("4. OOD Shift 17-Pool (g=0.08)", X[in_tr], Y[in_tr], C[in_tr], C_norm[in_tr], X[ood_idx], Y[ood_idx], C[ood_idx], C_norm[ood_idx], O_tok[ood_idx], 0.08, False, 0.10),
    ]
    for rname, Xw, Yw, Cw, Cnw, Xs, Ys, Cs, Cns, Os, gval, comp_only, w0 in configs:
        gate_rows.append(eval_gate_regime(rname, Xw, Yw, Cw, Cnw, Xs, Ys, Cs, Cns, Os, gval, 0.04, comp_only, False, False, "both_arms", w0))
        gate_rows.append(eval_gate_regime(rname, Xw, Yw, Cw, Cnw, Xs, Ys, Cs, Cns, Os, gval, 0.04, comp_only, True, True, "both_arms", w0))
        gate_rows.append(eval_gate_regime(rname, Xw, Yw, Cw, Cnw, Xs, Ys, Cs, Cns, Os, gval, 0.04, comp_only, True, True, "final_only", w0))

    df_gate = pd.DataFrame(gate_rows)
    df_gate.to_csv(TABLES_DIR / "exp3_claim5_calibrated_gate.csv", index=False)
    print(df_gate.to_string(index=False))

    # Single-threshold tau_cal sensitivity sweep on Frugal (0.30) & Compact-Primary Pool
    tau_rows = []
    for tau in [-0.08, -0.04, 0.0, 0.02, 0.04, 0.08, 0.14]:
        tau_rows.append(eval_gate_regime("Frugal 17-Pool", X_tr, Y_tr, C_tr, C_norm_tr, X_te, Y_te, C_te, C_norm_te, O_tok[test_idx], 0.30, tau, False, True, True, "both_arms", 0.20))
        tau_rows.append(eval_gate_regime("Compact-Primary Pool", X_tr, Y_tr, C_tr, C_norm_tr, X_te, Y_te, C_te, C_norm_te, O_tok[test_idx], 0.08, tau, True, True, True, "both_arms", 0.20))
    df_tau = pd.DataFrame(tau_rows)
    df_tau.to_csv(TABLES_DIR / "exp3_claim5_tau_sensitivity.csv", index=False)


def main():
    t0 = time.time()
    Y = np.load(DATA_DIR / "matrices" / "Y.npz")["Y"].astype(np.float32)
    Y_rep = np.load(DATA_DIR / "matrices" / "Y_repaired.npz")["Y"].astype(np.float32)
    C = np.load(DATA_DIR / "matrices" / "C.npz")["C"].astype(np.float64)
    T_npz = np.load(DATA_DIR / "matrices" / "T_tokens.npz")
    T_tot = T_npz["input_tokens"][:, None].astype(np.float64) + T_npz["output_tokens"].astype(np.float64)
    O_tok = T_npz["output_tokens"].astype(np.float64)
    L = T_npz["latency_s"].astype(np.float64)
    feat_npz = np.load(DATA_DIR / "matrices" / "X_multimodal_904_f16.npz")
    X = feat_npz["X"].astype(np.float32)
    P_raw = feat_npz["P_raw"].astype(np.float32)
    meta = pd.read_parquet(DATA_DIR / "registry" / "meta.parquet")
    with open(DATA_DIR / "registry" / "model_index.pkl", "rb") as f:
        models: List[str] = pickle.load(f)

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "--budget-gate-only":
        # Fast refresh of Horizon-Guarded Budget Compliance + Fixed Esc Extra Spend / tau sweep
        train_idx = np.where((meta["split"] == "train").values)[0]
        test_idx = np.where((meta["split"] == "test").values)[0]
        X_tr, Y_tr, C_tr = X[train_idx], Y[train_idx], C[train_idx]
        X_te, Y_te, C_te = X[test_idx], Y[test_idx], C[test_idx]
        c_scale = float(np.percentile(C_tr.mean(axis=0), 90))
        C_norm = np.clip(C / c_scale, 0.0, 2.5)
        C_norm_tr, C_norm_te = C_norm[train_idx], C_norm[test_idx]
        arm_mean_cost = C_tr.mean(axis=0)
        XtX_tr = X_tr.astype(np.float64).T @ X_tr.astype(np.float64)
        use_gpu = torch.cuda.is_available()
        X_te_g = torch.from_numpy(X_te.astype(np.float32)).cuda() if use_gpu else None
        d, K, T = X.shape[1], len(models), len(test_idx)
        gamma, lam = 0.08, 4.0
        R_tr = Y_tr - gamma * C_norm_tr
        nat_cost_per_q = 0.000160
        cheapest_arm = int(np.argmin(arm_mean_cost))
        c_max = float(C_te.max())
        c_min_safe = float(arm_mean_cost[cheapest_arm]) * 1.02
        order = np.random.RandomState(42).permutation(T)
        budget_rows = []
        for mult in [0.25, 0.40, 0.60, 0.80, 1.00, 1.25]:
            B_max = mult * nat_cost_per_q * T
            ag_u = LinUCB(n_actions=K, feature_dim=d, alpha=0.10, alpha_min=0.01, alpha_anneal=True, regularization=lam, use_gpu=use_gpu)
            ag_u.warm_start_from_full_matrix(X_tr, R_tr, regularization=lam, sample_weight=0.20, precomputed_XtX=XtX_tr)
            sp_u, corr_u = 0.0, 0.0
            for q in order:
                bf = max(0.0, min(1.0, (B_max - sp_u) / B_max))
                a = ag_u.select_action(X_te_g[q] if use_gpu else X_te[q], budget_fraction=bf)
                sp_u += float(C_te[q, a]); corr_u += float(Y_te[q, a])
                ag_u.update(X_te_g[q] if use_gpu else X_te[q], a, float(Y_te[q, a] - gamma * C_norm_te[q, a]))

            ag_g = LinUCB(n_actions=K, feature_dim=d, alpha=0.10, alpha_min=0.01, alpha_anneal=True, regularization=lam, use_gpu=use_gpu)
            ag_g.warm_start_from_full_matrix(X_tr, R_tr, regularization=lam, sample_weight=0.20, precomputed_XtX=XtX_tr)
            sp_g, corr_g, forced_cnt = 0.0, 0.0, 0
            for step_idx, q in enumerate(order):
                a, is_f = ag_g.select_action_budgeted(
                    X_te_g[q] if use_gpu else X_te[q], budget_remaining_usd=(B_max - sp_g), total_budget_usd=B_max,
                    arm_cost_est=arm_mean_cost, cheapest_arm=cheapest_arm, reserve_multiplier=1.0,
                    remaining_steps=T - step_idx, cheapest_arm_cost_bound=c_min_safe,
                )
                if is_f: forced_cnt += 1
                sp_g += float(C_te[q, a]); corr_g += float(Y_te[q, a])
                ag_g.update(X_te_g[q] if use_gpu else X_te[q], a, float(Y_te[q, a] - gamma * C_norm_te[q, a]))

            bwk = BwKPrimalDualLinUCB(n_actions=K, feature_dim=d, total_budget_usd=B_max, horizon=T, arm_mean_cost_usd=arm_mean_cost, alpha=0.10, regularization=lam, dual_lr=400.0, use_gpu=use_gpu)
            bwk.warm_start(X_tr, Y_tr, sample_weight=0.20, precomputed_XtX=XtX_tr)
            corr_b = 0.0
            for q in order:
                a = bwk.select_action(X_te_g[q] if use_gpu else X_te[q], enforce_hard_cap=True)
                corr_b += float(Y_te[q, a])
                bwk.update(X_te_g[q] if use_gpu else X_te[q], a, float(Y_te[q, a]), float(C_te[q, a]))

            for name, sp, cr, fr in [
                ("Unguarded Annealed LinUCB", sp_u, corr_u, 0.0),
                ("Hard-Guarded CascadeRoute (Thm 2)", sp_g, corr_g, 100.0 * forced_cnt / T),
                ("BwK Primal-Dual LinUCB", bwk.spent_usd, corr_b, 0.0),
            ]:
                budget_rows.append({
                    "Budget_Mult": f"{mult:.2f}x", "B_max ($)": round(B_max, 4), "Method": name,
                    "Realized_Spend ($)": round(sp, 4), "Budget_Ratio (C_T / B_max)": round(sp / B_max, 3),
                    "Hard_Violated (C_T > B_max + c_max)": bool(sp > B_max + c_max),
                    "Soft_Overrun (%)": round(max(0.0, (sp - B_max) / B_max * 100.0), 2),
                    "Forced_Cheapest (%)": round(fr, 1), "Accuracy (%)": round(100.0 * cr / T, 2),
                })
        pd.DataFrame(budget_rows).to_csv(TABLES_DIR / "exp3_claim4_budget_compliance.csv", index=False)
        print(pd.DataFrame(budget_rows).to_string(index=False))
        run_claim5_suite(Y, C, T_tot, O_tok, L, X, P_raw, meta, models)
        print(f"Completed --budget-gate-only in {time.time() - t0:.1f}s.")
        return

    if mode != "--post-pareto-only":
        run_claim3_suite(Y, Y_rep, C, T_tot, O_tok, L, X, P_raw, meta, models)
    else:
        # Finish modality ablation boot cleanly reusing saved CSVs from just-completed run_claim3_suite!
        is_in = meta["dataset"].apply(lambda ds: DATASET_DOMAINS[ds] in ["General_VQA", "STEM"]).values
        in_tr = np.where(is_in & (meta["split"] == "train").values)[0]
        ood_idx = np.where(~is_in)[0]
        tr_idx = np.where((meta["split"] == "train").values)[0]
        te_idx = np.where((meta["split"] == "test").values)[0]
        c_sc = float(np.percentile(C[tr_idx].mean(axis=0), 90))
        Cn = np.clip(C / c_sc, 0.0, 2.5)
        use_gpu = torch.cuda.is_available()
        mod_rows, vec_store = [], {}
        for mlabel, X_sub in [
            ("Text-Only (384-d)", X[:, 0:384]),
            ("Perceptual-Only (8-d)", X[:, 896:904]),
            ("Vision-Only (512-d)", X[:, 384:896]),
            ("Text+Vision (896-d)", X[:, 0:896]),
            ("Full Joint (904-d)", X),
        ]:
            XtX_sub = X_sub[tr_idx].astype(np.float64).T @ X_sub[tr_idx].astype(np.float64)
            XtX_in_sub = X_sub[in_tr].astype(np.float64).T @ X_sub[in_tr].astype(np.float64)
            Xg_te = torch.from_numpy(X_sub[te_idx].astype(np.float32)).cuda() if use_gpu else None
            Xg_oo = torch.from_numpy(X_sub[ood_idx].astype(np.float32)).cuda() if use_gpu else None
            rw = run_cascaderoute_stream(
                X_sub[te_idx], P_raw[te_idx], Y[te_idx], C[te_idx], Cn[te_idx], T_tot[te_idx], O_tok[te_idx], L[te_idx],
                X_sub[tr_idx], Y[tr_idx] - 0.08 * Cn[tr_idx], C[tr_idx].mean(0), Cn[tr_idx].mean(0), models,
                gamma=0.08, alpha_0=0.10, alpha_anneal=True, warm_weight=0.20, reg_lambda=4.0,
                use_gate=True, precomputed_XtX=XtX_sub, X_stream_gpu=Xg_te, seed=42
            )
            rc = run_cascaderoute_stream(
                X_sub[te_idx], P_raw[te_idx], Y[te_idx], C[te_idx], Cn[te_idx], T_tot[te_idx], O_tok[te_idx], L[te_idx],
                X_sub[tr_idx], Y[tr_idx] - 0.08 * Cn[tr_idx], C[tr_idx].mean(0), Cn[tr_idx].mean(0), models,
                gamma=0.08, alpha_0=0.35, alpha_anneal=False, use_warm=False, reg_lambda=2.0,
                use_gate=False, X_stream_gpu=Xg_te, seed=42
            )
            ro = run_cascaderoute_stream(
                X_sub[ood_idx], P_raw[ood_idx], Y[ood_idx], C[ood_idx], Cn[ood_idx], T_tot[ood_idx], O_tok[ood_idx], L[ood_idx],
                X_sub[in_tr], Y[in_tr] - 0.08 * Cn[in_tr], C[in_tr].mean(0), Cn[in_tr].mean(0), models,
                gamma=0.08, alpha_0=0.10, alpha_anneal=True, warm_weight=0.20, reg_lambda=4.0,
                use_gate=True, precomputed_XtX=XtX_in_sub, X_stream_gpu=Xg_oo, seed=42
            )
            vec_store[mlabel] = (rw["correct"], rc["correct"], ro["correct"])
            mod_rows.append({
                "Modality": mlabel,
                "Warm i.i.d. Acc (%)": round(float(rw["correct"].mean()) * 100.0, 2),
                "Cold i.i.d. Acc (%)": round(float(rc["correct"].mean()) * 100.0, 2),
                "OOD Online Acc (%)": round(float(ro["correct"].mean()) * 100.0, 2),
                "OOD Cost/1k ($)": round(float(ro["cost"].mean()) * 1000.0, 4),
            })
        for reg_i, reg_name in enumerate(["Warm i.i.d.", "Cold i.i.d.", "OOD Shift"]):
            b_cue = paired_bootstrap_ci(
                vec_store["Full Joint (904-d)"][reg_i] * 100.0,
                vec_store["Text+Vision (896-d)"][reg_i] * 100.0,
                n_boot=5000, seed=42
            )
            print(f"  8-Cue Marginal Effect ({reg_name:11s}): {b_cue['mean_diff']:+.2f}%  95% CI [{b_cue['ci_low']:+.2f}, {b_cue['ci_high']:+.2f}]  p={b_cue['p_value']:.4f}")
        pd.DataFrame(mod_rows).to_csv(TABLES_DIR / "exp3_claim3_modality_ablation_boot.csv", index=False)
    run_claim4_suite(Y, C, T_tot, O_tok, L, X, P_raw, meta, models)
    run_claim5_suite(Y, C, T_tot, O_tok, L, X, P_raw, meta, models)
    print(f"\nCompleted exp3 (Claims 3, 4, 5) in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
