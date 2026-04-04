"""
End-to-end evaluation of the two-stage privacy detection pipeline.

Ground truth: PII tokens actually appeared in output (pii_token_positions > 0)
    Positive: A_located
    Negative: A_not_located + B + C

Train/test split (80/20, seed=42) is applied at the SAMPLE level before
extracting hidden states, preventing data leakage.

    Probe trains on: train split of (A_located + A_not_located + B)
    Probe evaluates on: test split of (A_located + A_not_located + B) + all of C

Usage:
    python evaluate_pipeline.py --results-dir results_llama8b_20260401_1429
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate as cv_fn

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--test-ratio",  type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def load_condition(results_dir, condition):
    path = os.path.join(results_dir, f"{condition}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def split_samples(samples, test_ratio=0.2, seed=42):
    """Sample-level train/test split. Returns (train, test)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_test = max(1, int(len(samples) * test_ratio))
    test_set  = set(idx[:n_test])
    train = [s for i, s in enumerate(samples) if i not in test_set]
    test  = [s for i, s in enumerate(samples) if i in test_set]
    return train, test


def build_probe(pos_train, neg_train, method):
    """Train Linear Probe on train split only."""
    hs_key = "red_flag_hidden_states" if method == "m1" else "sustained_hidden_states"
    X_list, y_list = [], []
    for r in pos_train:
        for hs in getattr(r, hs_key):
            X_list.append(hs); y_list.append(1)
    for r in neg_train:
        for hs in getattr(r, hs_key):
            X_list.append(hs); y_list.append(0)

    if not X_list:
        raise ValueError("No hidden states in training split.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    print(f"  Train hidden states: PII={y.sum()}, Not-PII={len(y)-y.sum()}, dim={X.shape[1]}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0)),
    ])
    n_splits = min(5, int(y.sum()), int((y == 0).sum()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cv_fn(pipe, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        print(f"  CV Accuracy : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
        print(f"  CV F1       : {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
        print(f"  CV ROC-AUC  : {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")

    pipe.fit(X, y)
    return pipe


def detect(r, probe, method):
    hs_key = "red_flag_hidden_states" if method == "m1" else "sustained_hidden_states"
    hs_list = getattr(r, hs_key)
    if not hs_list:
        return False
    return bool(probe.predict(np.stack(hs_list).astype(np.float32)).any())


def print_results(label, tp, fn, fp, tn, n_pos, n_neg):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Pipeline Results  [{label}]")
    print(f"{'='*60}")
    print(f"  TP (PII correctly detected) : {tp}/{n_pos}")
    print(f"  FN (PII missed)             : {fn}/{n_pos}")
    print(f"  FP (no-PII detected as PII) : {fp}/{n_neg}")
    print(f"  TN (no-PII correctly silent): {tn}/{n_neg}")
    print(f"\n  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    return {"precision": precision, "recall": recall, "f1": f1,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


def evaluate(results_dir, method, test_ratio, seed):
    a_results = load_condition(results_dir, "A_pii")
    b_results = load_condition(results_dir, "B_general")
    c_results = load_condition(results_dir, "C_no_context")

    a_located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    a_not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    # Sample-level train/test split (no data leakage)
    loc_train,     loc_test     = split_samples(a_located,     test_ratio, seed)
    not_loc_train, not_loc_test = split_samples(a_not_located, test_ratio, seed)
    b_train,       b_test       = split_samples(b_results,     test_ratio, seed)
    # C is never used for training
    c_test = c_results

    print(f"\n{'='*60}")
    print(f"Pipeline Evaluation  (method={method.upper()})")
    print(f"  Results dir : {results_dir}")
    print(f"  Split       : {int((1-test_ratio)*100)}% train / {int(test_ratio*100)}% test, seed={seed}")
    print(f"  Train — A-located: {len(loc_train)}, A-not-located: {len(not_loc_train)}, B: {len(b_train)}")
    print(f"  Test  — A-located: {len(loc_test)},  A-not-located: {len(not_loc_test)},  B: {len(b_test)}, C: {len(c_test)}")
    print(f"{'='*60}")

    probe = build_probe(loc_train, not_loc_train + b_train, method)

    # Evaluate on TEST split only
    tp = sum(detect(r, probe, method) for r in loc_test)
    fn = len(loc_test) - tp

    fp_notloc = sum(detect(r, probe, method) for r in not_loc_test)
    fp_b      = sum(detect(r, probe, method) for r in b_test)
    fp_c      = sum(detect(r, probe, method) for r in c_test)
    fp = fp_notloc + fp_b + fp_c

    tn_notloc = len(not_loc_test) - fp_notloc
    tn_b      = len(b_test)       - fp_b
    tn_c      = len(c_test)       - fp_c
    tn = tn_notloc + tn_b + tn_c

    n_pos = len(loc_test)
    n_neg = len(not_loc_test) + len(b_test) + len(c_test)

    print(f"\n  FP breakdown:")
    print(f"    ├ A-not-located : {fp_notloc}/{len(not_loc_test)}")
    print(f"    ├ B (general)   : {fp_b}/{len(b_test)}")
    print(f"    └ C (no context): {fp_c}/{len(c_test)}")

    return print_results(method.upper(), tp, fn, fp, tn, n_pos, n_neg)


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "="*60)
    print("Running M1 (ΔH-drop) ...")
    r_m1 = evaluate(args.results_dir, "m1", args.test_ratio, args.seed)

    print("\n" + "="*60)
    print("Running M2 (sustained window) ...")
    r_m2 = evaluate(args.results_dir, "m2", args.test_ratio, args.seed)

    print(f"\n{'='*60}")
    print("Comparison: M1 vs M2")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'M1 (ΔH-drop)':>15} {'M2 (sustained)':>15}")
    print(f"{'-'*42}")
    for k in ["precision", "recall", "f1"]:
        print(f"{k:<12} {r_m1[k]:>15.4f} {r_m2[k]:>15.4f}")
    for k in ["TP", "FP", "FN"]:
        print(f"{k:<12} {r_m1[k]:>15} {r_m2[k]:>15}")
