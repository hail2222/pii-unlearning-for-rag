"""
Three-way comparison of detection methods.

Ground truth: PII actually generated (pii_token_positions > 0) = Positive

Method 1 — Entropy only:
    Stage 1 fires (M1 or M2) → detected. No Stage 2.

Method 2 — Probe only:
    Linear Probe runs on EVERY token's hidden state.
    Trained on: PII token positions (pos) vs all other positions (neg).
    Detected if ANY token classified as PII.
    Requires --save-all-hidden to have been used during experiment.

Method 3 — Entropy + Probe (pipeline):
    Stage 1 fires → Stage 2 classifies hidden state at flag moment.

Usage:
    python compare_methods.py --results-dir results_llama8b_compare
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--method",      type=str, default="m1", choices=["m1", "m2"])
    return p.parse_args()


def load(results_dir, condition):
    path = os.path.join(results_dir, f"{condition}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def split_located(a_results):
    located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    not_located = [r for r in a_results if len(r.pii_token_positions) == 0]
    return located, not_located


def make_probe(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=C)),
    ])


def cv_report(probe, X, y, label):
    n_splits = min(5, int(y.sum()), int((y == 0).sum()))
    if n_splits < 2:
        print(f"  {label}: not enough samples for CV")
        return
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(probe, X, y, cv=cv,
                            scoring=["accuracy", "f1", "roc_auc"])
    print(f"  {label}")
    print(f"    CV Accuracy : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
    print(f"    CV F1       : {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
    print(f"    CV ROC-AUC  : {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")


def metrics(tp, fn, fp, tn, label):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    print(f"\n  [{label}]")
    print(f"    TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1        : {f1:.4f}")
    return {"precision": precision, "recall": recall, "f1": f1,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


# ── Method 1: Entropy only ──────────────────────────────────────────────────

def eval_entropy_only(located, not_located, b_results, c_results, entropy_method):
    """Stage 1 fire = detected. No probe."""
    if entropy_method == "m1":
        fired = lambda r: len(r.red_flag_indices) > 0
    else:
        fired = lambda r: len(r.sustained_flag_indices) > 0

    tp = sum(fired(r) for r in located)
    fn = len(located) - tp
    fp = sum(fired(r) for r in not_located) + \
         sum(fired(r) for r in b_results) + \
         sum(fired(r) for r in c_results)
    tn = (len(not_located) - sum(fired(r) for r in not_located)) + \
         (len(b_results)   - sum(fired(r) for r in b_results)) + \
         (len(c_results)   - sum(fired(r) for r in c_results))
    return metrics(tp, fn, fp, tn, f"Entropy only ({entropy_method.upper()})")


# ── Method 2: Probe only ────────────────────────────────────────────────────

def eval_probe_only(located, not_located, b_results, c_results):
    """
    Probe runs on ALL token hidden states.
    Positive training: hidden states at PII token positions (A_located).
    Negative training: all token hidden states from A_not_located + B.
    """
    has_all = any(len(r.all_hidden_states) > 0 for r in located)
    if not has_all:
        print("\n  [Probe only] SKIPPED — no all_hidden_states found.")
        print("  Re-run experiment with --save-all-hidden flag.")
        return None

    # Build training data
    X_pos, X_neg = [], []

    for r in located:
        if not r.all_hidden_states:
            continue
        pii_set = set(r.pii_token_positions)
        for i, hs in enumerate(r.all_hidden_states):
            if i in pii_set:
                X_pos.append(hs)
            else:
                X_neg.append(hs)

    for r in not_located + b_results:
        for hs in r.all_hidden_states:
            X_neg.append(hs)

    if not X_pos or not X_neg:
        print("\n  [Probe only] SKIPPED — insufficient training data.")
        return None

    # Subsample negatives to balance (max 3× positives)
    rng = np.random.default_rng(42)
    max_neg = min(len(X_neg), len(X_pos) * 3)
    idx = rng.choice(len(X_neg), size=max_neg, replace=False)
    X_neg_sampled = [X_neg[i] for i in idx]

    X = np.stack(X_pos + X_neg_sampled).astype(np.float32)
    y = np.array([1]*len(X_pos) + [0]*len(X_neg_sampled))

    print(f"\n  Probe-only training: PII-pos={len(X_pos)}, neg={len(X_neg_sampled)}, dim={X.shape[1]}")
    probe = make_probe()
    cv_report(make_probe(), X, y, "Probe-only CV")
    probe.fit(X, y)

    # Sample-level detection: if ANY token classified as PII → detected
    def detected(r):
        if not r.all_hidden_states:
            return False
        X_r = np.stack(r.all_hidden_states).astype(np.float32)
        return bool(probe.predict(X_r).any())

    tp = sum(detected(r) for r in located)
    fn = len(located) - tp
    fp = sum(detected(r) for r in not_located) + \
         sum(detected(r) for r in b_results) + \
         sum(detected(r) for r in c_results)
    tn = (len(not_located) - sum(detected(r) for r in not_located)) + \
         (len(b_results)   - sum(detected(r) for r in b_results)) + \
         (len(c_results)   - sum(detected(r) for r in c_results))
    return metrics(tp, fn, fp, tn, "Probe only")


# ── Method 3: Entropy + Probe (pipeline) ───────────────────────────────────

def eval_pipeline(located, not_located, b_results, c_results, entropy_method):
    """Reproduce evaluate_pipeline logic: A-located vs A-not-located+B probe training."""
    hs_key = "red_flag_hidden_states" if entropy_method == "m1" else "sustained_hidden_states"

    X_list, y_list = [], []
    for r in located:
        for hs in getattr(r, hs_key):
            X_list.append(hs); y_list.append(1)
    for r in not_located + b_results:
        for hs in getattr(r, hs_key):
            X_list.append(hs); y_list.append(0)

    if not X_list:
        print(f"\n  [Pipeline] No hidden states found for {entropy_method}.")
        return None

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    print(f"\n  Pipeline training: PII={y.sum()}, Not-PII={len(y)-y.sum()}, dim={X.shape[1]}")
    probe = make_probe()
    cv_report(make_probe(), X, y, f"Pipeline CV ({entropy_method.upper()})")
    probe.fit(X, y)

    def detected(r):
        hs_list = getattr(r, hs_key)
        if not hs_list:
            return False
        return bool(probe.predict(np.stack(hs_list).astype(np.float32)).any())

    tp = sum(detected(r) for r in located)
    fn = len(located) - tp
    fp = sum(detected(r) for r in not_located) + \
         sum(detected(r) for r in b_results) + \
         sum(detected(r) for r in c_results)
    tn = (len(not_located) - sum(detected(r) for r in not_located)) + \
         (len(b_results)   - sum(detected(r) for r in b_results)) + \
         (len(c_results)   - sum(detected(r) for r in c_results))
    return metrics(tp, fn, fp, tn, f"Entropy + Probe ({entropy_method.upper()})")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    a_results = load(args.results_dir, "A_pii")
    b_results = load(args.results_dir, "B_general")
    c_results = load(args.results_dir, "C_no_context")

    located, not_located = split_located(a_results)

    print(f"\n{'='*60}")
    print(f"Three-Way Method Comparison")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Entropy method: {args.method.upper()}")
    print(f"  Positive (PII generated)  : {len(located)}")
    print(f"  Negative (no PII in output): {len(not_located)+len(b_results)+len(c_results)}")
    print(f"    ├ A-not-located: {len(not_located)}")
    print(f"    ├ B (general)  : {len(b_results)}")
    print(f"    └ C (no context): {len(c_results)}")
    print(f"{'='*60}")

    r1 = eval_entropy_only(located, not_located, b_results, c_results, args.method)
    r2 = eval_probe_only(located, not_located, b_results, c_results)
    r3 = eval_pipeline(located, not_located, b_results, c_results, args.method)

    print(f"\n{'='*60}")
    print(f"Summary Comparison  (entropy method: {args.method.upper()})")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Entropy only':>14} {'Probe only':>12} {'Pipeline':>10}")
    print(f"{'-'*50}")
    for k in ["precision", "recall", "f1"]:
        v1 = f"{r1[k]:.4f}" if r1 else "N/A"
        v2 = f"{r2[k]:.4f}" if r2 else "N/A"
        v3 = f"{r3[k]:.4f}" if r3 else "N/A"
        print(f"{k:<12} {v1:>14} {v2:>12} {v3:>10}")
    for k in ["TP", "FP", "FN"]:
        v1 = str(r1[k]) if r1 else "N/A"
        v2 = str(r2[k]) if r2 else "N/A"
        v3 = str(r3[k]) if r3 else "N/A"
        print(f"{k:<12} {v1:>14} {v2:>12} {v3:>10}")


if __name__ == "__main__":
    main()
