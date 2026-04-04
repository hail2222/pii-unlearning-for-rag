"""
Three-way comparison: Entropy only vs Probe only vs Entropy+Probe pipeline.

Ground truth: PII actually generated (pii_token_positions > 0) = Positive

Train/test split (80/20, seed=42) is applied at the SAMPLE level.
All three methods are evaluated on the SAME held-out test set.

    Train: A_located (pos) + A_not_located + B (neg)  — 80%
    Test:  A_located (pos) + A_not_located + B + C (neg) — 20% (C always test-only)

Requires --save-all-hidden for Probe only method.

Usage:
    python compare_methods.py --results-dir results_compare --method m1
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
    p.add_argument("--test-ratio",  type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def load(results_dir, condition):
    with open(os.path.join(results_dir, f"{condition}.pkl"), "rb") as f:
        return pickle.load(f)


def split_samples(samples, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_test = max(1, int(len(samples) * test_ratio))
    test_set = set(idx[:n_test])
    train = [s for i, s in enumerate(samples) if i not in test_set]
    test  = [s for i, s in enumerate(samples) if i in test_set]
    return train, test


def make_probe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0)),
    ])


def cv_report(X, y, label):
    n_splits = min(5, int(y.sum()), int((y == 0).sum()))
    if n_splits < 2:
        print(f"    CV skipped (insufficient samples)")
        return
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(make_probe(), X, y, cv=cv,
                            scoring=["accuracy", "f1", "roc_auc"])
    print(f"    CV Accuracy : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
    print(f"    CV F1       : {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
    print(f"    CV ROC-AUC  : {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")


def compute_metrics(label, tp, fn, fp, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) > 0 else 0.0
    print(f"\n  [{label}]")
    print(f"    TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"    Accuracy  : {accuracy:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1        : {f1:.4f}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


# ── Method 1: Entropy only ──────────────────────────────────────────────────

def eval_entropy_only(loc_test, neg_test, method):
    fired = (lambda r: len(r.red_flag_indices) > 0) if method == "m1" \
            else (lambda r: len(r.sustained_flag_indices) > 0)

    tp = sum(fired(r) for r in loc_test)
    fn = len(loc_test) - tp
    fp = sum(fired(r) for r in neg_test)
    tn = len(neg_test) - fp
    return compute_metrics(f"Entropy only ({method.upper()})", tp, fn, fp, tn)


# ── Method 2: Probe only ────────────────────────────────────────────────────

def eval_probe_only(loc_train, loc_test, neg_train, neg_test):
    has_all = any(len(r.all_hidden_states) > 0 for r in loc_train)
    if not has_all:
        print("\n  [Probe only] SKIPPED — re-run experiment with --save-all-hidden")
        return None

    # Build training data from TRAIN split only
    X_pos, X_neg = [], []
    for r in loc_train:
        pii_set = set(r.pii_token_positions)
        for i, hs in enumerate(r.all_hidden_states):
            if i in pii_set:
                X_pos.append(hs)
            else:
                X_neg.append(hs)
    for r in neg_train:
        for hs in r.all_hidden_states:
            X_neg.append(hs)

    # Subsample negatives (max 3× positives)
    rng = np.random.default_rng(42)
    max_neg = min(len(X_neg), len(X_pos) * 3)
    idx = rng.choice(len(X_neg), size=max_neg, replace=False)
    X_neg = [X_neg[i] for i in idx]

    X = np.stack(X_pos + X_neg).astype(np.float32)
    y = np.array([1]*len(X_pos) + [0]*len(X_neg))
    print(f"\n  Probe-only train: PII-pos={len(X_pos)}, neg={len(X_neg)}, dim={X.shape[1]}")
    cv_report(X, y, "Probe-only")
    probe = make_probe()
    probe.fit(X, y)

    def detected(r):
        if not r.all_hidden_states:
            return False
        return bool(probe.predict(np.stack(r.all_hidden_states).astype(np.float32)).any())

    # Evaluate on TEST split only
    tp = sum(detected(r) for r in loc_test)
    fn = len(loc_test) - tp
    fp = sum(detected(r) for r in neg_test)
    tn = len(neg_test) - fp
    return compute_metrics("Probe only", tp, fn, fp, tn)


# ── Method 3: Entropy + Probe (pipeline) ───────────────────────────────────

def eval_pipeline(loc_train, loc_test, neg_train, neg_test, method):
    hs_key = "red_flag_hidden_states" if method == "m1" else "sustained_hidden_states"

    X_list, y_list = [], []
    for r in loc_train:
        for hs in getattr(r, hs_key):
            X_list.append(hs); y_list.append(1)
    for r in neg_train:
        for hs in getattr(r, hs_key):
            X_list.append(hs); y_list.append(0)

    if not X_list:
        print(f"\n  [Pipeline] No hidden states in training split.")
        return None

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    print(f"\n  Pipeline train: PII={y.sum()}, Not-PII={len(y)-y.sum()}, dim={X.shape[1]}")
    cv_report(X, y, f"Pipeline ({method.upper()})")
    probe = make_probe()
    probe.fit(X, y)

    def detected(r):
        hs_list = getattr(r, hs_key)
        if not hs_list:
            return False
        return bool(probe.predict(np.stack(hs_list).astype(np.float32)).any())

    # Evaluate on TEST split only
    tp = sum(detected(r) for r in loc_test)
    fn = len(loc_test) - tp
    fp = sum(detected(r) for r in neg_test)
    tn = len(neg_test) - fp
    return compute_metrics(f"Entropy + Probe ({method.upper()})", tp, fn, fp, tn)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    a_results = load(args.results_dir, "A_pii")
    b_results = load(args.results_dir, "B_general")
    c_results = load(args.results_dir, "C_no_context")

    a_located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    a_not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    # Sample-level split — same split for all methods
    loc_train,     loc_test     = split_samples(a_located,     args.test_ratio, args.seed)
    not_loc_train, not_loc_test = split_samples(a_not_located, args.test_ratio, args.seed)
    b_train,       b_test       = split_samples(b_results,     args.test_ratio, args.seed)
    # C always test-only
    c_test = c_results

    neg_train = not_loc_train + b_train
    neg_test  = not_loc_test  + b_test + c_test  # C included in neg test

    print(f"\n{'='*60}")
    print(f"Three-Way Method Comparison  (entropy: {args.method.upper()})")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Split: {int((1-args.test_ratio)*100)}% train / {int(args.test_ratio*100)}% test, seed={args.seed}")
    print(f"  Train — A-located: {len(loc_train)}, neg (A-not+B): {len(neg_train)}")
    print(f"  Test  — A-located: {len(loc_test)},  neg (A-not+B+C): {len(neg_test)}")
    print(f"{'='*60}")

    r1 = eval_entropy_only(loc_test, neg_test, args.method)
    r2 = eval_probe_only(loc_train, loc_test, neg_train, neg_test)
    r3 = eval_pipeline(loc_train, loc_test, neg_train, neg_test, args.method)

    print(f"\n{'='*60}")
    print(f"Summary  (entropy: {args.method.upper()})")
    print(f"  Task: PII-generated text vs. everything else")
    print(f"  Test — Positive: {len(loc_test)}, Negative: {len(neg_test)}")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Entropy only':>14} {'Probe only':>12} {'Pipeline':>10}")
    print(f"{'-'*50}")
    for k in ["accuracy", "precision", "recall", "f1"]:
        v1 = f"{r1[k]:.4f}" if r1 else "N/A"
        v2 = f"{r2[k]:.4f}" if r2 else "N/A"
        v3 = f"{r3[k]:.4f}" if r3 else "N/A"
        print(f"{k:<12} {v1:>14} {v2:>12} {v3:>10}")
    print(f"{'-'*50}")
    for k in ["TP", "FP", "FN", "TN"]:
        v1 = str(r1[k]) if r1 else "N/A"
        v2 = str(r2[k]) if r2 else "N/A"
        v3 = str(r3[k]) if r3 else "N/A"
        print(f"{k:<12} {v1:>14} {v2:>12} {v3:>10}")


if __name__ == "__main__":
    main()
