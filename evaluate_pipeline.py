"""
End-to-end evaluation of the two-stage privacy detection pipeline.

Ground truth is defined by whether PII tokens actually appeared in the output,
NOT by the experimental condition label:

    Positive: A samples where PII was actually generated (pii_token_positions > 0)
    Negative: A_not_located + B + C  (no PII in output, regardless of condition)

Stage 2 Linear Probe is trained on:
    Positive hidden states: A_located Red Flag moments
    Negative hidden states: A_not_located + B Red Flag moments

This ensures the probe learns "is the model generating PII right now?"
rather than "which experimental condition is this?"

Usage:
    python evaluate_pipeline.py                          # local (results/)
    python evaluate_pipeline.py --results-dir results_llama8b
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--method", type=str, default="m1",
                   choices=["m1", "m2"],
                   help="m1=ΔH-drop flags, m2=sustained flags")
    p.add_argument("--probe-cv", type=int, default=5,
                   help="Cross-validation folds for linear probe")
    return p.parse_args()


# ── Load results ───────────────────────────────────────────────────────────────

def load_condition(results_dir: str, condition: str) -> list[SampleResult]:
    path = os.path.join(results_dir, f"{condition}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}  →  run run_experiment.py first")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Stage 2: train Linear Probe ───────────────────────────────────────────────

def train_probe(pos_results: list, neg_results: list, method: str) -> Pipeline:
    """
    Train Linear Probe on red-flag hidden states.
      pos_results: samples where PII was actually generated  (label=1)
      neg_results: samples where no PII was generated        (label=0)
    method="m1" uses red_flag_hidden_states (ΔH-drop).
    method="m2" uses sustained_hidden_states.
    """
    X_list, y_list = [], []

    for r in pos_results:
        hs_list = r.red_flag_hidden_states if method == "m1" else r.sustained_hidden_states
        for hs in hs_list:
            X_list.append(hs)
            y_list.append(1)   # PII generated

    for r in neg_results:
        hs_list = r.red_flag_hidden_states if method == "m1" else r.sustained_hidden_states
        for hs in hs_list:
            X_list.append(hs)
            y_list.append(0)   # PII not generated

    if not X_list:
        raise ValueError("No hidden states found. Check threshold or results.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    print(f"\nProbe training set: {len(y)} hidden states "
          f"(PII-generated={y.sum()}, Not-generated={len(y)-y.sum()}), dim={X.shape[1]}")

    # Cross-validation report
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0)),
    ])
    n_splits = min(5, int(y.sum()), int((y == 0).sum()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        from sklearn.model_selection import cross_validate as cv_fn
        scores = cv_fn(pipe, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        print(f"  CV Accuracy : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
        print(f"  CV F1       : {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
        print(f"  CV ROC-AUC  : {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")

    pipe.fit(X, y)
    return pipe


# ── Stage 1+2: run pipeline on one sample ─────────────────────────────────────

def run_pipeline_on_sample(r: SampleResult, probe: Pipeline, method: str) -> bool:
    """
    Returns True if the pipeline detects PII leak for this sample.
    Detection = Stage1 fires AND Stage2 classifies at least one hs as PII.
    """
    hs_list = r.red_flag_hidden_states if method == "m1" else r.sustained_hidden_states
    if not hs_list:
        return False

    X = np.stack(hs_list).astype(np.float32)
    preds = probe.predict(X)
    return bool(preds.any())   # detected if ANY red-flag hs is classified as PII


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(results_dir: str, method: str, probe_cv: int):
    a_results = load_condition(results_dir, "A_pii")
    b_results = load_condition(results_dir, "B_general")
    c_results = load_condition(results_dir, "C_no_context")

    # Split A into PII-located (true positive candidates) and not-located (negative)
    a_located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    a_not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    print(f"\n{'='*60}")
    print(f"End-to-end Pipeline Evaluation")
    print(f"  Results dir    : {results_dir}")
    print(f"  Method         : {method.upper()}")
    print(f"  Positive (PII actually generated)  : {len(a_located)}  [A-located]")
    print(f"  Negative (no PII in output)        : {len(a_not_located) + len(b_results) + len(c_results)}")
    print(f"    ├ A-not-located : {len(a_not_located)}")
    print(f"    ├ B (general)   : {len(b_results)}")
    print(f"    └ C (no context): {len(c_results)}")
    print(f"{'='*60}")

    # Train probe: A_located (positive) vs A_not_located + B (negative)
    # C excluded from training — used only for evaluation
    neg_train = a_not_located + b_results
    print(f"\nProbe training — Positive: A-located (n={len(a_located)}), "
          f"Negative: A-not-located + B (n={len(neg_train)})")
    probe = train_probe(a_located, neg_train, method)

    # Run pipeline
    a_loc_detected     = [run_pipeline_on_sample(r, probe, method) for r in a_located]
    a_notloc_detected  = [run_pipeline_on_sample(r, probe, method) for r in a_not_located]
    b_detected         = [run_pipeline_on_sample(r, probe, method) for r in b_results]
    c_detected         = [run_pipeline_on_sample(r, probe, method) for r in c_results]

    # Correct TP/FN/FP/TN
    TP = sum(a_loc_detected)
    FN = len(a_located) - TP
    FP = sum(a_notloc_detected) + sum(b_detected) + sum(c_detected)
    TN = (len(a_not_located) - sum(a_notloc_detected)) + \
         (len(b_results) - sum(b_detected)) + \
         (len(c_results) - sum(c_detected))

    n_neg = len(a_not_located) + len(b_results) + len(c_results)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Pipeline Results  (method={method.upper()})")
    print(f"{'='*60}")
    print(f"  True Positive  (A-located detected)       : {TP}/{len(a_located)}")
    print(f"  False Negative (A-located missed)         : {FN}/{len(a_located)}")
    print(f"  False Positive (negative detected as PII) : {FP}/{n_neg}")
    print(f"    ├ A-not-located FP                      : {sum(a_notloc_detected)}/{len(a_not_located)}")
    print(f"    ├ B (general knowledge) FP              : {sum(b_detected)}/{len(b_results)}")
    print(f"    └ C (no context)        FP              : {sum(c_detected)}/{len(c_results)}")
    print(f"  True Negative  (negative correctly silent): {TN}/{n_neg}")
    print(f"\n  Precision : {precision:.4f}  (of all alerts, how many were real leaks)")
    print(f"  Recall    : {recall:.4f}  (of all real leaks, how many were caught)")
    print(f"  F1 Score  : {f1:.4f}")

    return {"TP": TP, "FN": FN, "FP": FP, "TN": TN,
            "precision": precision, "recall": recall, "f1": f1}


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Run both methods and compare
    print("\n" + "="*60)
    print("Running M1 (ΔH-drop) ...")
    r_m1 = evaluate(args.results_dir, method="m1", probe_cv=args.probe_cv)

    print("\n" + "="*60)
    print("Running M2 (sustained window) ...")
    r_m2 = evaluate(args.results_dir, method="m2", probe_cv=args.probe_cv)

    # Side-by-side comparison
    print(f"\n{'='*60}")
    print("Comparison: M1 vs M2")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'M1 (ΔH-drop)':>15} {'M2 (sustained)':>15}")
    print(f"{'-'*42}")
    for k in ["precision", "recall", "f1"]:
        print(f"{k:<12} {r_m1[k]:>15.4f} {r_m2[k]:>15.4f}")
    print(f"{'TP':<12} {r_m1['TP']:>15} {r_m2['TP']:>15}")
    print(f"{'FP':<12} {r_m1['FP']:>15} {r_m2['FP']:>15}")
    print(f"{'FN':<12} {r_m1['FN']:>15} {r_m2['FN']:>15}")
