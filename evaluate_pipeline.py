"""
End-to-end evaluation of the two-stage privacy detection pipeline.

Stage 1: Entropy sensor (M1: ΔH-drop or M2: sustained low-entropy window)
          fires a Red Flag when it detects suspicious generation.
Stage 2: Linear Probe classifies the hidden state at Red Flag moment
          as PII (1) or General (0).

A sample is counted as DETECTED if:
    - Stage 1 fires at least one Red Flag, AND
    - Stage 2 classifies that hidden state as PII

Metrics (per condition):
    Condition A → ground truth = PII leak (positive)
    Condition B → ground truth = general knowledge (negative)
    Condition C → ground truth = no PII available (negative)

    True Positive  (TP): A sample detected as PII  → correct
    False Negative (FN): A sample NOT detected      → missed leak
    False Positive (FP): B or C sample detected as PII → false alarm
    True Negative  (TN): B or C sample NOT detected → correct silence

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


# ── Stage 2: train Linear Probe on A vs B ─────────────────────────────────────

def train_probe(a_results: list, b_results: list, method: str) -> Pipeline:
    """
    Train Linear Probe on red-flag hidden states.
    method="m1" uses red_flag_hidden_states (ΔH-drop).
    method="m2" uses sustained_hidden_states.
    """
    X_list, y_list = [], []

    for r in a_results:
        hs_list = r.red_flag_hidden_states if method == "m1" else r.sustained_hidden_states
        for hs in hs_list:
            X_list.append(hs)
            y_list.append(1)   # PII

    for r in b_results:
        hs_list = r.red_flag_hidden_states if method == "m1" else r.sustained_hidden_states
        for hs in hs_list:
            X_list.append(hs)
            y_list.append(0)   # General

    if not X_list:
        raise ValueError("No hidden states found. Check threshold or results.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    print(f"\nProbe training set: {len(y)} hidden states "
          f"(PII={y.sum()}, General={len(y)-y.sum()}), dim={X.shape[1]}")

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

    print(f"\n{'='*60}")
    print(f"End-to-end Pipeline Evaluation")
    print(f"  Results dir : {results_dir}")
    print(f"  Method      : {method.upper()}")
    print(f"  Samples     : A={len(a_results)}, B={len(b_results)}, C={len(c_results)}")
    print(f"{'='*60}")

    # Train probe on A vs B
    probe = train_probe(a_results, b_results, method)

    # Evaluate on all conditions
    a_detected = [run_pipeline_on_sample(r, probe, method) for r in a_results]
    b_detected = [run_pipeline_on_sample(r, probe, method) for r in b_results]
    c_detected = [run_pipeline_on_sample(r, probe, method) for r in c_results]

    TP = sum(a_detected)
    FN = len(a_results) - TP
    FP = sum(b_detected) + sum(c_detected)
    TN = (len(b_results) - sum(b_detected)) + (len(c_results) - sum(c_detected))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Pipeline Results  (method={method.upper()})")
    print(f"{'='*60}")
    print(f"  True Positive  (A detected as PII)       : {TP}/{len(a_results)}")
    print(f"  False Negative (A missed)                : {FN}/{len(a_results)}")
    print(f"  False Positive (B/C detected as PII)     : {FP}/{len(b_results)+len(c_results)}")
    print(f"    ├ B (general knowledge) FP             : {sum(b_detected)}/{len(b_results)}")
    print(f"    └ C (no context)        FP             : {sum(c_detected)}/{len(c_results)}")
    print(f"  True Negative  (B/C correctly silent)    : {TN}/{len(b_results)+len(c_results)}")
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
