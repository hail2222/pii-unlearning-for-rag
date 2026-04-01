"""
Linear Probe: classify hidden states at Red Flag moments as PII vs General Knowledge.

Labels:
  1 = PII         (Condition A red flag hidden states)
  0 = General     (Condition B red flag hidden states)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import numpy as np
from run_experiment import SampleResult  # needed for pickle deserialization
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR


def load_hidden_states(results_dir: str = RESULTS_DIR):
    """
    Load red-flag hidden states from condition A and B pickle files.
    Returns X (n_samples, hidden_size) and y (n_samples,) arrays.
    """
    X_list, y_list = [], []

    for condition, label in [("A_pii", 1), ("B_general", 0)]:
        pkl_path = os.path.join(results_dir, f"{condition}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Run run_experiment.py first: {pkl_path}")

        with open(pkl_path, "rb") as f:
            results = pickle.load(f)

        for r in results:
            for hs in r.red_flag_hidden_states:
                X_list.append(hs)
                y_list.append(label)

    if not X_list:
        raise ValueError("No red-flag hidden states found. Check threshold or results.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    print(f"Dataset: {X.shape[0]} samples (PII={y.sum()}, General={len(y)-y.sum()}), dim={X.shape[1]}")
    return X, y


def train_and_evaluate(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Train logistic regression with cross-validation and report metrics.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0)),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_estimator=True,
    )

    print(f"\n{'='*50}")
    print(f"Linear Probe ({n_splits}-fold CV)")
    print(f"  Accuracy : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
    print(f"  F1       : {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
    print(f"  ROC-AUC  : {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")

    # Final model trained on all data
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    print("\nFull-data classification report:")
    print(classification_report(y, y_pred, target_names=["General", "PII"]))

    return pipe, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    X, y = load_hidden_states(args.results_dir)
    model, scores = train_and_evaluate(X, y)
