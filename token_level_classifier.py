"""
Token-level PII classifier using entropy signal only.

For each generated token, predicts whether it is a PII token or not,
using only the token's entropy value and its local context window.

Ground truth: pii_token_positions (already labeled per sample)

This is Step 2 of the pipeline:
  Step 1: CNN-2ch (sample-level) → "this sample contains PII"
  Step 2: Token-level classifier  → "which tokens are PII"
  Step 3: Those tokens' hidden states → linear probe → final confirmation

Methods compared:
  1. Fixed threshold (entropy < T)              — baseline
  2. Adaptive threshold (entropy < mean - k*std) — per-sample adaptive
  3. Logistic Regression on [entropy_t]          — learned single feature
  4. Logistic Regression on window features      — learned context window
  5. 1D CNN on local window                      — learned non-linear

Usage:
    python token_level_classifier.py \
        --results-dir results_llama8b_20260401_1429
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, classification_report)

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--test-ratio",  type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--window",      type=int,   default=5,
                   help="Half-window size for context features (default: 5 → 11 tokens total)")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Data utils ─────────────────────────────────────────────────────────────────

def load(results_dir, condition):
    with open(os.path.join(results_dir, f"{condition}.pkl"), "rb") as f:
        return pickle.load(f)


def split_samples(samples, test_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_test = max(1, int(len(samples) * test_ratio))
    test_set = set(idx[:n_test])
    return ([s for i, s in enumerate(samples) if i not in test_set],
            [s for i, s in enumerate(samples) if i in test_set])


def extract_tokens(samples):
    """
    Extract (entropy_seq, pii_set) per sample.
    Returns list of (np.array of entropy values, set of PII token indices).
    """
    result = []
    for r in samples:
        entropy = np.array(r.entropy_seq, dtype=np.float32)
        pii_set = set(r.pii_token_positions)
        result.append((entropy, pii_set))
    return result


def build_token_dataset(token_data, window=5):
    """
    Build token-level dataset from list of (entropy_seq, pii_set).

    For each token t in each sample:
      - features: window around t → [e_{t-w}, ..., e_t, ..., e_{t+w}] (2w+1 values)
      - label: 1 if t in pii_set, else 0

    Returns X: (N_tokens, 2w+1), y: (N_tokens,)
    """
    X_list, y_list = [], []
    for entropy, pii_set in token_data:
        T = len(entropy)
        for t in range(T):
            # Extract window with zero-padding at boundaries
            window_feats = []
            for offset in range(-window, window + 1):
                idx = t + offset
                if 0 <= idx < T:
                    window_feats.append(entropy[idx])
                else:
                    window_feats.append(0.0)
            X_list.append(window_feats)
            y_list.append(1 if t in pii_set else 0)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    return {"f1": f1, "prec": prec, "rec": rec, "acc": acc,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


def print_metrics(name, m):
    print(f"\n  [{name}]")
    print(f"    TP={m['TP']:6d}  FN={m['FN']:6d}  FP={m['FP']:6d}  TN={m['TN']:6d}")
    print(f"    F1={m['f1']:.4f}  Prec={m['prec']:.4f}  Rec={m['rec']:.4f}  Acc={m['acc']:.4f}")


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ── Method 1: Fixed threshold ──────────────────────────────────────────────────

def eval_fixed_threshold(train_data, test_data, thresholds):
    """Try several fixed entropy thresholds, pick best on train, eval on test."""
    section("Method 1: Fixed Threshold (entropy < T)")

    # Build token-level labels for train
    train_tokens = build_token_dataset(train_data, window=0)  # single token
    X_train, y_train = train_tokens

    # Find best threshold on train
    best_t, best_f1 = None, -1
    for t in thresholds:
        y_pred = (X_train[:, 0] < t).astype(int)
        f1 = f1_score(y_train, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"  Best threshold on train: entropy < {best_t:.2f}  (train F1={best_f1:.4f})")

    # Eval on test
    test_tokens = build_token_dataset(test_data, window=0)
    X_test, y_test = test_tokens
    y_pred = (X_test[:, 0] < best_t).astype(int)
    m = compute_metrics(y_test, y_pred)
    print_metrics(f"Fixed threshold (entropy < {best_t:.2f})", m)
    return m, best_t


# ── Method 2: Adaptive threshold (per-sample mean - k*std) ────────────────────

def eval_adaptive_threshold(train_data, test_data, k_values):
    """Per-sample adaptive threshold: entropy < mean(seq) - k * std(seq)."""
    section("Method 2: Adaptive Threshold (entropy < mean - k*std)")

    def predict_adaptive(token_data, k):
        y_true_all, y_pred_all = [], []
        for entropy, pii_set in token_data:
            T = len(entropy)
            mean_e = entropy.mean()
            std_e  = entropy.std()
            cutoff = mean_e - k * std_e
            for t in range(T):
                y_true_all.append(1 if t in pii_set else 0)
                y_pred_all.append(1 if entropy[t] < cutoff else 0)
        return np.array(y_true_all), np.array(y_pred_all)

    # Find best k on train
    best_k, best_f1 = None, -1
    for k in k_values:
        y_true, y_pred = predict_adaptive(train_data, k)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"  k={k:.1f}  train F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_k = f1, k

    print(f"\n  Best k on train: {best_k}  (train F1={best_f1:.4f})")

    # Eval on test
    y_true, y_pred = predict_adaptive(test_data, best_k)
    m = compute_metrics(y_true, y_pred)
    print_metrics(f"Adaptive threshold (k={best_k})", m)
    return m, best_k


# ── Method 3: Logistic Regression on single entropy value ─────────────────────

def eval_lr_single(train_data, test_data):
    """Logistic Regression on single entropy value per token."""
    section("Method 3: Logistic Regression (single entropy value)")

    X_train, y_train = build_token_dataset(train_data, window=0)
    X_test,  y_test  = build_token_dataset(test_data,  window=0)

    # Subsample negatives (3x positives) to handle class imbalance
    rng = np.random.default_rng(42)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    max_neg = min(len(neg_idx), len(pos_idx) * 3)
    neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    X_tr, y_tr = X_train[idx], y_train[idx]

    print(f"  Train tokens: pos={len(pos_idx)}, neg={max_neg} (subsampled)")
    print(f"  Test  tokens: pos={int(y_test.sum())}, neg={int((y_test==0).sum())}")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=1.0))])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_test)
    m = compute_metrics(y_test, y_pred)
    print_metrics("LR (single entropy)", m)
    return m, pipe


# ── Method 4: Logistic Regression on window features ──────────────────────────

def eval_lr_window(train_data, test_data, window):
    """Logistic Regression on entropy window [t-w, ..., t, ..., t+w]."""
    section(f"Method 4: Logistic Regression (window={window}, {2*window+1} features)")

    X_train, y_train = build_token_dataset(train_data, window=window)
    X_test,  y_test  = build_token_dataset(test_data,  window=window)

    # Subsample negatives
    rng = np.random.default_rng(42)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    max_neg = min(len(neg_idx), len(pos_idx) * 3)
    neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    X_tr, y_tr = X_train[idx], y_train[idx]

    print(f"  Train tokens: pos={len(pos_idx)}, neg={max_neg} (subsampled)")
    print(f"  Test  tokens: pos={int(y_test.sum())}, neg={int((y_test==0).sum())}")

    # Grid search over C
    best_C, best_f1, best_pipe = None, -1, None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, C=C))])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["f1"])
        f1_cv = scores["test_f1"].mean()
        print(f"  C={C:<6}  CV F1={f1_cv:.4f}")
        if f1_cv > best_f1:
            best_f1, best_C = f1_cv, C
            best_pipe = Pipeline([("scaler", StandardScaler()),
                                  ("clf", LogisticRegression(max_iter=1000, C=C))])

    print(f"\n  Best C={best_C}")
    best_pipe.fit(X_tr, y_tr)
    y_pred = best_pipe.predict(X_test)
    m = compute_metrics(y_test, y_pred)
    print_metrics(f"LR window (C={best_C})", m)
    return m, best_pipe


# ── Method 5: 1D CNN on local window ──────────────────────────────────────────

class WindowCNN(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        # Input: (N, 1, 2w+1)
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(16, 1))
    def forward(self, x): return self.head(self.net(x)).squeeze(1)


class TokenDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def eval_cnn_window(train_data, test_data, window, epochs, batch_size, device):
    section(f"Method 5: 1D CNN (window={window}, {2*window+1} features)")

    X_train, y_train = build_token_dataset(train_data, window=window)
    X_test,  y_test  = build_token_dataset(test_data,  window=window)

    # Subsample negatives
    rng = np.random.default_rng(42)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    max_neg = min(len(neg_idx), len(pos_idx) * 3)
    neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    X_tr, y_tr = X_train[idx], y_train[idx]

    print(f"  Train tokens: pos={len(pos_idx)}, neg={max_neg} (subsampled)")
    print(f"  Test  tokens: pos={int(y_test.sum())}, neg={int((y_test==0).sum())}")

    pos_n = int((y_tr == 1).sum())
    neg_n = int((y_tr == 0).sum())

    train_ds = TokenDataset(X_tr, y_tr)
    test_ds  = TokenDataset(X_test, y_test)

    model = WindowCNN(window_size=2*window+1).to(device)
    pos_weight = torch.tensor([neg_n / pos_n], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_loader):.4f}")

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            p = (torch.sigmoid(model(X_b.to(device))) > 0.5).cpu().numpy().astype(int)
            preds.extend(p); labels.extend(y_b.numpy().astype(int))

    y_pred = np.array(preds)
    m = compute_metrics(y_test, y_pred)
    print_metrics("CNN window", m)
    return m, model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    a_results = load(args.results_dir, "A_pii")
    a_located  = [r for r in a_results if len(r.pii_token_positions) > 0]

    loc_train, loc_test = split_samples(a_located, args.test_ratio, args.seed)

    train_data = extract_tokens(loc_train)
    test_data  = extract_tokens(loc_test)

    # Count total PII tokens
    total_pii_train = sum(len(r.pii_token_positions) for r in loc_train)
    total_tok_train = sum(len(r.entropy_seq) for r in loc_train)
    total_pii_test  = sum(len(r.pii_token_positions) for r in loc_test)
    total_tok_test  = sum(len(r.entropy_seq) for r in loc_test)

    print(f"\n{'='*65}")
    print(f"  Token-level PII Classifier")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Split       : 80% train / 20% test, seed={args.seed}")
    print(f"  Train samples : {len(loc_train)}  "
          f"(tokens: {total_tok_train}, PII tokens: {total_pii_train}, "
          f"ratio: {total_pii_train/total_tok_train*100:.1f}%)")
    print(f"  Test  samples : {len(loc_test)}   "
          f"(tokens: {total_tok_test}, PII tokens: {total_pii_test}, "
          f"ratio: {total_pii_test/total_tok_test*100:.1f}%)")
    print(f"  Window size : {args.window} (→ {2*args.window+1} features)")
    print(f"{'='*65}")

    results = {}

    # Method 1: Fixed threshold
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
    m1, best_t = eval_fixed_threshold(train_data, test_data, thresholds)
    results["1. Fixed threshold"] = m1

    # Method 2: Adaptive threshold
    k_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    m2, best_k = eval_adaptive_threshold(train_data, test_data, k_values)
    results["2. Adaptive threshold"] = m2

    # Method 3: LR single
    m3, _ = eval_lr_single(train_data, test_data)
    results["3. LR (single)"] = m3

    # Method 4: LR window
    m4, _ = eval_lr_window(train_data, test_data, args.window)
    results["4. LR (window)"] = m4

    # Method 5: CNN window
    m5, _ = eval_cnn_window(train_data, test_data, args.window,
                             args.epochs, args.batch_size, args.device)
    results["5. CNN (window)"] = m5

    # Final summary
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY — Token-level PII Classification")
    print(f"  (A_located only, token-level labels from pii_token_positions)")
    print(f"{'='*65}")
    print(f"  {'Method':<25} {'F1':>7} {'Prec':>7} {'Rec':>7} {'TP':>7} {'FP':>7} {'FN':>7}")
    print(f"  {'-'*65}")
    for name, m in results.items():
        print(f"  {name:<25} {m['f1']:>7.4f} {m['prec']:>7.4f} {m['rec']:>7.4f}"
              f" {m['TP']:>7} {m['FP']:>7} {m['FN']:>7}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
