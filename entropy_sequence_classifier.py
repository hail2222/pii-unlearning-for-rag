"""
Entropy Sequence Classification for PII Detection.

Compares increasingly complex methods that use only the token-level
entropy sequence (no hidden states required):

  Step 1 — Feature-based Logistic Regression
            hand-crafted features from entropy_seq → LR

  Step 2 — 1D CNN
            raw entropy_seq (padded) → Conv1D classifier

Ground truth: A_located (PII actually generated) = Positive
              A_not_located + B + C              = Negative

Train/test split: 80/20 at sample level (seed=42).

Usage:
    python entropy_sequence_classifier.py --results-dir results_llama8b_20260401_1429
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR, MAX_NEW_TOKENS

MAX_LEN = MAX_NEW_TOKENS  # pad/truncate all sequences to this length


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--test-ratio",  type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=32)
    return p.parse_args()


# ── Load & split ───────────────────────────────────────────────────────────────

def load(results_dir, condition):
    with open(os.path.join(results_dir, f"{condition}.pkl"), "rb") as f:
        return pickle.load(f)


def split_samples(samples, test_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_test = max(1, int(len(samples) * test_ratio))
    test_set = set(idx[:n_test])
    train = [s for i, s in enumerate(samples) if i not in test_set]
    test  = [s for i, s in enumerate(samples) if i in test_set]
    return train, test


def pad_sequence(seq, max_len):
    seq = np.array(seq, dtype=np.float32)
    if len(seq) >= max_len:
        return seq[:max_len]
    return np.pad(seq, (0, max_len - len(seq)), constant_values=0.0)


# ── Feature extraction for LR ──────────────────────────────────────────────────

def extract_features(r: SampleResult) -> np.ndarray:
    """Hand-crafted features from entropy sequence."""
    seq = np.array(r.entropy_seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros(12, dtype=np.float32)

    delta = np.abs(np.diff(seq, prepend=seq[0]))
    below = (seq < 0.5).astype(float)

    # Longest run below threshold
    max_run = 0
    cur_run = 0
    for v in below:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    return np.array([
        seq.mean(),                    # mean entropy
        seq.std(),                     # std
        seq.min(),                     # min
        seq.max(),                     # max
        np.percentile(seq, 25),        # Q1
        np.median(seq),                # median
        below.mean(),                  # fraction below 0.5 nats
        float(max_run),                # longest run below 0.5
        delta.mean(),                  # mean absolute delta
        delta.max(),                   # max delta (sharpest drop)
        seq[:10].mean() if len(seq) >= 10 else seq.mean(),  # early entropy
        seq[-10:].mean() if len(seq) >= 10 else seq.mean(), # late entropy
    ], dtype=np.float32)


# ── Metrics helper ─────────────────────────────────────────────────────────────

def print_metrics(label, y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    print(f"\n  [{label}]")
    print(f"    Positive 맞춤 (TP) : {tp}/{n_pos} ({tp/n_pos*100:.1f}%)")
    print(f"    Positive 틀림 (FN) : {fn}/{n_pos} ({fn/n_pos*100:.1f}%)")
    print(f"    Negative 맞춤 (TN) : {tn}/{n_neg} ({tn/n_neg*100:.1f}%)")
    print(f"    Negative 틀림 (FP) : {fp}/{n_neg} ({fp/n_neg*100:.1f}%)")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1        : {f1:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


# ── Step 1: Feature-based Logistic Regression ──────────────────────────────────

def run_lr(train_pos, train_neg, test_pos, test_neg):
    print("\n" + "="*60)
    print("Step 1: Feature-based Logistic Regression")
    print("="*60)

    X_train = np.stack(
        [extract_features(r) for r in train_pos] +
        [extract_features(r) for r in train_neg]
    )
    y_train = np.array([1]*len(train_pos) + [0]*len(train_neg))

    X_test = np.stack(
        [extract_features(r) for r in test_pos] +
        [extract_features(r) for r in test_neg]
    )
    y_test = np.array([1]*len(test_pos) + [0]*len(test_neg))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0)),
    ])

    # CV on training set
    n_splits = min(5, int(y_train.sum()), int((y_train == 0).sum()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=["accuracy", "f1", "roc_auc"])
        print(f"\n  Train CV (n={len(y_train)}, features=12):")
        print(f"    CV Accuracy : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
        print(f"    CV F1       : {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
        print(f"    CV ROC-AUC  : {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n  Test set (n={len(y_test)}):")
    return print_metrics("LR on entropy features", y_test, y_pred)


# ── Step 2: 1D CNN ─────────────────────────────────────────────────────────────

class EntropyDataset(Dataset):
    def __init__(self, pos_samples, neg_samples, max_len):
        self.X, self.y = [], []
        for r in pos_samples:
            self.X.append(pad_sequence(r.entropy_seq, max_len))
            self.y.append(1)
        for r in neg_samples:
            self.X.append(pad_sequence(r.entropy_seq, max_len))
            self.y.append(0)
        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class EntropyCNN(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),                          # → (32, L/2)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),                          # → (64, L/4)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                  # → (128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.conv(x)).squeeze(1)


def run_cnn(train_pos, train_neg, test_pos, test_neg, epochs, batch_size, seed):
    print("\n" + "="*60)
    print("Step 2: 1D CNN on raw entropy sequence")
    print("="*60)

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}, seq_len={MAX_LEN}, epochs={epochs}")

    train_ds = EntropyDataset(train_pos, train_neg, MAX_LEN)
    test_ds  = EntropyDataset(test_pos,  test_neg,  MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    model = EntropyCNN(MAX_LEN).to(device)

    # Class weights for imbalanced data
    n_pos = len(train_pos)
    n_neg = len(train_neg)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\n  Train: {len(train_ds)} samples (pos={n_pos}, neg={n_neg})")

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_loader):.4f}")

    # Evaluation on test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy().astype(int))

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    print(f"\n  Test set (n={len(y_true)}):")
    return print_metrics("1D CNN on entropy sequence", y_true, y_pred)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    a_results = load(args.results_dir, "A_pii")
    b_results = load(args.results_dir, "B_general")
    c_results = load(args.results_dir, "C_no_context")

    a_located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    a_not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    # Sample-level train/test split
    loc_train,     loc_test     = split_samples(a_located,     args.test_ratio, args.seed)
    not_loc_train, not_loc_test = split_samples(a_not_located, args.test_ratio, args.seed)
    b_train,       b_test       = split_samples(b_results,     args.test_ratio, args.seed)
    c_test = c_results  # C always test-only

    neg_train = not_loc_train + b_train
    neg_test  = not_loc_test  + b_test + c_test

    print(f"\n{'='*60}")
    print(f"Entropy Sequence Classification")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Split: 80% train / 20% test, seed={args.seed}")
    print(f"  Train — pos: {len(loc_train)}, neg: {len(neg_train)}")
    print(f"  Test  — pos: {len(loc_test)},  neg: {len(neg_test)}")
    print(f"{'='*60}")

    r_lr  = run_lr(loc_train, neg_train, loc_test, neg_test)
    r_cnn = run_cnn(loc_train, neg_train, loc_test, neg_test,
                    args.epochs, args.batch_size, args.seed)

    # Summary
    print(f"\n{'='*60}")
    print("Summary: Entropy only  →  LR features  →  1D CNN")
    print(f"  Task: PII-generated text vs. everything else")
    print(f"  Test — Positive: {len(loc_test)}, Negative: {len(neg_test)}")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'LR (features)':>15} {'1D CNN':>10}")
    print(f"{'-'*38}")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"{k:<12} {r_lr[k]:>15.4f} {r_cnn[k]:>10.4f}")
    print(f"{'-'*38}")
    for k in ["TP", "FP", "FN", "TN"]:
        print(f"{k:<12} {r_lr[k]:>15} {r_cnn[k]:>10}")


if __name__ == "__main__":
    main()
