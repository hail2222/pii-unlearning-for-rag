"""
Real-time PII Detection Experiment

Simulates online (streaming) detection:
  At each token position T, the detector has only seen entropy[0:T].
  Can it predict "PII is being generated" before the PII tokens appear?

Two analyses:
  1. PII first-appearance distribution
     → At what token position does PII first appear? (tells us the deadline)

  2. Prefix-length sweep
     → Train a 1D CNN using only entropy[0:T] as input, for T in prefix_lengths.
     → Measure F1 at each T.
     → Compare with PII first-appearance to compute latency.

Usage:
    python realtime_detection_experiment.py \
        --results-dir results_llama8b_20260401_1429 \
        --epochs 30
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--test-ratio",  type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prefix-lengths", type=int, nargs="+",
                   default=[5, 10, 15, 20, 25, 30, 40, 50, 60, 80])
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


def pad_prefix(seq, prefix_len):
    """Take only the first prefix_len tokens, pad with 0 if shorter."""
    seq = np.array(seq, dtype=np.float32)
    seq = seq[:prefix_len]  # truncate to prefix
    if len(seq) < prefix_len:
        seq = np.pad(seq, (0, prefix_len - len(seq)), constant_values=0.0)
    return seq


# ── Dataset ────────────────────────────────────────────────────────────────────

class PrefixDataset(Dataset):
    """entropy[0:prefix_len] as input, shape (N, 1, prefix_len)."""
    def __init__(self, pos, neg, prefix_len):
        seqs, labels = [], []
        for r in pos:
            seqs.append(pad_prefix(r.entropy_seq, prefix_len))
            labels.append(1)
        for r in neg:
            seqs.append(pad_prefix(r.entropy_seq, prefix_len))
            labels.append(0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Model ──────────────────────────────────────────────────────────────────────

class PrefixCNN(nn.Module):
    """1D CNN that works on variable-length prefix (uses AdaptiveAvgPool)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)


# ── Trainer ────────────────────────────────────────────────────────────────────

def train_eval(model, train_ds, test_ds, pos_n, neg_n, epochs, batch_size, device):
    model = model.to(device)
    pos_weight = torch.tensor([neg_n / pos_n], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

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

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            p = (torch.sigmoid(model(X_batch.to(device))) > 0.5).cpu().numpy().astype(int)
            preds.extend(p)
            labels.extend(y_batch.numpy().astype(int))

    y_true = np.array(labels)
    y_pred = np.array(preds)
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


# ── Analysis 1: PII first-appearance distribution ──────────────────────────────

def analyze_pii_positions(a_located):
    print(f"\n{'='*62}")
    print(f"  Analysis 1: PII First-Appearance Position Distribution")
    print(f"  (n={len(a_located)} A_located samples)")
    print(f"{'='*62}")

    first_positions = []
    for r in a_located:
        if r.pii_token_positions:
            first_positions.append(r.pii_token_positions[0])

    arr = np.array(first_positions)
    print(f"  PII first token position:")
    print(f"    min    : {arr.min():.0f}")
    print(f"    mean   : {arr.mean():.1f}")
    print(f"    median : {np.median(arr):.1f}")
    print(f"    max    : {arr.max():.0f}")
    print(f"    std    : {arr.std():.1f}")

    # Percentile breakdown
    for pct in [10, 25, 50, 75, 90]:
        print(f"    p{pct:<2}    : {np.percentile(arr, pct):.1f}")

    # How many samples have PII before token T?
    print(f"\n  Cumulative: samples where PII starts before token T")
    print(f"  {'T':>5}  {'count':>7}  {'%':>7}")
    print(f"  {'-'*22}")
    for t in [5, 10, 15, 20, 25, 30, 40, 50, 60, 80]:
        count = int((arr < t).sum())
        pct_val = count / len(arr) * 100
        print(f"  {t:>5}  {count:>7}  {pct_val:>6.1f}%")

    return arr


# ── Analysis 2: Prefix-length sweep ───────────────────────────────────────────

def prefix_sweep(loc_train, loc_test, neg_train, neg_test,
                 prefix_lengths, epochs, batch_size, device):
    print(f"\n{'='*62}")
    print(f"  Analysis 2: Prefix-Length Sweep (1D CNN)")
    print(f"  Train: pos={len(loc_train)}, neg={len(neg_train)}")
    print(f"  Test : pos={len(loc_test)},  neg={len(neg_test)}")
    print(f"  Device: {device}, epochs={epochs}")
    print(f"{'='*62}")

    results = {}
    for T in prefix_lengths:
        print(f"\n  [Prefix T={T}]")
        train_ds = PrefixDataset(loc_train, neg_train, T)
        test_ds  = PrefixDataset(loc_test,  neg_test,  T)

        model = PrefixCNN()
        y_true, y_pred = train_eval(model, train_ds, test_ds,
                                    len(loc_train), len(neg_train),
                                    epochs, batch_size, device)
        m = compute_metrics(y_true, y_pred)
        results[T] = m
        print(f"  → F1={m['f1']:.4f}  Prec={m['prec']:.4f}  Rec={m['rec']:.4f}  "
              f"TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")

    return results


# ── Analysis 3: Per-sample latency ─────────────────────────────────────────────

def analyze_latency(loc_test, neg_test, prefix_results, pii_positions_arr):
    """
    For each prefix length T where detection fires (y_pred=1 for a positive sample),
    compute: latency = first_pii_position - T
      > 0 → detected T tokens BEFORE PII appears  ✅
      = 0 → detected exactly when PII starts       ⚠️
      < 0 → detected AFTER PII already appeared    ❌

    Simplified: use median PII first position vs T.
    """
    print(f"\n{'='*62}")
    print(f"  Analysis 3: Latency Analysis")
    print(f"  (latency = median PII position - prefix T)")
    print(f"  positive latency = detected BEFORE PII ✅")
    print(f"  negative latency = detected AFTER PII  ❌")
    print(f"{'='*62}")

    median_pii = float(np.median(pii_positions_arr))
    print(f"\n  Median PII first position: {median_pii:.1f} tokens")
    print(f"\n  {'T':>5}  {'F1':>7}  {'Recall':>8}  {'Latency':>10}  {'Feasible?':>10}")
    print(f"  {'-'*50}")

    for T, m in sorted(prefix_results.items()):
        latency = median_pii - T
        feasible = "✅ pre-PII" if latency > 0 else ("⚠️  same" if latency == 0 else "❌ post-PII")
        print(f"  {T:>5}  {m['f1']:>7.4f}  {m['rec']:>8.4f}  {latency:>+10.1f}  {feasible:>10}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    a_results = load(args.results_dir, "A_pii")
    b_results = load(args.results_dir, "B_general")
    c_results = load(args.results_dir, "C_no_context")

    a_located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    a_not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    loc_train,     loc_test     = split_samples(a_located,     args.test_ratio, args.seed)
    not_loc_train, not_loc_test = split_samples(a_not_located, args.test_ratio, args.seed)
    b_train,       b_test       = split_samples(b_results,     args.test_ratio, args.seed)
    c_test = c_results

    neg_train = not_loc_train + b_train
    neg_test  = not_loc_test  + b_test + c_test

    print(f"\n{'='*62}")
    print(f"  Real-time PII Detection Experiment")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Split       : 80% train / 20% test, seed={args.seed}")
    print(f"  Train       : pos={len(loc_train)}, neg={len(neg_train)}")
    print(f"  Test        : pos={len(loc_test)},  neg={len(neg_test)}")
    print(f"  Prefix lengths: {args.prefix_lengths}")
    print(f"{'='*62}")

    # Analysis 1: PII position distribution
    pii_positions = analyze_pii_positions(a_located)

    # Analysis 2: Prefix sweep
    prefix_results = prefix_sweep(loc_train, loc_test, neg_train, neg_test,
                                  args.prefix_lengths, args.epochs,
                                  args.batch_size, args.device)

    # Analysis 3: Latency summary
    analyze_latency(loc_test, neg_test, prefix_results, pii_positions)

    print(f"\n{'='*62}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Prefix T':>10}  {'F1':>7}  {'Precision':>10}  {'Recall':>8}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*55}")
    for T, m in sorted(prefix_results.items()):
        print(f"  {T:>10}  {m['f1']:>7.4f}  {m['prec']:>10.4f}  {m['rec']:>8.4f}"
              f"  {m['TP']:>5}  {m['FP']:>5}  {m['FN']:>5}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
