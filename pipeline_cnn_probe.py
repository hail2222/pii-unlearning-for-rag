"""
Two-stage Pipeline: CNN → Linear Probe

Experiment A: CNN-2ch (full 80-token entropy sequence) → Probe
Experiment B: Prefix CNN (K=5 tokens only)            → Probe

Stage 1 — CNN classifies each sample from entropy signal only.
Stage 2 — For CNN-positives, run a Linear Probe on red_flag_hidden_states.
           (red_flag_hidden_states = hidden states at ΔH-drop moments, pre-saved in pkl)

Fallback for samples flagged by CNN but with no hidden states:
  --fallback keep : treat as positive  (maximises recall)
  --fallback drop : treat as negative  (maximises precision)

Same 80/20 train/test split (seed=42) as all other experiments.

Usage:
    python pipeline_cnn_probe.py \\
        --results-dir results_llama8b_20260401_1429 \\
        --epochs 30 \\
        --device cuda \\
        --fallback keep \\
        2>&1 | tee results_llama8b_20260401_1429/pipeline_cnn_probe.log
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate as cv_fn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR, MAX_NEW_TOKENS

MAX_LEN = MAX_NEW_TOKENS  # 80


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir",  type=str,   default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--test-ratio",   type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--device",       type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prefix-k",     type=int,   default=5,
                   help="Token prefix length for Experiment B (default: 5)")
    p.add_argument("--fallback",     type=str,   choices=["keep", "drop"], default="keep",
                   help="What to predict when CNN=1 but no hidden states exist")
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


def pad_seq(seq, max_len):
    seq = np.array(seq, dtype=np.float32)
    if len(seq) >= max_len:
        return seq[:max_len]
    return np.pad(seq, (0, max_len - len(seq)), constant_values=0.0)


def delta_seq(seq, max_len):
    s = np.array(seq, dtype=np.float32)
    d = np.abs(np.diff(s, prepend=s[0] if len(s) > 0 else 0.0))
    return pad_seq(d, max_len)


def pad_prefix(seq, prefix_len):
    """Take only first prefix_len tokens, zero-pad if shorter."""
    seq = np.array(seq, dtype=np.float32)[:prefix_len]
    if len(seq) < prefix_len:
        seq = np.pad(seq, (0, prefix_len - len(seq)), constant_values=0.0)
    return seq


# ── Datasets ───────────────────────────────────────────────────────────────────

class TwoChannelDataset(Dataset):
    """CNN-2ch: entropy + Δentropy, shape (N, 2, MAX_LEN)."""
    def __init__(self, pos, neg):
        seqs, labels = [], []
        for r in pos:
            e = pad_seq(r.entropy_seq, MAX_LEN)
            d = delta_seq(r.entropy_seq, MAX_LEN)
            seqs.append(np.stack([e, d])); labels.append(1)
        for r in neg:
            e = pad_seq(r.entropy_seq, MAX_LEN)
            d = delta_seq(r.entropy_seq, MAX_LEN)
            seqs.append(np.stack([e, d])); labels.append(0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class PrefixDataset(Dataset):
    """Prefix CNN: entropy[0:K] only, shape (N, 1, K)."""
    def __init__(self, pos, neg, prefix_len):
        seqs, labels = [], []
        for r in pos:
            seqs.append(pad_prefix(r.entropy_seq, prefix_len)); labels.append(1)
        for r in neg:
            seqs.append(pad_prefix(r.entropy_seq, prefix_len)); labels.append(0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32).unsqueeze(1)  # (N,1,K)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Models ─────────────────────────────────────────────────────────────────────

class TwoChannelCNN(nn.Module):
    """Best-config CNN-2ch: channels=(64,128,256), kernels=(5,3,3), dropout=0.3."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64,  kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
    def forward(self, x): return self.head(self.net(x)).squeeze(1)


class PrefixCNN(nn.Module):
    """Lightweight 1D CNN for short prefix sequences."""
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
    def forward(self, x): return self.head(self.net(x)).squeeze(1)


# ── CNN trainer ────────────────────────────────────────────────────────────────

def train_cnn(model, train_ds, epochs, batch_size, device):
    """Train CNN with class-balanced BCE loss. Returns trained model."""
    pos_n = int(train_ds.y.sum().item())
    neg_n = len(train_ds) - pos_n
    pos_weight = torch.tensor([neg_n / pos_n], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total/len(loader):.4f}")
    return model


def predict_cnn(model, samples, dataset_cls, device, **ds_kwargs):
    """Return binary predictions (0/1) for a list of samples."""
    ds = dataset_cls(samples, [], **ds_kwargs) if ds_kwargs else dataset_cls(samples, [])
    # Need to handle empty neg list properly for datasets
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            p = (torch.sigmoid(model(X.to(device))) > 0.5).cpu().numpy().astype(int)
            preds.extend(p.tolist())
    return np.array(preds)


def predict_cnn_from_ds(model, ds, device):
    """Return binary predictions from a pre-built dataset (supports mixed pos+neg)."""
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            p = (torch.sigmoid(model(X.to(device))) > 0.5).cpu().numpy().astype(int)
            preds.extend(p.tolist())
            labels.extend(y.numpy().astype(int).tolist())
    return np.array(preds), np.array(labels)


# ── Probe ──────────────────────────────────────────────────────────────────────

def build_probe(pos_train, neg_train):
    """
    Train Linear Probe on red_flag_hidden_states.
    Positive: A_located train samples.
    Negative: A_not_located + B_general train samples.
    Returns fitted sklearn pipeline.
    """
    X_list, y_list = [], []
    for r in pos_train:
        for hs in r.red_flag_hidden_states:
            X_list.append(hs); y_list.append(1)
    for r in neg_train:
        for hs in r.red_flag_hidden_states:
            X_list.append(hs); y_list.append(0)

    if not X_list:
        raise ValueError("No red_flag_hidden_states found in training data. "
                         "Re-run experiment with hidden states saved.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    print(f"  Probe train hidden states: PII={y.sum()}, Not-PII={(y==0).sum()}, dim={X.shape[1]}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0)),
    ])
    k = min(5, int(y.sum()), int((y == 0).sum()))
    if k >= 2:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = cv_fn(pipe, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        print(f"  Probe CV Accuracy : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
        print(f"  Probe CV F1       : {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
        print(f"  Probe CV AUC      : {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")

    pipe.fit(X, y)
    return pipe


def probe_verify(sample, probe, fallback):
    """
    Run probe on red_flag_hidden_states of a single sample.
    Returns True if probe confirms PII, False otherwise.
    fallback: 'keep' → True if no hidden states (conservative)
              'drop' → False if no hidden states (aggressive filtering)
    """
    hs_list = sample.red_flag_hidden_states
    if not hs_list:
        return fallback == "keep"
    X = np.stack(hs_list).astype(np.float32)
    return bool(probe.predict(X).any())


# ── Pipeline evaluator ─────────────────────────────────────────────────────────

def evaluate_pipeline(
    label,
    cnn_preds_pos_test,   # CNN predictions on A_located test (array of 0/1)
    cnn_preds_neg_test,   # CNN predictions on neg test (array of 0/1), by segment
    pos_test,             # A_located test samples
    neg_segments,         # dict: name → list of samples (must match cnn_preds_neg_test order)
    probe,
    fallback,
):
    """
    Compute pipeline TP/FP/FN/TN and print detailed results.
    neg_segments: OrderedDict-like list of (name, samples) for FP breakdown.
    """
    print(f"\n{'='*62}")
    print(f"  Pipeline Results  [{label}]  (fallback={fallback})")
    print(f"{'='*62}")

    # Positives
    tp = fn = 0
    for i, r in enumerate(pos_test):
        if cnn_preds_pos_test[i] == 1:
            if probe_verify(r, probe, fallback):
                tp += 1
            else:
                fn += 1  # probe rejected → treat as negative → miss
        else:
            fn += 1  # CNN missed

    # Negatives — per segment
    fp_by_seg = {}
    tn_by_seg = {}
    neg_offset = 0
    for seg_name, seg_samples in neg_segments:
        n = len(seg_samples)
        seg_preds = cnn_preds_neg_test[neg_offset: neg_offset + n]
        fp_seg = tn_seg = 0
        for i, r in enumerate(seg_samples):
            if seg_preds[i] == 1:
                if probe_verify(r, probe, fallback):
                    fp_seg += 1
                else:
                    tn_seg += 1
            else:
                tn_seg += 1
        fp_by_seg[seg_name] = fp_seg
        tn_by_seg[seg_name] = tn_seg
        neg_offset += n

    fp = sum(fp_by_seg.values())
    tn = sum(tn_by_seg.values())
    n_pos = len(pos_test)
    n_neg = neg_offset

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"  TP (PII detected)           : {tp}/{n_pos}")
    print(f"  FN (PII missed)             : {fn}/{n_pos}")
    print(f"  FP (false alarm)            : {fp}/{n_neg}")
    for seg_name, fp_seg in fp_by_seg.items():
        n = dict(neg_segments)[seg_name]
        print(f"    ├ {seg_name:<22} : {fp_seg}/{len(n)}")
    print(f"  TN (correctly silent)       : {tn}/{n_neg}")
    print(f"\n  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")

    return {"f1": f1, "precision": prec, "recall": rec,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rdir = args.results_dir

    print("=" * 62)
    print("  Two-Stage Pipeline: CNN → Linear Probe")
    print(f"  Results dir : {rdir}")
    print(f"  Split       : 80% train / 20% test, seed={args.seed}")
    print(f"  Epochs      : {args.epochs}  Batch: {args.batch_size}  Device: {args.device}")
    print(f"  Prefix K    : {args.prefix_k}  Fallback: {args.fallback}")
    print("=" * 62)

    # ── Load data ──────────────────────────────────────────────────────────────
    a_results = load(rdir, "A_pii")
    b_results = load(rdir, "B_general")
    c_results = load(rdir, "C_no_context")

    a_located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    a_not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    loc_train, loc_test         = split_samples(a_located,     args.test_ratio, args.seed)
    notloc_train, notloc_test   = split_samples(a_not_located, args.test_ratio, args.seed)
    b_train, b_test             = split_samples(b_results,     args.test_ratio, args.seed)
    c_test = c_results  # C never used for training

    neg_train = notloc_train + b_train

    print(f"\n  Train — pos={len(loc_train)}, neg={len(neg_train)}")
    print(f"  Test  — pos={len(loc_test)},  "
          f"neg={len(notloc_test)+len(b_test)+len(c_test)} "
          f"(not_loc={len(notloc_test)}, B={len(b_test)}, C={len(c_test)})")

    neg_segments = [
        ("A-not-located", notloc_test),
        ("B (general)",   b_test),
        ("C (no context)", c_test),
    ]
    neg_test_all = notloc_test + b_test + c_test

    # ── Train Linear Probe (shared across both experiments) ────────────────────
    print("\n" + "=" * 62)
    print("  Training Linear Probe (red_flag_hidden_states)")
    print("=" * 62)
    probe = build_probe(loc_train, neg_train)

    # ══════════════════════════════════════════════════════════════════════════
    #  Experiment A: CNN-2ch (full 80 tokens) → Probe
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("  Experiment A: CNN-2ch (full sequence) → Probe")
    print("=" * 62)

    train_ds_a = TwoChannelDataset(loc_train, neg_train)
    test_ds_a  = TwoChannelDataset(loc_test,  neg_test_all)

    model_a = TwoChannelCNN()
    print(f"  Model params: {sum(p.numel() for p in model_a.parameters()):,}")
    model_a = train_cnn(model_a, train_ds_a, args.epochs, args.batch_size, args.device)

    # CNN predictions on test set
    preds_a, labels_a = predict_cnn_from_ds(model_a, test_ds_a, args.device)
    cnn_preds_pos_a = preds_a[:len(loc_test)]
    cnn_preds_neg_a = preds_a[len(loc_test):]

    # CNN-only metrics (before probe)
    f1_cnn_a   = f1_score(labels_a, preds_a, zero_division=0)
    prec_cnn_a = precision_score(labels_a, preds_a, zero_division=0)
    rec_cnn_a  = recall_score(labels_a, preds_a, zero_division=0)
    tp_a = int(((preds_a == 1) & (labels_a == 1)).sum())
    fp_a = int(((preds_a == 1) & (labels_a == 0)).sum())
    fn_a = int(((preds_a == 0) & (labels_a == 1)).sum())
    print(f"\n  [CNN-2ch only]  F1={f1_cnn_a:.4f}  Prec={prec_cnn_a:.4f}  Rec={rec_cnn_a:.4f}"
          f"  TP={tp_a}  FP={fp_a}  FN={fn_a}")

    # Pipeline metrics (CNN → Probe)
    results_a = evaluate_pipeline(
        "CNN-2ch → Probe",
        cnn_preds_pos_a, cnn_preds_neg_a,
        loc_test, neg_segments, probe, args.fallback,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  Experiment B: Prefix CNN (K=5) → Probe
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print(f"  Experiment B: Prefix CNN (K={args.prefix_k}) → Probe")
    print("=" * 62)

    train_ds_b = PrefixDataset(loc_train, neg_train, args.prefix_k)
    test_ds_b  = PrefixDataset(loc_test,  neg_test_all, args.prefix_k)

    model_b = PrefixCNN()
    print(f"  Model params: {sum(p.numel() for p in model_b.parameters()):,}")
    model_b = train_cnn(model_b, train_ds_b, args.epochs, args.batch_size, args.device)

    preds_b, labels_b = predict_cnn_from_ds(model_b, test_ds_b, args.device)
    cnn_preds_pos_b = preds_b[:len(loc_test)]
    cnn_preds_neg_b = preds_b[len(loc_test):]

    f1_cnn_b   = f1_score(labels_b, preds_b, zero_division=0)
    prec_cnn_b = precision_score(labels_b, preds_b, zero_division=0)
    rec_cnn_b  = recall_score(labels_b, preds_b, zero_division=0)
    tp_b = int(((preds_b == 1) & (labels_b == 1)).sum())
    fp_b_val = int(((preds_b == 1) & (labels_b == 0)).sum())
    fn_b = int(((preds_b == 0) & (labels_b == 1)).sum())
    print(f"\n  [Prefix CNN-{args.prefix_k} only]  F1={f1_cnn_b:.4f}  Prec={prec_cnn_b:.4f}  Rec={rec_cnn_b:.4f}"
          f"  TP={tp_b}  FP={fp_b_val}  FN={fn_b}")

    results_b = evaluate_pipeline(
        f"Prefix CNN (K={args.prefix_k}) → Probe",
        cnn_preds_pos_b, cnn_preds_neg_b,
        loc_test, neg_segments, probe, args.fallback,
    )

    # ── Final comparison ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print(f"  Test set: pos={len(loc_test)}, neg={len(neg_test_all)}")
    print("=" * 62)
    print(f"  {'Method':<38}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {'-'*72}")

    rows = [
        ("CNN-2ch only (80 tokens)",      f1_cnn_a,        prec_cnn_a,        rec_cnn_a,
         tp_a, fp_a, fn_a),
        ("CNN-2ch → Probe",               results_a["f1"], results_a["precision"], results_a["recall"],
         results_a["TP"], results_a["FP"], results_a["FN"]),
        (f"Prefix CNN-{args.prefix_k} only",  f1_cnn_b,   prec_cnn_b,        rec_cnn_b,
         tp_b, fp_b_val, fn_b),
        (f"Prefix CNN-{args.prefix_k} → Probe", results_b["f1"], results_b["precision"], results_b["recall"],
         results_b["TP"], results_b["FP"], results_b["FN"]),
    ]

    for name, f1, prec, rec, tp, fp, fn in rows:
        print(f"  {name:<38}  {f1:6.4f}  {prec:6.4f}  {rec:6.4f}  {tp:4d}  {fp:4d}  {fn:4d}")

    print("=" * 62)
    print("\nNote:")
    print(f"  Probe uses red_flag_hidden_states (ΔH-drop moments, layer 24, dim=4096)")
    print(f"  Fallback='{args.fallback}': CNN+ samples with no hidden states → "
          f"{'kept positive' if args.fallback == 'keep' else 'dropped to negative'}")

    # Coverage stats
    no_hs_pos = sum(1 for r in loc_test if not r.red_flag_hidden_states)
    no_hs_neg = sum(1 for r in neg_test_all if not r.red_flag_hidden_states)
    print(f"\n  Hidden state coverage:")
    print(f"    pos test with hidden states : {len(loc_test)-no_hs_pos}/{len(loc_test)} "
          f"({(len(loc_test)-no_hs_pos)/len(loc_test)*100:.1f}%)")
    print(f"    neg test with hidden states : {len(neg_test_all)-no_hs_neg}/{len(neg_test_all)} "
          f"({(len(neg_test_all)-no_hs_neg)/len(neg_test_all)*100:.1f}%)")


if __name__ == "__main__":
    main()
