"""
Two-stage Pipeline for PANORAMA: CNN (entropy) → Linear Probe (hidden states)

Loads PANORAMA results from run_panorama_experiment.py (Exp2 pkl files).
Splits samples into positive (pii_token_positions non-empty) and negative.

Stage 1: TwoChannelCNN on entropy + Δentropy sequence
Stage 2: Linear Probe on red_flag_hidden_states to filter false positives

Usage:
    python pipeline_panorama.py \\
        --results-dir results_panorama_20260421_0137 \\
        --device cuda \\
        --fallback keep
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import f1_score, precision_score, recall_score

from run_experiment import SampleResult
from config import MAX_NEW_TOKENS

MAX_LEN = MAX_NEW_TOKENS  # 80

ALL_CONTENT_TYPES = [
    "Article",
    "Online_Review",
    "Forum_Post",
    "Online_Ad",
    "Social_Media",
    "Blog_News_Article_Comment",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, required=True,
                   help="Directory with exp2_*.pkl files")
    p.add_argument("--test-ratio",  type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fallback",    type=str,   choices=["keep", "drop"], default="keep")
    p.add_argument("--max-samples", type=int,   default=None,
                   help="Limit total samples loaded (for quick testing)")
    p.add_argument("--train-dir",   type=str,   default=None,
                   help="If set, load train data from this dir (A_pii.pkl) instead of PANORAMA split. "
                        "Enables zero-shot transfer: train on UnlearnPII, test on PANORAMA.")
    return p.parse_args()


# ── Data utils ────────────────────────────────────────────────────────────────

def load_panorama_results(results_dir: str, max_samples: int = None):
    """Load all exp2_*.pkl files and return (all_results, per_type_dict)."""
    all_results = []
    per_type = {}
    for ct in ALL_CONTENT_TYPES:
        pkl_path = os.path.join(results_dir, f"exp2_{ct}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  [WARN] Not found: {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
        # Tag each result with its content type
        for r in results:
            r._content_type = ct
        per_type[ct] = results
        all_results.extend(results)
        pos = sum(1 for r in results if r.pii_token_positions)
        print(f"  [{ct:35s}] {len(results):4d} samples  (pos={pos}, neg={len(results)-pos})")
        if max_samples and len(all_results) >= max_samples:
            break
    if max_samples:
        all_results = all_results[:max_samples]
    return all_results, per_type


def stratified_split(results: list, test_ratio: float, seed: int):
    """
    Stratified split: keep pos/neg ratio in both train and test.
    Returns (train, test).
    """
    rng = np.random.default_rng(seed)
    pos = [r for r in results if r.pii_token_positions]
    neg = [r for r in results if not r.pii_token_positions]

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_test = max(1, int(len(pos) * test_ratio))
    n_neg_test = max(1, int(len(neg) * test_ratio))

    pos_test, pos_train = pos[:n_pos_test], pos[n_pos_test:]
    neg_test, neg_train = neg[:n_neg_test], neg[n_neg_test:]

    train = pos_train + neg_train
    test  = pos_test  + neg_test
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def pad_seq(seq, max_len):
    seq = np.array(seq, dtype=np.float32)
    if len(seq) >= max_len:
        return seq[:max_len]
    return np.pad(seq, (0, max_len - len(seq)), constant_values=0.0)


def delta_seq(seq, max_len):
    s = np.array(seq, dtype=np.float32)
    d = np.abs(np.diff(s, prepend=s[0] if len(s) > 0 else 0.0))
    return pad_seq(d, max_len)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TwoChannelDataset(Dataset):
    """2-channel: entropy + Δentropy. Shape (N, 2, MAX_LEN)."""
    def __init__(self, results: list):
        seqs, labels = [], []
        for r in results:
            e = pad_seq(r.entropy_seq, MAX_LEN)
            d = delta_seq(r.entropy_seq, MAX_LEN)
            seqs.append(np.stack([e, d]))
            labels.append(1 if r.pii_token_positions else 0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── CNN model ─────────────────────────────────────────────────────────────────

class TwoChannelCNN(nn.Module):
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


# ── CNN training ──────────────────────────────────────────────────────────────

def train_cnn(model, train_ds, epochs, batch_size, device):
    pos_n = int(train_ds.y.sum().item())
    neg_n = len(train_ds) - pos_n
    pos_weight = torch.tensor([neg_n / max(pos_n, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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


def predict_cnn(model, ds, device):
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            p = (torch.sigmoid(model(X.to(device))) > 0.5).cpu().numpy().astype(int)
            preds.extend(p.tolist())
            labels.extend(y.numpy().astype(int).tolist())
    return np.array(preds), np.array(labels)


# ── Linear Probe ──────────────────────────────────────────────────────────────

def build_probe(train_results: list):
    """Train probe on red_flag_hidden_states."""
    X_list, y_list = [], []
    for r in train_results:
        label = 1 if r.pii_token_positions else 0
        for hs in r.red_flag_hidden_states:
            X_list.append(hs)
            y_list.append(label)

    if not X_list:
        print("  [WARN] No red_flag_hidden_states found. Probe skipped.")
        return None

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    print(f"  Probe: {y.sum()} PII / {(y==0).sum()} non-PII hidden states  dim={X.shape[1]}")

    pipe = SkPipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0)),
    ])
    pipe.fit(X, y)
    return pipe


def probe_verify(r: SampleResult, probe, fallback: str) -> bool:
    if not r.red_flag_hidden_states:
        return fallback == "keep"
    X = np.stack(r.red_flag_hidden_states).astype(np.float32)
    return bool(probe.predict(X).any())


# ── Pipeline evaluation ───────────────────────────────────────────────────────

def evaluate_pipeline(test_results, cnn_preds, probe, fallback, label):
    y_true = np.array([1 if r.pii_token_positions else 0 for r in test_results])

    # CNN-only metrics
    cnn_f1   = f1_score(y_true, cnn_preds, zero_division=0)
    cnn_prec = precision_score(y_true, cnn_preds, zero_division=0)
    cnn_rec  = recall_score(y_true, cnn_preds, zero_division=0)
    print(f"\n  [CNN only]  F1={cnn_f1:.4f}  P={cnn_prec:.4f}  R={cnn_rec:.4f}")

    if probe is None:
        return {"cnn_f1": cnn_f1, "cnn_precision": cnn_prec, "cnn_recall": cnn_rec}

    # LR-only metrics (probe applied to all samples, no CNN filter)
    lr_preds = np.array([1 if probe_verify(r, probe, fallback) else 0 for r in test_results])
    lr_f1   = f1_score(y_true, lr_preds, zero_division=0)
    lr_prec = precision_score(y_true, lr_preds, zero_division=0)
    lr_rec  = recall_score(y_true, lr_preds, zero_division=0)
    print(f"  [LR  only]  F1={lr_f1:.4f}  P={lr_prec:.4f}  R={lr_rec:.4f}")

    # CNN → Probe pipeline
    pipeline_preds = []
    for i, r in enumerate(test_results):
        if cnn_preds[i] == 1:
            pipeline_preds.append(1 if probe_verify(r, probe, fallback) else 0)
        else:
            pipeline_preds.append(0)
    pipeline_preds = np.array(pipeline_preds)

    pip_f1   = f1_score(y_true, pipeline_preds, zero_division=0)
    pip_prec = precision_score(y_true, pipeline_preds, zero_division=0)
    pip_rec  = recall_score(y_true, pipeline_preds, zero_division=0)

    tp = int(((pipeline_preds == 1) & (y_true == 1)).sum())
    fp = int(((pipeline_preds == 1) & (y_true == 0)).sum())
    fn = int(((pipeline_preds == 0) & (y_true == 1)).sum())
    tn = int(((pipeline_preds == 0) & (y_true == 0)).sum())

    print(f"  [Pipeline]  F1={pip_f1:.4f}  P={pip_prec:.4f}  R={pip_rec:.4f}"
          f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return {
        "cnn_f1": cnn_f1, "cnn_precision": cnn_prec, "cnn_recall": cnn_rec,
        "lr_f1": lr_f1, "lr_precision": lr_prec, "lr_recall": lr_rec,
        "pipeline_f1": pip_f1, "pipeline_precision": pip_prec, "pipeline_recall": pip_rec,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    mode = "zero-shot (UnlearnPII→PANORAMA)" if args.train_dir else "PANORAMA-only"
    print("=" * 62)
    print(f"  PANORAMA Pipeline: TwoChannelCNN → Linear Probe  [{mode}]")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Train dir   : {args.train_dir or 'PANORAMA split'}")
    print(f"  Fallback    : {args.fallback}")
    print(f"  Device      : {args.device}")
    print("=" * 62)

    import json
    ckpt_suffix = "zeroshot" if args.train_dir else "panorama"
    ckpt_dir = os.path.join(args.results_dir, f"pipeline_ckpt_{ckpt_suffix}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load PANORAMA results (always used as test)
    print("\nLoading PANORAMA results...")
    all_results, per_type_results = load_panorama_results(args.results_dir, args.max_samples)

    pos_total = sum(1 for r in all_results if r.pii_token_positions)
    neg_total = len(all_results) - pos_total
    print(f"\nTotal PANORAMA: {len(all_results)} samples  (pos={pos_total}, neg={neg_total})")

    if args.train_dir:
        # Zero-shot: train on UnlearnPII A_pii, test on all PANORAMA
        train_pkl = os.path.join(args.train_dir, "A_pii.pkl")
        with open(train_pkl, "rb") as f:
            train = pickle.load(f)
        test = all_results
        print(f"Train (UnlearnPII): {len(train)} samples")
    else:
        # PANORAMA-only: stratified split
        train, test = stratified_split(all_results, args.test_ratio, args.seed)

    pos_train = sum(1 for r in train if r.pii_token_positions)
    pos_test  = sum(1 for r in test  if r.pii_token_positions)
    print(f"Train: {len(train)} (pos={pos_train}, neg={len(train)-pos_train})")
    print(f"Test : {len(test)}  (pos={pos_test},  neg={len(test)-pos_test})")

    # ── Stage 1: Train CNN ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Stage 1: Training TwoChannelCNN")
    print("=" * 62)

    cnn_ckpt   = os.path.join(ckpt_dir, "cnn.pt")
    preds_ckpt = os.path.join(ckpt_dir, "cnn_preds.pkl")

    train_ds = TwoChannelDataset(train)
    test_ds  = TwoChannelDataset(test)
    model    = TwoChannelCNN().to(args.device)

    if os.path.exists(cnn_ckpt) and os.path.exists(preds_ckpt):
        print(f"  Resuming: loading CNN from {cnn_ckpt}")
        model.load_state_dict(torch.load(cnn_ckpt, map_location=args.device))
        with open(preds_ckpt, "rb") as f:
            cnn_preds = pickle.load(f)
    else:
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        model = train_cnn(model, train_ds, args.epochs, args.batch_size, args.device)
        torch.save(model.state_dict(), cnn_ckpt)
        print(f"  CNN saved → {cnn_ckpt}")
        cnn_preds, _ = predict_cnn(model, test_ds, args.device)
        with open(preds_ckpt, "wb") as f:
            pickle.dump(cnn_preds, f)
        print(f"  CNN preds saved → {preds_ckpt}")

    # ── Stage 2: Train Probe ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Stage 2: Training Linear Probe on red_flag_hidden_states")
    print("=" * 62)

    probe_ckpt = os.path.join(ckpt_dir, "probe.pkl")
    if os.path.exists(probe_ckpt):
        print(f"  Resuming: loading probe from {probe_ckpt}")
        with open(probe_ckpt, "rb") as f:
            probe = pickle.load(f)
    else:
        probe = build_probe(train)
        if probe is not None:
            with open(probe_ckpt, "wb") as f:
                pickle.dump(probe, f)
            print(f"  Probe saved → {probe_ckpt}")

    # ── Evaluate Pipeline ─────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Pipeline Evaluation (Combined all content types)")
    print("=" * 62)
    metrics = evaluate_pipeline(test, cnn_preds, probe, args.fallback, "Combined")

    # ── Per-type evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Per-type Pipeline Evaluation (combined model → per-type test)")
    print("=" * 62)

    # Build index: test sample → index in cnn_preds
    test_idx_map = {id(r): i for i, r in enumerate(test)}

    per_type_metrics = {}
    for ct in ALL_CONTENT_TYPES:
        if ct not in per_type_results:
            continue
        # Filter test samples belonging to this content type
        ct_test = [r for r in test if getattr(r, "_content_type", None) == ct]
        if not ct_test:
            continue
        ct_idxs = [test_idx_map[id(r)] for r in ct_test]
        ct_preds = cnn_preds[ct_idxs]
        ct_metrics = evaluate_pipeline(ct_test, ct_preds, probe, args.fallback, ct)
        per_type_metrics[ct] = ct_metrics
        print(f"  Content type: {ct}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, "pipeline_metrics.json")
    full_metrics = {"combined": metrics, "per_type": per_type_metrics}
    with open(out_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print("=" * 62)
    print(f"  {'Content Type':<35}  {'CNN F1':>7}  {'LR F1':>7}  {'Pipe F1':>7}  {'Pipe P':>7}  {'Pipe R':>7}")
    print(f"  {'-'*77}")
    lr_f1_c   = metrics.get("lr_f1", float("nan"))
    pipe_f1_c = metrics.get("pipeline_f1", float("nan"))
    pipe_p_c  = metrics.get("pipeline_precision", float("nan"))
    pipe_r_c  = metrics.get("pipeline_recall", float("nan"))
    print(f"  {'Combined':<35}  {metrics['cnn_f1']:7.4f}  {lr_f1_c:7.4f}  {pipe_f1_c:7.4f}  {pipe_p_c:7.4f}  {pipe_r_c:7.4f}")
    for ct, m in per_type_metrics.items():
        lr_f1   = m.get("lr_f1", float("nan"))
        pipe_f1 = m.get("pipeline_f1", float("nan"))
        pipe_p  = m.get("pipeline_precision", float("nan"))
        pipe_r  = m.get("pipeline_recall", float("nan"))
        print(f"  {ct:<35}  {m['cnn_f1']:7.4f}  {lr_f1:7.4f}  {pipe_f1:7.4f}  {pipe_p:7.4f}  {pipe_r:7.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
