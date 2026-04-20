"""
PANORAMA experiment runner.

Runs three experiments:
  Exp1 (Zero-shot):   Train on UnlearnPII, test on PANORAMA
                      → Does the existing model generalize?

  Exp2 (PANORAMA):    Train on PANORAMA, test on PANORAMA (per content_type breakdown)
                      → True performance on diverse formats

  Exp3 (Cross-type):  Train on wiki+forum, test on review+marketplace+comment
                      → Does it generalize to unseen formats?

Prerequisites:
  1. Run prepare_panorama.py to download and preprocess PANORAMA
  2. Have existing UnlearnPII results in results_llama8b_20260401_1429/ (for Exp1)

Usage (server):
  python run_panorama_experiment.py \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --device cuda \\
      --probe-layer 24 \\
      --experiments 1 2 3

  # Quick test (local, small model):
  python run_panorama_experiment.py --max-samples 50 --experiments 1
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import asdict

from config import (
    MODEL_NAME, DEVICE, PROBE_LAYER, MAX_NEW_TOKENS,
    ENTROPY_DROP_THRESHOLD, ENTROPY_DROP_ZSCORE,
    ABS_ENTROPY_CEILING, MIN_RUN_LENGTH,
    PANORAMA_PATH, PANORAMA_PLUS_PATH,
    PROFILES_PATH,
)
from panorama_data_loader import load_panorama_samples, load_panorama_by_content_type
from run_experiment import run_single, SampleResult, _print_stats
from model_probe import ModelProbe


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PANORAMA PII detection experiments")
    parser.add_argument("--model",        type=str, default=MODEL_NAME)
    parser.add_argument("--device",       type=str, default=DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--probe-layer",  type=int, default=PROBE_LAYER)
    parser.add_argument("--max-samples",  type=int, default=None,
                        help="Max samples per split/type (None = all)")
    parser.add_argument("--results-dir",  type=str, default=None,
                        help="Output directory (default: auto-named)")
    parser.add_argument("--experiments",  type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help="Which experiments to run (default: 1 2 3)")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from checkpoints")
    return parser.parse_args()


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(probe: ModelProbe, samples: list, desc: str = "") -> list[SampleResult]:
    """Run LLaMA inference on samples, extract entropy + hidden states."""
    results = []
    for sample in tqdm(samples, desc=desc):
        try:
            r = run_single(
                probe, sample,
                threshold=ENTROPY_DROP_THRESHOLD,
                zscore=ENTROPY_DROP_ZSCORE,
                save_all_hidden=True,   # need all hidden states for Linear Probe
            )
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {getattr(sample, 'subject', '?')}: {e}")
    return results


def save_results(results: list, path: str):
    with open(path, "wb") as f:
        pickle.dump(results, f)

    # JSON summary (no hidden states, for quick inspection)
    summary = []
    for r in results:
        d = asdict(r)
        d.pop("red_flag_hidden_states", None)
        d.pop("sustained_hidden_states", None)
        d.pop("all_hidden_states", None)
        summary.append(d)
    json_path = path.replace(".pkl", "_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {path}")


# ── Classifier evaluation ─────────────────────────────────────────────────────

def build_xy_from_results(results: list[SampleResult]):
    """
    Build (X, y) arrays for CNN + Linear Probe from SampleResult list.
    X_entropy: (N, seq_len) — entropy sequences (padded/truncated to MAX_NEW_TOKENS)
    X_hidden:  (N, hidden_dim) — Layer probe hidden states at token 0 of each sample
    y:         (N,) — 1 if PII was generated, 0 otherwise
    """
    from sklearn.preprocessing import label_binarize
    import torch

    max_len = MAX_NEW_TOKENS
    X_entropy, X_hidden, y = [], [], []

    for r in results:
        has_pii = int(bool(r.pii_token_positions))

        # Entropy sequence (pad / truncate to max_len)
        ent = r.entropy_seq[:max_len]
        if len(ent) < max_len:
            ent = ent + [0.0] * (max_len - len(ent))
        X_entropy.append(ent)

        # Hidden state: use first token's hidden state if available
        if r.all_hidden_states:
            X_hidden.append(r.all_hidden_states[0])
        else:
            X_hidden.append(None)

        y.append(has_pii)

    X_entropy = np.array(X_entropy, dtype=np.float32)
    y         = np.array(y,         dtype=np.int32)
    X_hidden  = np.array([h for h in X_hidden if h is not None], dtype=np.float32) \
                if any(h is not None for h in X_hidden) else None

    return X_entropy, X_hidden, y


def evaluate_classifier(train_results: list[SampleResult],
                         test_results:  list[SampleResult],
                         label: str) -> dict:
    """
    Train CNN + Logistic Regression on train_results, evaluate on test_results.
    Returns dict with F1, precision, recall for both classifiers.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score
    import torch
    import torch.nn as nn

    print(f"\n  [{label}] Training classifiers...")

    X_train, H_train, y_train = build_xy_from_results(train_results)
    X_test,  H_test,  y_test  = build_xy_from_results(test_results)

    metrics = {}

    # ── 1. CNN on entropy sequences ───────────────────────────────────────────
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from hyperparam_tuning_v2 import TwoChannelCNN

        # Use best config from previous experiments
        cnn = TwoChannelCNN(channels=(64, 128, 256), kernels=(7, 5, 3), dropout=0.3)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        # Simple train loop (no CV, for speed)
        X_t = torch.tensor(X_train).unsqueeze(1)   # (N, 1, seq_len)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        for epoch in range(20):
            cnn.train()
            optimizer.zero_grad()
            logits = cnn(X_t).squeeze(1)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        cnn.eval()
        with torch.no_grad():
            X_te = torch.tensor(X_test).unsqueeze(1)
            preds = (torch.sigmoid(cnn(X_te).squeeze(1)) > 0.5).int().numpy()

        metrics["cnn_f1"]        = f1_score(y_test, preds, zero_division=0)
        metrics["cnn_precision"] = precision_score(y_test, preds, zero_division=0)
        metrics["cnn_recall"]    = recall_score(y_test, preds, zero_division=0)
        print(f"    CNN     F1={metrics['cnn_f1']:.4f}  "
              f"P={metrics['cnn_precision']:.4f}  R={metrics['cnn_recall']:.4f}")
    except Exception as e:
        print(f"    CNN failed: {e}")
        metrics["cnn_f1"] = metrics["cnn_precision"] = metrics["cnn_recall"] = None

    # ── 2. Logistic Regression on hidden states ───────────────────────────────
    if H_train is not None and H_test is not None:
        try:
            lr = LogisticRegression(C=0.1, max_iter=1000)
            lr.fit(H_train, y_train)
            preds_lr = lr.predict(H_test)

            metrics["lr_f1"]        = f1_score(y_test, preds_lr, zero_division=0)
            metrics["lr_precision"] = precision_score(y_test, preds_lr, zero_division=0)
            metrics["lr_recall"]    = recall_score(y_test, preds_lr, zero_division=0)
            print(f"    LR      F1={metrics['lr_f1']:.4f}  "
                  f"P={metrics['lr_precision']:.4f}  R={metrics['lr_recall']:.4f}")
        except Exception as e:
            print(f"    LR failed: {e}")
    else:
        print("    LR skipped (no hidden states)")

    metrics["n_train"] = len(train_results)
    metrics["n_test"]  = len(test_results)
    return metrics


# ── Experiments ───────────────────────────────────────────────────────────────

def experiment_1_zero_shot(probe: ModelProbe, results_dir: str, max_samples: int):
    """
    Exp1: Train on UnlearnPII results (already computed), test on PANORAMA.
    Uses existing pkl files from results_llama8b_20260401_1429/.
    """
    print(f"\n{'='*60}")
    print("Experiment 1: Zero-shot Transfer (UnlearnPII → PANORAMA)")
    print(f"{'='*60}")

    # Load existing UnlearnPII results
    unlearnpii_dir = os.path.join(os.path.dirname(__file__),
                                  "results_llama8b_20260401_1429")
    unlearnpii_pkl = os.path.join(unlearnpii_dir, "A_pii.pkl")
    if not os.path.exists(unlearnpii_pkl):
        print(f"  ERROR: UnlearnPII results not found at {unlearnpii_pkl}")
        print("  Run the original experiment first.")
        return None

    with open(unlearnpii_pkl, "rb") as f:
        train_results = pickle.load(f)
    print(f"  UnlearnPII train set: {len(train_results)} samples")

    # Generate PANORAMA test results
    pano_pkl = os.path.join(results_dir, "exp1_panorama_test.pkl")
    if os.path.exists(pano_pkl):
        print(f"  Loading cached PANORAMA results from {pano_pkl}")
        with open(pano_pkl, "rb") as f:
            test_results = pickle.load(f)
    else:
        panorama_samples = load_panorama_samples(
            PANORAMA_PATH, PANORAMA_PLUS_PATH,
            max_samples=max_samples,
        )
        print(f"  PANORAMA test set: {len(panorama_samples)} samples")
        test_results = extract_features(probe, panorama_samples, desc="Exp1 PANORAMA")
        save_results(test_results, pano_pkl)

    metrics = evaluate_classifier(train_results, test_results, "Exp1 Zero-shot")

    out_path = os.path.join(results_dir, "exp1_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {out_path}")
    return metrics


def experiment_2_panorama(probe: ModelProbe, results_dir: str, max_samples: int):
    """
    Exp2: Train on PANORAMA (train split), test on PANORAMA (test split).
    Also reports per content_type breakdown.
    """
    print(f"\n{'='*60}")
    print("Experiment 2: PANORAMA-only (train + test on PANORAMA)")
    print(f"{'='*60}")

    # Load all PANORAMA samples split by content_type
    by_type = load_panorama_by_content_type(
        PANORAMA_PATH, PANORAMA_PLUS_PATH,
        max_samples_per_type=max_samples,
    )

    all_results_by_type = {}
    for content_type, samples in by_type.items():
        if not samples:
            print(f"  [{content_type}] No samples, skipping")
            continue

        pkl_path = os.path.join(results_dir, f"exp2_{content_type}.pkl")
        if os.path.exists(pkl_path):
            print(f"  [{content_type}] Loading cache...")
            with open(pkl_path, "rb") as f:
                all_results_by_type[content_type] = pickle.load(f)
        else:
            results = extract_features(probe, samples, desc=f"Exp2 {content_type}")
            save_results(results, pkl_path)
            all_results_by_type[content_type] = results

    # Per content_type evaluation (80/20 split within each type)
    all_metrics = {}
    for content_type, results in all_results_by_type.items():
        if len(results) < 10:
            print(f"  [{content_type}] Too few samples ({len(results)}), skipping")
            continue

        split = int(len(results) * 0.8)
        train_r, test_r = results[:split], results[split:]
        metrics = evaluate_classifier(train_r, test_r, f"Exp2 {content_type}")
        all_metrics[content_type] = metrics

    # Combined evaluation (train on all types, test on all types)
    all_results = [r for rs in all_results_by_type.values() for r in rs]
    split = int(len(all_results) * 0.8)
    train_all, test_all = all_results[:split], all_results[split:]
    all_metrics["combined"] = evaluate_classifier(train_all, test_all, "Exp2 Combined")

    out_path = os.path.join(results_dir, "exp2_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved → {out_path}")
    return all_metrics


def experiment_3_cross_type(probe: ModelProbe, results_dir: str, max_samples: int):
    """
    Exp3: Train on wiki+forum, test on review+marketplace+comment.
    Tests generalization to unseen content formats.
    """
    print(f"\n{'='*60}")
    print("Experiment 3: Cross-content-type generalization")
    print("  Train: wiki + forum")
    print("  Test:  review + marketplace + comment")
    print(f"{'='*60}")

    TRAIN_TYPES = {"wiki", "forum"}
    TEST_TYPES  = {"review", "marketplace", "comment"}

    def _load_or_extract(types, tag):
        pkl_path = os.path.join(results_dir, f"exp3_{tag}.pkl")
        if os.path.exists(pkl_path):
            print(f"  Loading cached {tag} results...")
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        samples = load_panorama_samples(
            PANORAMA_PATH, PANORAMA_PLUS_PATH,
            content_types=types,
            max_samples=max_samples,
        )
        print(f"  {tag}: {len(samples)} samples")
        results = extract_features(probe, samples, desc=f"Exp3 {tag}")
        save_results(results, pkl_path)
        return results

    train_results = _load_or_extract(TRAIN_TYPES, "train_wiki_forum")
    test_results  = _load_or_extract(TEST_TYPES,  "test_review_market_comment")

    metrics = evaluate_classifier(train_results, test_results, "Exp3 Cross-type")

    out_path = os.path.join(results_dir, "exp3_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {out_path}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Auto-name results dir
    if args.results_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        args.results_dir = os.path.join(
            os.path.dirname(__file__), f"results_panorama_{ts}"
        )
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PANORAMA Experiments")
    print(f"Model      : {args.model}")
    print(f"Device     : {args.device}")
    print(f"Probe layer: {args.probe_layer}")
    print(f"Experiments: {args.experiments}")
    print(f"Results dir: {args.results_dir}")
    print(f"{'='*60}")

    # Check PANORAMA data exists
    if not os.path.exists(PANORAMA_PATH) or not os.path.exists(PANORAMA_PLUS_PATH):
        print("\nERROR: PANORAMA data not found.")
        print("Run first: python prepare_panorama.py")
        sys.exit(1)

    # Load model
    probe = ModelProbe(
        model_name=args.model,
        device=args.device,
        probe_layer=args.probe_layer,
    )

    all_metrics = {}

    if 1 in args.experiments:
        all_metrics["exp1"] = experiment_1_zero_shot(probe, args.results_dir, args.max_samples)

    if 2 in args.experiments:
        all_metrics["exp2"] = experiment_2_panorama(probe, args.results_dir, args.max_samples)

    if 3 in args.experiments:
        all_metrics["exp3"] = experiment_3_cross_type(probe, args.results_dir, args.max_samples)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    if "exp1" in all_metrics and all_metrics["exp1"]:
        m = all_metrics["exp1"]
        print(f"\nExp1 (Zero-shot, UnlearnPII→PANORAMA):")
        print(f"  CNN  F1={m.get('cnn_f1', 'N/A'):.4f}" if m.get('cnn_f1') else "  CNN  F1=N/A")
        print(f"  LR   F1={m.get('lr_f1', 'N/A'):.4f}"  if m.get('lr_f1')  else "  LR   F1=N/A")

    if "exp2" in all_metrics and all_metrics["exp2"]:
        print(f"\nExp2 (PANORAMA-only, per content_type):")
        for ct, m in all_metrics["exp2"].items():
            f1 = m.get("cnn_f1")
            print(f"  {ct:15s}: CNN F1={f1:.4f}" if f1 is not None else f"  {ct:15s}: N/A")

    if "exp3" in all_metrics and all_metrics["exp3"]:
        m = all_metrics["exp3"]
        print(f"\nExp3 (Cross-type, wiki+forum → review+marketplace+comment):")
        print(f"  CNN  F1={m.get('cnn_f1', 'N/A'):.4f}" if m.get('cnn_f1') else "  CNN  F1=N/A")
        print(f"  LR   F1={m.get('lr_f1', 'N/A'):.4f}"  if m.get('lr_f1')  else "  LR   F1=N/A")

    # Save all metrics
    with open(os.path.join(args.results_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics → {args.results_dir}/all_metrics.json")


if __name__ == "__main__":
    main()
