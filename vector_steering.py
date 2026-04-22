"""
Vector Steering Defense for PII Suppression

Two-pass pipeline:
  Pass 1 (Detection):
    - Generate normally with generate_with_probes()
    - CNN classifies entropy sequence → PII candidate
    - LR (probe) verifies red_flag_hidden_states → confirmed PII

  Pass 2 (Steering, only for confirmed PII):
    - Re-generate same prompt with forward hook active
    - Hook: h_layer += alpha * v_steer  (applied from token 0)
    - Check if original PII string is absent from new generation

Steering vector:
    v_pii     = mean(red_flag_hidden_states | PII train samples)
    v_non_pii = mean(red_flag_hidden_states | non-PII train samples)
    v_steer   = normalize(v_non_pii - v_pii)

Metrics:
  - Detection rate     : CNN+LR detected / total PII samples
  - Suppression rate   : PII removed / CNN+LR detected
  - Overall rate       : PII removed / total PII samples
  - False trigger rate : CNN+LR fired on non-PII / total non-PII

Usage:
    python vector_steering.py \\
        --results-dir results_panorama_20260421_1149 \\
        --pipeline-ckpt results_panorama_20260421_1149/pipeline_ckpt_panorama \\
        --model-name meta-llama/Llama-3.1-8B-Instruct \\
        --alpha 20.0 \\
        --device cuda
"""

import os
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from model_probe import ModelProbe
from config import MODEL_NAME, PROBE_LAYER, MAX_NEW_TOKENS, DEVICE
from pipeline_panorama import (
    load_panorama_results, stratified_split,
    TwoChannelCNN, TwoChannelDataset, predict_cnn,
    probe_verify,
)


# ── Steering vector ───────────────────────────────────────────────────────────

def compute_steering_vector(train_results: list) -> np.ndarray:
    """v_steer = normalize(mean(non-PII red_flag_hs) - mean(PII red_flag_hs))"""
    pii_hs, non_pii_hs = [], []
    for r in train_results:
        hs_list = getattr(r, "red_flag_hidden_states", [])
        if not hs_list:
            continue
        if r.pii_token_positions:
            pii_hs.extend(hs_list)
        else:
            non_pii_hs.extend(hs_list)

    if not pii_hs:
        raise ValueError("No PII red_flag_hidden_states in training data.")

    v_pii = np.mean(np.stack(pii_hs), axis=0).astype(np.float32)
    if non_pii_hs:
        v_non_pii = np.mean(np.stack(non_pii_hs), axis=0).astype(np.float32)
        v_steer = v_non_pii - v_pii
    else:
        print("  [WARN] No non-PII hidden states; using -v_pii direction.")
        v_steer = -v_pii

    norm = np.linalg.norm(v_steer)
    if norm > 1e-8:
        v_steer = v_steer / norm

    print(f"  PII hidden states    : {len(pii_hs)}")
    print(f"  Non-PII hidden states: {len(non_pii_hs)}")
    print(f"  ||v_steer||          : {np.linalg.norm(v_steer):.4f}")
    return v_steer


# ── Pass 1: CNN + LR detection (reuse pipeline checkpoint) ───────────────────

def load_pipeline(ckpt_dir: str, device: str):
    """Load saved CNN and probe from pipeline_panorama.py checkpoint."""
    cnn = TwoChannelCNN().to(device)
    cnn.load_state_dict(torch.load(
        os.path.join(ckpt_dir, "cnn.pt"), map_location=device
    ))
    cnn.eval()

    with open(os.path.join(ckpt_dir, "probe.pkl"), "rb") as f:
        probe_clf = pickle.load(f)

    print(f"  CNN + probe loaded from {ckpt_dir}")
    return cnn, probe_clf


def detect_with_pipeline(results: list, cnn, probe_clf, device: str, fallback: str = "keep"):
    """
    Run CNN on entropy sequences, then LR on red_flag_hidden_states.
    Returns list of bool: True = detected as PII.
    """
    ds = TwoChannelDataset(results)
    cnn_preds, _ = predict_cnn(cnn, ds, device)

    pipeline_preds = []
    for i, r in enumerate(results):
        if cnn_preds[i] == 1:
            pipeline_preds.append(probe_verify(r, probe_clf, fallback))
        else:
            pipeline_preds.append(False)
    return pipeline_preds


# ── Pass 2: Steered re-generation ────────────────────────────────────────────

def generate_steered(
    probe: ModelProbe,
    prompt: str,
    steering_vec: torch.Tensor,
    alpha: float,
    steer_from_token: int = 0,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Re-generate prompt with steering hook active from token `steer_from_token`.
    Returns the generated text.
    """
    if probe.is_instruct:
        messages = [{"role": "user", "content": prompt}]
        formatted = probe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    inputs = probe.tokenizer(formatted, return_tensors="pt").to(probe.device)
    current_ids = inputs["input_ids"]

    sv = steering_vec.to(probe.device)
    step_counter = {"n": 0}

    def steering_hook(module, input, output):
        if step_counter["n"] < steer_from_token:
            return output
        if isinstance(output, torch.Tensor):
            hidden = output.clone()
            if hidden.dim() == 3:
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * sv
            elif hidden.dim() == 2:
                hidden[-1, :] = hidden[-1, :] + alpha * sv
            return hidden
        else:
            raw = output[0]
            hidden = raw.clone()
            if hidden.dim() == 3:
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * sv
            elif hidden.dim() == 2:
                hidden[-1, :] = hidden[-1, :] + alpha * sv
            return (hidden,) + output[1:]

    layer = probe.model.model.layers[probe.probe_layer]
    hook_handle = layer.register_forward_hook(steering_hook)

    generated_ids = []
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = probe.model(input_ids=current_ids)
                step_logits = outputs.logits[0, -1, :]
                next_token_id = int(step_logits.argmax())
                generated_ids.append(next_token_id)
                step_counter["n"] += 1

                if next_token_id == probe.tokenizer.eos_token_id:
                    break

                current_ids = torch.cat(
                    [current_ids,
                     torch.tensor([[next_token_id]], device=probe.device)],
                    dim=1,
                )
    finally:
        hook_handle.remove()

    return probe.tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def steer_from(r, default: int = 0) -> int:
    """Return the token index to start steering from (first red_flag - 1)."""
    if r.red_flag_indices:
        return max(0, min(r.red_flag_indices) - 1)
    return default


def get_pii_string(r) -> str:
    if not r.pii_token_positions or not r.tokens:
        return ""
    return "".join(
        r.tokens[i] for i in r.pii_token_positions if i < len(r.tokens)
    ).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",    required=True)
    parser.add_argument("--pipeline-ckpt",  required=True,
                        help="Dir with cnn.pt and probe.pkl (from pipeline_panorama.py)")
    parser.add_argument("--model-name",     default=MODEL_NAME)
    parser.add_argument("--alpha",          type=float, default=20.0,
                        help="Single alpha value (used when --alpha-sweep not set)")
    parser.add_argument("--alpha-sweep",    type=float, nargs="+", default=None,
                        help="Sweep multiple alpha values, e.g. --alpha-sweep 5 10 20 40 80")
    parser.add_argument("--steer-from",     type=int,   default=0,
                        help="Apply steering from this token index (0 = always on)")
    parser.add_argument("--test-ratio",     type=float, default=0.2)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--device",         type=str,   default=DEVICE)
    parser.add_argument("--n-samples",      type=int,   default=None,
                        help="Max PII/non-PII test samples each")
    args = parser.parse_args()

    print("=" * 62)
    print("  Vector Steering Defense  (CNN+LR → Steer)")
    print(f"  Results dir   : {args.results_dir}")
    print(f"  Pipeline ckpt : {args.pipeline_ckpt}")
    print(f"  Model         : {args.model_name}")
    print(f"  Alpha         : {args.alpha}")
    print(f"  Steer from    : token {args.steer_from}")
    print(f"  Device        : {args.device}")
    print("=" * 62)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading PANORAMA results...")
    all_results, _ = load_panorama_results(args.results_dir)
    train, test = stratified_split(all_results, args.test_ratio, args.seed)

    pii_test = [r for r in test if r.pii_token_positions]
    neg_test = [r for r in test if not r.pii_token_positions]
    if args.n_samples:
        pii_test = pii_test[:args.n_samples]
        neg_test = neg_test[:args.n_samples]

    print(f"  Train: {len(train)}  |  Test PII: {len(pii_test)}  |  Test non-PII: {len(neg_test)}")

    # ── Steering vector ───────────────────────────────────────────────────────
    print("\nComputing steering vector from train data...")
    v_steer_np = compute_steering_vector(train)
    v_steer = torch.tensor(v_steer_np, dtype=torch.float16)

    # ── Load CNN + LR pipeline ────────────────────────────────────────────────
    print("\nLoading CNN + LR pipeline checkpoint...")
    cnn, probe_clf = load_pipeline(args.pipeline_ckpt, args.device)

    # ── Pass 1: Detect with CNN + LR ─────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Pass 1: CNN + LR Detection")
    print("=" * 62)

    pii_detected  = detect_with_pipeline(pii_test, cnn, probe_clf, args.device)
    neg_detected  = detect_with_pipeline(neg_test, cnn, probe_clf, args.device)

    n_pii_detected = sum(pii_detected)
    n_false_trigger = sum(neg_detected)
    detection_rate  = n_pii_detected / len(pii_test) if pii_test else 0
    false_trig_rate = n_false_trigger / len(neg_test) if neg_test else 0

    print(f"  PII detected        : {n_pii_detected}/{len(pii_test)}  ({detection_rate:.1%})")
    print(f"  False triggers      : {n_false_trigger}/{len(neg_test)}  ({false_trig_rate:.1%})")

    # ── Load model for re-generation ─────────────────────────────────────────
    print("\nLoading language model for steered re-generation...")
    lm_probe = ModelProbe(
        model_name=args.model_name,
        device=args.device,
        probe_layer=PROBE_LAYER,
    )

    # ── Pass 2: Steered re-generation (single or sweep) ──────────────────────
    alpha_list = args.alpha_sweep if args.alpha_sweep else [args.alpha]
    detected_pii = [(r, steer_from(r, args.steer_from)) for r, d in zip(pii_test, pii_detected) if d]

    n_pii = len(pii_test)
    all_alpha_metrics = {}
    sweep_summary = []

    for alpha in alpha_list:
        print("\n" + "=" * 62)
        print(f"  Pass 2: Steered Re-generation  alpha={alpha}  ({n_pii_detected} samples)")
        print("=" * 62)

        n_suppressed = 0
        sample_logs = []
        per_type_counts = {}  # {ct: {"n_detected": , "n_suppressed": , "n_pii": }}

        for r, sf in tqdm(detected_pii, desc=f"alpha={alpha}"):
            pii_str = get_pii_string(r)
            if not pii_str:
                continue
            ct = getattr(r, "_content_type", "unknown")

            new_text = generate_steered(
                lm_probe, r.prompt, v_steer, alpha,
                steer_from_token=sf,
            )

            suppressed = pii_str.lower() not in new_text.lower()
            if suppressed:
                n_suppressed += 1

            if ct not in per_type_counts:
                per_type_counts[ct] = {"n_detected": 0, "n_suppressed": 0}
            per_type_counts[ct]["n_detected"] += 1
            if suppressed:
                per_type_counts[ct]["n_suppressed"] += 1

            sample_logs.append({
                "content_type": ct,
                "pii_string": pii_str,
                "suppressed": suppressed,
                "steer_from": sf,
                "original_text_snippet": r.generated_text[:100],
                "steered_text_snippet": new_text[:100],
            })

        suppression_rate = n_suppressed / n_pii_detected if n_pii_detected > 0 else 0
        overall_rate     = n_suppressed / n_pii if n_pii > 0 else 0

        # Per-type suppression rates
        n_pii_per_type = {}
        for r in pii_test:
            ct = getattr(r, "_content_type", "unknown")
            n_pii_per_type[ct] = n_pii_per_type.get(ct, 0) + 1

        per_type_metrics = {}
        for ct, counts in per_type_counts.items():
            nd = counts["n_detected"]
            ns = counts["n_suppressed"]
            np_ct = n_pii_per_type.get(ct, 0)
            per_type_metrics[ct] = {
                "n_pii": np_ct,
                "n_detected": nd,
                "n_suppressed": ns,
                "detection_rate": nd / np_ct if np_ct > 0 else 0,
                "suppression_rate": ns / nd if nd > 0 else 0,
                "overall_rate": ns / np_ct if np_ct > 0 else 0,
            }

        all_alpha_metrics[str(alpha)] = {
            "alpha": alpha,
            "n_pii_detected": n_pii_detected,
            "n_suppressed": n_suppressed,
            "detection_rate": detection_rate,
            "suppression_rate": suppression_rate,
            "overall_suppression_rate": overall_rate,
            "false_trigger_rate": false_trig_rate,
            "per_type": per_type_metrics,
            "sample_logs": sample_logs,
        }
        sweep_summary.append((alpha, detection_rate, suppression_rate, overall_rate))

        print(f"  alpha={alpha:6.1f}  suppress={suppression_rate:.1%}  overall={overall_rate:.1%}")
        print(f"  {'Content Type':<35}  {'Detected':>8}  {'Suppressed':>10}  {'Overall':>8}")
        print(f"  {'-'*65}")
        for ct, m in per_type_metrics.items():
            print(f"  {ct:<35}  {m['n_detected']:>8}  {m['n_suppressed']:>9} ({m['suppression_rate']:.1%})  {m['overall_rate']:.1%}")

        # ── Save after each alpha (crash-safe) ───────────────────────────────
        out_path = os.path.join(args.results_dir, "steering_metrics.json")
        metrics_so_far = {
            "n_pii_test": n_pii,
            "n_neg_test": len(neg_test),
            "n_pii_detected": n_pii_detected,
            "n_false_trigger": n_false_trigger,
            "detection_rate": detection_rate,
            "false_trigger_rate": false_trig_rate,
            "alpha_results": all_alpha_metrics,
        }
        with open(out_path, "w") as f:
            json.dump(metrics_so_far, f, indent=2)
        print(f"  Saved → {out_path}")

    print("\n" + "=" * 62)
    print("  STEERING RESULTS SUMMARY")
    print("=" * 62)
    print(f"  Total PII test samples  : {n_pii}")
    print(f"  ├─ Detected (CNN+LR)    : {n_pii_detected}  ({detection_rate:.1%})")
    print(f"  └─ False trigger        : {n_false_trigger}/{len(neg_test)}  ({false_trig_rate:.1%})")
    print()
    print(f"  {'Alpha':>8}  {'Suppress/Detected':>18}  {'Overall':>8}")
    print(f"  {'-'*40}")
    for alpha, det, sup, ov in sweep_summary:
        n_sup = all_alpha_metrics[str(alpha)]["n_suppressed"]
        print(f"  {alpha:8.1f}  {n_sup}/{n_pii_detected} ({sup:.1%})          {ov:.1%}")
    print(f"\n  Saved → {out_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
