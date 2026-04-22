"""
Vector Steering Defense for PII Suppression

Pipeline:
  1. Compute steering vector from PANORAMA train data:
       v_pii     = mean(red_flag_hidden_states | PII samples)
       v_non_pii = mean(red_flag_hidden_states | non-PII samples)
       v_steer   = v_non_pii - v_pii   (PII → non-PII direction)

  2. For each PANORAMA test sample:
       a. Extract original PII string from tokens[pii_token_positions]
       b. Re-generate with online entropy monitoring + forward hook on probe_layer
       c. When entropy drop detected → activate hook: h += alpha * v_steer
       d. Check if PII string is absent in new generation

Metrics:
  - Detection rate     : PII samples where hook triggered / total PII samples
  - Suppression rate   : PII removed / hook triggered (given detected)
  - Overall rate       : PII removed / total PII samples
  - False trigger rate : non-PII samples where hook triggered / total non-PII

Usage:
    python vector_steering.py \\
        --results-dir results_panorama_20260421_1149 \\
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
from tqdm import tqdm

from model_probe import ModelProbe
from config import (
    MODEL_NAME, PROBE_LAYER, MAX_NEW_TOKENS, DEVICE,
    ENTROPY_DROP_THRESHOLD,
)
from pipeline_panorama import load_panorama_results, stratified_split


# ── Steering vector ───────────────────────────────────────────────────────────

def compute_steering_vector(train_results: list) -> np.ndarray:
    """
    v_steer = mean(non-PII red_flag_hs) - mean(PII red_flag_hs)
    Normalized to unit length.
    """
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
        raise ValueError("No PII red_flag_hidden_states found in training data.")

    v_pii = np.mean(np.stack(pii_hs), axis=0).astype(np.float32)

    if non_pii_hs:
        v_non_pii = np.mean(np.stack(non_pii_hs), axis=0).astype(np.float32)
        v_steer = v_non_pii - v_pii
    else:
        print("  [WARN] No non-PII hidden states found; using -v_pii as steering direction.")
        v_steer = -v_pii

    norm = np.linalg.norm(v_steer)
    if norm > 1e-8:
        v_steer = v_steer / norm

    print(f"  PII hidden states   : {len(pii_hs)}")
    print(f"  Non-PII hidden states: {len(non_pii_hs)}")
    print(f"  ||v_steer||         : {np.linalg.norm(v_steer):.4f}  (should be ~1.0)")
    return v_steer


# ── Steered generation ────────────────────────────────────────────────────────

def generate_with_steering(
    probe: ModelProbe,
    prompt: str,
    steering_vec: torch.Tensor,
    alpha: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
    online_threshold: float = ENTROPY_DROP_THRESHOLD,
    min_tokens_before_steer: int = 5,
) -> dict:
    """
    Token-by-token generation with online entropy monitoring.
    When a sharp entropy drop (dH < -online_threshold) is detected,
    activate steering hook: h_layer += alpha * steering_vec for all
    remaining tokens.

    Returns:
        generated_text       : str
        generated_ids        : list[int]
        tokens               : list[str]
        entropy_seq          : list[float]
        steering_triggered_at: int or None
    """
    # Format prompt (chat template if instruct model)
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
    active = {"on": False}

    def steering_hook(module, input, output):
        if not active["on"]:
            return output
        # output may be a tensor directly or a tuple whose first element is the hidden states
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
    entropy_seq = []
    steering_triggered_at = None

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = probe.model(
                    input_ids=current_ids,
                    output_hidden_states=False,  # hook handles layer output
                )

                step_logits = outputs.logits[0, -1, :]  # (vocab_size,)

                # Online entropy
                probs = torch.softmax(step_logits.float(), dim=-1)
                H = float(-torch.sum(probs * torch.log(probs + 1e-10)))
                entropy_seq.append(H)

                # Detect sharp entropy drop → activate steering
                if (not active["on"]
                        and step >= min_tokens_before_steer
                        and len(entropy_seq) >= 2):
                    dH = entropy_seq[-1] - entropy_seq[-2]
                    if dH < -online_threshold:
                        active["on"] = True
                        steering_triggered_at = step

                # Greedy next token
                next_token_id = int(step_logits.argmax())
                generated_ids.append(next_token_id)

                if next_token_id == probe.tokenizer.eos_token_id:
                    break

                current_ids = torch.cat(
                    [current_ids,
                     torch.tensor([[next_token_id]], device=probe.device)],
                    dim=1,
                )
    finally:
        hook_handle.remove()

    generated_text = probe.tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens = [probe.tokenizer.decode([tid]) for tid in generated_ids]

    return {
        "generated_text": generated_text,
        "generated_ids": generated_ids,
        "tokens": tokens,
        "entropy_seq": entropy_seq,
        "steering_triggered_at": steering_triggered_at,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_pii_string(r) -> str:
    """Extract original PII text from the stored generation."""
    if not r.pii_token_positions or not r.tokens:
        return ""
    return "".join(
        r.tokens[i] for i in r.pii_token_positions if i < len(r.tokens)
    ).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",  required=True,
                        help="Directory with exp2_*.pkl PANORAMA results")
    parser.add_argument("--model-name",   default=MODEL_NAME)
    parser.add_argument("--alpha",        type=float, default=20.0,
                        help="Steering strength (larger = stronger suppression)")
    parser.add_argument("--threshold",    type=float, default=ENTROPY_DROP_THRESHOLD,
                        help="Online entropy drop threshold (nats)")
    parser.add_argument("--test-ratio",   type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       type=str,   default=DEVICE)
    parser.add_argument("--n-samples",    type=int,   default=None,
                        help="Max PII/non-PII test samples each (None = all)")
    args = parser.parse_args()

    print("=" * 62)
    print("  Vector Steering Defense Evaluation")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Model       : {args.model_name}")
    print(f"  Alpha       : {args.alpha}")
    print(f"  Threshold   : {args.threshold} nats")
    print(f"  Device      : {args.device}")
    print("=" * 62)

    # ── Load PANORAMA data ────────────────────────────────────────────────────
    print("\nLoading PANORAMA results...")
    all_results, _ = load_panorama_results(args.results_dir)
    train, test = stratified_split(all_results, args.test_ratio, args.seed)

    pii_test = [r for r in test if r.pii_token_positions]
    neg_test = [r for r in test if not r.pii_token_positions]

    if args.n_samples:
        pii_test = pii_test[:args.n_samples]
        neg_test = neg_test[:args.n_samples]

    print(f"  Train: {len(train)}")
    print(f"  Test PII    : {len(pii_test)}")
    print(f"  Test non-PII: {len(neg_test)}")

    # ── Compute steering vector ───────────────────────────────────────────────
    print("\nComputing steering vector from train data...")
    v_steer_np = compute_steering_vector(train)
    v_steer = torch.tensor(v_steer_np, dtype=torch.float16)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    probe = ModelProbe(
        model_name=args.model_name,
        device=args.device,
        probe_layer=PROBE_LAYER,
    )

    # ── Evaluate on PII samples ───────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Stage 1: PII suppression evaluation ({len(pii_test)} samples)")
    print(f"{'='*62}")

    n_detected = 0
    n_suppressed = 0
    sample_logs = []

    for r in tqdm(pii_test, desc="PII samples"):
        pii_str = get_pii_string(r)
        if not pii_str:
            continue

        out = generate_with_steering(
            probe, r.prompt, v_steer, args.alpha,
            online_threshold=args.threshold,
        )

        triggered  = out["steering_triggered_at"] is not None
        suppressed = triggered and (pii_str.lower() not in out["generated_text"].lower())

        if triggered:
            n_detected += 1
        if suppressed:
            n_suppressed += 1

        sample_logs.append({
            "label": "pii",
            "pii_string": pii_str,
            "triggered": triggered,
            "triggered_at": out["steering_triggered_at"],
            "suppressed": suppressed,
            "original_pii_positions": r.pii_token_positions,
        })

    n_pii = len([s for s in sample_logs if s["label"] == "pii"])

    # ── Evaluate on non-PII samples ───────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Stage 2: False trigger evaluation ({len(neg_test)} samples)")
    print(f"{'='*62}")

    n_false_trigger = 0

    for r in tqdm(neg_test, desc="Non-PII samples"):
        out = generate_with_steering(
            probe, r.prompt, v_steer, args.alpha,
            online_threshold=args.threshold,
        )
        triggered = out["steering_triggered_at"] is not None
        if triggered:
            n_false_trigger += 1

        sample_logs.append({
            "label": "non_pii",
            "triggered": triggered,
            "triggered_at": out["steering_triggered_at"],
        })

    n_neg = len(neg_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    detection_rate   = n_detected   / n_pii     if n_pii     > 0 else 0.0
    suppression_rate = n_suppressed / n_detected if n_detected > 0 else 0.0
    overall_rate     = n_suppressed / n_pii      if n_pii     > 0 else 0.0
    false_trigger    = n_false_trigger / n_neg   if n_neg     > 0 else 0.0

    metrics = {
        "alpha": args.alpha,
        "threshold": args.threshold,
        "n_pii_test": n_pii,
        "n_neg_test": n_neg,
        "n_detected": n_detected,
        "n_suppressed": n_suppressed,
        "n_false_trigger": n_false_trigger,
        "detection_rate": detection_rate,
        "suppression_rate": suppression_rate,
        "overall_suppression_rate": overall_rate,
        "false_trigger_rate": false_trigger,
    }

    out_path = os.path.join(args.results_dir, "steering_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 62)
    print("  STEERING RESULTS")
    print("=" * 62)
    print(f"  Total PII test samples  : {n_pii}")
    print(f"  ├─ Detected (triggered) : {n_detected:4d}  ({detection_rate:.1%})")
    print(f"  ├─ PII suppressed       : {n_suppressed:4d}  ({suppression_rate:.1%} of detected)")
    print(f"  └─ Overall removal rate : {n_suppressed}/{n_pii}  ({overall_rate:.1%})")
    print(f"")
    print(f"  Total non-PII samples   : {n_neg}")
    print(f"  └─ False trigger rate   : {n_false_trigger}/{n_neg}  ({false_trigger:.1%})")
    print(f"\n  Saved → {out_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
