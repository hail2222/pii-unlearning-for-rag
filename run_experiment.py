"""
Main experiment runner.
Runs all 3 conditions and saves per-sample results to results/.

Local debug:
    python run_experiment.py

Server (Llama-3.1-8B, 200 samples):
    python run_experiment.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --device cuda \
        --max-samples 200 \
        --probe-layer 24 \
        --results-dir results_llama8b
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

from config import (
    RESULTS_DIR, ENTROPY_DROP_THRESHOLD, ENTROPY_DROP_ZSCORE, MAX_NEW_TOKENS,
    ABS_ENTROPY_CEILING, MIN_RUN_LENGTH,
    MODEL_NAME, DEVICE, PROBE_LAYER, MAX_SAMPLES,
)
from data_loader import load_all, Sample
from model_probe import ModelProbe
from entropy_utils import (
    compute_entropy_sequence,
    compute_delta_entropy,
    detect_red_flags,
    detect_sustained_flags,
    find_pii_token_positions,
    compute_lead_time,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Entropy PII detection experiment")
    parser.add_argument("--model",       type=str, default=MODEL_NAME,
                        help="HuggingFace model name")
    parser.add_argument("--device",      type=str, default=DEVICE,
                        choices=["cpu", "cuda"],
                        help="Device to run on")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help="Max samples per condition (None = all)")
    parser.add_argument("--probe-layer", type=int, default=PROBE_LAYER,
                        help="Layer index for hidden state extraction")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help="Directory to save results")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip already-completed conditions (resume interrupted run)")
    return parser.parse_args()


@dataclass
class SampleResult:
    # Input
    condition: str
    subject: str
    pii_types: list[str]
    prompt: str
    answer: str

    # Generation
    generated_text: str
    tokens: list[str]
    generated_ids: list[int]

    # Entropy signals
    entropy_seq: list[float]
    delta_entropy: list[float]

    # Method 1: adaptive ΔH drop
    red_flag_indices: list[int]
    lead_times: list[int]

    # Method 2: sustained low-entropy window
    sustained_flag_indices: list[int]
    sustained_lead_times: list[int]

    # PII location
    pii_token_positions: list[int]

    # Hidden states at red flag moments (Method 1): list of shape-(hidden_size,) arrays
    red_flag_hidden_states: list = field(default_factory=list)
    # Hidden states at sustained flag moments (Method 2)
    sustained_hidden_states: list = field(default_factory=list)


def run_single(probe: ModelProbe, sample: Sample, threshold: float, zscore: float) -> SampleResult:
    # Generate
    out = probe.generate_with_probes(sample.prompt, max_new_tokens=MAX_NEW_TOKENS)

    # Entropy
    entropy_seq   = compute_entropy_sequence(out["logits"])
    delta_entropy = compute_delta_entropy(entropy_seq)

    # Method 1: adaptive ΔH drop
    red_flags  = detect_red_flags(delta_entropy, threshold, zscore=zscore)

    # Method 2: sustained low-entropy window
    sus_flags  = detect_sustained_flags(entropy_seq, ABS_ENTROPY_CEILING, MIN_RUN_LENGTH)

    # PII token positions
    pii_values_ids = []
    for val in sample.pii_values:
        candidates = [
            probe.tokenizer.encode(val, add_special_tokens=False),
            probe.tokenizer.encode(" " + val, add_special_tokens=False),
        ]
        pii_values_ids.append(candidates)
    pii_positions = find_pii_token_positions(out["generated_ids"], pii_values_ids)

    # Lead-times for both methods
    lead_times     = compute_lead_time(red_flags, pii_positions)
    sus_lead_times = compute_lead_time(sus_flags, pii_positions)

    # Hidden states
    rf_hidden = [
        out["hidden_states"][i].numpy()
        for i in red_flags if i < len(out["hidden_states"])
    ]
    sus_hidden = [
        out["hidden_states"][i].numpy()
        for i in sus_flags if i < len(out["hidden_states"])
    ]

    return SampleResult(
        condition=sample.condition,
        subject=sample.subject or "",
        pii_types=sample.pii_types,
        prompt=sample.prompt,
        answer=sample.answer,
        generated_text=out["generated_text"],
        tokens=out["tokens"],
        generated_ids=out["generated_ids"],
        entropy_seq=entropy_seq.tolist(),
        delta_entropy=delta_entropy.tolist(),
        red_flag_indices=red_flags,
        lead_times=lead_times,
        sustained_flag_indices=sus_flags,
        sustained_lead_times=sus_lead_times,
        pii_token_positions=pii_positions,
        red_flag_hidden_states=rf_hidden,
        sustained_hidden_states=sus_hidden,
    )


def _print_stats(results: list, n_total: int):
    n_with_pii = sum(1 for r in results if r.pii_token_positions)
    print(f"  PII located          : {n_with_pii}/{n_total}")

    n_m1    = sum(1 for r in results if r.red_flag_indices)
    lead_m1 = [lt for r in results for lt in r.lead_times]
    print(f"  [M1 ΔH-drop]  flags : {n_m1}/{n_total}", end="")
    if lead_m1:
        print(f"  | lead-time mean={np.mean(lead_m1):.1f}, median={np.median(lead_m1):.1f}")
    else:
        print()

    n_m2    = sum(1 for r in results if r.sustained_flag_indices)
    lead_m2 = [lt for r in results for lt in r.sustained_lead_times]
    print(f"  [M2 sustained] flags : {n_m2}/{n_total}", end="")
    if lead_m2:
        print(f"  | lead-time mean={np.mean(lead_m2):.1f}, median={np.median(lead_m2):.1f}")
    else:
        print()


def _save_checkpoint(results: list, condition: str, results_dir: str):
    """Save after every sample so interrupted runs can be resumed."""
    ckpt_path = os.path.join(results_dir, f"{condition}_checkpoint.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(results, f)


def _load_checkpoint(condition: str, results_dir: str) -> list:
    ckpt_path = os.path.join(results_dir, f"{condition}_checkpoint.pkl")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)
    return []


def run_all(args):
    os.makedirs(args.results_dir, exist_ok=True)

    # Print run config
    print(f"\n{'='*60}")
    print(f"Model      : {args.model}")
    print(f"Device     : {args.device}")
    print(f"Max samples: {args.max_samples}")
    print(f"Probe layer: {args.probe_layer}")
    print(f"Results dir: {args.results_dir}")
    print(f"{'='*60}")

    probe = ModelProbe(
        model_name=args.model,
        device=args.device,
        probe_layer=args.probe_layer,
    )
    all_data = load_all(max_samples=args.max_samples)
    all_results: dict[str, list[SampleResult]] = {}

    for condition, samples in all_data.items():
        print(f"\n{'='*60}")
        print(f"Condition: {condition} ({len(samples)} samples)")

        # Resume: load checkpoint and skip already-done samples
        done = _load_checkpoint(condition, args.results_dir) if args.resume else []
        start_idx = len(done)
        if start_idx > 0:
            print(f"  Resuming from sample {start_idx} (checkpoint found)")
        results = list(done)

        for sample in tqdm(samples[start_idx:], desc=condition, initial=start_idx, total=len(samples)):
            try:
                r = run_single(probe, sample,
                               threshold=ENTROPY_DROP_THRESHOLD,
                               zscore=ENTROPY_DROP_ZSCORE)
                results.append(r)
                # Checkpoint every sample
                _save_checkpoint(results, condition, args.results_dir)
            except Exception as e:
                print(f"  [ERROR] {getattr(sample, 'subject', '?')}: {e}")

        all_results[condition] = results

        # Final save: full pickle
        pkl_path = os.path.join(args.results_dir, f"{condition}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)

        # JSON summary (no hidden states)
        summary = []
        for r in results:
            d = asdict(r)
            d.pop("red_flag_hidden_states")
            d.pop("sustained_hidden_states")
            summary.append(d)
        json_path = os.path.join(args.results_dir, f"{condition}_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Saved → {pkl_path}")
        _print_stats(results, len(results))

        # Remove checkpoint after successful completion
        ckpt_path = os.path.join(args.results_dir, f"{condition}_checkpoint.pkl")
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    return all_results


if __name__ == "__main__":
    args = parse_args()
    run_all(args)
