"""
Download and preprocess PANORAMA dataset from HuggingFace.

Run this ONCE on the server before running experiments.

Prerequisites:
  pip install datasets huggingface_hub
  huggingface-cli login          # need HF account + accept dataset terms at:
                                 # https://huggingface.co/datasets/srirxml/PANORAMA
                                 # https://huggingface.co/datasets/srirxml/PANORAMA-Plus

Output files (saved to DATA_DIR):
  panorama/panorama.json         - list of {id, content_type, content}
  panorama/panorama_plus.json    - dict: profile_id → profile attributes

Usage:
  python prepare_panorama.py
  python prepare_panorama.py --output-dir /custom/path --max-profiles 500
"""

import os
import json
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Download and preprocess PANORAMA")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "../data/panorama"),
                        help="Directory to save preprocessed files")
    parser.add_argument("--max-profiles", type=int, default=None,
                        help="Limit number of profiles (for quick testing)")
    return parser.parse_args()


def download_panorama(output_dir: str, max_profiles: int = None):
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Download PANORAMA-Plus (profile attributes) ─────────────────────────
    print("\n[1/2] Downloading PANORAMA-Plus (profile attributes)...")
    plus_dataset = load_dataset("srirxml/PANORAMA-Plus", split="train")

    profiles = {}
    for row in tqdm(plus_dataset, desc="Processing profiles"):
        profile_id = row.get("Unique ID", "").strip()
        if not profile_id:
            continue
        profiles[profile_id] = {k: v for k, v in row.items()}

        if max_profiles and len(profiles) >= max_profiles:
            break

    plus_path = os.path.join(output_dir, "panorama_plus.json")
    with open(plus_path, "w") as f:
        json.dump(profiles, f)
    print(f"  Saved {len(profiles)} profiles → {plus_path}")

    # ── 2. Download PANORAMA (content texts) ───────────────────────────────────
    print("\n[2/2] Downloading PANORAMA (content texts)...")
    pano_dataset = load_dataset("srirxml/PANORAMA", split="train")

    # Only keep rows whose profile_id appears in PANORAMA-Plus
    valid_ids = set(profiles.keys())

    rows = []
    skipped = 0
    for row in tqdm(pano_dataset, desc="Processing content rows"):
        # Actual HuggingFace column names: 'id', 'content-type', 'text'
        profile_id = (row.get("id") or row.get("synthetic_profile_id") or "").strip()
        if not profile_id or profile_id not in valid_ids:
            skipped += 1
            continue
        rows.append({
            "synthetic_profile_id": profile_id,
            "content_type":         row.get("content-type", "").strip(),
            "content":              row.get("text", "").strip(),
        })

    pano_path = os.path.join(output_dir, "panorama.json")
    with open(pano_path, "w") as f:
        json.dump(rows, f)

    # ── 3. Summary ─────────────────────────────────────────────────────────────
    print(f"\n  Saved {len(rows)} content rows → {pano_path}")
    print(f"  Skipped {skipped} rows (profile not found)")

    content_type_counts = {}
    for r in rows:
        ct = r["content_type"]
        content_type_counts[ct] = content_type_counts.get(ct, 0) + 1
    print("\n  Content type breakdown:")
    for ct, count in sorted(content_type_counts.items()):
        print(f"    {ct:15s}: {count:,}")

    return pano_path, plus_path


def verify_output(pano_path: str, plus_path: str):
    """Quick sanity check: print 2 example samples with detectable PII."""
    import re
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from panorama_data_loader import load_panorama_samples

    print("\n── Verification (first 2 samples with PII) ──────────────────────")
    samples = load_panorama_samples(pano_path, plus_path, max_samples=2)
    if not samples:
        print("  WARNING: No samples with detectable PII found. Check data.")
        return

    for s in samples:
        print(f"\n  Subject  : {s.subject}")
        print(f"  PII type : {s.pii_types[0]}")
        print(f"  PII value: {s.pii_values[0]}")
        print(f"  Prompt (truncated):")
        print(f"    {s.prompt[:250]}...")
    print()


if __name__ == "__main__":
    args = parse_args()
    pano_path, plus_path = download_panorama(args.output_dir, args.max_profiles)
    verify_output(pano_path, plus_path)

    print("="*60)
    print("PANORAMA preprocessing complete.")
    print(f"Files saved to: {args.output_dir}")
    print("\nNext step: run experiments")
    print("  python run_panorama_experiment.py --device cuda \\")
    print("    --model meta-llama/Llama-3.1-8B-Instruct \\")
    print("    --probe-layer 24")
