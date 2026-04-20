"""
PANORAMA data loader.

Converts PANORAMA dataset into our Sample format (same as UnlearnPII's data_loader.py).

Data sources:
  - PANORAMA      (srirxml/PANORAMA):      synthetic_profile_id | content_type | content
  - PANORAMA-Plus (srirxml/PANORAMA-Plus): full profile attributes (36 fields)

Pipeline:
  1. Join PANORAMA content with PANORAMA-Plus profile on profile_id
  2. For each (profile, content) pair: find which PII values appear in the content
  3. Generate a question asking about that PII
  4. Return Sample(prompt=context+question, answer=pii_value, ...)

Content types: wiki | social | forum | review | comment | marketplace
"""

import re
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from data_loader import Sample, _make_rag_prompt

# ── PII field definitions ──────────────────────────────────────────────────────
# (panorama_plus_field, pii_type_label, question_to_ask)
PII_FIELDS = [
    ("Email Address",          "email",           "What email address is mentioned in this text?"),
    ("Work Email",             "work_email",       "What work email address is referenced in this text?"),
    ("Phone Number",           "phone",            "What phone number is mentioned in this text?"),
    ("Work Phone",             "work_phone",       "What work phone number is referenced in this text?"),
    ("National ID",            "national_id",      "What national ID number is referenced in this text?"),
    ("Passport Number",        "passport",         "What passport number is mentioned in this text?"),
    ("Driver's License",       "drivers_license",  "What driver's license number is mentioned in this text?"),
    ("Annual Salary",          "salary",           "What salary information is disclosed in this text?"),
    ("Net Worth",              "net_worth",        "What net worth or financial information is mentioned in this text?"),
    ("Credit Score",           "credit_score",     "What credit score is referenced in this text?"),
    ("Birth Date",             "dob",              "What birth date is mentioned in this text?"),
    ("Address",                "address",          "What home address is mentioned in this text?"),
    ("Social Media Handles",   "social_handle",    "What social media handle is mentioned in this text?"),
    ("Blood Type",             "blood_type",       "What blood type is mentioned in this text?"),
    ("Allergies",              "allergies",        "What allergies are mentioned in this text?"),
    ("Disability",             "disability",       "What disability or medical condition is mentioned in this text?"),
]

# Actual content-type values from HuggingFace dataset
ALL_CONTENT_TYPES = {
    "Article",
    "Social Media",
    "Forum Post",
    "Online Review",
    "Blog/News Article Comment",
    "Online Ad",
}


# ── Matching helpers ───────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Strip separators for fuzzy matching (e.g. phone formats)."""
    return re.sub(r'[\s\-\(\)\.\/]+', '', text.lower())


def _find_pii_in_content(content: str, profile: dict) -> list[tuple[str, str]]:
    """
    Returns list of (pii_type, pii_value) for PII values that actually appear
    in the content text. Uses both direct and normalized matching.
    Only includes values with length > 3 to avoid false positives.
    """
    found = []
    content_lower = content.lower()
    content_norm  = _normalize(content)

    for field_name, pii_type, _ in PII_FIELDS:
        value = profile.get(field_name, "")
        if not value or not str(value).strip():
            continue

        value_str  = str(value).strip()
        value_norm = _normalize(value_str)

        # Skip very short values (avoid noise)
        if len(value_str) < 4:
            continue

        # Direct substring match (case-insensitive)
        if value_str.lower() in content_lower:
            found.append((pii_type, value_str))
            continue

        # Normalized match (handles format differences like "303-555-0198" vs "3035550198")
        if len(value_norm) >= 6 and value_norm in content_norm:
            found.append((pii_type, value_str))

    return found


def _get_full_name(profile: dict) -> str:
    first = profile.get("First Name", "").strip()
    last  = profile.get("Last Name",  "").strip()
    return f"{first} {last}".strip()


def _get_question(pii_type: str) -> str:
    for _, pt, question in PII_FIELDS:
        if pt == pii_type:
            return question
    return "What personal information is disclosed in this text?"


# ── Main loader ───────────────────────────────────────────────────────────────

def load_panorama_samples(
    panorama_path: str,
    panorama_plus_path: str,
    content_types: Optional[set[str]] = None,
    max_samples: Optional[int] = None,
    pii_per_sample: int = 1,
) -> list[Sample]:
    """
    Load PANORAMA samples in our Sample format.

    Args:
        panorama_path:      Path to preprocessed PANORAMA JSON (list of dicts)
        panorama_plus_path: Path to preprocessed PANORAMA-Plus JSON (dict: id → profile)
        content_types:      Filter to specific content types (default: all 6)
        max_samples:        Cap total samples returned
        pii_per_sample:     How many PII types to generate questions for per content (default 1)

    Returns:
        list[Sample] — same format as load_condition_a() in data_loader.py
    """
    if content_types is None:
        content_types = ALL_CONTENT_TYPES

    with open(panorama_path) as f:
        panorama_rows = json.load(f)   # list of {id, content_type, content}

    with open(panorama_plus_path) as f:
        profiles = json.load(f)         # dict: profile_id → profile dict

    samples = []

    for row in panorama_rows:
        if max_samples and len(samples) >= max_samples:
            break

        content_type = row.get("content_type", "")
        if content_type not in content_types:
            continue

        profile_id = row.get("synthetic_profile_id") or row.get("id")
        content    = row.get("content", "").strip()
        if not content or not profile_id:
            continue

        profile = profiles.get(profile_id)
        if not profile:
            continue

        # Find which PII values actually appear in this content
        pii_found = _find_pii_in_content(content, profile)
        if not pii_found:
            continue   # Skip content with no detectable PII

        # Generate one sample per PII type found (up to pii_per_sample)
        for pii_type, pii_value in pii_found[:pii_per_sample]:
            question = _get_question(pii_type)
            prompt   = _make_rag_prompt(content, question)
            name     = _get_full_name(profile)

            samples.append(Sample(
                condition="A_pii",
                prompt=prompt,
                answer=pii_value,
                pii_types=[pii_type],
                pii_values=[pii_value],
                subject=name,
            ))

            if max_samples and len(samples) >= max_samples:
                break

    return samples


def load_panorama_by_content_type(
    panorama_path: str,
    panorama_plus_path: str,
    max_samples_per_type: Optional[int] = None,
) -> dict[str, list[Sample]]:
    """
    Load PANORAMA samples split by content_type.
    Returns dict: content_type → list[Sample]
    Used for Experiment B (per-type breakdown).
    """
    result = {}
    for ct in ALL_CONTENT_TYPES:
        result[ct] = load_panorama_samples(
            panorama_path,
            panorama_plus_path,
            content_types={ct},
            max_samples=max_samples_per_type,
        )
        print(f"  [{ct:15s}] {len(result[ct])} samples with detectable PII")
    return result


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import PANORAMA_PATH, PANORAMA_PLUS_PATH

    samples = load_panorama_samples(
        PANORAMA_PATH,
        PANORAMA_PLUS_PATH,
        max_samples=20,
    )

    print(f"\nLoaded {len(samples)} samples")
    for i, s in enumerate(samples[:3]):
        print(f"\n{'='*60}")
        print(f"Subject   : {s.subject}")
        print(f"PII type  : {s.pii_types}")
        print(f"PII value : {s.pii_values}")
        print(f"Prompt    :\n{s.prompt[:400]}...")
