"""
Loads and formats data for the three experimental conditions.

Condition A (PII):         RAG context + OneHop question → answer contains PII
Condition B (General):     World-fact question, no context → answer is general knowledge
Condition C (No-context):  Same as A but context stripped → model must guess
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from config import PROFILES_PATH, WORLD_FACTS_PATH, MAX_SAMPLES


@dataclass
class Sample:
    condition: str          # "A_pii" | "B_general" | "C_no_context"
    prompt: str             # full input string fed to the model
    answer: str             # expected ground-truth answer text
    pii_types: list[str] = field(default_factory=list)   # e.g. ["phone_number", "email_address"]
    pii_values: list[str] = field(default_factory=list)  # e.g. ["332-111-2233", "m.f@libero.it"]
    subject: Optional[str] = None


# ── Prompt templates ───────────────────────────────────────────────────────────

def _profile_to_context(profile: dict) -> str:
    """Convert a person profile dict into a readable context paragraph."""
    fields = [
        ("Name",              profile.get("full_name")),
        ("Date of Birth",     profile.get("DOB")),
        ("Occupation",        profile.get("Occupation")),
        ("Home Address",      profile.get("home_address")),
        ("Work Address",      profile.get("work_address")),
        ("Phone Number",      profile.get("phone_number")),
        ("Email",             profile.get("email_address")),
        ("Partner",           profile.get("partner_name")),
        ("Bank",              profile.get("bank_name")),
        ("Bank Account",      profile.get("bank_account_number")),
        ("Credit Card",       profile.get("credit_card_nr")),
        ("Health Insurance",  profile.get("health_insurance_nr")),
        ("Hospital",          profile.get("hospital_name")),
        ("Doctor",            profile.get("doctor_name")),
        ("Disease",           profile.get("disease")),
        ("Treatment",         profile.get("treatment")),
    ]
    lines = [f"{k}: {v}" for k, v in fields if v]
    return "\n".join(lines)


def _make_rag_prompt(context: str, question: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def _make_direct_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


# ── Loaders ───────────────────────────────────────────────────────────────────

def _extract_pii_values(profile: dict) -> list[str]:
    """Extract the actual string values for the pii_picked fields."""
    values = []
    for key in profile.get("pii_picked", []):
        val = profile.get(key)
        if val:
            values.append(str(val))
    return values


def load_condition_a(max_samples=MAX_SAMPLES) -> list[Sample]:
    """Condition A: model reads full PII profile and answers OneHop question."""
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)

    if max_samples:
        profiles = profiles[:max_samples]

    samples = []
    for p in profiles:
        context = _profile_to_context(p)
        prompt  = _make_rag_prompt(context, p["question"])
        samples.append(Sample(
            condition="A_pii",
            prompt=prompt,
            answer=p["answer"],
            pii_types=p.get("pii_picked", []),
            pii_values=_extract_pii_values(p),
            subject=p.get("full_name"),
        ))
    return samples


def load_condition_b(max_samples=MAX_SAMPLES) -> list[Sample]:
    """Condition B: general world-knowledge question, no context."""
    with open(WORLD_FACTS_PATH) as f:
        facts = json.load(f)

    if max_samples:
        facts = facts[:max_samples]

    samples = []
    for item in facts:
        prompt = _make_direct_prompt(item["question"])
        samples.append(Sample(
            condition="B_general",
            prompt=prompt,
            answer=item["answer"],
        ))
    return samples


def load_condition_c(max_samples=MAX_SAMPLES) -> list[Sample]:
    """Condition C: same PII question as A but without context (out-of-context)."""
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)

    if max_samples:
        profiles = profiles[:max_samples]

    samples = []
    for p in profiles:
        prompt = _make_direct_prompt(p["question"])
        samples.append(Sample(
            condition="C_no_context",
            prompt=prompt,
            answer=p["answer"],
            pii_types=p.get("pii_picked", []),
            pii_values=_extract_pii_values(p),
            subject=p.get("full_name"),
        ))
    return samples


def load_all(max_samples=MAX_SAMPLES) -> dict[str, list[Sample]]:
    return {
        "A_pii":        load_condition_a(max_samples),
        "B_general":    load_condition_b(max_samples),
        "C_no_context": load_condition_c(max_samples),
    }


if __name__ == "__main__":
    data = load_all()
    for cond, samples in data.items():
        print(f"\n{'='*60}")
        print(f"Condition {cond}: {len(samples)} samples")
        s = samples[0]
        print(f"Subject : {s.subject}")
        print(f"PII types: {s.pii_types}")
        print(f"Answer  : {s.answer[:80]}...")
        print(f"Prompt (first 300 chars):\n{s.prompt[:300]}...")
