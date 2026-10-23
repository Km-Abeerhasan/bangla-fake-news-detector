"""
tests/test_claims.py
Unit tests verifying Stage 2 proposition splitting, modality, and provenance detection.
"""

from datetime import datetime
from src.ingest.loader import load_claim_text
from src.claims.processor import deconstruct_claim

def test_atomic_splitting_and_provenance():
    # Compound claim: provenance assertion + prediction
    raw_text = "পানি উন্নয়ন বোর্ড সতর্কবার্তা জারি করেছে এবং ২৪ ঘণ্টার মধ্যে বাঁধ ভেঙে যেতে পারে।"
    claim_input = load_claim_text(raw_text, claim_id="CLM-010")

    atomics = deconstruct_claim(claim_input)

    assert len(atomics) == 2
    # First atomic proposition: Institutional provenance
    assert atomics[0].is_provenance is True
    # Second atomic proposition: Forward prediction
    assert atomics[1].claim_type == "prediction"
    assert len(atomics[1].quantities) > 0  # Matches "২৪"

def test_opinion_modality():
    raw_text = "আমার মতে তিস্তা প্রকল্পের কাজ অত্যন্ত ধীরগতির।"
    claim_input = load_claim_text(raw_text, claim_id="CLM-011")

    atomics = deconstruct_claim(claim_input)
    assert len(atomics) == 1
    assert atomics[0].claim_type == "opinion"

if __name__ == "__main__":
    test_atomic_splitting_and_provenance()
    test_opinion_modality()
    print("All Stage 2 claim deconstruction tests passed successfully!")
