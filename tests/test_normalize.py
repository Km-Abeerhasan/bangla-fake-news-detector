"""
tests/test_normalize.py
Unit tests verifying Bengali ligature preservation and audit logging.
"""

from src.ingest.normalize import normalize_text

def test_bengali_ligature_preservation():
    # Valid Bengali conjuncts: 'তিস্তা', 'যুক্তাক্ষর', 'ব্রহ্মপুত্র'
    sample = "তিস্তা নদীর বাঁধ ভেঙে উত্তরাঞ্চল প্লাবিত"
    cleaned, log = normalize_text(sample)
    assert cleaned == sample
    assert len(cleaned) == len(sample)

def test_zero_width_artifact_removal():
    # Embedded U+200B (Zero-width space) and excess whitespace
    noisy = "তিস্তা\u200B   নদী   "
    cleaned, log = normalize_text(noisy)
    assert cleaned == "তিস্তা নদী"
    assert any(entry["op"] == "strip_artifact" for entry in log)
    assert any(entry["op"] == "whitespace_canonicalization" for entry in log)

if __name__ == "__main__":
    test_bengali_ligature_preservation()
    test_zero_width_artifact_removal()
    print("All normalization unit tests passed successfully!")
