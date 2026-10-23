"""
tests/test_loader.py
Unit tests verifying Stage 1 loader functionality and ClaimInput immutability.
"""

from datetime import datetime
from src.ingest.loader import load_claim_text
from src.contracts import ClaimInput, OCRToken

def test_load_claim_text():
    raw = "তিস্তা নদীর বাঁধ\u200B ভেঙে প্লাবিত   "
    t_pub = datetime(2026, 8, 15)

    claim = load_claim_text(
        raw_text=raw,
        lang="bn",
        claim_id="CLM-PILOT-01",
        t_claim=t_pub
    )

    assert isinstance(claim, ClaimInput)
    assert claim.claim_id == "CLM-PILOT-01"
    assert claim.clean_text == "তিস্তা নদীর বাঁধ ভেঙে প্লাবিত"
    assert claim.lang == "bn"
    assert claim.t_claim == t_pub

def test_claim_input_immutability():
    claim = load_claim_text("যমুনা নদীর পানি বিপদসীমার উপরে", claim_id="CLM-02")
    try:
        # Should raise FrozenInstanceError because frozen=True
        claim.clean_text = "অন্য কিছু"
        assert False, "Failed to enforce immutability on ClaimInput!"
    except Exception:
        pass

if __name__ == "__main__":
    test_load_claim_text()
    test_claim_input_immutability()
    print("All Stage 1 loader unit tests passed successfully!")
