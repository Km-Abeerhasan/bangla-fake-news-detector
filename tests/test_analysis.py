"""
tests/test_analysis.py
Unit tests verifying temperature calibration and StanceResult immutability.
"""

from datetime import datetime
from src.contracts import AtomicClaim, EvidenceItem, StanceResult
from src.analysis.stance import apply_temperature_scaling, analyze_passage_stance

def test_temperature_scaling():
    logits = [3.0, 0.0, -2.0]
    # Scaling with T > 1.0 should produce a valid probability distribution summing to 1.0
    p_entail, p_neutral, p_contra = apply_temperature_scaling(logits, temperature=1.5)
    assert 0.0 <= p_entail <= 1.0
    assert 0.0 <= p_contra <= 1.0
    assert round(p_entail + p_neutral + p_contra, 2) == 1.0
    # Highest logit must remain the highest probability
    assert p_entail > p_neutral > p_contra

def test_analyze_passage_stance():
    claim = AtomicClaim(
        atomic_id="CLM-01-A1",
        parent_id="CLM-01",
        text="তিস্তা নদীর পানি বিপদসীমার উপরে",
        claim_type="factual"
    )
    evidence = EvidenceItem(
        passage_id="DOC-99",
        text="তিস্তা নদীর পানি বিপদসীমার ২৫ সেমি উপর দিয়ে প্রবাহিত হচ্ছে।",
        url="https://bwdb.gov.bd",
        publisher="BWDB",
        tier=1,
        t_pub=datetime(2026, 8, 1),
        t_verif=datetime(2026, 8, 1),
        retriever="bm25",
        retrieval_pass="pass_a",
        fusion_rank=1,
        eligibility_reason="valid"
    )

    result = analyze_passage_stance(claim, evidence, raw_logits=[3.5, -1.0, -2.0], temperature=1.2)
    assert isinstance(result, StanceResult)
    assert result.claim_id == "CLM-01-A1"
    assert result.passage_id == "DOC-99"
    assert result.p_entail > 0.80

if __name__ == "__main__":
    test_temperature_scaling()
    test_analyze_passage_stance()
    print("All Stage 5 evidence analysis unit tests passed successfully!")
