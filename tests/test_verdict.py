"""
tests/test_verdict.py
Unit tests verifying the 9-priority deterministic decision table and rule tracking.
"""

from datetime import datetime
from src.contracts import AtomicClaim, EvidenceItem, StanceResult
from src.verdict.aggregator import aggregate_verdict

def make_claim(is_prov: bool = False) -> AtomicClaim:
    return AtomicClaim(
        atomic_id="CLM-01",
        parent_id="P-01",
        text="তিস্তা নদীর বাঁধ ভেঙে গেছে",
        claim_type="factual",
        is_provenance=is_prov
    )

def make_item(pid: str) -> EvidenceItem:
    return EvidenceItem(
        passage_id=pid,
        text="Sample",
        url="https://news.com",
        publisher="Daily",
        tier=3,
        t_pub=datetime(2026, 8, 1),
        t_verif=datetime(2026, 8, 1),
        retriever="hybrid",
        retrieval_pass="pass_a",
        fusion_rank=1,
        eligibility_reason="valid"
    )

def test_independent_support_rule():
    claim = make_claim()
    items = (make_item("doc1"), make_item("doc2"))
    stances = (
        StanceResult("CLM-01", "doc1", 0.85, 0.10, 0.05, (0.85, 0.10, 0.05)),
        StanceResult("CLM-01", "doc2", 0.90, 0.05, 0.05, (0.90, 0.05, 0.05)),
    )
    v = aggregate_verdict(claim, items, stances, independence_count=2)
    assert v.label == "SUPPORTED"
    assert v.rule_fired == "RULE_5_INDEPENDENT_SUPPORT"
    assert len(v.supporting_ids) == 2

def test_provenance_absence_rule():
    claim = make_claim(is_prov=True)
    v = aggregate_verdict(claim, (), (), independence_count=0, absence_status="confirmed_within_scope")
    assert v.label == "MISLEADING"
    assert v.rule_fired == "RULE_1_PROVENANCE_ABSENCE"
    assert "false_attribution" in v.secondary_flags

def test_numeric_distortion_rule():
    claim = make_claim()
    items = (make_item("doc1"),)
    stances = (StanceResult("CLM-01", "doc1", 0.80, 0.10, 0.10, (0.80, 0.10, 0.10)),)
    v = aggregate_verdict(claim, items, stances, independence_count=1, has_numeric_mismatch=True)
    assert v.label == "MISLEADING"
    assert v.rule_fired == "RULE_7_NUMERIC_DISTORTION"

if __name__ == "__main__":
    test_independent_support_rule()
    test_provenance_absence_rule()
    test_numeric_distortion_rule()
    print("All Stage 6 verdict aggregator unit tests passed successfully!")
