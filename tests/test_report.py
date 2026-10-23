"""
tests/test_report.py
Unit tests verifying Stage 8 citation audit and templated verification reporting.
"""

from datetime import datetime
from src.contracts import ClaimInput, AtomicClaim, EvidenceItem, Verdict, GateDecision
from src.report.generator import audit_citations, generate_verification_report

def test_citation_audit():
    evidence = (
        EvidenceItem("doc1", "text", "http://a.com", "A", 1, datetime.now(), datetime.now(), "bm25", "pass_a", 1, "valid"),
        EvidenceItem("doc2", "text", "http://b.com", "B", 2, datetime.now(), datetime.now(), "dense_e5", "pass_a", 2, "valid"),
    )
    valid, bad = audit_citations(("doc1", "doc2"), evidence)
    assert valid is True
    assert len(bad) == 0

    valid_fail, bad_fail = audit_citations(("doc1", "phantom_doc"), evidence)
    assert valid_fail is False
    assert "phantom_doc" in bad_fail

def test_generate_verification_report():
    c_in = ClaimInput("CLM-1", "কাঁচা পাঠ্য", "পরিষ্কার পাঠ্য", "bn")
    atomics = (AtomicClaim("CLM-1-A1", "CLM-1", "পরিষ্কার পাঠ্য", "factual"),)
    evidence = (
        EvidenceItem("doc1", "text", "http://a.com", "A", 1, datetime.now(), datetime.now(), "hybrid", "hybrid_fusion", 1, "valid"),
    )
    v = Verdict(
        label="SUPPORTED",
        pipeline_status="ANSWER",
        calibrated_conf=0.92,
        absence_status="unknown",
        supporting_ids=("doc1",),
        contradicting_ids=(),
        t_evaluation=datetime.now(),
        verification_mode="retrospective",
        rule_fired="RULE_5_INDEPENDENT_SUPPORT",
        independence_count=1,
        citation_chains=("doc1",),
        secondary_flags=()
    )
    gate = GateDecision(
        pipeline_status="ANSWER",
        reason="Passed threshold",
        verdict=v,
        features={"overall_reliability": 0.85}
    )

    report = generate_verification_report(c_in, atomics, evidence, v, gate)
    assert report["project"] == "BiN (Audit-Then-Believe)"
    assert report["verdict"]["label"] == "SUPPORTED"
    assert report["citation_audit"]["passed"] is True
    assert len(report["evidence_docket"]) == 1

if __name__ == "__main__":
    test_citation_audit()
    test_generate_verification_report()
    print("All Stage 8 report & citation audit unit tests passed successfully!")
