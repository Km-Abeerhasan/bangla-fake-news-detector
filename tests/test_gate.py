"""
tests/test_gate.py
Unit tests verifying multi-signal gate computation and calibrated abstention.
"""

from datetime import datetime
from src.contracts import Verdict, EvidenceItem, StanceResult
from src.gate.evaluator import compute_reliability_features, evaluate_gate

def dummy_verdict(label="SUPPORTED", indep_count=2):
    return Verdict(
        label=label,
        pipeline_status="ANSWER",
        calibrated_conf=0.90,
        absence_status="unknown",
        supporting_ids=("p1",),
        contradicting_ids=(),
        t_evaluation=datetime.now(),
        verification_mode="retrospective",
        rule_fired="RULE_5_INDEPENDENT_SUPPORT",
        independence_count=indep_count,
        citation_chains=("p1",),
        secondary_flags=()
    )

def dummy_item(pid="p1", tier=1):
    return EvidenceItem(
        passage_id=pid,
        text="Evidence",
        url="https://gov.bd",
        publisher="Gov",
        tier=tier,
        t_pub=datetime(2026, 8, 1),
        t_verif=datetime(2026, 8, 1),
        retriever="hybrid",
        retrieval_pass="hybrid_fusion",
        fusion_rank=1,
        eligibility_reason="valid"
    )

def test_gate_answer_high_reliability():
    v = dummy_verdict()
    items = (dummy_item("p1", tier=1),)
    stances = (StanceResult("CLM-01", "p1", 0.90, 0.05, 0.05, (0.90, 0.05, 0.05)),)
    
    decision = evaluate_gate(v, items, stances, tau_abstain=0.65)
    assert decision.pipeline_status == "ANSWER"
    assert decision.features["overall_reliability"] > 0.70

def test_gate_abstain_low_reliability():
    v = dummy_verdict(label="INSUFFICIENT_EVIDENCE", indep_count=0)
    # Empty evidence represents retrieval failure
    decision = evaluate_gate(v, (), (), tau_abstain=0.65)
    assert decision.pipeline_status == "ABSTAIN"
    assert decision.features["overall_reliability"] == 0.0

if __name__ == "__main__":
    test_gate_answer_high_reliability()
    test_gate_abstain_low_reliability()
    print("All Stage 7 reliability gate unit tests passed successfully!")
