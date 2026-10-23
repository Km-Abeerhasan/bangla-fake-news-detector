"""
src/report/generator.py
Stage 8: Templated Verification Report & Multi-Tier Citation Audit.
Outputs auditable verification certificates without ungrounded free-form generation.
"""

from typing import Tuple, Dict, Any, List
from datetime import datetime
import json

from src.contracts import ClaimInput, AtomicClaim, Verdict, GateDecision, EvidenceItem


def audit_citations(
    citation_ids: Tuple[str, ...],
    evidence_items: Tuple[EvidenceItem, ...]
) -> Tuple[bool, List[str]]:
    """
    Audits cited passage IDs against retrieved evidence pool.
    Returns (is_valid, list_of_missing_or_invalid_ids).
    """
    available_ids = {item.passage_id for item in evidence_items}
    invalid_ids = [cid for cid in citation_ids if cid not in available_ids]
    return (len(invalid_ids) == 0, invalid_ids)


def generate_verification_report(
    claim_input: ClaimInput,
    atomic_claims: Tuple[AtomicClaim, ...],
    evidence_items: Tuple[EvidenceItem, ...],
    verdict: Verdict,
    gate_decision: GateDecision
) -> Dict[str, Any]:
    """
    Constructs a deterministic, fully auditable verification certificate.
    """
    citations_valid, invalid_citations = audit_citations(verdict.citation_chains, evidence_items)

    evidence_summary = [
        {
            "passage_id": e.passage_id,
            "publisher": e.publisher,
            "tier": e.tier,
            "url": e.url,
            "t_pub": e.t_pub.isoformat(),
            "retrieval_pass": e.retrieval_pass,
            "fusion_rank": e.fusion_rank
        }
        for e in evidence_items
    ]

    report = {
        "report_id": f"REP-{verdict.t_evaluation.strftime('%Y%m%d%H%M%S')}-{claim_input.claim_id}",
        "project": "BiN (Audit-Then-Believe)",
        "timestamp": verdict.t_evaluation.isoformat(),
        "input_claim": {
            "claim_id": claim_input.claim_id,
            "clean_text": claim_input.clean_text,
            "lang": claim_input.lang,
            "t_claim": claim_input.t_claim.isoformat() if claim_input.t_claim else None
        },
        "atomic_propositions": [
            {
                "atomic_id": a.atomic_id,
                "text": a.text,
                "claim_type": a.claim_type,
                "is_provenance": a.is_provenance
            }
            for a in atomic_claims
        ],
        "gate_status": gate_decision.pipeline_status,
        "gate_reason": gate_decision.reason,
        "reliability_signals": gate_decision.features,
        "verdict": {
            "label": verdict.label,
            "calibrated_confidence": verdict.calibrated_conf,
            "rule_fired": verdict.rule_fired,
            "independence_count": verdict.independence_count,
            "verification_mode": verdict.verification_mode,
            "secondary_flags": list(verdict.secondary_flags)
        },
        "citation_audit": {
            "passed": citations_valid,
            "cited_passages": list(verdict.citation_chains),
            "unresolved_citations": invalid_citations
        },
        "evidence_docket": evidence_summary
    }

    return report
