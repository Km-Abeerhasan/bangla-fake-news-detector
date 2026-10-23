"""
src/verdict/aggregator.py
Stage 6: Deterministic Verdict Decision Aggregator.
Implements the 9-priority deterministic decision table, logging rule_fired and citation chains.
"""

from typing import Tuple, List, Optional
from datetime import datetime

from src.contracts import (
    Verdict,
    VerdictLabel,
    PipelineStatus,
    AbsenceStatus,
    VerificationMode,
    EvidenceItem,
    StanceResult,
    AtomicClaim
)


def aggregate_verdict(
    claim: AtomicClaim,
    evidence_items: Tuple[EvidenceItem, ...],
    stances: Tuple[StanceResult, ...],
    independence_count: int,
    absence_status: AbsenceStatus = "unknown",
    mode: VerificationMode = "retrospective",
    has_numeric_mismatch: bool = False
) -> Verdict:
    """
    Evaluates evidence through the deterministic 9-priority decision tree.
    """
    supporting_ids: List[str] = []
    contradicting_ids: List[str] = []

    for s in stances:
        if s.p_entail >= 0.70:
            supporting_ids.append(s.passage_id)
        elif s.p_contra >= 0.70:
            contradicting_ids.append(s.passage_id)

    n_sup = len(supporting_ids)
    n_contra = len(contradicting_ids)
    secondary_flags: List[str] = []

    # Priority 1: Provenance claim missing from official registry
    if claim.is_provenance and absence_status == "confirmed_within_scope":
        return Verdict(
            label="MISLEADING",
            pipeline_status="ANSWER",
            calibrated_conf=0.92,
            absence_status=absence_status,
            supporting_ids=(),
            contradicting_ids=tuple(contradicting_ids),
            t_evaluation=datetime.now(),
            verification_mode=mode,
            rule_fired="RULE_1_PROVENANCE_ABSENCE",
            independence_count=independence_count,
            citation_chains=tuple(contradicting_ids),
            secondary_flags=("false_attribution",)
        )

    # Priority 2: Empty evidence pool
    if not evidence_items or len(stances) == 0:
        return Verdict(
            label="INSUFFICIENT_EVIDENCE",
            pipeline_status="ANSWER",
            calibrated_conf=0.90,
            absence_status="search_incomplete",
            supporting_ids=(),
            contradicting_ids=(),
            t_evaluation=datetime.now(),
            verification_mode=mode,
            rule_fired="RULE_2_INSUFFICIENT_SEARCH",
            independence_count=0,
            citation_chains=(),
            secondary_flags=()
        )

    # Priority 7: Numeric / scale distortion detected
    if has_numeric_mismatch:
        return Verdict(
            label="MISLEADING",
            pipeline_status="ANSWER",
            calibrated_conf=0.88,
            absence_status=absence_status,
            supporting_ids=tuple(supporting_ids),
            contradicting_ids=tuple(contradicting_ids),
            t_evaluation=datetime.now(),
            verification_mode=mode,
            rule_fired="RULE_7_NUMERIC_DISTORTION",
            independence_count=independence_count,
            citation_chains=tuple(supporting_ids + contradicting_ids),
            secondary_flags=("scale_or_date_distortion",)
        )

    # Priority 6: Evidence Conflict
    if n_sup >= 1 and n_contra >= 1:
        return Verdict(
            label="CONFLICTING",
            pipeline_status="ANSWER",
            calibrated_conf=0.85,
            absence_status=absence_status,
            supporting_ids=tuple(supporting_ids),
            contradicting_ids=tuple(contradicting_ids),
            t_evaluation=datetime.now(),
            verification_mode=mode,
            rule_fired="RULE_6_EVIDENCE_CONFLICT",
            independence_count=independence_count,
            citation_chains=tuple(supporting_ids + contradicting_ids),
            secondary_flags=()
        )

    # Priority 4: Independent Contradiction
    if n_contra >= 2 and independence_count >= 2:
        return Verdict(
            label="REFUTED",
            pipeline_status="ANSWER",
            calibrated_conf=0.94,
            absence_status=absence_status,
            supporting_ids=(),
            contradicting_ids=tuple(contradicting_ids),
            t_evaluation=datetime.now(),
            verification_mode=mode,
            rule_fired="RULE_4_INDEPENDENT_CONTRADICTION",
            independence_count=independence_count,
            citation_chains=tuple(contradicting_ids),
            secondary_flags=()
        )

    # Priority 5: Independent Support
    if n_sup >= 2 and independence_count >= 2:
        return Verdict(
            label="SUPPORTED",
            pipeline_status="ANSWER",
            calibrated_conf=0.93,
            absence_status=absence_status,
            supporting_ids=tuple(supporting_ids),
            contradicting_ids=(),
            t_evaluation=datetime.now(),
            verification_mode=mode,
            rule_fired="RULE_5_INDEPENDENT_SUPPORT",
            independence_count=independence_count,
            citation_chains=tuple(supporting_ids),
            secondary_flags=()
        )

    # Priority 8: Single Source Support
    if n_sup == 1:
        return Verdict(
            label="SUPPORTED",
            pipeline_status="ANSWER",
            calibrated_conf=0.72,
            absence_status=absence_status,
            supporting_ids=tuple(supporting_ids),
            contradicting_ids=(),
            t_evaluation=datetime.now(),
            verification_mode=mode,
            rule_fired="RULE_8_SINGLE_SOURCE_SUPPORT",
            independence_count=1,
            citation_chains=tuple(supporting_ids),
            secondary_flags=("single_source_caveat",)
        )

    # Priority 9: Low Confidence Fallback
    return Verdict(
        label="INSUFFICIENT_EVIDENCE",
        pipeline_status="ANSWER",
        calibrated_conf=0.60,
        absence_status=absence_status,
        supporting_ids=tuple(supporting_ids),
        contradicting_ids=tuple(contradicting_ids),
        t_evaluation=datetime.now(),
        verification_mode=mode,
        rule_fired="RULE_9_LOW_CONFIDENCE_FALLBACK",
        independence_count=independence_count,
        citation_chains=(),
        secondary_flags=()
    )
