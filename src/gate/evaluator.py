"""
src/gate/evaluator.py
Stage 7: Multi-Signal Reliability Gate & Selective Classification Module.
Evaluates epistemic uncertainty and triggers calibrated abstention to optimize AURC.
"""

from typing import Tuple, Dict
import math

from src.contracts import Verdict, GateDecision, EvidenceItem, StanceResult, PipelineStatus

TAU_ABSTAIN_THRESHOLD = 0.65


def compute_reliability_features(
    verdict: Verdict,
    items: Tuple[EvidenceItem, ...],
    stances: Tuple[StanceResult, ...]
) -> Dict[str, float]:
    """
    Computes epistemic signal features for selective classification.
    """
    if not items or not stances:
        return {
            "max_stance_conf": 0.0,
            "independence_ratio": 0.0,
            "avg_tier_weight": 0.0,
            "overall_reliability": 0.0
        }

    # Signal 1: Max Stance Confidence
    max_stance = max(max(s.calibrated_dist) for s in stances)

    # Signal 2: Publisher Independence Ratio
    indep_ratio = min(1.0, verdict.independence_count / max(1, len(items)))

    # Signal 3: Source Tier Weighting (Tier 1 = 1.0, Tier 2 = 0.8, Tier 3 = 0.6, Tier 4 = 0.3)
    tier_weights = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.3}
    avg_tier = sum(tier_weights.get(item.tier, 0.3) for item in items) / len(items)

    # Composite reliability score
    reliability = (0.45 * max_stance) + (0.35 * avg_tier) + (0.20 * indep_ratio)

    return {
        "max_stance_conf": round(max_stance, 4),
        "independence_ratio": round(indep_ratio, 4),
        "avg_tier_weight": round(avg_tier, 4),
        "overall_reliability": round(reliability, 4)
    }


def evaluate_gate(
    verdict: Verdict,
    items: Tuple[EvidenceItem, ...],
    stances: Tuple[StanceResult, ...],
    tau_abstain: float = TAU_ABSTAIN_THRESHOLD,
    force_human_review: bool = False
) -> GateDecision:
    """
    Decides whether to output the verdict, abstain, or escalate to human review.
    """
    features = compute_reliability_features(verdict, items, stances)
    rel_score = features["overall_reliability"]

    if force_human_review:
        return GateDecision(
            pipeline_status="HUMAN_REVIEW",
            reason="Triggered by safety-critical override or high discrepancy flag.",
            verdict=verdict,
            features=features
        )

    if verdict.label == "INSUFFICIENT_EVIDENCE" and rel_score < tau_abstain:
        return GateDecision(
            pipeline_status="ABSTAIN",
            reason=f"Reliability score ({rel_score}) below threshold ({tau_abstain}). Epistemic coverage deficit.",
            verdict=verdict,
            features=features
        )

    return GateDecision(
        pipeline_status="ANSWER",
        reason=f"Reliability score ({rel_score}) meets threshold ({tau_abstain}).",
        verdict=verdict,
        features=features
    )
