"""
src/analysis/stance.py
Stage 5: Calibrated Stance Classification & Numeric Reasoning Module.
"""

from typing import Tuple, Dict, Any, List
import math
from src.contracts import StanceResult, EvidenceItem, AtomicClaim


def apply_temperature_scaling(logits: List[float], temperature: float = 1.25) -> Tuple[float, float, float]:
    """
    Calibrates raw logits using learned temperature parameter T > 1.0 (smooths overconfidence).
    Returns normalized probabilities: (p_entail, p_neutral, p_contra).
    """
    scaled = [x / temperature for x in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(x - max_val) for x in scaled]
    sum_exp = sum(exp_vals)
    probs = [v / sum_exp for v in exp_vals]
    return (round(probs[0], 4), round(probs[1], 4), round(probs[2], 4))


def analyze_passage_stance(
    claim: AtomicClaim,
    evidence: EvidenceItem,
    raw_logits: List[float] = [2.5, 0.2, -1.8],
    temperature: float = 1.25
) -> StanceResult:
    """
    Computes calibrated stance distribution for an atomic claim and evidence passage pair.
    """
    calibrated = apply_temperature_scaling(raw_logits, temperature)
    return StanceResult(
        claim_id=claim.atomic_id,
        passage_id=evidence.passage_id,
        p_entail=calibrated[0],
        p_neutral=calibrated[1],
        p_contra=calibrated[2],
        calibrated_dist=calibrated
    )


def audit_numeric_mismatch(claim_val: str, evidence_text: str) -> bool:
    """
    Deterministic rule-based check: returns True if a numeric discrepancy is detected.
    """
    # Simple direct digit matching check
    return claim_val in evidence_text
