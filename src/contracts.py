"""
X-MiND: Typed Data Contracts (v21)
Immutable, frozen dataclasses defining stage interfaces across S1-S8.
Repository: bin-audit-then-believe
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Tuple
from datetime import datetime

VerdictLabel = Literal["SUPPORTED", "REFUTED", "MISLEADING", "CONFLICTING", "INSUFFICIENT_EVIDENCE"]
PipelineStatus = Literal["ANSWER", "NON_FACTUAL", "RETRIEVAL_DEFICIT", "ABSTAIN", "HUMAN_REVIEW"]
ClaimType = Literal["factual", "opinion", "satire", "prediction"]
AbsenceStatus = Literal["confirmed_within_scope", "search_incomplete", "unknown"]
VerificationMode = Literal["contemporaneous", "retrospective"]
RetrievalPass = Literal["pass_a", "pass_b", "hybrid_fusion"]

@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)

@dataclass(frozen=True)
class ClaimInput:
    claim_id: str
    raw_text: str
    clean_text: str
    lang: str
    t_event: Optional[datetime] = None
    t_claim: Optional[datetime] = None  # Optional; fallbacks to earliest archive date
    raw_media_path: Optional[str] = None
    ocr_tokens: Tuple[OCRToken, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class AtomicClaim:
    atomic_id: str
    parent_id: str
    text: str
    claim_type: ClaimType
    entities: Tuple[str, ...] = field(default_factory=tuple)
    quantities: Tuple[Dict[str, str], ...] = field(default_factory=tuple)
    t_event_anchor: Optional[datetime] = None
    is_provenance: bool = False

@dataclass(frozen=True)
class EvidenceItem:
    passage_id: str
    text: str
    url: str
    publisher: str
    tier: int  # 1: Official, 2: Fact-Checker, 3: Mainstream, 4: Other
    t_pub: datetime
    t_verif: datetime
    retriever: Literal["bm25", "dense_e5", "hybrid"]
    retrieval_pass: RetrievalPass
    fusion_rank: int
    eligibility_reason: str
    cluster_id: Optional[str] = None

@dataclass(frozen=True)
class StanceResult:
    claim_id: str
    passage_id: str
    p_entail: float
    p_neutral: float
    p_contra: float
    calibrated_dist: Tuple[float, float, float]  # (entail, neutral, contra)

@dataclass(frozen=True)
class Verdict:
    label: Optional[VerdictLabel]
    pipeline_status: PipelineStatus
    calibrated_conf: float
    absence_status: AbsenceStatus
    supporting_ids: Tuple[str, ...]
    contradicting_ids: Tuple[str, ...]
    t_evaluation: datetime
    verification_mode: VerificationMode
    rule_fired: str
    independence_count: int
    citation_chains: Tuple[str, ...]
    secondary_flags: Tuple[str, ...]

@dataclass(frozen=True)
class GateDecision:
    pipeline_status: PipelineStatus
    reason: str
    verdict: Optional[Verdict]
    features: Dict[str, float]
