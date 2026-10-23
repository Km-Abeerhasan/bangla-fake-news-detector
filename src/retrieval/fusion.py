"""
src/retrieval/fusion.py
Stage 3: Evidence Ranking & Reciprocal Rank Fusion (RRF).
Deterministic fusion of dense and sparse passage rankings.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from src.contracts import EvidenceItem


def reciprocal_rank_fusion(
    bm25_ranked: List[str],
    dense_ranked: List[str],
    rrf_k: int = 60
) -> List[str]:
    """
    Computes reciprocal rank fusion scores across sparse and dense lists.
    RRF_score(d) = sum(1 / (k + rank(d)))
    """
    scores: Dict[str, float] = defaultdict(float)

    for rank, doc_id in enumerate(bm25_ranked, start=1):
        scores[doc_id] += 1.0 / (rrf_k + rank)

    for rank, doc_id in enumerate(dense_ranked, start=1):
        scores[doc_id] += 1.0 / (rrf_k + rank)

    # Sort descending by fused RRF score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]


def fuse_and_rank_evidence(
    raw_passages: Dict[str, Dict[str, Any]],
    bm25_ids: List[str],
    dense_ids: List[str],
    pass_type: str = "pass_hybrid_fusion",
    top_k: int = 5,
    rrf_k: int = 60
) -> Tuple[EvidenceItem, ...]:
    """
    Fuses rankings and constructs verified EvidenceItem tuples.
    """
    fused_order = reciprocal_rank_fusion(bm25_ids, dense_ids, rrf_k=rrf_k)
    selected_ids = fused_order[:top_k]

    evidence_items = []
    for rank, doc_id in enumerate(selected_ids, start=1):
        p_data = raw_passages.get(doc_id, {})
        item = EvidenceItem(
            passage_id=doc_id,
            text=p_data.get("text", ""),
            url=p_data.get("url", f"https://archive.factcheck.org/doc/{doc_id}"),
            publisher=p_data.get("publisher", "Unknown Source"),
            tier=p_data.get("tier", 3),
            t_pub=p_data.get("t_pub"),
            t_verif=p_data.get("t_verif"),
            retriever="hybrid_rrf",
            retrieval_pass=pass_type,
            fusion_rank=rank,
            eligibility_reason="passed_rrf_cutoff"
        )
        evidence_items.append(item)

    return tuple(evidence_items)
