"""
src/retrieval/fusion.py
Reciprocal Rank Fusion (RRF) and Contrastive Query Expansion for Bengali Evidence Retrieval.
"""

from typing import Dict, List, Any
from collections import defaultdict
from src.contracts import EvidenceItem


def build_contrastive_query(claim_text: str) -> str:
    """
    Expands input claim with contrastive debunking keywords (গুজব, ভুয়া, সত্যতা, দাবি).
    """
    contrastive_terms = ["গুজব", "ভুয়া", "অসত্য", "সত্যতা", "দাবি"]
    expansion = " ".join(contrastive_terms)
    return f"{claim_text} {expansion}".strip()


def compute_rrf_scores(
    bm25_ranks: List[str],
    dense_ranks: List[str],
    rrf_k: int = 60
) -> Dict[str, float]:
    """
    Computes reciprocal rank fusion scores across BM25 and dense retrieval rankings.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    for rank, doc_id in enumerate(bm25_ranks):
        rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)

    for rank, doc_id in enumerate(dense_ranks):
        rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)

    return dict(rrf_scores)


def reciprocal_rank_fusion(*ranking_lists, rrf_k: int = 60, k: int = 60, **kwargs) -> List[str]:
    """
    Universal Reciprocal Rank Fusion helper compatible with hybrid_engine.
    Accepts both positional ranking lists and keyword ranking arguments (bm25_ranked, dense_ranked, etc.).
    Returns a list of document IDs sorted in descending order of RRF score.
    """
    effective_k = kwargs.get("k", kwargs.get("rrf_k", rrf_k if rrf_k != 60 else k))
    
    # Collect lists from both positional and keyword arguments
    all_ranking_lists = list(ranking_lists)
    for kw_key, kw_val in kwargs.items():
        if kw_key not in ("k", "rrf_k") and isinstance(kw_val, (list, tuple)):
            all_ranking_lists.append(kw_val)

    rrf_scores: Dict[str, float] = defaultdict(float)
    for rank_list in all_ranking_lists:
        for rank, doc_id in enumerate(rank_list):
            rrf_scores[doc_id] += 1.0 / (effective_k + rank + 1)

    sorted_ranked_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
    return sorted_ranked_ids


def fuse_and_rank_evidence(
    raw_passages: Dict[str, Dict[str, Any]],
    bm25_ids: List[str],
    dense_ids: List[str],
    pass_type: str = "hybrid_fusion",
    top_k: int = 5,
    rrf_k: int = 60
) -> List[EvidenceItem]:
    """
    Fuses candidate documents, ranks via RRF, and packages into complete EvidenceItem objects.
    """
    rrf_scores = compute_rrf_scores(bm25_ids, dense_ids, rrf_k=rrf_k)
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

    evidence_items = []
    for rank_idx, doc_id in enumerate(sorted_doc_ids, start=1):
        p_data = raw_passages.get(doc_id, {})
        item = EvidenceItem(
            passage_id=doc_id,
            text=p_data.get("text", ""),
            url=p_data.get("url", ""),
            publisher=p_data.get("publisher", "Unknown"),
            tier=p_data.get("tier", 3),
            t_pub=p_data.get("t_pub", "2026-08-01T00:00:00"),
            t_verif=p_data.get("t_verif", "2026-08-01T00:00:00"),
            retriever="hybrid",
            retrieval_pass=pass_type,
            fusion_rank=rank_idx,
            eligibility_reason=p_data.get("eligibility_reason", "passed_temporal_audit"),
            cluster_id=p_data.get("cluster_id", f"c_{doc_id}")
        )
        evidence_items.append(item)

    return evidence_items
