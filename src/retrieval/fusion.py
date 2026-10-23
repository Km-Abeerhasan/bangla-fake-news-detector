"""
src/retrieval/fusion.py
Stage 3: Hybrid Retrieval & Reciprocal Rank Fusion (RRF).
Combines BM25 and Dense rankings, logging retrieval passes and fusion ranks into EvidenceItem.
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime
from src.contracts import EvidenceItem, RetrievalPass

RRF_K_DEFAULT = 60

# Contrastive cues for Pass B balanced refutation queries
CONTRASTIVE_TERMS_BN = ["গুজব", "ভুয়া", "বিভ্রান্তিকর", "সত্য নয়", "অস্বীকৃতি"]


def build_contrastive_query(claim_text: str) -> str:
    """
    Constructs a Pass B contrastive query to balance evidence discovery.
    """
    return f"{claim_text} {' '.join(CONTRASTIVE_TERMS_BN[:3])}"


def compute_rrf_scores(
    bm25_ranked_ids: List[str],
    dense_ranked_ids: List[str],
    rrf_k: int = RRF_K_DEFAULT
) -> Dict[str, float]:
    """
    Computes reciprocal rank fusion scores across BM25 and Dense candidate lists.
    RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: Dict[str, float] = {}

    for rank, doc_id in enumerate(bm25_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    for rank, doc_id in enumerate(dense_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    return scores


def fuse_and_rank_evidence(
    raw_passages: Dict[str, Dict],
    bm25_ids: List[str],
    dense_ids: List[str],
    pass_type: RetrievalPass,
    top_k: int = 5
) -> Tuple[EvidenceItem, ...]:
    """
    Fuses rankings, assigns fusion ranks, and packages passages into immutable EvidenceItem contracts.
    """
    rrf_scores = compute_rrf_scores(bm25_ids, dense_ids)
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    fused_items: List[EvidenceItem] = []
    for rank, pid in enumerate(sorted_ids, start=1):
        item_data = raw_passages[pid]
        fused_items.append(
            EvidenceItem(
                passage_id=pid,
                text=item_data["text"],
                url=item_data.get("url", ""),
                publisher=item_data.get("publisher", "unknown"),
                tier=item_data.get("tier", 4),
                t_pub=item_data.get("t_pub", datetime.now()),
                t_verif=item_data.get("t_verif", datetime.now()),
                retriever="hybrid",
                retrieval_pass=pass_type,
                fusion_rank=rank,
                eligibility_reason="passed_contemporaneous_filter",
                cluster_id=item_data.get("cluster_id")
            )
        )

    return tuple(fused_items)
