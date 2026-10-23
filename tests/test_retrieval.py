"""
tests/test_retrieval.py
Unit tests verifying RRF scoring, contrastive query building, and EvidenceItem packaging.
"""

from datetime import datetime
from src.retrieval.fusion import (
    build_contrastive_query,
    compute_rrf_scores,
    fuse_and_rank_evidence
)
from src.contracts import EvidenceItem

def test_contrastive_query_builder():
    claim = "তিস্তা নদীর বাঁধ ভেঙে গেছে"
    query_b = build_contrastive_query(claim)
    assert "গুজব" in query_b
    assert "ভুয়া" in query_b

def test_rrf_scoring_and_fusion():
    bm25_ranks = ["doc1", "doc2", "doc3"]
    dense_ranks = ["doc2", "doc1", "doc4"]

    scores = compute_rrf_scores(bm25_ranks, dense_ranks, rrf_k=60)
    # doc1 and doc2 appear in both lists; both must outrank doc3 and doc4
    assert scores["doc1"] > scores["doc3"]
    assert scores["doc2"] > scores["doc4"]

def test_fuse_and_rank_evidence():
    dummy_passages = {
        "doc1": {"text": "বিডব্লিউডিবি সতর্কবার্তা জারি করেনি।", "publisher": "BWDB", "tier": 1},
        "doc2": {"text": "তিস্তা নদীর বাঁধ অক্ষত রয়েছে।", "publisher": "DailyStar", "tier": 3}
    }
    
    items = fuse_and_rank_evidence(
        raw_passages=dummy_passages,
        bm25_ids=["doc1", "doc2"],
        dense_ids=["doc1", "doc2"],
        pass_type="hybrid_fusion",
        top_k=2
    )

    assert len(items) == 2
    assert isinstance(items[0], EvidenceItem)
    assert items[0].fusion_rank == 1
    assert items[0].retriever == "hybrid"
    assert items[0].retrieval_pass == "hybrid_fusion"
    assert items[0].tier == 1

if __name__ == "__main__":
    test_contrastive_query_builder()
    test_rrf_scoring_and_fusion()
    test_fuse_and_rank_evidence()
    print("All Stage 3 retrieval fusion unit tests passed successfully!")
