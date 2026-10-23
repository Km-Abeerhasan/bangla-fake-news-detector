"""
tests/test_audit.py
Unit tests verifying publisher independence counting and temporal leakage filters.
"""

from datetime import datetime
from src.contracts import EvidenceItem
from src.audit.clustering import count_independent_publishers, filter_temporal_leakage

def make_dummy_item(pid: str, url: str, pub: str, t_pub: datetime) -> EvidenceItem:
    return EvidenceItem(
        passage_id=pid,
        text="Sample text",
        url=url,
        publisher=pub,
        tier=3,
        t_pub=t_pub,
        t_verif=datetime(2026, 9, 1),
        retriever="hybrid",
        retrieval_pass="hybrid_fusion",
        fusion_rank=1,
        eligibility_reason="valid"
    )

def test_independence_counting():
    items = (
        make_dummy_item("1", "https://www.prothomalo.com/news1", "Prothom Alo", datetime(2026, 8, 1)),
        make_dummy_item("2", "https://prothomalo.com/news2", "Prothom Alo", datetime(2026, 8, 1)),
        make_dummy_item("3", "https://thedailystar.net/news3", "Daily Star", datetime(2026, 8, 1)),
    )
    # 3 articles, but only 2 unique domains
    assert count_independent_publishers(items) == 2

def test_temporal_leakage_filter():
    t_claim = datetime(2026, 8, 10)
    items = (
        make_dummy_item("pre", "https://a.com", "A", datetime(2026, 8, 5)),
        make_dummy_item("post", "https://b.com", "B", datetime(2026, 8, 15)),
    )
    
    contemp_filtered = filter_temporal_leakage(items, t_claim=t_claim, t_verif=datetime(2026, 9, 1), mode="contemporaneous")
    assert len(contemp_filtered) == 1
    assert contemp_filtered[0].passage_id == "pre"

    retro_filtered = filter_temporal_leakage(items, t_claim=t_claim, t_verif=datetime(2026, 9, 1), mode="retrospective")
    assert len(retro_filtered) == 2

if __name__ == "__main__":
    test_independence_counting()
    test_temporal_leakage_filter()
    print("All Stage 4 evidence audit unit tests passed successfully!")
