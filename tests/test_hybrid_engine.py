"""
tests/test_hybrid_engine.py
Unit tests verifying BengaliHybridRetriever end-to-end execution on Apple Silicon MPS.
"""

from datetime import datetime
from src.retrieval.hybrid_engine import BengaliHybridRetriever

def test_hybrid_engine_ranking():
    corpus = {
        "doc_teesta": {
            "text": "পানি উন্নয়ন বোর্ড জানায় ভারী বর্ষণ ও উজানের পাহাড়ি ঢলে তিস্তা ব্যারেজের ৪৪টি জলকপাট খুলে দেওয়া হয়েছে।",
            "publisher": "BWDB Official",
            "tier": 1,
            "url": "https://bwdb.gov.bd/flood",
            "t_pub": datetime(2026, 8, 9)
        },
        "doc_bb": {
            "text": "বাংলাদেশ ব্যাংক ৫০০ টাকার নোট বাতিলের তথ্যকে সম্পূর্ণ গুজব বলে প্রেস বিজ্ঞপ্তি দিয়েছে।",
            "publisher": "Bangladesh Bank Official",
            "tier": 1,
            "url": "https://bb.org.bd/rumor",
            "t_pub": datetime(2026, 9, 2)
        },
        "doc_unrelated": {
            "text": "সুন্দরবনের নদী ও বনাঞ্চল রক্ষায় নতুন নিরাপত্তা কমিটি গঠন করা হয়েছে।",
            "publisher": "Daily Star",
            "tier": 3,
            "url": "https://thedailystar.net/forest",
            "t_pub": datetime(2026, 8, 10)
        }
    }

    engine = BengaliHybridRetriever()
    query = "তিস্তা নদীর বাঁধের জলকপাট খুলে দেওয়ার বিষয়ে সরকারি তথ্য"
    
    ranked_evidence = engine.retrieve(query=query, corpus=corpus, top_k=2)
    assert len(ranked_evidence) == 2
    assert ranked_evidence[0].passage_id == "doc_teesta"
    assert ranked_evidence[0].fusion_rank == 1
    assert ranked_evidence[0].tier == 1

if __name__ == "__main__":
    test_hybrid_engine_ranking()
    print("All Hybrid Dense-Sparse Retrieval engine tests passed successfully!")
