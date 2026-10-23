"""
tests/test_pipeline.py
End-to-end integration test executing Stages 1-8 through BiNPipeline.
"""

from datetime import datetime
from src.pipeline import BiNPipeline

def test_full_pipeline_run():
    pipeline = BiNPipeline(tau_abstain=0.60)
    
    # টেস্ট ইনপুট: সত্য যাচাই করার জন্য একটি বাস্তবমুখী বাংলা সংবাদ দাবি
    sample_text = "তিস্তা নদীর বাঁধ খুলে দেওয়ায় নিম্নাঞ্চল প্লাবিত হয়েছে।"
    
    # সিমুলেটেড বিশ্বস্ত প্রমাণপত্র
    evidence_pool = {
        "doc101": {
            "text": "পানি উন্নয়ন বোর্ড নিশ্চিত করেছে তিস্তা ব্যারেজের সবকটি জলকপাট খুলে দেওয়া হয়েছে।",
            "publisher": "Prothom Alo",
            "tier": 3,
            "url": "https://prothomalo.com/news/101",
            "t_pub": datetime(2026, 8, 10)
        },
        "doc102": {
            "text": "উজানের ঢলে তিস্তা অববাহিকার চরাঞ্চল ও নিম্নাঞ্চল প্লাবিত হয়েছে বলে জানিয়েছে প্রশাসন।",
            "publisher": "Daily Star",
            "tier": 3,
            "url": "https://thedailystar.net/news/102",
            "t_pub": datetime(2026, 8, 10)
        }
    }
    
    report = pipeline.run_pipeline(
        raw_text=sample_text,
        evidence_pool=evidence_pool,
        t_claim=datetime(2026, 8, 12),
        claim_id="TEST-E2E-01"
    )

    # ভেরিফিকেশন যাচাই
    assert report["project"] == "BiN (Audit-Then-Believe)"
    assert report["gate_status"] == "ANSWER"
    assert report["verdict"]["label"] == "SUPPORTED"
    assert report["verdict"]["rule_fired"] == "RULE_5_INDEPENDENT_SUPPORT"
    assert report["citation_audit"]["passed"] is True
    assert len(report["evidence_docket"]) == 2

if __name__ == "__main__":
    test_full_pipeline_run()
    print("All End-to-End Pipeline tests executed and passed successfully!")
