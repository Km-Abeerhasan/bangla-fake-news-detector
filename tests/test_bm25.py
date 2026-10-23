"""
tests/test_bm25.py
Unit tests verifying Bengali BM25 lexical tokenization and retrieval scoring.
"""

from src.retrieval.bm25_search import BengaliBM25Searcher, tokenize_bengali

def test_bengali_tokenization():
    text = "তিস্তা নদীর বাঁধ খুলে দেওয়া হয়েছে, পানি দ্রুত বাড়ছে!"
    tokens = tokenize_bengali(text)
    assert "তিস্তা" in tokens
    assert "বাঁধ" in tokens
    assert "হয়েছে" not in tokens  # Stopword removed

def test_bm25_ranking():
    corpus = {
        "doc1": "তিস্তা ব্যারেজের সবকটি জলকপাট খুলে দেওয়া হয়েছে ভারী বর্ষণের কারণে।",
        "doc2": "বাংলাদেশ ব্যাংক ৫০০ টাকার নোট বাতিলের তথ্যকে ভুয়া এবং গুজব হিসেবে উল্লেখ করেছে।",
        "doc3": "সুন্দরবনের নদী ও ম্যানগ্রোভ বনাঞ্চল রক্ষায় নতুন নিরাপত্তা কমিটি গঠন করা হয়েছে।"
    }

    searcher = BengaliBM25Searcher(corpus)
    
    # Query 1: তিস্তা নদী সংক্রান্ত সার্চ
    results = searcher.search("তিস্তা ব্যারেজ জলকপাট প্লাবিত", top_k=2)
    assert len(results) == 2
    top_doc_id, top_score = results[0]
    assert top_doc_id == "doc1"
    assert top_score > 0.0

    # Query 2: বাংলাদেশ ব্যাংক ৫০০ টাকার নোট সার্চ
    results_bb = searcher.search("বাংলাদেশ ব্যাংক ৫০০ টাকার নোট গুজব", top_k=2)
    assert results_bb[0][0] == "doc2"

if __name__ == "__main__":
    test_bengali_tokenization()
    test_bm25_ranking()
    print("All Bengali BM25 Lexical Retrieval unit tests passed successfully!")
