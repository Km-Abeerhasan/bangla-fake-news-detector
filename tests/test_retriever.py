"""
tests/test_retriever.py
Unit tests verifying MultilingualDenseRetriever embedding computation on MPS.
"""

import torch
from src.models.retriever import MultilingualDenseRetriever

def test_dense_encoder_dimensions_and_norm():
    retriever = MultilingualDenseRetriever()
    texts = [
        "তিস্তা ব্যারেজের সবকটি জলকপাট খুলে দেওয়া হয়েছে।",
        "বাংলাদেশ ব্যাংক ৫০০ টাকার নোট বাতিল করেনি।"
    ]
    
    embeddings = retriever.encode(texts, is_query=False)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 2
    # Check L2 normalization (norm should be ~1.0)
    norms = torch.norm(embeddings, p=2, dim=1)
    for n in norms:
        assert abs(n.item() - 1.0) < 1e-4

    print(f"Retriever test passed on device: {retriever.device} with shape: {embeddings.shape}")

if __name__ == "__main__":
    test_dense_encoder_dimensions_and_norm()
    print("All Dense Retriever unit tests passed successfully!")
