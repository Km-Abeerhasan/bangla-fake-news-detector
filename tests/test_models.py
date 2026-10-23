"""
tests/test_models.py
Unit tests verifying hardware acceleration and IndicXNLI inference wrapper.
"""

import torch
from src.models.device import get_compute_device
from src.models.stance_classifier import IndicXNLIClassifier

def test_device_detection():
    device = get_compute_device()
    assert device.type in ["mps", "cuda", "cpu"]
    print(f"Device test passed with device: {device}")

def test_stance_classifier_fallback():
    classifier = IndicXNLIClassifier(use_fallback=True)
    
    premise_contra = "বাংলাদেশ ব্যাংক ৫০০ টাকার নোট বাতিল সংক্রান্ত কোনো বিজ্ঞপ্তি জারি করেনি। এটি ভিত্তিহীন গুজব।"
    hypo = "বাংলাদেশ ব্যাংক ৫০০ টাকার নোট বাতিল ঘোষণা করেছে।"
    logits_contra = classifier.predict_logits(premise_contra, hypo)
    
    # Verify contradiction logit is highest
    assert logits_contra[2] > logits_contra[0]

    premise_entail = "পানি উন্নয়ন বোর্ড নিশ্চিত করেছে তিস্তা ব্যারেজের জলকপাট খুলে দেওয়া হয়েছে।"
    hypo_entail = "তিস্তা নদীর বাঁধ খুলে দেওয়া হয়েছে।"
    logits_entail = classifier.predict_logits(premise_entail, hypo_entail)
    
    # Verify entailment logit is highest
    assert logits_entail[0] > logits_entail[2]

if __name__ == "__main__":
    test_device_detection()
    test_stance_classifier_fallback()
    print("All Model & MPS inference tests passed successfully!")
