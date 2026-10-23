"""
tests/test_ocr_plugin.py
Unit tests verifying OCR token confidence auditing and recovery routing.
"""

from src.ingest.ocr_plugin import audit_ocr_token, process_raw_ocr_stream

def test_token_recovery_thresholds():
    high_conf = audit_ocr_token("প্লাবিত", 0.85, (10, 10, 50, 20))
    mid_conf = audit_ocr_token("বাঁধ", 0.45, (65, 10, 40, 20))
    low_conf = audit_ocr_token("ভেঙে", 0.20, (110, 10, 40, 20))

    assert high_conf.action == "ACCEPTED"
    assert mid_conf.action == "NEEDS_PREPROCESSING"
    assert low_conf.action == "ROUTED_TO_REVIEW"

def test_process_raw_ocr_stream():
    sample_detections = [
        {"text": "তিস্তা", "conf": 0.92, "bbox": (0, 0, 30, 15)},
        {"text": "নদীর", "conf": 0.88, "bbox": (35, 0, 25, 15)},
        {"text": "পানি", "conf": 0.50, "bbox": (65, 0, 20, 15)},
        {"text": "[noise]", "conf": 0.15, "bbox": (90, 0, 10, 15)}
    ]
    
    text, tokens, stats = process_raw_ocr_stream(sample_detections)
    
    assert len(tokens) == 4
    assert stats["ACCEPTED"] == 2
    assert stats["NEEDS_PREPROCESSING"] == 1
    assert stats["ROUTED_TO_REVIEW"] == 1
    assert "তিস্তা নদীর পানি" in text

if __name__ == "__main__":
    test_token_recovery_thresholds()
    test_process_raw_ocr_stream()
    print("All Stage 1 OCR recovery unit tests passed successfully!")
