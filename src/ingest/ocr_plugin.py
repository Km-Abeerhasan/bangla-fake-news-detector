"""
src/ingest/ocr_plugin.py
Stage 1: Multimodal OCR Extraction & Token Recovery Module.
Extracts text from Bengali/English screenshot cards, maps bounding boxes,
and classifies token recovery status to prevent uncalibrated noise propagation.
"""

from typing import List, Tuple, Dict, Any, Literal
from dataclasses import dataclass
from src.contracts import OCRToken

RecoveryAction = Literal["ACCEPTED", "NEEDS_PREPROCESSING", "ROUTED_TO_REVIEW"]

CONFIDENCE_ACCEPT_THRESHOLD = 0.60
CONFIDENCE_RECOVERY_THRESHOLD = 0.35


@dataclass(frozen=True)
class AuditedOCRToken:
    token: OCRToken
    action: RecoveryAction


def audit_ocr_token(text: str, confidence: float, bbox: Tuple[int, int, int, int]) -> AuditedOCRToken:
    """
    Evaluates individual OCR tokens against calibrated confidence thresholds.
    """
    token = OCRToken(text=text, confidence=confidence, bbox=bbox)
    
    if confidence >= CONFIDENCE_ACCEPT_THRESHOLD:
        action: RecoveryAction = "ACCEPTED"
    elif confidence >= CONFIDENCE_RECOVERY_THRESHOLD:
        action = "NEEDS_PREPROCESSING"
    else:
        action = "ROUTED_TO_REVIEW"
        
    return AuditedOCRToken(token=token, action=action)


def process_raw_ocr_stream(
    raw_detections: List[Dict[str, Any]]
) -> Tuple[str, Tuple[OCRToken, ...], Dict[str, int]]:
    """
    Processes a list of raw OCR detections [{'text': ..., 'conf': ..., 'bbox': (...)}].
    Returns reconstructed clean text, frozen OCRToken tuple, and recovery action statistics.
    """
    accepted_words: List[str] = []
    frozen_tokens: List[OCRToken] = []
    stats: Dict[str, int] = {
        "ACCEPTED": 0,
        "NEEDS_PREPROCESSING": 0,
        "ROUTED_TO_REVIEW": 0
    }

    for item in raw_detections:
        audited = audit_ocr_token(
            text=item["text"],
            confidence=float(item.get("conf", 0.0)),
            bbox=item.get("bbox", (0, 0, 0, 0))
        )
        stats[audited.action] += 1
        frozen_tokens.append(audited.token)
        
        # Accepted tokens form the immediate baseline clean text
        if audited.action == "ACCEPTED":
            accepted_words.append(audited.token.text)
        elif audited.action == "NEEDS_PREPROCESSING":
            # Included with flagged audit tracking
            accepted_words.append(audited.token.text)

    reconstructed_text = " ".join(accepted_words).strip()
    return reconstructed_text, tuple(frozen_tokens), stats
