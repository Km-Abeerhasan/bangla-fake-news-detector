"""
src/ingest/loader.py
Stage 1: Input Ingestion & Normalization Loader.
Converts raw claim strings or multimodal image inputs into immutable ClaimInput contracts.
"""

from typing import Optional, Tuple
from datetime import datetime
import uuid

from src.contracts import ClaimInput, OCRToken
from src.ingest.normalize import normalize_text


def load_claim_text(
    raw_text: str,
    lang: str = "bn",
    claim_id: Optional[str] = None,
    t_claim: Optional[datetime] = None,
    t_event: Optional[datetime] = None,
    raw_media_path: Optional[str] = None,
    ocr_tokens: Tuple[OCRToken, ...] = ()
) -> ClaimInput:
    """
    Ingests and normalizes claim text, preserving ligatures and recording changes.
    """
    if not claim_id:
        claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"

    clean_text, audit_log = normalize_text(raw_text)

    return ClaimInput(
        claim_id=claim_id,
        raw_text=raw_text,
        clean_text=clean_text,
        lang=lang,
        t_event=t_event,
        t_claim=t_claim,
        raw_media_path=raw_media_path,
        ocr_tokens=ocr_tokens
    )
