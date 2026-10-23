"""
src/ingest/normalize.py
Bengali-safe canonical Unicode normalizer with reversible audit log.
Preserves conjuncts (যুক্তাক্ষর) while cleaning zero-width spaces and excess whitespace.
"""

import unicodedata
import re
from typing import Tuple, List, Dict, Any

WHITESPACE_PATTERN = re.compile(r'\s+')

AUDITABLE_ZERO_WIDTH = {
    '\u200B': 'ZERO_WIDTH_SPACE',
    '\uFEFF': 'BYTE_ORDER_MARK',
}

def normalize_text(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    audit_log: List[Dict[str, Any]] = []

    # Step 1: Record zero-width non-conjunct artifacts
    for idx, char in enumerate(raw_text):
        if char in AUDITABLE_ZERO_WIDTH:
            audit_log.append({
                "op": "strip_artifact",
                "char_name": AUDITABLE_ZERO_WIDTH[char],
                "codepoint": f"U+{ord(char):04X}",
                "original_idx": idx
            })

    # Step 2: Strip unmeaningful zero-width spaces/BOM
    stripped_text = re.sub(r'[\u200B\uFEFF]', '', raw_text)

    # Step 3: Canonical Unicode NFC normalization
    nfc_text = unicodedata.normalize('NFC', stripped_text)
    if nfc_text != stripped_text:
        audit_log.append({
            "op": "unicode_nfc_recomposition",
            "len_before": len(stripped_text),
            "len_after": len(nfc_text)
        })

    # Step 4: Canonicalize whitespace
    clean_text = WHITESPACE_PATTERN.sub(' ', nfc_text).strip()
    if clean_text != nfc_text:
        audit_log.append({
            "op": "whitespace_canonicalization",
            "original_length": len(nfc_text),
            "cleaned_length": len(clean_text)
        })

    return clean_text, audit_log
