"""
src/claims/processor.py
Stage 2: Claim Deconstruction, Proposition Splitting & Metadata Structuring.
Parses ClaimInput into discrete, typed AtomicClaim contracts.
"""

import re
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import uuid

from src.contracts import ClaimInput, AtomicClaim, ClaimType

# Conjunctions and punctuation marking Bengali clause boundaries
SPLIT_PATTERN = re.compile(r'[\।\!\?\;]+|\s+(?:এবং|ও|কিন্তু|অথবা|কিংবা)\s+')

# Provenance attribution patterns (e.g., "দাবি করেছে", "বিজ্ঞপ্তি জারি করেছে")
PROVENANCE_KEYWORDS = [
    "বিজ্ঞপ্তি", "ঘোষণা", "দাবি", "জানিয়েছে", "জানায়", "কর্তৃক", "সতর্কবার্তা"
]

# Modality indicators
OPINION_MARKERS = ["আমার মতে", "মনে হয়", "হতে পারে", "সুন্দর", "ভালো", "খারাপ"]
SATIRE_MARKERS = ["ব্যঙ্গাত্মক", "মজা করে", "হাস্যকর"]
PREDICTION_MARKERS = ["ভেঙে যেতে পারে", "হতে পারে", "আগামী", "ঘণ্টার মধ্যে", "সম্ভাবনা আছে", "ঝুঁকি রয়েছে"]

# Numeric patterns for Bengali and English digits
NUMERIC_PATTERN = re.compile(r'(\d+(?:\.\d+)?|[০-৯]+(?:\.[০-৯]+)?)')


def classify_modality(text: str) -> ClaimType:
    """
    Classifies proposition modality into factual, opinion, satire, or prediction.
    """
    for marker in SATIRE_MARKERS:
        if marker in text:
            return "satire"
    for marker in OPINION_MARKERS:
        if marker in text:
            return "opinion"
    for marker in PREDICTION_MARKERS:
        if marker in text:
            return "prediction"
    return "factual"


def detect_provenance(text: str) -> bool:
    """
    Detects if the proposition asserts an institutional attribution or issuance.
    """
    return any(keyword in text for keyword in PROVENANCE_KEYWORDS)


def extract_quantities(text: str) -> Tuple[Dict[str, str], ...]:
    """
    Extracts numerical figures from text for arithmetic verification.
    """
    matches = NUMERIC_PATTERN.findall(text)
    return tuple({"raw_value": m} for m in matches)


def deconstruct_claim(claim_input: ClaimInput) -> Tuple[AtomicClaim, ...]:
    """
    Deconstructs a normalized ClaimInput into one or more immutable AtomicClaim records.
    """
    raw_clauses = SPLIT_PATTERN.split(claim_input.clean_text)
    clauses = [c.strip() for c in raw_clauses if c and len(c.strip()) > 3]

    if not clauses:
        clauses = [claim_input.clean_text]

    atomic_claims: List[AtomicClaim] = []

    for idx, clause in enumerate(clauses):
        claim_type = classify_modality(clause)
        is_prov = detect_provenance(clause)
        quantities = extract_quantities(clause)

        atomic = AtomicClaim(
            atomic_id=f"{claim_input.claim_id}-A{idx+1}",
            parent_id=claim_input.claim_id,
            text=clause,
            claim_type=claim_type,
            entities=(),  # Extracted downstream or via entity dictionary
            quantities=quantities,
            t_event_anchor=claim_input.t_event,
            is_provenance=is_prov
        )
        atomic_claims.append(atomic)

    return tuple(atomic_claims)
