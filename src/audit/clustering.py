"""
src/audit/clustering.py
Stage 4: Evidence Deduplication, Independence Counting & Temporal Leakage Filter.
"""

from typing import Tuple, List, Set, Dict
from datetime import datetime
from urllib.parse import urlparse
from src.contracts import EvidenceItem, VerificationMode


def extract_domain(url: str, fallback_publisher: str) -> str:
    """Extracts base publisher domain for independence clustering."""
    if url.startswith("http"):
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    return fallback_publisher.strip().lower()


def count_independent_publishers(items: Tuple[EvidenceItem, ...]) -> int:
    """Counts unique independent publisher domains across retrieved evidence."""
    unique_domains: Set[str] = set()
    for item in items:
        domain = extract_domain(item.url, item.publisher)
        if domain:
            unique_domains.add(domain)
    return len(unique_domains)


def filter_temporal_leakage(
    items: Tuple[EvidenceItem, ...],
    t_claim: datetime,
    t_verif: datetime,
    mode: VerificationMode
) -> Tuple[EvidenceItem, ...]:
    """
    Enforces point-in-time leakage constraints:
    - Contemporaneous: t_pub <= t_claim
    - Retrospective: t_pub <= t_verif
    """
    eligible: List[EvidenceItem] = []
    cutoff = t_claim if mode == "contemporaneous" else t_verif

    for item in items:
        if item.t_pub <= cutoff:
            eligible.append(item)

    return tuple(eligible)
