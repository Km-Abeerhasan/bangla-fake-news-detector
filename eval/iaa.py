"""
eval/iaa.py
Inter-Annotator Agreement (IAA) Harness for BiN (Audit-Then-Believe).
Calculates Krippendorff's Alpha (Nominal Metric) and Cohen's Kappa
across human fact-checkers for the 5 decoupled verdict categories.
"""

from typing import List, Dict, Any, Tuple
from collections import Counter


def compute_cohens_kappa(rater1: List[str], rater2: List[str]) -> float:
    """
    Computes Cohen's Kappa for two annotators across nominal labels.
    """
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0

    n = len(rater1)
    # Observed agreement
    p_o = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    # Expected agreement
    counts1 = Counter(rater1)
    counts2 = Counter(rater2)
    categories = set(rater1) | set(rater2)

    p_e = sum((counts1[c] / n) * (counts2[c] / n) for c in categories)

    if p_e == 1.0:
        return 1.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def compute_krippendorff_alpha_nominal(matrix: List[List[str]]) -> float:
    """
    Computes Krippendorff's Alpha for nominal categories with arbitrary raters.
    Input matrix: List of items, where each item is a list of ratings from annotators.
    """
    all_units = matrix
    if not all_units:
        return 0.0

    # Count values per unit
    pair_counts = 0
    observed_mismatches = 0.0

    category_counts: Counter = Counter()
    total_ratings = 0

    for unit in all_units:
        m = len(unit)
        if m < 2:
            continue
        pair_counts += m * (m - 1)
        for val in unit:
            category_counts[val] += 1
            total_ratings += 1

        for i in range(m):
            for j in range(i + 1, m):
                if unit[i] != unit[j]:
                    observed_mismatches += 2.0  # symmetric difference

    if pair_counts == 0 or total_ratings <= 1:
        return 1.0

    # D_o: Observed disagreement
    d_o = observed_mismatches / pair_counts

    # D_e: Expected disagreement
    expected_mismatches = 0.0
    for c1, n1 in category_counts.items():
        for c2, n2 in category_counts.items():
            if c1 != c2:
                expected_mismatches += n1 * n2

    d_e = expected_mismatches / (total_ratings * (total_ratings - 1))

    if d_e == 0.0:
        return 1.0

    alpha = 1.0 - (d_o / d_e)
    return round(alpha, 4)
