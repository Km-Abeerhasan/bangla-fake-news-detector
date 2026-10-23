"""
eval/agreement.py
Inter-Annotator Agreement Auditor using Krippendorff's Alpha (Nominal Metric).
Validates multi-annotator concordance on the 50-claim pilot benchmark.
"""

from typing import List, Dict, Any
from collections import Counter


def compute_krippendorff_alpha_nominal(matrix: List[List[Any]]) -> float:
    """
    Computes Krippendorff's alpha for nominal data across N units and R raters.
    Handles missing values represented by None.
    """
    n_units = len(matrix)
    if n_units == 0:
        return 0.0

    # Count values per unit
    unit_counters = []
    all_values = set()
    total_pairable_ratings = 0

    for row in matrix:
        valid_vals = [v for v in row if v is not None]
        counter = Counter(valid_vals)
        unit_counters.append(counter)
        all_values.update(valid_vals)
        m = len(valid_vals)
        if m > 1:
            total_pairable_ratings += m * (m - 1)

    if total_pairable_ratings == 0:
        return 0.0

    # Observed disagreement (Do)
    observed_disagreements = 0.0
    for counter in unit_counters:
        m = sum(counter.values())
        if m > 1:
            for val_i in counter:
                for val_j in counter:
                    if val_i != val_j:
                        observed_disagreements += (counter[val_i] * counter[val_j]) / (m - 1)

    # Marginals across all pairable units
    marginal_counts: Counter = Counter()
    for counter in unit_counters:
        m = sum(counter.values())
        if m > 1:
            for val, count in counter.items():
                marginal_counts[val] += count

    total_n = sum(marginal_counts.values())
    if total_n <= 1:
        return 0.0

    # Expected disagreement (De)
    expected_disagreements = 0.0
    for val_i in all_values:
        for val_j in all_values:
            if val_i != val_j:
                expected_disagreements += marginal_counts[val_i] * marginal_counts[val_j]

    expected_disagreements /= (total_n - 1)

    if expected_disagreements == 0.0:
        return 1.0  # Perfect agreement

    alpha = 1.0 - (observed_disagreements / expected_disagreements)
    return round(alpha, 4)
