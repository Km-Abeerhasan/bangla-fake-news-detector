"""
tests/test_agreement.py
Unit tests verifying Krippendorff's Alpha calculation on sample annotation matrices.
"""

from eval.agreement import compute_krippendorff_alpha_nominal

def test_perfect_agreement():
    # 3 claims, 2 raters, complete consensus
    matrix = [
        ["SUPPORTED", "SUPPORTED"],
        ["REFUTED", "REFUTED"],
        ["MISLEADING", "MISLEADING"]
    ]
    alpha = compute_krippendorff_alpha_nominal(matrix)
    assert alpha == 1.0

def test_partial_disagreement():
    # 4 claims, 2 raters with realistic partial agreement
    matrix = [
        ["SUPPORTED", "SUPPORTED"],
        ["REFUTED", "REFUTED"],
        ["MISLEADING", "CONFLICTING"],
        ["SUPPORTED", "SUPPORTED"]
    ]
    alpha = compute_krippendorff_alpha_nominal(matrix)
    assert 0.60 < alpha < 1.0

if __name__ == "__main__":
    test_perfect_agreement()
    test_partial_disagreement()
    print("All Krippendorff's alpha unit tests passed successfully!")
