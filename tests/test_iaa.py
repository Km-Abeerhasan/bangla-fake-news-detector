"""
tests/test_iaa.py
Unit tests verifying Inter-Annotator Agreement calculation accuracy.
"""

from eval.iaa import compute_cohens_kappa, compute_krippendorff_alpha_nominal

def test_perfect_agreement():
    r1 = ["SUPPORTED", "REFUTED", "MISLEADING", "CONFLICTING", "INSUFFICIENT_EVIDENCE"]
    r2 = ["SUPPORTED", "REFUTED", "MISLEADING", "CONFLICTING", "INSUFFICIENT_EVIDENCE"]
    
    kappa = compute_cohens_kappa(r1, r2)
    assert kappa == 1.0000

    matrix = [[a, b] for a, b in zip(r1, r2)]
    alpha = compute_krippendorff_alpha_nominal(matrix)
    assert alpha == 1.0000
    print(f"Perfect agreement verified: Alpha={alpha}, Kappa={kappa}")

def test_high_agreement():
    matrix = [
        ["SUPPORTED", "SUPPORTED"],
        ["REFUTED", "REFUTED"],
        ["MISLEADING", "MISLEADING"],
        ["CONFLICTING", "CONFLICTING"],
        ["SUPPORTED", "REFUTED"]  # single mismatch
    ]
    alpha = compute_krippendorff_alpha_nominal(matrix)
    assert alpha > 0.60
    print(f"High agreement test passed: Alpha={alpha}")

if __name__ == "__main__":
    test_perfect_agreement()
    test_high_agreement()
    print("All Inter-Annotator Agreement (IAA) unit tests passed successfully!")
