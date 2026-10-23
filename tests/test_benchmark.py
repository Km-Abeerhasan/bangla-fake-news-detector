"""
tests/test_benchmark.py
Unit tests verifying benchmark metrics computation and coverage evaluation.
"""

from eval.benchmark import compute_classification_metrics, evaluate_dataset

def test_metrics_computation():
    y_true = ["SUPPORTED", "REFUTED", "MISLEADING", "SUPPORTED"]
    y_pred = ["SUPPORTED", "REFUTED", "SUPPORTED", "SUPPORTED"]

    results = compute_classification_metrics(y_true, y_pred)
    assert 0.0 <= results["macro_f1"] <= 1.0
    assert results["total_evaluated"] == 4
    # SUPPORTED class: TP=2, FP=1, FN=0 -> Precision=2/3, Recall=1.0
    assert results["per_class"]["SUPPORTED"]["recall"] == 1.0

def test_evaluate_pilot_dataset():
    report = evaluate_dataset("annotation/pilot/claims_50.json", tau_abstain=0.60)
    assert report["total_claims"] >= 2
    assert report["answered_claims"] > 0
    assert "metrics" in report
    assert report["coverage_rate"] > 0.0

if __name__ == "__main__":
    test_metrics_computation()
    test_evaluate_pilot_dataset()
    print("All Benchmark evaluation unit tests passed successfully!")
