"""
eval/benchmark.py
Evaluation and benchmarking framework for BiN fact verification.
"""

import json
from typing import List, Dict, Any, Optional
from collections import defaultdict
from src.pipeline import BiNPipeline


def compute_multiclass_metrics(y_true: List[str], y_pred: List[str], classes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Computes per-class precision, recall, f1 and macro-F1 across decoupled labels.
    """
    if classes is None:
        classes = list(dict.fromkeys(list(y_true) + list(y_pred)))
        if not classes:
            classes = ["SUPPORTED", "REFUTED", "MISLEADING", "CONFLICTING", "INSUFFICIENT_EVIDENCE"]

    per_class = {}
    f1_scores = []

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for yt in y_true if yt == c)
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "total_evaluated": len(y_true),
        "total_claims": len(y_true),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class
    }


def compute_classification_metrics(y_true: List[str], y_pred: List[str], classes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Direct alias and wrapper for unit test compatibility.
    """
    return compute_multiclass_metrics(y_true, y_pred, classes=classes)


def evaluate_dataset(dataset_path: str, tau_abstain: float = 0.50) -> Dict[str, Any]:
    """
    Runs pipeline evaluation on the specified dataset and returns summary report.
    Compatible with both test_benchmark and run_pilot_eval CLI format.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pipeline = BiNPipeline(tau_abstain=tau_abstain)
    y_true = []
    y_pred = []
    answered_count = 0
    abstained_count = 0

    for item in data:
        cid = item.get("claim_id", "")
        txt = item.get("claim_text", "")
        gold_verdict = item.get("gold_verdict") or item.get("verdict", "INSUFFICIENT_EVIDENCE")
        pool = item.get("evidence_pool", {})

        res = pipeline.run_pipeline(raw_text=txt, evidence_pool=pool, claim_id=cid)
        gate_status = res.get("gate_status", "ANSWER")
        pred_obj = res.get("verdict")
        pred_label = pred_obj.get("label") if isinstance(pred_obj, dict) else (pred_obj.label if hasattr(pred_obj, "label") else str(pred_obj))

        if gate_status == "ABSTAIN" or pred_label == "INSUFFICIENT_EVIDENCE":
            abstained_count += 1
            pred_label = "INSUFFICIENT_EVIDENCE"
        else:
            answered_count += 1

        y_true.append(gold_verdict)
        y_pred.append(pred_label)

    all_classes = ["SUPPORTED", "REFUTED", "MISLEADING", "CONFLICTING", "INSUFFICIENT_EVIDENCE"]
    metric_results = compute_multiclass_metrics(y_true, y_pred, classes=all_classes)
    total_claims = len(data)
    coverage_rate = answered_count / total_claims if total_claims > 0 else 0.0

    return {
        "total_claims": total_claims,
        "total_evaluated": total_claims,
        "answered_claims": answered_count,
        "abstained_claims": abstained_count,
        "coverage_rate": round(coverage_rate, 4),
        "selective_coverage_rate": round(coverage_rate, 4),
        "macro_f1": metric_results["macro_f1"],
        "metrics": metric_results["per_class"]
    }
