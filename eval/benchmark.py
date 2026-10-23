"""
eval/benchmark.py
Standardized Evaluation Harness for BiN (Audit-Then-Believe).
Calculates Decoupled 5-Class Macro-F1, Selective Accuracy, and Abstention Rates.
"""

import json
from typing import Dict, Any, List
from src.pipeline import BiNPipeline


def compute_multiclass_metrics(y_true: List[str], y_pred: List[str], classes: List[str]) -> Dict[str, Any]:
    metrics = {}
    f1_sum = 0.0

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(1 for yt in y_true if yt == c)
        }
        f1_sum += f1

    macro_f1 = f1_sum / len(classes) if classes else 0.0
    return {"per_class": metrics, "macro_f1": round(macro_f1, 4)}


def evaluate_dataset(dataset_path: str = "annotation/pilot/claims_50.json", tau_abstain: float = 0.60) -> Dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        claims_data = json.load(f)

    pipeline = BiNPipeline(tau_abstain=tau_abstain)
    classes = ["SUPPORTED", "REFUTED", "MISLEADING", "CONFLICTING", "INSUFFICIENT_EVIDENCE"]

    y_true = []
    y_pred = []
    abstained_count = 0

    for item in claims_data:
        gold_verdict = item.get("gold_verdict", "INSUFFICIENT_EVIDENCE")
        claim_text = item.get("claim_text", "")
        evidence_pool = item.get("evidence_pool")
        if evidence_pool is None:
            passages = item.get("passages", [])
            evidence_pool = {p.get("passage_id", f"p_{i}"): p for i, p in enumerate(passages)} if isinstance(passages, list) else {}

        result = pipeline.run_pipeline(
            raw_text=claim_text,
            evidence_pool=evidence_pool,
            claim_id=item.get("claim_id")
        )

        gate_status = result.get("gate_status")
        verdict_obj = result.get("verdict")
        pred_label = "INSUFFICIENT_EVIDENCE"
        if isinstance(verdict_obj, dict):
            pred_label = verdict_obj.get("label", verdict_obj.get("predicted_label", "INSUFFICIENT_EVIDENCE"))
        elif hasattr(verdict_obj, "label"):
            pred_label = verdict_obj.label
        elif isinstance(verdict_obj, str):
            pred_label = verdict_obj

        if gate_status == "ABSTAIN" or pred_label is None or pred_label == "INSUFFICIENT_EVIDENCE":
            abstained_count += 1
            y_pred.append("INSUFFICIENT_EVIDENCE")
        else:
            y_pred.append(pred_label)

        y_true.append(gold_verdict)

    eval_results = compute_multiclass_metrics(y_true, y_pred, classes)
    total = len(claims_data)
    answered = total - abstained_count

    return {
        "dataset": dataset_path,
        "total_claims": total,
        "answered_claims": answered,
        "answered_decisions": answered,
        "abstained_claims": abstained_count,
        "abstained_decisions": abstained_count,
        "selective_coverage_rate": round(answered / total, 4) if total > 0 else 0.0,
        "macro_f1": eval_results["macro_f1"],
        "metrics": eval_results["per_class"]
    }
