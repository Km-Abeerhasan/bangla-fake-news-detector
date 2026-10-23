"""
eval/benchmark.py
Evaluation Benchmark Harness for BiN (Audit-Then-Believe).
Runs pipeline across annotated datasets, evaluates selective classification,
and calculates Macro-F1 across the 5 decoupled verdict classes.
"""

from typing import List, Dict, Any, Tuple
import json
from collections import defaultdict
from datetime import datetime

from src.pipeline import BiNPipeline


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: Tuple[str, ...] = ("SUPPORTED", "REFUTED", "MISLEADING", "CONFLICTING", "INSUFFICIENT_EVIDENCE")
) -> Dict[str, Any]:
    """
    Computes per-class Precision, Recall, F1 and overall Macro-F1.
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            tp[true_label] += 1
        else:
            fp[pred_label] += 1
            fn[true_label] += 1

    per_class = {}
    f1_scores = []

    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(classes) if classes else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "total_evaluated": len(y_true)
    }


def evaluate_dataset(
    dataset_path: str,
    tau_abstain: float = 0.60
) -> Dict[str, Any]:
    """
    Loads dataset, executes BiNPipeline on each item, and generates benchmark report.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    pipeline = BiNPipeline(tau_abstain=tau_abstain)

    y_true: List[str] = []
    y_pred: List[str] = []
    answered_count = 0
    abstain_count = 0

    for item in records:
        gold = item["gold_verdict"]
        # Format passage pool for pipeline input
        evidence_pool = {
            p["passage_id"]: {
                "text": p["text"],
                "publisher": p["publisher"],
                "tier": p["tier"],
                "url": p["url"],
                "t_pub": datetime.fromisoformat(p["t_pub"])
            }
            for p in item["passages"]
        }

        report = pipeline.run_pipeline(
            raw_text=item["claim_text"],
            evidence_pool=evidence_pool,
            t_claim=datetime.fromisoformat(item["t_claim"]),
            claim_id=item["claim_id"]
        )

        gate_status = report["gate_status"]
        if gate_status == "ANSWER" and report.get("verdict"):
            answered_count += 1
            pred = report["verdict"]["label"]
            y_true.append(gold)
            y_pred.append(pred)
        else:
            abstain_count += 1

    metrics = compute_classification_metrics(y_true, y_pred)
    coverage = answered_count / len(records) if records else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "total_claims": len(records),
        "answered_claims": answered_count,
        "abstained_claims": abstain_count,
        "coverage_rate": round(coverage, 4),
        "metrics": metrics
    }
