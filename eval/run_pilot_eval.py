"""
eval/run_pilot_eval.py
CLI Runner executing benchmark harness on the 50-claim pilot dataset.
Prints formatted 5-class metrics table and outputs auditable JSON report.
"""

import json
from pathlib import Path
from eval.benchmark import evaluate_dataset

def main():
    dataset_path = "annotation/pilot/claims_50.json"
    print(f"[*] Running BiN Pipeline Benchmark on: {dataset_path} ...")
    
    report = evaluate_dataset(dataset_path=dataset_path, tau_abstain=0.50)
    
    total = report.get("total_claims", 50)
    answered = report.get("answered_claims", report.get("answered_decisions", 0))
    abstained = report.get("abstained_claims", report.get("abstained_decisions", 0))
    cov_rate = report.get("selective_coverage_rate", report.get("coverage_rate", 0.0))
    macro_f1 = report.get("macro_f1", 0.0)
    metrics = report.get("metrics", {})

    print("\n=======================================================")
    print("       BiN (Audit-Then-Believe) - PILOT BENCHMARK       ")
    print("=======================================================")
    print(f"Total Evaluated Claims  : {total}")
    print(f"Answered Decisions      : {answered}")
    print(f"Abstained Decisions     : {abstained}")
    print(f"Selective Coverage Rate : {cov_rate * 100:.1f}%")
    print(f"Macro-F1 (5 Classes)    : {macro_f1:.4f}")
    print("-------------------------------------------------------")
    print(f"{'Class Label':<22} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-------------------------------------------------------")

    for cls_name, vals in metrics.items():
        p = vals.get("precision", 0.0)
        r = vals.get("recall", 0.0)
        f1 = vals.get("f1", 0.0)
        print(f"{cls_name:<22} | {p:<9.4f} | {r:<8.4f} | {f1:<8.4f}")

    print("=======================================================")

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "pilot_benchmark_report.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[+] Full audit report saved to: {out_file}\n")

if __name__ == "__main__":
    main()
