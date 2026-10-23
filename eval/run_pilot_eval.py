"""
eval/run_pilot_eval.py
Executes the evaluation benchmark harness on the pilot dataset
and prints a formatted academic report table.
"""

import json
from pathlib import Path
from eval.benchmark import evaluate_dataset

def main():
    dataset_path = "annotation/pilot/claims_50.json"
    print(f"[*] Running BiN Pipeline Benchmark on: {dataset_path} ...")

    report = evaluate_dataset(dataset_path=dataset_path, tau_abstain=0.60)

    # রেজাল্ট JSON ফাইলে সংরক্ষণ
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pilot_benchmark_report.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "="*65)
    print("      BiN (Audit-Then-Believe) — PILOT BENCHMARK REPORT      ")
    print("="*65)
    print(f"Total Evaluated Claims:  {report['total_claims']}")
    print(f"Answered Decisions:      {report['answered_claims']}")
    print(f"Abstained Decisions:     {report['abstained_claims']}")
    print(f"Selective Coverage Rate: {report['coverage_rate'] * 100:.1f}%\n")

    print(f"Macro-F1 (5 Decoupled Classes): {report['metrics']['macro_f1']:.4f}\n")
    print(f"{'Class Label':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)

    for label, scores in report['metrics']['per_class'].items():
        print(f"{label:<25} | {scores['precision']:<10.4f} | {scores['recall']:<10.4f} | {scores['f1']:<10.4f}")

    print("="*65)
    print(f"[+] Full audit report saved to: {output_file}\n")

if __name__ == "__main__":
    main()
