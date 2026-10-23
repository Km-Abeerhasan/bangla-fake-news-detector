"""
tests/test_pilot_dataset.py
Validates integrity of pilot dataset entries against core verification constraints.
"""

import json
from pathlib import Path

def test_pilot_dataset_schema():
    dataset_path = Path("annotation/pilot/claims_50.json")
    assert dataset_path.exists(), "Pilot dataset file does not exist!"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 2

    required_fields = {"claim_id", "claim_text", "claim_type", "gold_verdict", "passages"}
    valid_verdicts = {"SUPPORTED", "REFUTED", "MISLEADING", "CONFLICTING", "INSUFFICIENT_EVIDENCE"}

    for item in data:
        for rf in required_fields:
            assert rf in item, f"Missing required field {rf} in {item.get('claim_id')}"
        assert item["gold_verdict"] in valid_verdicts
        assert isinstance(item["passages"], list)
        for p in item["passages"]:
            assert p["tier"] in [1, 2, 3, 4]
            assert p["stance"] in ["entailment", "neutral", "contradiction"]

if __name__ == "__main__":
    test_pilot_dataset_schema()
    print("All pilot dataset validation tests passed successfully!")
