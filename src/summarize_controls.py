"""Collect diagnostic-control scores and case-overlap counts from executed outputs."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT / "results/controls"


def main():
    experiments = {
        "operational_RF": "operational_only",
        "operational_plus_network_RF": "operational_plus_network_train",
        "activity_majority": "activity_majority_train_only",
        "activity_one_hot_RF": "activity_one_hot",
    }
    rows = []
    for dataset in ("synthetic", "bpi2017", "bpi2019", "data_quality_issues"):
        directory = CONTROL_ROOT / dataset
        evidence = json.loads((directory / "audit.json").read_text())
        scores = pd.read_csv(directory / "rework_audit_metrics.csv").set_index(
            "experiment"
        )
        test = evidence["windows"]["test"]
        row = {
            "dataset": dataset,
            "test_events": test["events"],
            "test_positives": test["rework_counts"].get("1", 0),
        }
        row.update(
            {
                name: float(scores.loc[experiment, "f1"])
                for name, experiment in experiments.items()
            }
        )
        row.update(evidence["case_overlap"])
        rows.append(row)
    destination = CONTROL_ROOT / "activity_baseline_comparison.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
