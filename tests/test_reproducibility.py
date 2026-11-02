"""Check input integrity, temporal ordering, and numerical scenario consistency."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproducibilityTests(unittest.TestCase):
    def test_manifest_matches_included_inputs(self):
        module = load_module("reproduce", ROOT / "reproduce.py")
        module.verify_inputs()

    def test_mixed_timestamps_are_kept_and_sorted(self):
        module = load_module("event_log_analysis", ROOT / "src/event_log_analysis.py")
        # A mixed-format reader must retain valid timestamps and preserve ties.
        source = pd.DataFrame(
            {
                "case_id": ["same_case"] * 3,
                "event_id": [3, 1, 2],
                "resource_id": ["resource"] * 3,
                "activity": ["C", "A", "B"],
                "start_ts": [
                    "2020-01-02T00:00:00Z",
                    "2020-01-01 00:00:00+00:00",
                    "2020-01-01T00:00:00.000Z",
                ],
                "end_ts": [
                    "2020-01-02T00:01:00Z",
                    "2020-01-01 00:01:00+00:00",
                    "2020-01-01T00:01:00.000Z",
                ],
            }
        )
        for column in (
            "cycle_time_min",
            "waiting_time_min",
            "queue_before",
            "wip_before",
            "defect_flag",
            "rework_flag",
        ):
            source[column] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            source.to_csv(path, index=False)
            result = module.load_dataset(path)
        self.assertEqual(result.event_id.tolist(), [1, 2, 3])

    def test_seed_blocks_and_headline_scenario_changes(self):
        replications = pd.read_csv(ROOT / "results/scenario_replications.csv")
        self.assertEqual(len(replications), 180)
        self.assertEqual(
            replications.groupby("scenario").replication.nunique().tolist(), [30] * 6
        )
        self.assertTrue(
            replications.groupby(["scenario", "replication"]).size().eq(1).all()
        )
        summary = pd.read_csv(ROOT / "results/scenario_results.csv").set_index(
            "scenario"
        )
        baseline, final = summary.loc["Baseline"], summary.loc["Full Framework"]
        reductions = [
            (1 - final[column] / baseline[column]) * 100
            for column in ("Lead_Time", "WIP", "Defect_Rate", "Rework_Rate")
        ]
        values = reductions + [(final.Throughput / baseline.Throughput - 1) * 100]
        np.testing.assert_allclose(
            values, [32.40, 45.09, 27.63, 45.52, 71.73], atol=0.005
        )
        for column, raw_column in [
            ("Lead_Time", "lead_time_mean"),
            ("WIP", "wip_mean"),
            ("Throughput", "throughput_per_hour"),
        ]:
            np.testing.assert_allclose(
                summary[column].sort_index(),
                replications.groupby("scenario")[raw_column].mean().sort_index(),
                rtol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
