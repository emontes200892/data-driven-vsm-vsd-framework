#!/usr/bin/env python3
"""Reproduce the event-log diagnostics and Monte Carlo scenario experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_inputs() -> None:
    """Fail before computation when an input is missing or has changed."""
    manifest = json.loads((ROOT / "datasets/input_manifest.json").read_text())
    for item in manifest:
        path = ROOT / item["file"]
        if not path.is_file():
            raise ValueError(f"Missing input: {item['file']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
            raise ValueError(f"Input does not match the manifest: {item['file']}")
    print(f"Verified {len(manifest)} input files against SHA-256 manifest.", flush=True)


def commands_for(stage: str) -> list[list[str]]:
    scripts = {
        "synthetic": ["vsm_kpis.py", "network_features.py", "synthetic_ml.py"],
        "scenarios": ["monte_carlo_scenarios.py"],
        "real": ["event_log_analysis.py", "event_log_ml.py"],
        "statistics": ["scenario_statistics.py"],
        "tables": ["generate_tables.py"],
        "figures": ["generate_figures.py"],
    }
    if stage == "verify":
        return []
    if stage == "controls":
        inputs = [("synthetic", "data/synthetic_event_log.csv")]
        inputs += [
            (name, f"datasets/prepared/{name}_prepared.csv")
            for name in ("bpi2017", "bpi2019", "data_quality_issues")
        ]
        commands = [
            [
                "src/diagnostic_controls.py",
                "--input",
                path,
                "--output",
                f"results/controls/{name}",
                "--timestamp-parser",
                "mixed",
                *([] if name == "synthetic" else ["--class-weight", "balanced"]),
            ]
            for name, path in inputs
        ]
        return commands + [["src/summarize_controls.py"]]
    if stage == "all":
        return [
            command
            for name in (
                "synthetic",
                "scenarios",
                "real",
                "controls",
                "statistics",
                "tables",
                "figures",
            )
            for command in commands_for(name)
        ]
    return [[f"src/{name}"] for name in scripts[stage]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=(
            "verify",
            "synthetic",
            "scenarios",
            "real",
            "controls",
            "statistics",
            "tables",
            "figures",
            "all",
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check inputs and show commands."
    )
    args = parser.parse_args()
    verify_inputs()
    commands = commands_for(args.stage)
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] python {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    if not args.dry_run:
        print(f"Completed stage: {args.stage}", flush=True)


if __name__ == "__main__":
    main()
