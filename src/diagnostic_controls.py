"""Audit existing event-log labels. This does not create a future-event forecast.

Usage:
  python src/diagnostic_controls.py --input data/synthetic_event_log.csv --output results/controls/synthetic
  python src/diagnostic_controls.py --input datasets/prepared/bpi2017_prepared.csv --output results/controls/bpi2017

All model metrics come from the supplied input, not from manuscript constants.
Graphs and activity-majority rules are fitted on the chronological training rows.
Original input files are never overwritten.
"""

from pathlib import Path
import argparse, importlib.util, json, hashlib, platform
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "reported_real_ml", HERE / "event_log_ml.py"
)
reported = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reported)
NET = reported.NETWORK_FEATURES
BASE = reported.BASE_FEATURES


def metrics(y, p):
    return {
        "f1": float(f1_score(y, p, zero_division=0)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "accuracy": float(accuracy_score(y, p)),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    }


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [plain(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def audit(
    path,
    out,
    n_estimators=300,
    drop_invalid=False,
    class_weight=None,
    timestamp_parser="reported",
):
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    required = {"case_id", "activity", "start_ts", "end_ts", "rework_flag"}
    missing = required - set(df)
    if missing:
        raise ValueError(
            f"Missing observed fields: {sorted(missing)}; do not fill missing labels with zero."
        )
    for c in ["start_ts", "end_ts"]:
        kwargs = {"format": "mixed"} if timestamp_parser == "mixed" else {}
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, **kwargs)
    raw_rows = len(df)
    bad = df[list(required)].isna().any(axis=1)
    non_time_missing = df[["case_id", "activity", "rework_flag"]].isna().any(axis=1)
    if non_time_missing.any():
        raise ValueError(
            "Missing case/activity/label values must be reviewed, not filled or dropped silently."
        )
    excluded_count = int(bad.sum())
    if excluded_count and not drop_invalid:
        raise ValueError(
            f"{excluded_count} rows have invalid timestamps. Use --drop-invalid-timestamps to match the reported script explicitly; exclusions will be recorded."
        )
    df = df.loc[~bad].copy()
    df["rework_flag"] = pd.to_numeric(df.rework_flag, errors="raise")
    if not set(df.rework_flag.unique()).issubset({0, 1}):
        raise ValueError("rework_flag must be binary; confirm its definition.")
    df = df.sort_values(
        "start_ts", kind="stable" if timestamp_parser == "mixed" else "quicksort"
    ).reset_index(drop=True)
    n = len(df)
    a = int(n * 0.7)
    b = int(n * 0.85)
    if min(a, b - a, n - b) < 2:
        raise ValueError("Too few rows for chronological 70/15/15 audit.")
    train, valid, test = df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
    cutoff0 = valid.start_ts.min()
    cutoff1 = test.start_ts.min()
    ntrain = reported.build_training_network_features(train)
    nfull = reported.build_training_network_features(df)
    ntrain.to_csv(out / "network_train_only.csv", index=False)
    nfull.to_csv(out / "network_full_descriptive.csv", index=False)
    merged = ntrain.merge(nfull, on="activity", suffixes=("_train", "_full"))
    changed = {
        c: int(
            (
                ~np.isclose(
                    merged[c + "_train"], merged[c + "_full"], rtol=1e-8, atol=1e-10
                )
            ).sum()
        )
        for c in NET
    }
    labelstats = (
        df.groupby("activity")
        .rework_flag.agg(events="size", positives="sum", label_values="nunique")
        .reset_index()
    )
    labelstats["positive_rate"] = labelstats.positives / labelstats.events
    labelstats.to_csv(out / "rework_by_activity.csv", index=False)
    signature = (
        ntrain.groupby(NET, dropna=False)
        .activity.agg(list)
        .reset_index(name="activities")
    )
    signature["activity_count"] = signature.activities.map(len)
    signature.to_csv(out / "network_signature_collisions.csv", index=False)

    def window(d):
        return {
            "events": len(d),
            "cases": d.case_id.nunique(),
            "start_min": d.start_ts.min(),
            "start_max": d.start_ts.max(),
            "rework_counts": d.rework_flag.value_counts().to_dict(),
        }

    evidence = {
        "input": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_rows": raw_rows,
        "excluded_timestamp_parse_rows": excluded_count,
        "timestamp_parser": timestamp_parser,
        "ordering": "stable start_ts"
        if timestamp_parser == "mixed"
        else "reported quicksort start_ts",
        "class_weight": class_weight,
        "scope": "Retrospective event-log label classification, not future rework prediction.",
        "split": "chronological rows by start_ts: 70/15/15; validation reserved, no model selection performed",
        "windows": {
            "train": window(train),
            "validation": window(valid),
            "test": window(test),
        },
        "case_overlap": {
            "train_validation": len(set(train.case_id) & set(valid.case_id)),
            "train_test": len(set(train.case_id) & set(test.case_id)),
            "validation_test": len(set(valid.case_id) & set(test.case_id)),
        },
        "training_end_ts_at_or_after_validation_start": int(
            (train.end_ts >= cutoff0).sum()
        ),
        "pre_test_rows_end_ts_at_or_after_test_start": int(
            (df.iloc[:b].end_ts >= cutoff1).sum()
        ),
        "missing_operational_fields": [c for c in BASE if c not in df],
        "constant_operational_fields": [
            c for c in BASE if c in df and df[c].nunique(dropna=False) <= 1
        ],
        "training_graph_activities": len(ntrain),
        "full_graph_activities": len(nfull),
        "changed_training_node_features_using_full_log": changed,
        "rework_deterministic_within_activity_on_full_input": bool(
            (labelstats.label_values <= 1).all()
        ),
        "activities_with_unique_training_network_signature": int(
            signature.loc[signature.activity_count.eq(1), "activity_count"].sum()
        ),
        "all_network_signature_activity_counts": signature.activity_count.tolist(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
        },
        "cautions": [
            "Current-event durations and end timestamps may not be available at event start.",
            "A fixed train-only activity signature can reconstruct activity identity and an activity-defined label.",
            "An audit on synthetic input does not reproduce or validate BPI2017 metrics.",
        ],
    }
    if "delay_flag" in df and "waiting_time_min" in df:
        evidence["delay_flag_equals_waiting_time_over_15"] = bool(
            df.delay_flag.eq(df.waiting_time_min.gt(15).astype(int)).all()
        )
    if "Rework" in set(df.activity):
        evidence["rework_flag_equals_named_Rework_activity"] = bool(
            df.rework_flag.eq(df.activity.eq("Rework").astype(int)).all()
        )
    results = []
    predictions = test[
        ["case_id", "activity", "start_ts", "end_ts", "rework_flag"]
    ].copy()
    y = train.rework_flag.astype(int)
    yt = test.rework_flag.astype(int)
    if y.nunique() < 2 or yt.nunique() < 2:
        evidence["status"] = "single_class_target; model audit not evaluable"
    else:
        majority = train.groupby("activity").rework_flag.mean().ge(0.5).astype(int)
        p = test.activity.map(majority).fillna(int(y.mean() >= 0.5)).astype(int)
        results.append({"experiment": "activity_majority_train_only", **metrics(yt, p)})
        predictions["pred_activity_majority"] = p.to_numpy()
        available = [
            c
            for c in BASE
            if c in df and pd.to_numeric(df[c], errors="coerce").notna().all()
        ]
        for condition in [
            "operational_only",
            "network_only_train",
            "operational_plus_network_train",
            "activity_one_hot",
        ]:
            if condition == "activity_one_hot":
                xt = pd.get_dummies(train.activity, dtype=float)
                xe = pd.get_dummies(test.activity, dtype=float).reindex(
                    columns=xt.columns, fill_value=0
                )
                features = ["one-hot activity (training vocabulary)"]
            else:
                tr = reported.merge_training_network_features(train, ntrain)
                te = reported.merge_training_network_features(test, ntrain)
                features = (available if condition != "network_only_train" else []) + (
                    NET if condition != "operational_only" else []
                )
                if not features:
                    continue
                xt = tr[features].astype(float)
                xe = te[features].astype(float)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=12,
                random_state=42,
                n_jobs=2,
                class_weight=class_weight,
            )
            model.fit(xt, y)
            p = model.predict(xe)
            results.append(
                {
                    "experiment": condition,
                    "features": "|".join(features),
                    **metrics(yt, p),
                }
            )
            predictions["pred_" + condition] = p
            predictions["prob_" + condition] = model.predict_proba(xe)[:, 1]
        evidence["status"] = "executed; see metrics and saved predictions"
    pd.DataFrame(results).to_csv(out / "rework_audit_metrics.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    (out / "audit.json").write_text(
        json.dumps(plain(evidence), indent=2, allow_nan=False), encoding="utf8"
    )
    print(
        json.dumps(
            {"status": evidence["status"], "output": str(out), "metrics": results},
            indent=2,
        )
    )
    return evidence, results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--drop-invalid-timestamps", action="store_true")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p.add_argument(
        "--timestamp-parser", choices=["reported", "mixed"], default="reported"
    )
    a = p.parse_args()
    audit(
        a.input,
        a.output,
        a.trees,
        a.drop_invalid_timestamps,
        None if a.class_weight == "none" else a.class_weight,
        a.timestamp_parser,
    )
