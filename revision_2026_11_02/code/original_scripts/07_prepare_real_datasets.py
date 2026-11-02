from pathlib import Path
import zipfile
import pandas as pd
import pm4py


DATASETS_DIR = Path("datasets")
EXTRACTED_DIR = DATASETS_DIR / "extracted"
PREPARED_DIR = DATASETS_DIR / "prepared"

PREPARED_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)


TARGETS = {
    "clean_manufacturing": {
        "zip_path": DATASETS_DIR / "Cleaned Event Log.zip",
        "xes_in_zip": "Cleaned Event Log/00001dab-09fc-4439-b784-4c9a4a3dedcb.xes",
        "output": PREPARED_DIR / "clean_manufacturing_prepared.csv",
    },
    "data_quality_issues": {
        "zip_path": DATASETS_DIR / "Data Quality Issues Event Log.zip",
        "xes_in_zip": "Data Quality Issues Event Log/MainProcess.xes",
        "output": PREPARED_DIR / "data_quality_issues_prepared.csv",
    },
}


def extract_target_xes(zip_path: Path, xes_in_zip: str, dataset_name: str) -> Path:
    out_dir = EXTRACTED_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / Path(xes_in_zip).name

    print(f"Extracting/overwriting: {xes_in_zip}")

    with zipfile.ZipFile(zip_path) as z:
        with z.open(xes_in_zip) as src, open(out_path, "wb") as dst:
            dst.write(src.read())

    return out_path


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def standardize_event_log(xes_path: Path, max_events: int = 250_000) -> pd.DataFrame:
    print(f"Reading: {xes_path}")

    log = pm4py.read_xes(str(xes_path))
    df = pm4py.convert_to_dataframe(log)

    print("Raw shape:", df.shape)
    print("Raw columns:", list(df.columns))

    if len(df) == 0:
        raise ValueError(f"Empty dataframe after reading {xes_path.name}")

    if len(df) > max_events:
        df = df.head(max_events).copy()

    case_col = find_col(df, [
        "case:concept:name",
        "case_id",
        "case:Case ID",
        "case:case_id",
        "Case ID",
    ])

    activity_col = find_col(df, [
        "concept:name",
        "activity",
        "Activity",
        "event",
        "Event",
    ])

    resource_col = find_col(df, [
        "org:resource",
        "resource",
        "Resource",
        "user",
        "User",
    ])

    timestamp_col = find_col(df, [
        "time:timestamp",
        "timestamp",
        "Timestamp",
        "start_timestamp",
        "Start Timestamp",
        "complete_timestamp",
        "Complete Timestamp",
    ])

    if activity_col is None:
        raise ValueError(f"No activity column found in {xes_path.name}")

    if timestamp_col is None:
        raise ValueError(f"No timestamp column found in {xes_path.name}")

    if case_col is None:
        print("WARNING: no case column found. Creating synthetic case_id = single_trace_1")
        df["case_id_synthetic"] = "single_trace_1"
        case_col = "case_id_synthetic"

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[case_col, activity_col, timestamp_col]).copy()

    df = df.sort_values([case_col, timestamp_col]).reset_index(drop=True)

    df["next_ts"] = df.groupby(case_col)[timestamp_col].shift(-1)
    df["prev_ts"] = df.groupby(case_col)[timestamp_col].shift(1)

    df["cycle_time_min"] = (
        (df["next_ts"] - df[timestamp_col]).dt.total_seconds() / 60
    )

    df["waiting_time_min"] = (
        (df[timestamp_col] - df["prev_ts"]).dt.total_seconds() / 60
    )

    df["cycle_time_min"] = df["cycle_time_min"].fillna(0).clip(lower=0)
    df["waiting_time_min"] = df["waiting_time_min"].fillna(0).clip(lower=0)

    out = pd.DataFrame()

    out["case_id"] = df[case_col].astype(str)
    out["event_id"] = range(1, len(df) + 1)
    out["activity"] = df[activity_col].astype(str)

    if resource_col is not None:
        out["resource_id"] = df[resource_col].fillna("unknown").astype(str)
    else:
        out["resource_id"] = "unknown"

    out["machine_id"] = "unknown"
    out["product_family"] = "unknown"

    out["start_ts"] = df[timestamp_col]
    out["end_ts"] = df["next_ts"].fillna(df[timestamp_col])

    out["cycle_time_min"] = df["cycle_time_min"]
    out["waiting_time_min"] = df["waiting_time_min"]

    out["queue_before"] = 0
    out["wip_before"] = 0

    out["shift"] = out["start_ts"].dt.hour.apply(
        lambda h: "morning" if 6 <= h < 14 else "evening" if 14 <= h < 22 else "night"
    )

    out["defect_flag"] = 0

    out["rework_flag"] = (
        out.groupby("case_id")["activity"]
        .transform(lambda x: x.duplicated().astype(int))
        .astype(int)
    )

    return out


def main():
    for dataset_name, cfg in TARGETS.items():
        print("\n" + "=" * 80)
        print(dataset_name)

        xes_path = extract_target_xes(
            zip_path=cfg["zip_path"],
            xes_in_zip=cfg["xes_in_zip"],
            dataset_name=dataset_name,
        )

        prepared = standardize_event_log(xes_path)
        prepared.to_csv(cfg["output"], index=False)

        print(f"Saved: {cfg['output']}")
        print(f"Shape: {prepared.shape}")
        print(f"Activities: {prepared['activity'].nunique()}")
        print(f"Cases: {prepared['case_id'].nunique()}")


if __name__ == "__main__":
    main()
