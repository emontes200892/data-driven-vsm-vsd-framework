from pathlib import Path
import pandas as pd
import networkx as nx


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "bpi2017": ROOT / "datasets" / "prepared" / "bpi2017_prepared.csv",
    "bpi2019": ROOT / "datasets" / "prepared" / "bpi2019_prepared.csv",
    "data_quality_issues": ROOT / "datasets" / "prepared" / "data_quality_issues_prepared.csv",
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["start_ts"] = pd.to_datetime(df["start_ts"], errors="coerce", utc=True)
    df["end_ts"] = pd.to_datetime(df["end_ts"], errors="coerce", utc=True)

    required = [
        "case_id",
        "event_id",
        "activity",
        "resource_id",
        "start_ts",
        "end_ts",
        "cycle_time_min",
        "waiting_time_min",
        "queue_before",
        "wip_before",
        "defect_flag",
        "rework_flag",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    df = df.dropna(subset=["case_id", "activity", "start_ts", "end_ts"])
    df = df.sort_values(["case_id", "start_ts"]).reset_index(drop=True)

    return df


def generate_dataset_summary(df: pd.DataFrame, out_dir: Path) -> None:
    obs_hours = max(
        (df["end_ts"].max() - df["start_ts"].min()).total_seconds() / 3600,
        1e-9,
    )

    case_summary = (
        df.groupby("case_id")
        .agg(
            first_start=("start_ts", "min"),
            last_end=("end_ts", "max"),
            events=("event_id", "count"),
            total_processing_min=("cycle_time_min", "sum"),
            total_waiting_min=("waiting_time_min", "sum"),
            rework_events=("rework_flag", "sum"),
            defects=("defect_flag", "sum"),
        )
        .reset_index()
    )

    case_summary["lead_time_min"] = (
        case_summary["last_end"] - case_summary["first_start"]
    ).dt.total_seconds() / 60

    case_summary["value_added_ratio"] = (
        case_summary["total_processing_min"]
        / case_summary["lead_time_min"].replace(0, pd.NA)
    )

    case_summary.to_csv(out_dir / "case_summary.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "events": len(df),
                "cases": df["case_id"].nunique(),
                "activities": df["activity"].nunique(),
                "resources": df["resource_id"].nunique(),
                "observation_hours": obs_hours,
                "mean_case_lead_time_min": case_summary["lead_time_min"].mean(),
                "median_case_lead_time_min": case_summary["lead_time_min"].median(),
                "mean_processing_time_min": case_summary["total_processing_min"].mean(),
                "mean_waiting_time_min": case_summary["total_waiting_min"].mean(),
                "mean_value_added_ratio": case_summary["value_added_ratio"].mean(),
                "throughput_cases_per_hour": df["case_id"].nunique() / obs_hours,
                "throughput_events_per_hour": len(df) / obs_hours,
                "defect_rate": df["defect_flag"].mean(),
                "rework_rate": df["rework_flag"].mean(),
            }
        ]
    )

    summary.to_csv(out_dir / "dataset_summary.csv", index=False)


def generate_vsm_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    obs_hours = max(
        (df["end_ts"].max() - df["start_ts"].min()).total_seconds() / 3600,
        1e-9,
    )

    stage = (
        df.groupby("activity")
        .agg(
            events=("event_id", "count"),
            cases=("case_id", "nunique"),
            CT_min=("cycle_time_min", "mean"),
            WT_min=("waiting_time_min", "mean"),
            WIP=("wip_before", "mean"),
            Queue=("queue_before", "mean"),
            Defect_Rate=("defect_flag", "mean"),
            Rework_Rate=("rework_flag", "mean"),
        )
        .reset_index()
    )

    stage["Throughput_events_per_hour"] = stage["events"] / obs_hours
    stage["Total_Time_min"] = stage["CT_min"] + stage["WT_min"]

    stage = stage.sort_values("events", ascending=False)
    stage.to_csv(out_dir / "vsm_stage_kpis.csv", index=False)


def generate_network_features(df: pd.DataFrame, out_dir: Path) -> None:
    graph = nx.DiGraph()

    for _, g in df.groupby("case_id"):
        acts = g.sort_values("start_ts")["activity"].tolist()

        for a, b in zip(acts[:-1], acts[1:]):
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, weight=1)

    if graph.number_of_nodes() == 0:
        pd.DataFrame().to_csv(out_dir / "network_activity_features.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "dfg_edges.csv", index=False)
        return

    undirected = graph.to_undirected()

    degree = nx.degree_centrality(undirected)
    betweenness = nx.betweenness_centrality(undirected, weight=None)
    closeness = nx.closeness_centrality(undirected)

    try:
        eigenvector = nx.eigenvector_centrality(undirected, max_iter=1000)
    except Exception:
        eigenvector = {node: 0 for node in graph.nodes()}

    features = pd.DataFrame(
        {
            "activity": list(graph.nodes()),
            "degree_centrality": [degree.get(n, 0) for n in graph.nodes()],
            "betweenness_centrality": [betweenness.get(n, 0) for n in graph.nodes()],
            "closeness_centrality": [closeness.get(n, 0) for n in graph.nodes()],
            "eigenvector_centrality": [eigenvector.get(n, 0) for n in graph.nodes()],
        }
    )

    features = features.sort_values("betweenness_centrality", ascending=False)
    features.to_csv(out_dir / "network_activity_features.csv", index=False)

    edges = pd.DataFrame(
        [
            {
                "source": u,
                "target": v,
                "frequency": data["weight"],
            }
            for u, v, data in graph.edges(data=True)
        ]
    )

    edges = edges.sort_values("frequency", ascending=False)
    edges.to_csv(out_dir / "dfg_edges.csv", index=False)

    global_metrics = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "weakly_connected_components": nx.number_weakly_connected_components(graph),
    }

    pd.DataFrame([global_metrics]).to_csv(out_dir / "network_global_metrics.csv", index=False)


def run_dataset(dataset_name: str, input_path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"Running dataset: {dataset_name}")
    print(f"Input: {input_path}")

    out_dir = ROOT / "results" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)

    print(f"Events: {len(df):,}")
    print(f"Cases: {df['case_id'].nunique():,}")
    print(f"Activities: {df['activity'].nunique():,}")

    generate_dataset_summary(df, out_dir)
    generate_vsm_metrics(df, out_dir)
    generate_network_features(df, out_dir)

    print(f"Saved results to: {out_dir}")


def main():
    for dataset_name, input_path in DATASETS.items():
        if not input_path.exists():
            print(f"Skipping {dataset_name}. File not found: {input_path}")
            continue

        run_dataset(dataset_name, input_path)


if __name__ == "__main__":
    main()
