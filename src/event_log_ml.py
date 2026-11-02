from pathlib import Path
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "bpi2017": ROOT / "datasets" / "prepared" / "bpi2017_prepared.csv",
    "bpi2019": ROOT / "datasets" / "prepared" / "bpi2019_prepared.csv",
    "data_quality_issues": ROOT
    / "datasets"
    / "prepared"
    / "data_quality_issues_prepared.csv",
}

BASE_FEATURES = [
    "cycle_time_min",
    "waiting_time_min",
    "queue_before",
    "wip_before",
]

NETWORK_FEATURES = [
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
]

# Direct target-defining variables are excluded from the corresponding task.
TARGET_EXCLUSIONS = {
    "delay_proxy": {"waiting_time_min"},
    "bottleneck_proxy": {"cycle_time_min", "waiting_time_min"},
    "rework_proxy": set(),
}


def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def evaluate(model, X_train, y_train, X_test, y_test):
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    try:
        prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        prob = np.zeros(len(y_test))

    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": safe_auc(y_test, prob),
    }


def build_training_network_features(train_df):
    """
    Build the activity-transition network ONLY from the chronological
    training partition. No validation/test event is used here.
    """
    G = nx.DiGraph()

    # Add every activity observed during training, including isolated nodes.
    G.add_nodes_from(train_df["activity"].dropna().unique().tolist())

    for _, case_df in train_df.groupby("case_id", sort=False):
        activities = (
            case_df.sort_values("start_ts", kind="stable")["activity"]
            .dropna()
            .astype(str)
            .tolist()
        )

        for source, target in zip(activities[:-1], activities[1:]):
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)

    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["activity"] + NETWORK_FEATURES)

    # Use an undirected projection for the same family of structural indicators
    # employed by the descriptive CNA layer.
    H = G.to_undirected()

    degree = nx.degree_centrality(H)
    betweenness = nx.betweenness_centrality(H, weight=None)
    closeness = nx.closeness_centrality(H)

    try:
        eigenvector = nx.eigenvector_centrality(H, max_iter=1000)
    except Exception:
        eigenvector = {node: 0.0 for node in H.nodes()}

    return pd.DataFrame(
        {
            "activity": list(H.nodes()),
            "degree_centrality": [degree.get(n, 0.0) for n in H.nodes()],
            "betweenness_centrality": [betweenness.get(n, 0.0) for n in H.nodes()],
            "closeness_centrality": [closeness.get(n, 0.0) for n in H.nodes()],
            "eigenvector_centrality": [eigenvector.get(n, 0.0) for n in H.nodes()],
        }
    )


def merge_training_network_features(df, training_network):
    """
    Attach training-only network features.

    Activities unseen during training receive zero centrality values rather
    than being characterized using future observations.
    """
    out = df.copy().merge(training_network, on="activity", how="left")

    for c in NETWORK_FEATURES:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def add_proxy_labels(train, test, training_network):
    """Build proxy labels using training data only.

    Delay is excessive waiting (training Q90). Bottlenecks are activities whose
    training-only mean cycle time or mean waiting time is in the upper decile.
    The variables defining each target are excluded from its predictor set.
    """
    train = train.copy()
    test = test.copy()

    delay_threshold = train["waiting_time_min"].quantile(0.90)
    train["delay_proxy"] = (train["waiting_time_min"] >= delay_threshold).astype(int)
    test["delay_proxy"] = (test["waiting_time_min"] >= delay_threshold).astype(int)

    activity_stats = train.groupby("activity", as_index=False).agg(
        mean_cycle_time_min=("cycle_time_min", "mean"),
        mean_waiting_time_min=("waiting_time_min", "mean"),
    )
    if not activity_stats.empty:
        cycle_threshold = activity_stats["mean_cycle_time_min"].quantile(0.90)
        waiting_threshold = activity_stats["mean_waiting_time_min"].quantile(0.90)
        critical_activities = set(
            activity_stats.loc[
                (activity_stats["mean_cycle_time_min"] >= cycle_threshold)
                | (activity_stats["mean_waiting_time_min"] >= waiting_threshold),
                "activity",
            ]
        )
    else:
        cycle_threshold = np.nan
        waiting_threshold = np.nan
        critical_activities = set()

    train["bottleneck_proxy"] = train["activity"].isin(critical_activities).astype(int)
    test["bottleneck_proxy"] = test["activity"].isin(critical_activities).astype(int)
    train["rework_proxy"] = (
        pd.to_numeric(train["rework_flag"], errors="coerce").fillna(0).astype(int)
    )
    test["rework_proxy"] = (
        pd.to_numeric(test["rework_flag"], errors="coerce").fillna(0).astype(int)
    )
    return (
        train,
        test,
        delay_threshold,
        cycle_threshold,
        waiting_threshold,
        critical_activities,
    )


def get_features(target, use_network):
    features = BASE_FEATURES.copy()

    if use_network:
        features += NETWORK_FEATURES

    exclusions = TARGET_EXCLUSIONS.get(target, set())
    return [f for f in features if f not in exclusions]


def run_ml(dataset_name, input_path):
    print("\n" + "=" * 80)
    print(f"Running temporally split diagnostic classification for: {dataset_name}")

    out_dir = ROOT / "results" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df["start_ts"] = pd.to_datetime(
        df["start_ts"], errors="coerce", utc=True, format="mixed"
    )
    df["end_ts"] = pd.to_datetime(
        df["end_ts"], errors="coerce", utc=True, format="mixed"
    )

    df = df.dropna(subset=["case_id", "activity", "start_ts", "end_ts"])
    df = df.sort_values("start_ts", kind="stable").reset_index(drop=True)

    for c in BASE_FEATURES:
        if c not in df.columns:
            raise ValueError(
                f"Missing operational feature {c}; document and prepare it explicitly."
            )
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "rework_flag" not in df.columns:
        raise ValueError(
            "Missing rework_flag: an unobserved target must not be filled with zero."
        )
    if not set(
        pd.to_numeric(df["rework_flag"], errors="raise").dropna().unique()
    ).issubset({0, 1}):
        raise ValueError("rework_flag must be binary.")

    # ------------------------------------------------------------------
    # Chronological 70/15/15 split BEFORE network construction/thresholds
    # ------------------------------------------------------------------
    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train_raw = df.iloc[:train_end].copy()
    validation_raw = df.iloc[train_end:validation_end].copy()
    test_raw = df.iloc[validation_end:].copy()

    print(
        f"Temporal split: train={len(train_raw):,}, "
        f"validation={len(validation_raw):,}, "
        f"test={len(test_raw):,}"
    )

    import json, hashlib

    audit_metadata = {
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "timestamp_parser": "pandas mixed-format UTC; stable temporal ordering",
        "scope": "current-event diagnostic classification, not a future-event forecast",
        "train_events": len(train_raw),
        "validation_events": len(validation_raw),
        "test_events": len(test_raw),
        "train_test_case_overlap": len(set(train_raw.case_id) & set(test_raw.case_id)),
        "validation_used_for_selection": False,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(audit_metadata, indent=2))

    # -------------------------------------------------------------
    # CRITICAL: structural features are reconstructed from TRAIN ONLY
    # -------------------------------------------------------------
    training_network = build_training_network_features(train_raw)

    training_network_path = out_dir / "network_activity_features_train_only.csv"
    training_network.to_csv(training_network_path, index=False)

    print(f"Training-only network saved to: {training_network_path}")

    train = merge_training_network_features(train_raw, training_network)
    validation = merge_training_network_features(validation_raw, training_network)
    test = merge_training_network_features(test_raw, training_network)

    # Validation remains untouched/reserved. It is deliberately not used
    # to estimate proxy thresholds or structural features in this experiment.
    _ = validation

    (
        train,
        test,
        delay_threshold,
        bottleneck_cycle_threshold,
        bottleneck_waiting_threshold,
        critical_activities,
    ) = add_proxy_labels(train, test, training_network)

    print(f"Training-only delay threshold (Q90 waiting time): {delay_threshold:.6f}")
    print(
        f"Training-only bottleneck cycle threshold (Q90 activity mean): "
        f"{bottleneck_cycle_threshold:.6f}"
    )
    print(
        f"Training-only bottleneck waiting threshold (Q90 activity mean): "
        f"{bottleneck_waiting_threshold:.6f}"
    )
    print(f"Training-only critical activities: {len(critical_activities)}")

    targets = [
        "delay_proxy",
        "bottleneck_proxy",
        "rework_proxy",
    ]

    rows = []

    for target in targets:
        for use_network in [False, True]:
            features = get_features(target, use_network)

            print(
                f"{dataset_name} | {target} | "
                f"network={use_network} | features={features}"
            )

            X_train = train[features]
            y_train = train[target]
            X_test = test[features]
            y_test = test[target]

            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                ),
            }

            for model_name, model in models.items():
                metrics = evaluate(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )

                common = {
                    "dataset": dataset_name,
                    "target": target,
                    "model": model_name,
                    "network_features": use_network,
                    "features": "|".join(features),
                    "train_events": len(train),
                    "test_events": len(test),
                    "delay_threshold_train_only": delay_threshold,
                    "bottleneck_cycle_threshold_train_only": bottleneck_cycle_threshold,
                    "bottleneck_waiting_threshold_train_only": bottleneck_waiting_threshold,
                }

                if metrics is None:
                    row = {
                        **common,
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "roc_auc": np.nan,
                        "note": "single_class_target",
                    }
                else:
                    row = {
                        **common,
                        **metrics,
                        "note": "",
                    }

                rows.append(row)

    results = pd.DataFrame(rows)

    result_path = out_dir / "ml_results.csv"
    results.to_csv(result_path, index=False)

    print("\nResults:")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: {result_path}")


def main():
    all_results = []

    for dataset_name, input_path in DATASETS.items():
        if not input_path.exists():
            print(f"Skipping {dataset_name}. Missing file: {input_path}")
            continue

        run_ml(dataset_name, input_path)

        out_file = ROOT / "results" / dataset_name / "ml_results.csv"

        if out_file.exists():
            all_results.append(pd.read_csv(out_file))

    if all_results:
        combined = pd.concat(
            all_results,
            ignore_index=True,
        )

        combined_path = ROOT / "results" / "real_datasets_ml_results.csv"

        combined.to_csv(
            combined_path,
            index=False,
        )

        print("\n" + "=" * 80)
        print("Combined diagnostic results saved to:")
        print(combined_path)


if __name__ == "__main__":
    main()
