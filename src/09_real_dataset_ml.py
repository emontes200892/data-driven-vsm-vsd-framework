from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "bpi2017": ROOT / "datasets" / "prepared" / "bpi2017_prepared.csv",
    "bpi2019": ROOT / "datasets" / "prepared" / "bpi2019_prepared.csv",
    "data_quality_issues": ROOT / "datasets" / "prepared" / "data_quality_issues_prepared.csv",
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


def add_proxy_labels(df, network):
    df = df.merge(network, on="activity", how="left")

    for c in NETWORK_FEATURES:
        df[c] = df[c].fillna(0)

    df["delay_proxy"] = (
        df["cycle_time_min"] >= df["cycle_time_min"].quantile(0.90)
    ).astype(int)

    activity_criticality = (
        network.set_index("activity")["betweenness_centrality"]
        if len(network)
        else pd.Series(dtype=float)
    )

    if len(activity_criticality) > 0:
        threshold = activity_criticality.quantile(0.90)
        critical_activities = set(activity_criticality[activity_criticality >= threshold].index)
        df["bottleneck_proxy"] = df["activity"].isin(critical_activities).astype(int)
    else:
        df["bottleneck_proxy"] = 0

    df["rework_proxy"] = df["rework_flag"].astype(int)

    return df


def run_ml(dataset_name, input_path):
    print("\n" + "=" * 80)
    print(f"Running ML for: {dataset_name}")

    out_dir = ROOT / "results" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    network_path = out_dir / "network_activity_features.csv"

    if not network_path.exists():
        raise FileNotFoundError(f"Missing network file: {network_path}")

    df = pd.read_csv(input_path)
    network = pd.read_csv(network_path)

    df["start_ts"] = pd.to_datetime(df["start_ts"], errors="coerce", utc=True)
    df["end_ts"] = pd.to_datetime(df["end_ts"], errors="coerce", utc=True)

    df = df.dropna(subset=["case_id", "activity", "start_ts", "end_ts"])
    df = df.sort_values("start_ts").reset_index(drop=True)

    df = add_proxy_labels(df, network)

    for c in BASE_FEATURES + NETWORK_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    targets = [
        "delay_proxy",
        "bottleneck_proxy",
        "rework_proxy",
    ]

    n = len(df)
    train_end = int(n * 0.70)
    test_start = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    test = df.iloc[test_start:].copy()

    rows = []

    for target in targets:
        for use_network in [False, True]:
            features = BASE_FEATURES.copy()

            if use_network:
                features += NETWORK_FEATURES

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
                metrics = evaluate(model, X_train, y_train, X_test, y_test)

                if metrics is None:
                    rows.append({
                        "dataset": dataset_name,
                        "target": target,
                        "model": model_name,
                        "network_features": use_network,
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "roc_auc": np.nan,
                        "note": "single_class_target",
                    })
                else:
                    rows.append({
                        "dataset": dataset_name,
                        "target": target,
                        "model": model_name,
                        "network_features": use_network,
                        **metrics,
                        "note": "",
                    })

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "ml_results.csv", index=False)

    print(results.round(4).to_string(index=False))


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
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(ROOT / "results" / "real_datasets_ml_results.csv", index=False)

        print("\n" + "=" * 80)
        print("Combined real-dataset ML results saved to:")
        print(ROOT / "results" / "real_datasets_ml_results.csv")


if __name__ == "__main__":
    main()
