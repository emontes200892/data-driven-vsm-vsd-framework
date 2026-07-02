from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# ==========================================================
# Load data
# ==========================================================

df = pd.read_csv(
    DATA / "synthetic_event_log.csv",
    parse_dates=["start_ts", "end_ts"]
)

df = df.sort_values("start_ts").reset_index(drop=True)

network = pd.read_csv(
    RESULTS / "network_activity_features.csv"
)

df = df.merge(
    network,
    on="activity",
    how="left"
)

# ==========================================================
# Feature sets
# ==========================================================

base_features = [
    "cycle_time_min",
    "waiting_time_min",
    "queue_before",
    "wip_before",
]

network_features = [
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
]

targets = [
    "defect_flag",
    "rework_flag",
    "bottleneck_flag",
    "delay_flag",
]

# ==========================================================
# Temporal split
# ==========================================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
test = df.iloc[val_end:]

# ==========================================================
# Helper
# ==========================================================

def evaluate_model(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    try:
        prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, prob)
    except:
        roc = np.nan

    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc,
        "pr_auc": np.nan,
    }

# ==========================================================
# Experiments
# ==========================================================

rows = []

for target in targets:

    y_train = train[target]
    y_test = test[target]

    for use_network in [False, True]:

        features = base_features.copy()

        if use_network:
            features += network_features

        X_train = train[features]
        X_test = test[features]

        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
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

        for name, model in models.items():

            metrics = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            rows.append({
                "target": target,
                "model": name,
                "network_features": use_network,
                **metrics
            })

# ==========================================================
# Save
# ==========================================================

results = pd.DataFrame(rows)

results.to_csv(
    RESULTS / "ml_results.csv",
    index=False
)

print("\nML RESULTS\n")
print(results.round(4))
