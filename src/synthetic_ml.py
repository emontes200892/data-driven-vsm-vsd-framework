"""Synthetic current-event diagnostic comparisons with train-only graph features.
Direct variables defining delay/bottleneck flags are excluded. This script does
not create a future-event forecasting task.
"""

from pathlib import Path
import importlib.util, json
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
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "real_ml", Path(__file__).with_name("event_log_ml.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pd.read_csv(ROOT / "data/synthetic_event_log.csv")
d["start_ts"] = pd.to_datetime(d.start_ts, utc=True, format="mixed")
d["end_ts"] = pd.to_datetime(d.end_ts, utc=True, format="mixed")
d = d.sort_values("start_ts", kind="stable").reset_index(drop=True)
a = int(0.7 * len(d))
b = int(0.85 * len(d))
train = d.iloc[:a].copy()
test = d.iloc[b:].copy()
network = m.build_training_network_features(train)
network.to_csv(OUT / "network_activity_features_train_only.csv", index=False)
train = m.merge_training_network_features(train, network)
test = m.merge_training_network_features(test, network)
exclusions = {
    "defect_flag": set(),
    "rework_flag": set(),
    "delay_flag": {"waiting_time_min"},
    "bottleneck_flag": {"cycle_time_min", "waiting_time_min"},
}
rows = []
preds = []
for target, exclude in exclusions.items():
    for use_network in [False, True]:
        features = [
            x
            for x in m.BASE_FEATURES + (m.NETWORK_FEATURES if use_network else [])
            if x not in exclude
        ]
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300, max_depth=12, random_state=42, n_jobs=2
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                n_jobs=2,
            ),
        }
        for name, model in models.items():
            if train[target].nunique() < 2 or test[target].nunique() < 2:
                rows.append(
                    {
                        "target": target,
                        "model": name,
                        "network_features": use_network,
                        "features": "|".join(features),
                        "f1": np.nan,
                        "note": "single_class_target",
                    }
                )
                continue
            model.fit(train[features], train[target])
            p = model.predict(test[features])
            pr = model.predict_proba(test[features])[:, 1]
            y = test[target]
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "network_features": use_network,
                    "features": "|".join(features),
                    "accuracy": accuracy_score(y, p),
                    "precision": precision_score(y, p, zero_division=0),
                    "recall": recall_score(y, p, zero_division=0),
                    "f1": f1_score(y, p, zero_division=0),
                    "roc_auc": roc_auc_score(y, pr),
                    "note": "retrospective diagnostic task",
                }
            )
            q = test[["case_id", "activity", "start_ts"]].copy()
            q["target"] = target
            q["model"] = name
            q["network_features"] = use_network
            q["y_true"] = y.to_numpy()
            q["prediction"] = p
            q["probability"] = pr
            preds.append(q)
pd.DataFrame(rows).to_csv(OUT / "ml_results.csv", index=False)
pd.concat(preds, ignore_index=True).to_csv(
    OUT / "synthetic_diagnostic_predictions.csv", index=False
)
(OUT / "synthetic_ml_protocol.json").write_text(
    json.dumps(
        {
            "scope": "current-event diagnostic classification",
            "graph": "training rows only",
            "validation_used": False,
            "excluded_direct_target_variables": {
                k: sorted(v) for k, v in exclusions.items()
            },
        },
        indent=2,
    )
)
print(
    pd.DataFrame(rows)[["target", "model", "network_features", "f1"]].to_string(
        index=False
    )
)
