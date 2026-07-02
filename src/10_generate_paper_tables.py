from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "bpi2017": "BPI2017",
    "bpi2019": "BPI2019",
    "data_quality_issues": "Data Quality Issues",
}


def safe_read(path):
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    return pd.read_csv(path)


def pct(x):
    return 100 * x if pd.notna(x) else np.nan


# ------------------------------------------------------------
# Table 1. Dataset characteristics
# ------------------------------------------------------------
rows = []

for folder, name in DATASETS.items():
    df = safe_read(RESULTS / folder / "dataset_summary.csv")
    if df is None:
        continue

    r = df.iloc[0]
    rows.append({
        "Dataset": name,
        "Events": int(r["events"]),
        "Cases": int(r["cases"]),
        "Activities": int(r["activities"]),
        "Resources": int(r["resources"]),
        "Observation Hours": round(r["observation_hours"], 2),
    })

table_dataset = pd.DataFrame(rows)
table_dataset.to_csv(OUT / "table_dataset_characteristics.csv", index=False)
table_dataset.to_latex(
    OUT / "table_dataset_characteristics.tex",
    index=False,
    escape=False,
    float_format="%.2f"
)


# ------------------------------------------------------------
# Table 2. VSM KPIs
# ------------------------------------------------------------
rows = []

for folder, name in DATASETS.items():
    df = safe_read(RESULTS / folder / "dataset_summary.csv")
    if df is None:
        continue

    r = df.iloc[0]
    rows.append({
        "Dataset": name,
        "Mean Lead Time (min)": round(r["mean_case_lead_time_min"], 2),
        "Median Lead Time (min)": round(r["median_case_lead_time_min"], 2),
        "Mean Processing Time (min)": round(r["mean_processing_time_min"], 2),
        "VA Ratio (%)": round(pct(r["mean_value_added_ratio"]), 2),
        "Throughput (cases/hour)": round(r["throughput_cases_per_hour"], 4),
        "Defect Rate (%)": round(pct(r["defect_rate"]), 2),
        "Rework Rate (%)": round(pct(r["rework_rate"]), 2),
    })

table_vsm = pd.DataFrame(rows)
table_vsm.to_csv(OUT / "table_vsm_kpis.csv", index=False)
table_vsm.to_latex(
    OUT / "table_vsm_kpis.tex",
    index=False,
    escape=False,
    float_format="%.2f"
)


# ------------------------------------------------------------
# Table 3. Network global metrics
# ------------------------------------------------------------
critical_activity = {
    "bpi2017": "W_Call incomplete files",
    "bpi2019": "Record Invoice Receipt",
    "data_quality_issues": "/vgr/pick_up_and_transport",
}

rows = []

for folder, name in DATASETS.items():
    df = safe_read(RESULTS / folder / "network_global_metrics.csv")
    if df is None:
        continue

    r = df.iloc[0]
    rows.append({
        "Dataset": name,
        "Nodes": int(r["nodes"]),
        "Edges": int(r["edges"]),
        "Density": round(r["density"], 4),
        "Weakly Connected Components": int(r["weakly_connected_components"]),
        "Critical Activity": critical_activity.get(folder, ""),
    })

table_network = pd.DataFrame(rows)
table_network.to_csv(OUT / "table_network_metrics.csv", index=False)
table_network.to_latex(
    OUT / "table_network_metrics.tex",
    index=False,
    escape=False,
    float_format="%.4f"
)


# ------------------------------------------------------------
# Table 4. Real dataset ML results
# Best F1 without networks vs with networks
# ------------------------------------------------------------
real_ml = safe_read(RESULTS / "real_datasets_ml_results.csv")

if real_ml is not None:
    rows = []

    for (dataset, target), g in real_ml.groupby(["dataset", "target"]):
        no_net = g[g["network_features"] == False]
        yes_net = g[g["network_features"] == True]

        if no_net.empty or yes_net.empty:
            continue

        best_no = no_net.loc[no_net["f1"].idxmax()]
        best_yes = yes_net.loc[yes_net["f1"].idxmax()]

        f1_no = best_no["f1"]
        f1_yes = best_yes["f1"]

        improvement = ((f1_yes - f1_no) / f1_no * 100) if f1_no > 0 else np.nan

        rows.append({
            "Dataset": DATASETS.get(dataset, dataset),
            "Target": target.replace("_proxy", "").replace("_", " ").title(),
            "Best Model (No Network)": best_no["model"],
            "F1 (No Network)": round(f1_no, 4),
            "Best Model (Network)": best_yes["model"],
            "F1 (Network)": round(f1_yes, 4),
            "Improvement (%)": round(improvement, 2) if pd.notna(improvement) else "N/A",
        })

    table_ml_real = pd.DataFrame(rows)
    table_ml_real.to_csv(OUT / "table_real_ml_results.csv", index=False)
    table_ml_real.to_latex(
        OUT / "table_real_ml_results.tex",
        index=False,
        escape=False,
        float_format="%.4f"
    )


# ------------------------------------------------------------
# Table 5. Synthetic ML results
# ------------------------------------------------------------
synthetic_ml = safe_read(RESULTS / "ml_results.csv")

if synthetic_ml is not None:
    rows = []

    for target, g in synthetic_ml.groupby("target"):
        no_net = g[g["network_features"] == False]
        yes_net = g[g["network_features"] == True]

        if no_net.empty or yes_net.empty:
            continue

        best_no = no_net.loc[no_net["f1"].idxmax()]
        best_yes = yes_net.loc[yes_net["f1"].idxmax()]

        f1_no = best_no["f1"]
        f1_yes = best_yes["f1"]

        improvement = ((f1_yes - f1_no) / f1_no * 100) if f1_no > 0 else np.nan

        rows.append({
            "Target": target.replace("_flag", "").replace("_", " ").title(),
            "Best Model (No Network)": best_no["model"],
            "F1 (No Network)": round(f1_no, 4),
            "Best Model (Network)": best_yes["model"],
            "F1 (Network)": round(f1_yes, 4),
            "Improvement (%)": round(improvement, 2) if pd.notna(improvement) else "N/A",
        })

    table_ml_synthetic = pd.DataFrame(rows)
    table_ml_synthetic.to_csv(OUT / "table_synthetic_ml_results.csv", index=False)
    table_ml_synthetic.to_latex(
        OUT / "table_synthetic_ml_results.tex",
        index=False,
        escape=False,
        float_format="%.4f"
    )


# ------------------------------------------------------------
# Table 6. Ablation results
# ------------------------------------------------------------
ablation = safe_read(RESULTS / "ablation_results.csv")

if ablation is not None:
    preferred_order = [
        "Baseline",
        "Manual VSM",
        "PM-VSM",
        "PM-VSM-XGB",
        "PM-VSM-XGB-DT",
        "Full Framework",
    ]

    ablation["scenario"] = pd.Categorical(
        ablation["scenario"],
        categories=preferred_order,
        ordered=True
    )

    ablation = ablation.sort_values("scenario")

    baseline = ablation[ablation["scenario"] == "Baseline"].iloc[0]

    rows = []
    for _, r in ablation.iterrows():
        lead_improvement = (baseline["Lead_Time"] - r["Lead_Time"]) / baseline["Lead_Time"] * 100
        throughput_improvement = (r["Throughput"] - baseline["Throughput"]) / baseline["Throughput"] * 100
        wip_reduction = (baseline["WIP"] - r["WIP"]) / baseline["WIP"] * 100
        defect_reduction = (baseline["Defect_Rate"] - r["Defect_Rate"]) / baseline["Defect_Rate"] * 100
        rework_reduction = (baseline["Rework_Rate"] - r["Rework_Rate"]) / baseline["Rework_Rate"] * 100

        rows.append({
            "Scenario": r["scenario"],
            "Lead Time": round(r["Lead_Time"], 2),
            "Throughput": round(r["Throughput"], 2),
            "WIP": round(r["WIP"], 2),
            "Defect Rate (%)": round(pct(r["Defect_Rate"]), 2),
            "Rework Rate (%)": round(pct(r["Rework_Rate"]), 2),
            "Lead Time Reduction (%)": round(lead_improvement, 2),
            "Throughput Gain (%)": round(throughput_improvement, 2),
            "WIP Reduction (%)": round(wip_reduction, 2),
            "Defect Reduction (%)": round(defect_reduction, 2),
            "Rework Reduction (%)": round(rework_reduction, 2),
        })

    table_ablation = pd.DataFrame(rows)
    table_ablation.to_csv(OUT / "table_ablation.csv", index=False)
    table_ablation.to_latex(
        OUT / "table_ablation.tex",
        index=False,
        escape=False,
        float_format="%.2f"
    )


# ------------------------------------------------------------
# Table 7. Scenario / Digital Twin results
# ------------------------------------------------------------
scenarios = safe_read(RESULTS / "scenario_results.csv")

if scenarios is not None:
    preferred_order = [
        "Baseline",
        "Manual VSM",
        "PM-VSM",
        "PM-VSM-XGB",
        "PM-VSM-XGB-DT",
        "Full Framework",
    ]

    scenarios["scenario"] = pd.Categorical(
        scenarios["scenario"],
        categories=preferred_order,
        ordered=True
    )

    scenarios = scenarios.sort_values("scenario")

    rows = []
    for _, r in scenarios.iterrows():
        rows.append({
            "Scenario": r["scenario"],
            "Lead Time": round(r["Lead_Time"], 2),
            "Lead Time SD": round(r["Lead_Time_SD"], 3),
            "Lead Time CI95": round(r["Lead_Time_CI95"], 3),
            "Throughput": round(r["Throughput"], 2),
            "WIP": round(r["WIP"], 2),
            "Defect Rate (%)": round(pct(r["Defect_Rate"]), 2),
            "Rework Rate (%)": round(pct(r["Rework_Rate"]), 2),
        })

    table_scenarios = pd.DataFrame(rows)
    table_scenarios.to_csv(OUT / "table_scenarios.csv", index=False)
    table_scenarios.to_latex(
        OUT / "table_scenarios.tex",
        index=False,
        escape=False,
        float_format="%.2f"
    )


print("\nPaper tables generated in:")
print(OUT)

print("\nGenerated files:")
for f in sorted(OUT.glob("*")):
    print(" -", f.name)
