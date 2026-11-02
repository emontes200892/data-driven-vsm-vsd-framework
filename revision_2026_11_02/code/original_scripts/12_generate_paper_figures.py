from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "paper_tables"
OUT = RESULTS / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)


def read_table(name):
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_csv(path)


def savefig(name):
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {path}")


# ============================================================
# Fig. 2. VSM KPI comparison
# ============================================================
vsm = read_table("table_vsm_kpis.csv")

plt.figure(figsize=(9, 5))
plt.bar(vsm["Dataset"], vsm["VA Ratio (%)"])
plt.ylabel("Value-Added Ratio (%)")
plt.xlabel("Dataset")
plt.title("Value-Added Ratio by Dataset")
plt.xticks(rotation=20, ha="right")
savefig("fig_2a_vsm_va_ratio.png")


plt.figure(figsize=(9, 5))
plt.bar(vsm["Dataset"], vsm["Rework Rate (%)"])
plt.ylabel("Rework Rate (%)")
plt.xlabel("Dataset")
plt.title("Rework Rate by Dataset")
plt.xticks(rotation=20, ha="right")
savefig("fig_2b_vsm_rework_rate.png")


# ============================================================
# Fig. 3. Network global summary
# ============================================================
network = read_table("table_network_metrics.csv")

plt.figure(figsize=(9, 5))
plt.bar(network["Dataset"], network["Density"])
plt.ylabel("Network Density")
plt.xlabel("Dataset")
plt.title("Activity Network Density by Dataset")
plt.xticks(rotation=20, ha="right")
savefig("fig_3_network_density.png")


# ============================================================
# Fig. 4. Leakage-controlled matched-model comparison
#         Only evaluable targets are shown.
# ============================================================
ml = read_table("table_real_ml_results.csv")

required_cols = {
    "Dataset",
    "Target",
    "Model",
    "F1 (No CNA)",
    "F1 (CNA)",
}

missing_cols = required_cols.difference(ml.columns)

if missing_cols:
    raise ValueError(
        "table_real_ml_results.csv is missing required columns: "
        + ", ".join(sorted(missing_cols))
    )


# Convert metric columns explicitly to numeric.
# N/A entries become NaN and are excluded from the comparison.
ml["F1 (No CNA)"] = pd.to_numeric(
    ml["F1 (No CNA)"],
    errors="coerce"
)

ml["F1 (CNA)"] = pd.to_numeric(
    ml["F1 (CNA)"],
    errors="coerce"
)


# Keep only targets for which both feature configurations
# are genuinely evaluable.
ml_eval = ml.dropna(
    subset=["F1 (No CNA)", "F1 (CNA)"]
).copy()


if ml_eval.empty:
    raise ValueError(
        "No evaluable ML comparisons were found in "
        "table_real_ml_results.csv"
    )


ml_eval["Label"] = (
    ml_eval["Dataset"].astype(str)
    + "\n"
    + ml_eval["Target"].astype(str)
    + "\n"
    + ml_eval["Model"].astype(str)
)

x = list(range(len(ml_eval)))
width = 0.38

plt.figure(figsize=(13, 6))

plt.bar(
    [i - width / 2 for i in x],
    ml_eval["F1 (No CNA)"],
    width,
    label="Without CNA",
)

plt.bar(
    [i + width / 2 for i in x],
    ml_eval["F1 (CNA)"],
    width,
    label="With CNA",
)

plt.ylabel("F1-score")
plt.xlabel("Dataset / Target / Model")

plt.title(
    "Leakage-Controlled Predictive Performance: "
    "Matched Models Without and With CNA Features"
)

plt.xticks(
    x,
    ml_eval["Label"],
    rotation=45,
    ha="right"
)

plt.ylim(0, 1.08)
plt.legend()

savefig("fig_4_ml_f1_cna_comparison.png")


# ============================================================
# Fig. 5. Cumulative framework configurations
# ============================================================
scenarios = read_table("table_scenarios.csv")


# ------------------------------------------------------------
# Fig. 5a. Lead Time
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    scenarios["Scenario"],
    scenarios["Lead Time"],
    marker="o"
)

plt.ylabel("Lead Time")
plt.xlabel("Cumulative Framework Configuration")

plt.title(
    "Lead Time Across Cumulative Framework Configurations"
)

plt.xticks(rotation=30, ha="right")

savefig("fig_5a_digital_twin_lead_time.png")


# ------------------------------------------------------------
# Fig. 5b. Throughput
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    scenarios["Scenario"],
    scenarios["Throughput"],
    marker="o"
)

plt.ylabel("Throughput")
plt.xlabel("Cumulative Framework Configuration")

plt.title(
    "Throughput Across Cumulative Framework Configurations"
)

plt.xticks(rotation=30, ha="right")

savefig("fig_5b_digital_twin_throughput.png")


# ------------------------------------------------------------
# Fig. 5c. WIP
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    scenarios["Scenario"],
    scenarios["WIP"],
    marker="o"
)

plt.ylabel("WIP")
plt.xlabel("Cumulative Framework Configuration")

plt.title(
    "WIP Across Cumulative Framework Configurations"
)

plt.xticks(rotation=30, ha="right")

savefig("fig_5c_digital_twin_wip.png")


# ============================================================
# Fig. 6. Cumulative framework evaluation
# ============================================================
ablation = read_table("table_ablation.csv")

plt.figure(figsize=(10, 5))

plt.plot(
    ablation["Scenario"],
    ablation["Lead Time Reduction (%)"],
    marker="o",
    label="Lead Time Reduction",
)

plt.plot(
    ablation["Scenario"],
    ablation["Throughput Gain (%)"],
    marker="o",
    label="Throughput Gain",
)

plt.plot(
    ablation["Scenario"],
    ablation["WIP Reduction (%)"],
    marker="o",
    label="WIP Reduction",
)

plt.ylabel("Improvement Relative to Baseline (%)")
plt.xlabel("Cumulative Framework Configuration")

plt.title(
    "Cumulative Framework Evaluation: "
    "Improvement Relative to Baseline"
)

plt.xticks(rotation=30, ha="right")
plt.legend()

savefig("fig_6_ablation_improvement.png")


print("\nAll paper figures generated in:")
print(OUT)
