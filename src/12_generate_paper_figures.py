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


# ------------------------------------------------------------
# Fig. 2. VSM KPI comparison
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Fig. 3. Network global summary
# ------------------------------------------------------------
network = read_table("table_network_metrics.csv")

plt.figure(figsize=(9, 5))
plt.bar(network["Dataset"], network["Density"])
plt.ylabel("Network Density")
plt.xlabel("Dataset")
plt.title("Activity Network Density by Dataset")
plt.xticks(rotation=20, ha="right")
savefig("fig_3_network_density.png")


# ------------------------------------------------------------
# Fig. 4. ML F1: no network vs network
# ------------------------------------------------------------
ml = read_table("table_real_ml_results.csv")

ml["Label"] = ml["Dataset"] + " - " + ml["Target"]

x = range(len(ml))
width = 0.38

plt.figure(figsize=(12, 6))
plt.bar([i - width / 2 for i in x], ml["F1 (No Network)"], width, label="No Network")
plt.bar([i + width / 2 for i in x], ml["F1 (Network)"], width, label="Network")
plt.ylabel("F1-score")
plt.xlabel("Dataset / Target")
plt.title("Predictive Performance With and Without Network Features")
plt.xticks(x, ml["Label"], rotation=45, ha="right")
plt.ylim(0, 1.08)
plt.legend()
savefig("fig_4_ml_f1_network_comparison.png")


# ------------------------------------------------------------
# Fig. 5. Digital twin scenario comparison
# ------------------------------------------------------------
scenarios = read_table("table_scenarios.csv")

plt.figure(figsize=(10, 5))
plt.plot(scenarios["Scenario"], scenarios["Lead Time"], marker="o")
plt.ylabel("Lead Time")
plt.xlabel("Scenario")
plt.title("Lead Time Across Digital Twin Scenarios")
plt.xticks(rotation=30, ha="right")
savefig("fig_5a_digital_twin_lead_time.png")

plt.figure(figsize=(10, 5))
plt.plot(scenarios["Scenario"], scenarios["Throughput"], marker="o")
plt.ylabel("Throughput")
plt.xlabel("Scenario")
plt.title("Throughput Across Digital Twin Scenarios")
plt.xticks(rotation=30, ha="right")
savefig("fig_5b_digital_twin_throughput.png")

plt.figure(figsize=(10, 5))
plt.plot(scenarios["Scenario"], scenarios["WIP"], marker="o")
plt.ylabel("WIP")
plt.xlabel("Scenario")
plt.title("WIP Across Digital Twin Scenarios")
plt.xticks(rotation=30, ha="right")
savefig("fig_5c_digital_twin_wip.png")


# ------------------------------------------------------------
# Fig. 6. Ablation improvement
# ------------------------------------------------------------
ablation = read_table("table_ablation.csv")

plt.figure(figsize=(10, 5))
plt.plot(ablation["Scenario"], ablation["Lead Time Reduction (%)"], marker="o", label="Lead Time Reduction")
plt.plot(ablation["Scenario"], ablation["Throughput Gain (%)"], marker="o", label="Throughput Gain")
plt.plot(ablation["Scenario"], ablation["WIP Reduction (%)"], marker="o", label="WIP Reduction")
plt.ylabel("Improvement vs Baseline (%)")
plt.xlabel("Scenario")
plt.title("Ablation Study: Incremental Improvement by Framework Component")
plt.xticks(rotation=30, ha="right")
plt.legend()
savefig("fig_6_ablation_improvement.png")


print("\nAll paper figures generated in:")
print(OUT)
