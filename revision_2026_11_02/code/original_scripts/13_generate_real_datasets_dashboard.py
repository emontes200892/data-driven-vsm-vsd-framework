from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "paper_tables"
OUT = RESULTS / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

OUTFILE = OUT / "fig_real_datasets_experimental_results_dashboard.png"


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def short_dataset_name(name):
    mapping = {
        "BPI2017": "BPI2017",
        "BPI2019": "BPI2019",
        "Data Quality Issues": "DQI",
        "data_quality_issues": "DQI",
        "bpi2017": "BPI2017",
        "bpi2019": "BPI2019",
    }
    return mapping.get(name, name)


vsm = read_csv(TABLES / "table_vsm_kpis.csv")
network = read_csv(TABLES / "table_network_metrics.csv")
real_ml = read_csv(TABLES / "table_real_ml_results.csv")
scenarios = read_csv(TABLES / "table_scenarios.csv")
ablation = read_csv(TABLES / "table_ablation.csv")

# ------------------------------------------------------------
# Figure canvas
# ------------------------------------------------------------
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.34, hspace=0.58)

fig.suptitle(
    "Real-World Datasets: Experimental Results Summary",
    fontsize=20,
    fontweight="bold",
    y=0.98
)
fig.text(
    0.5, 0.935,
    "BPI2017 · BPI2019 · Data Quality Issues",
    ha="center",
    fontsize=13
)

# ------------------------------------------------------------
# A. Dataset overview
# Different KPIs have incompatible units/scales, so a table is used instead
# of plotting them on a common numerical axis.
# ------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis("off")

overview = vsm.copy()
overview["Mean Lead Time (hours)"] = (
    pd.to_numeric(overview["Mean Lead Time (min)"], errors="coerce") / 60.0
)
overview["Short"] = overview["Dataset"].map(short_dataset_name)

dataset_order_overview = ["BPI2017", "BPI2019", "DQI"]
overview = overview.set_index("Short").reindex(dataset_order_overview)

kpi_rows = [
    ("Mean LT (hours)", "Mean Lead Time (hours)", "{:.1f}"),
    ("VA Ratio (%)", "VA Ratio (%)", "{:.1f}"),
    ("Rework Rate (%)", "Rework Rate (%)", "{:.1f}"),
    ("Defect Rate (%)", "Defect Rate (%)", "{:.1f}"),
    ("Throughput (cases/h)", "Throughput (cases/hour)", "{:.2f}"),
]

cell_text = []
for label, col, fmt in kpi_rows:
    cell_text.append(
        [label] + [fmt.format(overview.loc[d, col]) for d in dataset_order_overview]
    )

tbl = ax1.table(
    cellText=cell_text,
    colLabels=["Indicator"] + dataset_order_overview,
    cellLoc="center",
    colLoc="center",
    loc="center",
    colWidths=[0.43, 0.19, 0.19, 0.19],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.75)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight="bold")
    if c == 0 and r > 0:
        cell.set_text_props(ha="left")

ax1.set_title("A. Dataset Overview\nKey Indicators", fontweight="bold", pad=14)

# ------------------------------------------------------------
# B. Network density
# ------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])

net = network.copy()
net["Short"] = net["Dataset"].map(short_dataset_name)

bars = ax2.bar(net["Short"], net["Density"])
for bar in bars:
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.015,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax2.set_title("B. Activity Network Density", fontweight="bold")
ax2.set_ylabel("Density")
ax2.set_ylim(0, max(0.7, net["Density"].max() * 1.25))
ax2.grid(axis="y", linestyle="--", alpha=0.35)
ax2.spines[["top", "right"]].set_visible(False)

# ------------------------------------------------------------
# C. Predictive performance
# Leakage-controlled matched-model comparison: without vs with CNA
# ------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])

ml = real_ml.copy()

required_ml_cols = {
    "Dataset",
    "Target",
    "Model",
    "F1 (No CNA)",
    "F1 (CNA)",
}
missing_ml_cols = required_ml_cols.difference(ml.columns)
if missing_ml_cols:
    raise ValueError(
        "table_real_ml_results.csv is missing required columns: "
        + ", ".join(sorted(missing_ml_cols))
    )

# The dashboard uses Random Forest as a single matched-model summary to keep
# the panel readable. The complete table/standalone figure reports both RF
# and XGBoost. Delay is retained because it is no longer a trivial task after
# leakage control.
ml = ml[
    (ml["Model"] == "Random Forest")
    & (ml["Target"].isin(["Delay", "Bottleneck", "Rework"]))
].copy()

dataset_order = ["BPI2017", "BPI2019", "Data Quality Issues"]
target_order = ["Delay", "Bottleneck", "Rework"]

ml["Dataset"] = pd.Categorical(
    ml["Dataset"], categories=dataset_order, ordered=True
)
ml["Target"] = pd.Categorical(
    ml["Target"], categories=target_order, ordered=True
)
ml = ml.sort_values(["Dataset", "Target"]).reset_index(drop=True)

ml["Dataset Short"] = ml["Dataset"].astype(str).map(short_dataset_name)
ml["Label"] = ml["Dataset Short"] + "\n" + ml["Target"].astype(str)

x = np.arange(len(ml))
width = 0.36

bars1 = ax3.bar(
    x - width / 2,
    ml["F1 (No CNA)"],
    width,
    label="Without CNA"
)
bars2 = ax3.bar(
    x + width / 2,
    ml["F1 (CNA)"],
    width,
    label="With CNA"
)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.025,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=7
        )

ax3.set_title(
    "C. Predictive Performance (Random Forest)\n"
    "Leakage-Controlled: Without vs With CNA",
    fontweight="bold"
)
ax3.set_ylabel("F1-score")
ax3.set_ylim(0, 1.18)
ax3.set_xticks(x)
ax3.set_xticklabels(ml["Label"], fontsize=7, rotation=45, ha="right")
ax3.legend(fontsize=8, frameon=False, ncol=2, loc="upper center")
ax3.grid(axis="y", linestyle="--", alpha=0.35)
ax3.spines[["top", "right"]].set_visible(False)

# ------------------------------------------------------------
# D. Cumulative framework configurations
# ------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 0])

sc = scenarios.copy()
scenario_labels = [
    "Baseline",
    "Manual\nVSM",
    "PM-VSM",
    "PM-VSM\nXGB",
    "PM-VSM\nXGB-DT",
    "Full\nFramework",
]

x = np.arange(len(sc))

line1 = ax4.plot(
    x,
    sc["Lead Time"],
    marker="o",
    linewidth=2,
    label="Lead Time"
)

ax4.set_ylabel("Lead Time")
ax4.set_xticks(x)
ax4.set_xticklabels(scenario_labels, fontsize=7.5)
ax4.set_title("D. Cumulative Framework Configurations\nLead Time & Throughput", fontweight="bold", pad=10)
ax4.grid(axis="y", linestyle="--", alpha=0.35)
ax4.spines[["top"]].set_visible(False)

for xi, yi in zip(x, sc["Lead Time"]):
    ax4.text(xi, yi + 1.2, f"{yi:.1f}", ha="center", fontsize=8)

ax4b = ax4.twinx()
line2 = ax4b.plot(
    x,
    sc["Throughput"],
    marker="s",
    linewidth=2,
    linestyle="--",
    label="Throughput"
)
ax4b.set_ylabel("Throughput")
ax4b.spines[["top"]].set_visible(False)

for xi, yi in zip(x, sc["Throughput"]):
    ax4b.text(xi, yi + 0.18, f"{yi:.2f}", ha="center", fontsize=8)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, fontsize=8, frameon=True, loc="upper center")

# ------------------------------------------------------------
# E. Cumulative framework improvement
# ------------------------------------------------------------
ax5 = fig.add_subplot(gs[1, 1])

abl = ablation.copy()
abl = abl[abl["Scenario"] != "Baseline"].copy()

x = np.arange(len(abl))
width = 0.25

bars1 = ax5.bar(x - width, abl["Lead Time Reduction (%)"], width, label="Lead Time Reduction")
bars2 = ax5.bar(x, abl["Throughput Gain (%)"], width, label="Throughput Gain")
bars3 = ax5.bar(x + width, abl["WIP Reduction (%)"], width, label="WIP Reduction")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.2,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=7
        )

ax5.set_title("E. Cumulative Framework Evaluation\nImprovement vs Baseline (%)", fontweight="bold", pad=10)
ax5.set_ylabel("Improvement (%)")
ax5.set_xticks(x)
ax5.set_xticklabels(
    ["Manual\nVSM", "PM-VSM", "PM-VSM\nXGB", "PM-VSM\nXGB-DT", "Full\nFramework"],
    fontsize=8
)
ax5.legend(fontsize=8, frameon=False, ncol=1, loc="upper left")
ax5.grid(axis="y", linestyle="--", alpha=0.35)
ax5.spines[["top", "right"]].set_visible(False)

# ------------------------------------------------------------
# F. Structurally relevant activities
# Activity names are reported from table_network_metrics.csv; the dashboard does not
# claim an independent centrality ranking beyond what is present in that table.
# ------------------------------------------------------------
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

ax6.set_title("F. Structurally Relevant Activities\nfrom Network Analysis", fontweight="bold", pad=10)

crit = network[["Dataset", "Critical Activity", "Nodes", "Edges", "Density"]].copy()
crit["Dataset"] = crit["Dataset"].map(short_dataset_name)

y_positions = [0.80, 0.55, 0.30]

for y, (_, row) in zip(y_positions, crit.iterrows()):
    ax6.text(
        0.02,
        y,
        row["Dataset"],
        fontsize=12,
        fontweight="bold",
        transform=ax6.transAxes
    )
    ax6.text(
        0.25,
        y,
        row["Critical Activity"],
        fontsize=10,
        transform=ax6.transAxes
    )
    ax6.text(
        0.25,
        y - 0.09,
        f"Nodes={int(row['Nodes'])}, Edges={int(row['Edges'])}, Density={row['Density']:.3f}",
        fontsize=9,
        color="dimgray",
        transform=ax6.transAxes
    )
    ax6.plot(
        [0.02, 0.95],
        [y - 0.15, y - 0.15],
        transform=ax6.transAxes,
        linewidth=0.8,
        alpha=0.25
    )

ax6.text(
    0.02,
    0.08,
    "Note: reported activities summarize structurally relevant nodes identified in the activity-network analysis.",
    fontsize=8,
    color="dimgray",
    transform=ax6.transAxes
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
fig.subplots_adjust(top=0.88, bottom=0.08)
plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
plt.savefig(OUTFILE.with_suffix(".pdf"), bbox_inches="tight")
plt.close()

print(f"Generated PNG: {OUTFILE}")
print(f"Generated PDF: {OUTFILE.with_suffix('.pdf')}")
