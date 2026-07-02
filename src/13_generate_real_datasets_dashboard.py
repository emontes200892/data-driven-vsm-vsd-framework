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
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.28, hspace=0.42)

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
# ------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

overview = vsm.copy()
overview["Mean Lead Time (hours)"] = overview["Mean Lead Time (min)"] / 60.0

metrics = [
    "Mean Lead Time (hours)",
    "VA Ratio (%)",
    "Rework Rate (%)",
    "Defect Rate (%)",
    "Throughput (cases/hour)",
]

x = np.arange(len(metrics))
width = 0.24

for i, (_, row) in enumerate(overview.iterrows()):
    values = [row[m] for m in metrics]
    offset = (i - 1) * width
    bars = ax1.bar(x + offset, values, width, label=short_dataset_name(row["Dataset"]))
    for bar in bars:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(overview[metrics].max()) * 0.015,
            f"{h:.1f}" if h >= 1 else f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0
        )

ax1.set_title("A. Dataset Overview: Key Indicators", fontweight="bold")
ax1.set_ylabel("Value")
ax1.set_xticks(x)
ax1.set_xticklabels(
    ["Mean LT\n(hours)", "VA Ratio\n(%)", "Rework\n(%)", "Defect\n(%)", "Throughput\n(cases/h)"],
    rotation=18,
    ha="right"
)
ax1.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.spines[["top", "right"]].set_visible(False)

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
# Keep only Bottleneck and Rework, because Delay is trivial
# ------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])

ml = real_ml.copy()
ml = ml[ml["Target"].isin(["Bottleneck", "Rework"])].copy()
ml["Dataset Short"] = ml["Dataset"].map(short_dataset_name)
ml["Label"] = ml["Dataset Short"] + "\n" + ml["Target"]

x = np.arange(len(ml))
width = 0.36

bars1 = ax3.bar(
    x - width / 2,
    ml["F1 (No Network)"],
    width,
    label="No Network"
)
bars2 = ax3.bar(
    x + width / 2,
    ml["F1 (Network)"],
    width,
    label="With Network"
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
            fontsize=8
        )

ax3.set_title(
    "C. Predictive Performance\nNetwork Features vs No Network",
    fontweight="bold"
)
ax3.set_ylabel("F1-score")
ax3.set_ylim(0, 1.18)
ax3.set_xticks(x)
ax3.set_xticklabels(ml["Label"], fontsize=8)
ax3.legend(fontsize=8, frameon=False, ncol=2, loc="upper center")
ax3.grid(axis="y", linestyle="--", alpha=0.35)
ax3.spines[["top", "right"]].set_visible(False)

# ------------------------------------------------------------
# D. Digital twin scenarios
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
ax4.set_xticklabels(scenario_labels, fontsize=8)
ax4.set_title("D. Digital Twin Scenarios: Lead Time & Throughput", fontweight="bold")
ax4.grid(axis="y", linestyle="--", alpha=0.35)
ax4.spines[["top"]].set_visible(False)

for xi, yi in zip(x, sc["Lead Time"]):
    ax4.text(xi, yi + 1.2, f"{yi:.1f}", ha="center", fontsize=8)

ax4b = ax4.twinx()
line2 = ax4b.plot(
    x,
    sc["Throughput"],
    marker="o",
    linewidth=2,
    linestyle="-",
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
# E. Ablation improvement
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

ax5.set_title("E. Ablation Study: Improvement vs Baseline (%)", fontweight="bold")
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
# F. Critical activities
# This uses the manually selected critical activity names from table_network_metrics.csv.
# If centrality values are available later, replace the placeholders with actual values.
# ------------------------------------------------------------
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

ax6.set_title("F. Critical Activities by Network Analysis", fontweight="bold", pad=12)

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
    "Note: critical activities correspond to the highest structural relevance identified in the activity networks.",
    fontsize=8,
    color="dimgray",
    transform=ax6.transAxes
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
plt.savefig(OUTFILE.with_suffix(".pdf"), bbox_inches="tight")
plt.close()

print(f"Generated PNG: {OUTFILE}")
print(f"Generated PDF: {OUTFILE.with_suffix('.pdf')}")
