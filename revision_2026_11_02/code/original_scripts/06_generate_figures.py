import os
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

FIG_DIR = "figures"
RESULTS_DIR = "results"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 500,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans"
})

COLORS = {
    "Baseline": "#6C5CE7",
    "Manual VSM": "#3498DB",
    "PM-VSM": "#1ABC9C",
    "PM-VSM-XGB": "#82C91E",
    "PM-VSM-XGB-DT": "#F39C12",
    "Full Framework": "#E74C3C"
}

ML_COLORS = {
    "RF": "#3498DB",
    "XGB": "#F39C12",
    "Proxy": "#7F8C8D"
}

def clean_label(x, width=12):
    return "\n".join(textwrap.wrap(str(x), width=width))

def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"No encontré columnas: {candidates}. Columnas disponibles: {list(df.columns)}")

def savefig(filename, dpi=500):
    plt.savefig(os.path.join(FIG_DIR, filename + ".png"), bbox_inches="tight", dpi=dpi)
    plt.savefig(os.path.join(FIG_DIR, filename + ".pdf"), bbox_inches="tight")
    plt.close()

def add_bar_labels(ax, fmt="{:.2f}", fontsize=8, offset=3):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=fontsize, padding=offset)

def simplify_target(x):
    x = str(x).lower()
    if "defect" in x:
        return "Defect"
    if "rework" in x:
        return "Rework"
    if "bottleneck" in x:
        return "Bottleneck"
    if "delay" in x:
        return "Delay"
    return str(x).replace("_flag", "").title()

def simplify_model(x):
    x = str(x)
    xl = x.lower()
    if "random" in xl or "forest" in xl or x.upper() == "RF":
        return "RF"
    if "xgb" in xl or "xgboost" in xl:
        return "XGB"
    if "proxy" in xl:
        return "Proxy"
    return x[:8]

# =========================
# Load data
# =========================

ablation = pd.read_csv(f"{RESULTS_DIR}/ablation_results.csv")
vsm = pd.read_csv(f"{RESULTS_DIR}/vsm_stage_kpis.csv")
ml = pd.read_csv(f"{RESULTS_DIR}/ml_results.csv")
network = pd.read_csv(f"{RESULTS_DIR}/network_activity_features.csv")

scenario_col = find_col(ablation, ["scenario", "configuration", "group"])
activity_col = find_col(vsm, ["activity", "stage"])

metric_map = {
    "Lead Time": find_col(ablation, ["Lead_Time", "lead_time_mean", "lead_time", "LT"]),
    "Throughput": find_col(ablation, ["Throughput", "throughput_per_hour", "throughput", "TH"]),
    "WIP": find_col(ablation, ["WIP", "wip_mean"]),
    "Defect Rate": find_col(ablation, ["Defect_Rate", "defect_rate"]),
    "Rework Rate": find_col(ablation, ["Rework_Rate", "rework_rate"]),
}

vsm_metric_map = {
    "Cycle Time": find_col(vsm, ["CT_min", "cycle_time_min", "ct_min"]),
    "Waiting Time": find_col(vsm, ["WT_min", "waiting_time_min", "wt_min"]),
    "WIP": find_col(vsm, ["WIP", "wip_before", "wip_mean"]),
    "Defect Rate": find_col(vsm, ["Defect_Rate", "defect_rate"]),
    "Bottleneck Rate": find_col(vsm, ["Bottleneck_Rate", "bottleneck_rate"]),
}

target_col = find_col(ml, ["target", "task"])
model_col = find_col(ml, ["model"])
f1_col = find_col(ml, ["f1", "f1_score", "f1-score"])

net_activity_col = find_col(network, ["activity", "node"])
centrality_cols = {
    "Degree": find_col(network, ["degree_centrality"]),
    "Betweenness": find_col(network, ["betweenness_centrality"]),
    "Closeness": find_col(network, ["closeness_centrality"]),
    "Eigenvector": find_col(network, ["eigenvector_centrality"]),
}

order = [
    "Baseline",
    "Manual VSM",
    "PM-VSM",
    "PM-VSM-XGB",
    "PM-VSM-XGB-DT",
    "Full Framework"
]

ablation[scenario_col] = pd.Categorical(
    ablation[scenario_col],
    categories=order,
    ordered=True
)
ablation = ablation.sort_values(scenario_col).reset_index(drop=True)

# =========================
# Individual ablation figures
# =========================

for title, col in metric_map.items():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [COLORS.get(str(x), "#607D8B") for x in ablation[scenario_col].astype(str)]

    ax.bar(
        ablation[scenario_col].astype(str),
        ablation[col],
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    ax.set_title(f"Ablation study: {title}", fontweight="bold")
    ax.set_ylabel(title)
    ax.set_xticks(range(len(ablation)))
    ax.set_xticklabels([clean_label(x, 11) for x in ablation[scenario_col].astype(str)])

    if "rate" in title.lower():
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    add_bar_labels(ax, "{:.3f}" if "Rate" in title else "{:.2f}")
    fig.subplots_adjust(bottom=0.18)
    savefig(f"paper_ablation_{title.lower().replace(' ', '_')}")

# =========================
# Individual VSM figures
# =========================

for title, col in vsm_metric_map.items():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    vals = vsm[col].values
    colors = plt.cm.viridis(np.linspace(0.20, 0.85, len(vals)))

    ax.bar(
        vsm[activity_col],
        vals,
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    ax.set_title(f"Current-state VSM stage KPI: {title}", fontweight="bold")
    ax.set_ylabel(title)
    ax.set_xticks(range(len(vsm)))
    ax.set_xticklabels([clean_label(x, 11) for x in vsm[activity_col]], rotation=0)

    if "rate" in title.lower():
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    add_bar_labels(ax, "{:.3f}" if "Rate" in title else "{:.2f}")
    fig.subplots_adjust(bottom=0.20)
    savefig(f"paper_vsm_{title.lower().replace(' ', '_')}")

# =========================
# ML individual grouped figure
# =========================

ml_plot = ml.dropna(subset=[f1_col]).copy()
ml_plot["task"] = ml_plot[target_col].apply(simplify_target)
ml_plot["model_short"] = ml_plot[model_col].apply(simplify_model)

ml_pivot = (
    ml_plot
    .pivot_table(index="task", columns="model_short", values=f1_col, aggfunc="max")
    .reindex(["Defect", "Rework", "Bottleneck", "Delay"])
    .dropna(how="all")
)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(ml_pivot.index))
models = list(ml_pivot.columns)
width = 0.8 / max(len(models), 1)

for j, model in enumerate(models):
    vals = ml_pivot[model].values
    xpos = x - 0.4 + width / 2 + j * width
    ax.bar(
        xpos,
        vals,
        width=width,
        label=model,
        color=ML_COLORS.get(model, "#7F8C8D"),
        edgecolor="black",
        linewidth=0.5
    )

ax.set_title("Machine Learning performance by task", fontweight="bold")
ax.set_ylabel("F1-score")
ax.set_ylim(0, 1.12)
ax.set_xticks(x)
ax.set_xticklabels(ml_pivot.index)
ax.legend(frameon=False, ncol=len(models), loc="upper left")

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

savefig("paper_ml_f1_grouped")

# =========================
# Network individual figure
# =========================

fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True)

for ax, (name, col) in zip(axes, centrality_cols.items()):
    tmp = network[[net_activity_col, col]].sort_values(col, ascending=True).tail(8)
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(tmp)))

    ax.barh(
        tmp[net_activity_col],
        tmp[col],
        color=colors,
        edgecolor="black",
        linewidth=0.4
    )

    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Centrality")

axes[0].set_ylabel("Activity")
fig.suptitle("Activity network centrality indicators", fontsize=14, fontweight="bold")
fig.subplots_adjust(top=0.82, left=0.12, right=0.98, wspace=0.28)
savefig("paper_network_centralities")

# =========================
# Final dashboard figure
# =========================

fig = plt.figure(figsize=(15.5, 8.8))
gs = fig.add_gridspec(2, 3)

fig.suptitle(
    "Synthetic Manufacturing Dataset: Experimental Results",
    fontsize=18,
    fontweight="bold",
    y=0.965
)

# A. Overall framework score
ax1 = fig.add_subplot(gs[0, 0])

norm_df = ablation[[scenario_col] + list(metric_map.values())].copy()

for name, col in metric_map.items():
    if name == "Throughput":
        norm_df[name] = norm_df[col] / norm_df[col].max()
    else:
        norm_df[name] = 1 - (norm_df[col] / norm_df[col].max())

summary_cols = ["Lead Time", "Throughput", "WIP", "Defect Rate", "Rework Rate"]
norm_df["Overall Score"] = norm_df[summary_cols].mean(axis=1) * 100
norm_df = norm_df.sort_values("Overall Score", ascending=True)

ax1.barh(
    norm_df[scenario_col].astype(str),
    norm_df["Overall Score"],
    color=[COLORS.get(str(x), "#607D8B") for x in norm_df[scenario_col].astype(str)],
    edgecolor="black",
    linewidth=0.5
)

ax1.set_title("A. Overall Framework Score", fontweight="bold")
ax1.set_xlabel("Normalized performance score (%)")
ax1.set_xlim(0, 100)

for i, v in enumerate(norm_df["Overall Score"]):
    ax1.text(v + 1.3, i, f"{v:.1f}", va="center", fontsize=9, clip_on=False)

# B. Full framework vs baseline
ax2 = fig.add_subplot(gs[0, 1])

baseline = ablation[ablation[scenario_col].astype(str) == "Baseline"].iloc[0]
full = ablation[ablation[scenario_col].astype(str) == "Full Framework"].iloc[0]

improvements = {
    "Lead\nTime": (baseline[metric_map["Lead Time"]] - full[metric_map["Lead Time"]]) / baseline[metric_map["Lead Time"]] * 100,
    "WIP": (baseline[metric_map["WIP"]] - full[metric_map["WIP"]]) / baseline[metric_map["WIP"]] * 100,
    "Defect\nRate": (baseline[metric_map["Defect Rate"]] - full[metric_map["Defect Rate"]]) / baseline[metric_map["Defect Rate"]] * 100,
    "Rework\nRate": (baseline[metric_map["Rework Rate"]] - full[metric_map["Rework Rate"]]) / baseline[metric_map["Rework Rate"]] * 100,
    "Throughput": (full[metric_map["Throughput"]] - baseline[metric_map["Throughput"]]) / baseline[metric_map["Throughput"]] * 100,
}

imp_names = list(improvements.keys())
imp_vals = list(improvements.values())

ax2.bar(
    imp_names,
    imp_vals,
    color=["#3498DB", "#66BB6A", "#FF7043", "#EC407A", "#F39C12"],
    edgecolor="black",
    linewidth=0.5
)

ax2.set_title("B. Full Framework vs Baseline", fontweight="bold")
ax2.set_ylabel("Relative improvement (%)")
ax2.axhline(0, color="black", linewidth=0.8)
ymax = max(imp_vals) * 1.24
ax2.set_ylim(0, ymax)

for i, v in enumerate(imp_vals):
    ax2.text(i, v + ymax * 0.025, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, clip_on=False)

# C. Flow performance trend
ax3 = fig.add_subplot(gs[0, 2])

x = np.arange(len(ablation))
lt = ablation[metric_map["Lead Time"]]
th = ablation[metric_map["Throughput"]]
short_names = ["Base", "Manual", "PM", "PM+XGB", "PM+DT", "Full"]

ax3.plot(x, lt, marker="o", linewidth=2.6, color="#E74C3C", label="Lead Time")
ax3.set_ylabel("Lead Time", color="#E74C3C")
ax3.tick_params(axis="y", labelcolor="#E74C3C")
ax3.set_xticks(x)
ax3.set_xticklabels(short_names)

ax3b = ax3.twinx()
ax3b.plot(x, th, marker="s", linewidth=2.6, color="#1E88E5", label="Throughput")
ax3b.set_ylabel("Throughput", color="#1E88E5")
ax3b.tick_params(axis="y", labelcolor="#1E88E5")

ax3.set_title("C. Flow Performance Trend", fontweight="bold")

# D. Current-state VSM time structure
ax4 = fig.add_subplot(gs[1, 0])

ct_col = vsm_metric_map["Cycle Time"]
wt_col = vsm_metric_map["Waiting Time"]

vsm_tmp = vsm.copy()
vsm_tmp["total_time"] = vsm_tmp[ct_col] + vsm_tmp[wt_col]
vsm_tmp = vsm_tmp.sort_values("total_time", ascending=True)

ax4.barh(vsm_tmp[activity_col], vsm_tmp[ct_col], label="Cycle Time", color="#3498DB")
ax4.barh(vsm_tmp[activity_col], vsm_tmp[wt_col], left=vsm_tmp[ct_col], label="Waiting Time", color="#F39C12")

ax4.set_title("D. Current-State VSM Time Structure", fontweight="bold")
ax4.set_xlabel("Minutes")
ax4.legend(frameon=False, loc="lower right")

# E. ML grouped performance
ax5 = fig.add_subplot(gs[1, 1])

ml_pivot_dash = ml_pivot.copy()
x = np.arange(len(ml_pivot_dash.index))
models = list(ml_pivot_dash.columns)
width = 0.8 / max(len(models), 1)

for j, model in enumerate(models):
    vals = ml_pivot_dash[model].values
    xpos = x - 0.4 + width / 2 + j * width
    ax5.bar(
        xpos,
        vals,
        width=width,
        label=model,
        color=ML_COLORS.get(model, "#7F8C8D"),
        edgecolor="black",
        linewidth=0.5
    )

ax5.set_title("E. ML Predictive Performance", fontweight="bold")
ax5.set_ylabel("F1-score")
ax5.set_ylim(0, 1.12)
ax5.set_xticks(x)
ax5.set_xticklabels(ml_pivot_dash.index)
ax5.legend(frameon=False, loc="upper left", ncol=len(models))

for container in ax5.containers:
    ax5.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

# F. Network criticality
ax6 = fig.add_subplot(gs[1, 2])

bet_col = centrality_cols["Betweenness"]
tmp = (
    network[[net_activity_col, bet_col]]
    .sort_values(bet_col, ascending=True)
    .tail(6)
)
colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(tmp)))

ax6.barh(
    tmp[net_activity_col],
    tmp[bet_col],
    color=colors,
    edgecolor="black",
    linewidth=0.5
)

ax6.set_title("F. Critical Activities\nby Betweenness", fontweight="bold", pad=10)
ax6.set_xlabel("Betweenness centrality")
ax6.set_xlim(0, tmp[bet_col].max() * 1.18)
ax6.tick_params(axis="y", labelsize=8)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for txt in ax.texts:
        txt.set_clip_on(False)

fig.subplots_adjust(
    top=0.88,
    bottom=0.09,
    left=0.10,
    right=0.97,
    hspace=0.55,
    wspace=0.52
)

plt.savefig(
    os.path.join(FIG_DIR, "paper_synthetic_results_final_dashboard.png"),
    bbox_inches="tight",
    dpi=500
)

plt.savefig(
    os.path.join(FIG_DIR, "paper_synthetic_results_final_dashboard.pdf"),
    bbox_inches="tight"
)

plt.close()

print("Publication-ready figures generated successfully.")
print(f"Main dashboard PNG: {FIG_DIR}/paper_synthetic_results_final_dashboard.png")
print(f"Main dashboard PDF: {FIG_DIR}/paper_synthetic_results_final_dashboard.pdf")
