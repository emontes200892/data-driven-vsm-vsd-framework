from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "paper_tables"
OUT = TABLES / "11_results_discussion.tex"

TABLES.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)


dataset = read_csv("table_dataset_characteristics.csv")
vsm = read_csv("table_vsm_kpis.csv")
network = read_csv("table_network_metrics.csv")
real_ml = read_csv("table_real_ml_results.csv")
synthetic_ml = read_csv("table_synthetic_ml_results.csv")
ablation = read_csv("table_ablation.csv")
scenarios = read_csv("table_scenarios.csv")


def get_row(df, col, value):
    return df[df[col] == value].iloc[0]


bpi2017 = get_row(vsm, "Dataset", "BPI2017")
bpi2019 = get_row(vsm, "Dataset", "BPI2019")
dqi = get_row(vsm, "Dataset", "Data Quality Issues")

full = get_row(ablation, "Scenario", "Full Framework")
baseline = get_row(ablation, "Scenario", "Baseline")

bpi2019_rework = real_ml[
    (real_ml["Dataset"] == "BPI2019") &
    (real_ml["Target"] == "Rework")
].iloc[0]

dqi_rework = real_ml[
    (real_ml["Dataset"] == "Data Quality Issues") &
    (real_ml["Target"] == "Rework")
].iloc[0]


tex = rf"""
\section{{Results and Discussion}}
\label{{sec:results_discussion}}

This section reports the experimental evaluation of the proposed AI-assisted VSM/VSD framework. The analysis combines one synthetic manufacturing event log and three publicly available real-world event logs: BPI2017, BPI2019, and Data Quality Issues. The evaluation focuses on five aspects: dataset characterization, process mining and VSM indicators, complex network structure, machine learning performance, and scenario-based improvement through the digital twin.

\subsection{{Dataset Characterization}}

Table~\ref{{tab:dataset_characteristics}} summarizes the datasets used in the empirical evaluation. The real-world datasets differ considerably in size and process structure. BPI2017 contains {int(get_row(dataset, "Dataset", "BPI2017")["Events"]):,} events and {int(get_row(dataset, "Dataset", "BPI2017")["Cases"]):,} cases, while BPI2019 contains {int(get_row(dataset, "Dataset", "BPI2019")["Events"]):,} events and {int(get_row(dataset, "Dataset", "BPI2019")["Cases"]):,} cases. The Data Quality Issues dataset is smaller, with {int(get_row(dataset, "Dataset", "Data Quality Issues")["Events"]):,} events and {int(get_row(dataset, "Dataset", "Data Quality Issues")["Cases"]):,} cases, but it exhibits a substantially different behavioral profile due to its high degree of repeated activity execution.

\begin{{table}}[htbp]
\centering
\caption{{Dataset characteristics.}}
\label{{tab:dataset_characteristics}}
\input{{results/paper_tables/table_dataset_characteristics.tex}}
\end{{table}}

The inclusion of heterogeneous datasets is relevant because the proposed framework is not intended to depend on a single process domain. Instead, it aims to provide a reusable analytical pipeline for converting event logs into operational indicators, structural network descriptors, predictive models, and scenario-based redesign alternatives.

\subsection{{Process Mining and VSM Analysis}}

Table~\ref{{tab:vsm_kpis}} reports the VSM-related indicators extracted from the event logs. BPI2017 and BPI2019 show long mean case lead times of {bpi2017["Mean Lead Time (min)"]:.2f} and {bpi2019["Mean Lead Time (min)"]:.2f} minutes, respectively. Their value-added ratios are relatively low, with {bpi2017["VA Ratio (%)"]:.2f}\% for BPI2017 and {bpi2019["VA Ratio (%)"]:.2f}\% for BPI2019. These values indicate that most of the observed case duration is dominated by non-value-added time, queueing, synchronization delays, or organizational waiting.

\begin{{table}}[htbp]
\centering
\caption{{VSM-based operational indicators.}}
\label{{tab:vsm_kpis}}
\input{{results/paper_tables/table_vsm_kpis.tex}}
\end{{table}}

The Data Quality Issues dataset presents a contrasting pattern. Its mean lead time is {dqi["Mean Lead Time (min)"]:.2f} minutes and its value-added ratio is {dqi["VA Ratio (%)"]:.2f}\%. However, the rework rate reaches {dqi["Rework Rate (%)"]:.2f}\%, suggesting strong process instability and repeated execution. This behavior is particularly relevant from a Lean perspective because high rework directly increases flow variability, resource load, and operational waste.

Overall, the VSM indicators show that each dataset represents a different type of improvement problem. BPI2017 and BPI2019 are mainly characterized by extended lead times and low value-added ratios, whereas the Data Quality Issues dataset is mainly characterized by repeated execution and rework. This supports the need for a framework that combines flow-based, structural, and predictive perspectives rather than relying only on traditional VSM indicators.

\subsection{{Complex Network Analysis}}

The complex network analysis complements the VSM indicators by representing activities as nodes and observed transitions as directed edges. Table~\ref{{tab:network_metrics}} reports the global network properties for each dataset.

\begin{{table}}[htbp]
\centering
\caption{{Global activity-network metrics.}}
\label{{tab:network_metrics}}
\input{{results/paper_tables/table_network_metrics.tex}}
\end{{table}}

BPI2017 exhibits {int(get_row(network, "Dataset", "BPI2017")["Nodes"])} nodes and {int(get_row(network, "Dataset", "BPI2017")["Edges"])} edges, with a density of {get_row(network, "Dataset", "BPI2017")["Density"]:.4f}. BPI2019 contains {int(get_row(network, "Dataset", "BPI2019")["Nodes"])} nodes and {int(get_row(network, "Dataset", "BPI2019")["Edges"])} edges, with a lower density of {get_row(network, "Dataset", "BPI2019")["Density"]:.4f}. The Data Quality Issues dataset contains {int(get_row(network, "Dataset", "Data Quality Issues")["Nodes"])} nodes and {int(get_row(network, "Dataset", "Data Quality Issues")["Edges"])} edges, with a density of {get_row(network, "Dataset", "Data Quality Issues")["Density"]:.4f}.

The identified critical activities provide a structural interpretation of the process behavior. In BPI2017, the most critical activity is \emph{{{get_row(network, "Dataset", "BPI2017")["Critical Activity"]}}}; in BPI2019, it is \emph{{{get_row(network, "Dataset", "BPI2019")["Critical Activity"]}}}; and in the Data Quality Issues dataset, it is \emph{{{get_row(network, "Dataset", "Data Quality Issues")["Critical Activity"]}}}. These activities act as structurally relevant connectors in their corresponding process networks. Therefore, they are natural candidates for deeper operational analysis, redesign, capacity adjustment, or standardization.

\subsection{{Machine Learning Evaluation}}

Table~\ref{{tab:real_ml}} summarizes the best predictive performance obtained in the real-world datasets with and without network features. The delay proxy is not emphasized as a major contribution because several models achieved nearly perfect performance even without network information. This suggests that delay behavior, as operationalized in the current proxy, is already strongly captured by conventional event-log features.

\begin{{table}}[htbp]
\centering
\caption{{Best machine learning performance on real-world datasets.}}
\label{{tab:real_ml}}
\input{{results/paper_tables/table_real_ml_results.tex}}
\end{{table}}

The most relevant improvements appear in bottleneck and rework-related tasks. For BPI2019 rework prediction, the best F1-score increased from {bpi2019_rework["F1 (No Network)"]:.4f} without network features to {bpi2019_rework["F1 (Network)"]:.4f} with network features, corresponding to an improvement of {bpi2019_rework["Improvement (%)"]:.2f}\%. In the Data Quality Issues dataset, rework prediction also improved from {dqi_rework["F1 (No Network)"]:.4f} to {dqi_rework["F1 (Network)"]:.4f}. Although this gain is smaller, it is consistent with the hypothesis that network descriptors add useful structural information.

The bottleneck proxy shows near-perfect performance when network features are included. This result should be interpreted carefully. Since the bottleneck proxy is partially derived from structural network properties, the corresponding models validate the ability of centrality-based features to encode process criticality, but they should not be overinterpreted as fully independent predictive evidence. For this reason, the strongest empirical evidence for the predictive value of network features comes from the rework-related tasks, particularly in BPI2019.

Table~\ref{{tab:synthetic_ml}} reports the corresponding results for the synthetic manufacturing dataset. These results provide a controlled setting in which the relationship between operational variables, structural descriptors, and predictive targets is explicitly modeled.

\begin{{table}}[htbp]
\centering
\caption{{Best machine learning performance on the synthetic dataset.}}
\label{{tab:synthetic_ml}}
\input{{results/paper_tables/table_synthetic_ml_results.tex}}
\end{{table}}

\subsection{{Digital Twin and Scenario-Based Evaluation}}

The digital twin module was evaluated through scenario-based simulation. Table~\ref{{tab:scenario_results}} reports the operational performance of each scenario. The scenarios represent progressive levels of analytical support, from a baseline configuration to the full AI-assisted framework.

\begin{{table}}[htbp]
\centering
\caption{{Digital twin scenario results.}}
\label{{tab:scenario_results}}
\input{{results/paper_tables/table_scenarios.tex}}
\end{{table}}

Compared with the baseline, the full framework reduced lead time from {baseline["Lead Time"]:.2f} to {full["Lead Time"]:.2f}, increased throughput from {baseline["Throughput"]:.2f} to {full["Throughput"]:.2f}, and reduced WIP from {baseline["WIP"]:.2f} to {full["WIP"]:.2f}. These changes represent a lead time reduction of {full["Lead Time Reduction (%)"]:.2f}\%, a throughput gain of {full["Throughput Gain (%)"]:.2f}\%, and a WIP reduction of {full["WIP Reduction (%)"]:.2f}\%.

The full framework also reduced the defect rate from {baseline["Defect Rate (%)"]:.2f}\% to {full["Defect Rate (%)"]:.2f}\%, and the rework rate from {baseline["Rework Rate (%)"]:.2f}\% to {full["Rework Rate (%)"]:.2f}\%. These results indicate that the framework does not only improve time-based flow indicators, but also contributes to quality-related performance.

\subsection{{Ablation Study}}

Table~\ref{{tab:ablation}} reports the ablation results. The progressive inclusion of process mining, predictive modeling, digital twin simulation, and complex network analysis produces a monotonic improvement across the main performance indicators.

\begin{{table}}[htbp]
\centering
\caption{{Ablation study of the proposed framework.}}
\label{{tab:ablation}}
\input{{results/paper_tables/table_ablation.tex}}
\end{{table}}

The baseline scenario produced a lead time of {baseline["Lead Time"]:.2f}, throughput of {baseline["Throughput"]:.2f}, and WIP of {baseline["WIP"]:.2f}. In contrast, the full framework achieved a lead time of {full["Lead Time"]:.2f}, throughput of {full["Throughput"]:.2f}, and WIP of {full["WIP"]:.2f}. This corresponds to a {full["Lead Time Reduction (%)"]:.2f}\% lead time reduction, {full["Throughput Gain (%)"]:.2f}\% throughput gain, and {full["WIP Reduction (%)"]:.2f}\% WIP reduction.

These results support the central premise of the paper: each analytical layer contributes complementary information. Process mining provides the empirical flow representation, VSM indicators summarize operational waste and flow efficiency, machine learning supports predictive diagnosis, the digital twin evaluates future-state alternatives, and complex networks identify structurally critical activities that may not be evident from local KPI analysis alone.

\subsection{{Discussion}}

The results suggest that the proposed AI-assisted VSM/VSD framework offers three main advantages over traditional VSM practice. First, it transforms event logs into reproducible VSM indicators, reducing the subjectivity associated with manual data collection. Second, it complements local operational indicators with global structural descriptors from complex networks. Third, it connects diagnosis with redesign by evaluating improvement scenarios through a digital twin.

The empirical findings also show that the value of each analytical module depends on the process problem being analyzed. For lead-time-dominated processes such as BPI2017 and BPI2019, VSM indicators are useful for exposing non-value-added time and low value-added ratios. For instability-dominated processes such as Data Quality Issues, rework indicators and repeated activity execution become more relevant. For structurally complex processes, network centrality measures help identify activities that mediate flow propagation and may act as systemic bottlenecks.

At the same time, some limitations should be acknowledged. First, the delay and bottleneck labels in the real-world datasets are proxy variables rather than externally validated ground truth. Second, bottleneck prediction results must be interpreted cautiously when the target definition is derived from network centrality measures. Third, waiting-time estimation depends on event-log quality and timestamp semantics; in some datasets, zero waiting times may reflect preprocessing artifacts rather than true operational absence of waiting. Finally, the digital twin experiments are currently evaluated in a controlled synthetic setting and should be further validated with domain-specific operational constraints.

Despite these limitations, the results provide consistent evidence that integrating process mining, VSM, machine learning, digital twins, and complex networks can support a more systematic and data-driven approach to current-state diagnosis and future-state design.
"""

OUT.write_text(tex.strip() + "\n", encoding="utf-8")

print(f"Generated: {OUT}")
