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

def get_ml_row(dataset_name, target_name, model_name):
    rows = real_ml[
        (real_ml["Dataset"] == dataset_name) &
        (real_ml["Target"] == target_name) &
        (real_ml["Model"] == model_name)
    ]
    if rows.empty:
        raise ValueError(
            f"Missing ML result for dataset={dataset_name}, "
            f"target={target_name}, model={model_name}"
        )
    return rows.iloc[0]


# Random Forest is used for matched-model examples in the narrative.
bpi2017_delay_rf = get_ml_row("BPI2017", "Delay", "Random Forest")
bpi2017_bottleneck_rf = get_ml_row("BPI2017", "Bottleneck", "Random Forest")
bpi2017_rework_rf = get_ml_row("BPI2017", "Rework", "Random Forest")

bpi2019_delay_rf = get_ml_row("BPI2019", "Delay", "Random Forest")
bpi2019_bottleneck_rf = get_ml_row("BPI2019", "Bottleneck", "Random Forest")
bpi2019_rework_rf = get_ml_row("BPI2019", "Rework", "Random Forest")

dqi_delay_rf = get_ml_row("Data Quality Issues", "Delay", "Random Forest")
dqi_bottleneck_rf = get_ml_row("Data Quality Issues", "Bottleneck", "Random Forest")
dqi_rework_rf = get_ml_row("Data Quality Issues", "Rework", "Random Forest")


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

Table~\ref{{tab:real_ml}} reports matched-model comparisons for the real-world
datasets, contrasting each classifier without and with Complex Network Analysis
(CNA) features. The evaluation follows a chronological 70/15/15 split. Network
representations and proxy thresholds are estimated exclusively from the training
partition. In addition, variables used directly to construct each operational
proxy are excluded from the corresponding predictor set. Delay is defined from
the training-only 90th percentile of waiting time and therefore
\texttt{{waiting\_time\_min}} is excluded from delay prediction. Bottlenecks are
defined at the activity level from training-only mean cycle and waiting times;
consequently, both \texttt{{cycle\_time\_min}} and
\texttt{{waiting\_time\_min}} are excluded from bottleneck prediction. The
rework target is obtained from the observed \texttt{{rework\_flag}}, which is
not included among the predictors.

\begin{{table}}[htbp]
\centering
\caption{{Matched-model machine learning performance on real-world datasets
without and with CNA features. $\Delta$F1 denotes the absolute difference
between the CNA and no-CNA configurations. N/A indicates that the target was
not evaluable because the held-out task became single-class under the
training-derived operational proxy definition.}}
\label{{tab:real_ml}}
\input{{results/paper_tables/table_real_ml_results.tex}}
\end{{table}}

The predictive tasks were not equally supported by all event logs. In BPI2017
and BPI2019, the available waiting-time signal lacks sufficient variability
under the adopted preprocessing and timestamp semantics to construct
informative delay and operational bottleneck targets. These comparisons are
therefore reported as N/A rather than being interpreted as predictive results.
This behavior illustrates an important practical limitation of data-driven
VSM/VSD: the analytical tasks that can be supported depend on the semantics and
information content of the available event log.

The Data Quality Issues dataset supports all three operational prediction
tasks. For Random Forest, adding CNA features increases delay F1-score from
{dqi_delay_rf["F1 (No CNA)"]:.4f} to {dqi_delay_rf["F1 (CNA)"]:.4f}
($\Delta$F1={dqi_delay_rf["Delta F1"]:.4f}) and bottleneck F1-score from
{dqi_bottleneck_rf["F1 (No CNA)"]:.4f} to
{dqi_bottleneck_rf["F1 (CNA)"]:.4f}
($\Delta$F1={dqi_bottleneck_rf["Delta F1"]:.4f}). For rework, the improvement
is smaller, from {dqi_rework_rf["F1 (No CNA)"]:.4f} to
{dqi_rework_rf["F1 (CNA)"]:.4f}
($\Delta$F1={dqi_rework_rf["Delta F1"]:.4f}). Because the delay and bottleneck
targets are defined from operational variables that are excluded from their
respective predictor sets, these results provide evidence that structural
network descriptors can supply complementary information for selected
operational diagnosis tasks.

Rework prediction can also be evaluated in BPI2017 and BPI2019. Using Random
Forest, the F1-score changes from
{bpi2017_rework_rf["F1 (No CNA)"]:.4f} to
{bpi2017_rework_rf["F1 (CNA)"]:.4f} in BPI2017 and from
{bpi2019_rework_rf["F1 (No CNA)"]:.4f} to
{bpi2019_rework_rf["F1 (CNA)"]:.4f} in BPI2019. These large gains require
cautious interpretation. In BPI2017, the observed rework indicator is
activity-specific, with rework events concentrated in \emph{{O\_Returned}}.
BPI2019 similarly associates rework with a subset of change-related
activities. Since activity-level network descriptors can act as structural
signatures of activity identity, part of the predictive gain can reflect this
activity--target association rather than a generally transferable causal
relationship between network centrality and rework. Accordingly, these two
results are retained as dataset-specific predictive findings but are not used
as the primary evidence for cross-domain generalization.

Table~\ref{{tab:synthetic_ml}} reports the corresponding results for the
synthetic manufacturing dataset. These experiments provide a controlled
setting in which the relationships among operational variables, structural
descriptors, and predictive targets are explicitly modeled.

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

Because the ablation configurations are cumulative, the observed progression should be interpreted as framework-level incremental performance rather than as isolated causal effects attributable to individual analytical components. Process mining provides the empirical flow representation, VSM indicators summarize operational waste and flow efficiency, machine learning supports predictive diagnosis, the digital twin evaluates future-state alternatives, and complex networks provide structural descriptors that complement local KPI analysis. A factorial or component-removal design would be required to quantify the independent and interaction effects of these capabilities.

\subsection{{Discussion}}

The results suggest that the proposed data-driven VSM/VSD framework offers
three main advantages over traditional VSM practice. First, it transforms
event logs into reproducible operational indicators and current-state
representations, reducing dependence on manual data collection. Second, it
complements local operational indicators with structural descriptors derived
from activity networks. Third, it connects diagnosis with redesign by
evaluating candidate interventions through Digital Twin simulation before
their outcomes are mapped to a Future-State Value Stream Design.

The real-world experiments also show that predictive validation in
process-oriented analytics is constrained by event-log semantics. BPI2017 and
BPI2019 do not provide sufficient waiting-time variability, under the adopted
event interpretation, to support informative delay and operational bottleneck
targets. Reporting these tasks as unavailable avoids creating artificial
classification problems from degenerate proxies. In contrast, the Data
Quality Issues dataset provides sufficient variability to evaluate delay,
bottleneck, and rework prediction under the same leakage-controlled temporal
protocol. The improvements obtained after adding CNA features in this dataset
indicate that structural descriptors can provide information complementary to
the operational predictors for selected tasks.

The rework experiments further illustrate why high predictive performance
must be interpreted in relation to label construction and process semantics.
In BPI2017, rework is concentrated in the activity \emph{{O\_Returned}}, while
in BPI2019 it is associated with a subset of change-related activities.
Consequently, activity-level network descriptors can encode structural
signatures that make these labels highly separable. The resulting gains are
valid for the evaluated event logs, but they should not be interpreted as
evidence that the same level of rework prediction will generalize to processes
where rework is not strongly associated with activity identity. The more
heterogeneous Data Quality Issues dataset therefore provides the more
conservative evidence of incremental CNA value for rework prediction.

The simulation experiments provide a complementary form of validation. The
cumulative configurations show progressive improvements in lead time,
throughput, WIP, defect rate, and rework rate as analytical capabilities are
incorporated into the workflow. However, because the ablation sequence is
cumulative, these differences quantify framework-level performance across
successive configurations rather than the isolated causal contribution of
each component. Stronger attribution would require factorial,
component-removal, or interaction-focused experimental designs.

Several limitations should therefore be acknowledged. First, the delay and
bottleneck labels are operational proxies rather than externally validated
ground truth. Second, the availability of informative proxies depends on event
attributes and timestamp semantics, as demonstrated by the unavailable delay
and bottleneck tasks in BPI2017 and BPI2019. Third, activity-level structural
features may partially encode activity identity, particularly when target
labels are concentrated in specific activities. Fourth, the Digital Twin
evaluation is conducted in a controlled synthetic manufacturing setting and
does not yet constitute an implemented industrial intervention. Finally, the
current experiments evaluate an offline analytical workflow; recurrent or
online recalibration remains a future extension.

Despite these limitations, the combined results support the feasibility of a
traceable analytical path from event-log reconstruction to diagnosis,
simulation-based intervention evaluation, and Future-State Value Stream
Design. The evidence does not imply that every analytical capability produces
uniform gains for every dataset or task. Rather, it shows how Process Mining,
Explainable Machine Learning, Complex Network Analysis, and Digital Twin
simulation can be organized as complementary decision-support mechanisms,
with their applicability and contribution evaluated according to the
information available in each operational context.
"""

OUT.write_text(tex.strip() + "\n", encoding="utf-8")

print(f"Generated: {OUT}")
