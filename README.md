# A Data-Driven Value Stream Mapping and Design Framework Integrating Process Mining, Explainable Machine Learning, Digital Twins, and Complex Networks

This repository contains the complete experimental implementation supporting the paper:

> **A Data-Driven Value Stream Mapping and Design Framework Integrating Process Mining, Explainable Machine Learning, Digital Twins, and Complex Networks**

The proposed framework transforms traditional Value Stream Mapping (VSM) from a static diagnostic tool into a data-driven decision-support system capable of integrating Process Mining, Explainable Machine Learning, Digital Twins, and Complex Network Analysis for continuous process improvement and future-state Value Stream Design (VSD).

---

## Overview

The framework automatically:

- Discovers process flows from event logs using Process Mining.
- Generates data-driven current-state Value Stream Maps.
- Extracts operational and structural process indicators.
- Identifies bottlenecks, delays, defects, and rework through Machine Learning.
- Provides explainability using SHAP-based explanations.
- Evaluates alternative future-state designs through Digital Twin simulation.
- Quantifies structural process behavior using Complex Network Analysis.
- Supports evidence-based Value Stream Design decisions.

The repository contains all scripts required to reproduce the complete experimental workflow reported in the manuscript, including synthetic experimentation, real-world validation, ablation studies, statistical validation, figure generation, and paper artifacts.

---

# Experimental Evaluation

The framework was evaluated using:

## Synthetic Manufacturing Dataset

A synthetic event log specifically designed for:

- Controlled experimentation
- Current-state VSM generation
- Process Mining validation
- Digital Twin calibration
- Future-state scenario evaluation
- Ablation analysis
- Statistical validation

## Real-World Event Logs

The framework was further validated using:

- BPI Challenge 2017
- BPI Challenge 2019
- Data Quality Issues (DQI)

These datasets were used to evaluate:

- Cross-domain generalization
- Scalability
- Robustness under heterogeneous process conditions
- Predictive performance of network-enhanced machine learning models

---

# Repository Structure

```text
.
├── data/
│   ├── synthetic_event_log.csv
│   └── route_parameters.csv
│
├── datasets/
│   ├── BPI Challenge 2017
│   ├── BPI Challenge 2019
│   ├── Data Quality Issues
│   ├── extracted/
│   └── prepared/
│
├── src/
│
│   #Synthetic Manufacturing Pipeline
│   ├── 00_generate_synthetic_dataset.py
│   ├── 01_vsm_kpis.py
│   ├── 02_network_features.py
│   ├── 03_ml_experiments.py
│   ├── 04_digital_twin_scenarios.py
│   ├── 05_ablation_and_statistics.py
│
│   #Real-World Dataset Pipeline
│   ├── 07_prepare_real_datasets.py
│   ├── 08_run_dataset_experiment.py
│   ├── 09_real_dataset_ml.py
├── results/
│   ├── synthetic/
│   ├── bpi2017/
│   ├── bpi2019/
│   ├── data_quality_issues/
│   │
│   ├── ablation_results.csv
│   ├── case_summary.csv
│   ├── dfg_edges.csv
│   ├── ml_results.csv
│   ├── network_activity_features.csv
│   ├── real_datasets_ml_results.csv
│   ├── scenario_replications.csv
│   ├── scenario_results.csv
│   ├── statistical_tests.txt
│   └── vsm_stage_kpis.csv
│
├── pdf/
│   └── manuscript.pdf
│
├── README.md
└── requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-user/data-driven-vsm-vsd-framework.git

cd data-driven-vsm-vsd-framework
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Complete Experimental Pipeline

## 1. Generate Synthetic Manufacturing Data

```bash
python src/00_generate_synthetic_dataset.py
```

Generates:

- Synthetic event log
- Route parameters
- Ground-truth process information

---

## 2. Generate Process Mining and VSM Indicators

```bash
python src/01_vsm_kpis.py
```

Computes:

- Lead Time
- Cycle Time
- Waiting Time
- Work-In-Process
- Defect Rate
- Rework Rate
- Throughput

---

## 3. Extract Complex Network Features

```bash
python src/02_network_features.py
```

Builds:

- Activity Networks
- Resource Networks
- Value-Flow Networks

Extracts:

- Degree Centrality
- Betweenness Centrality
- Closeness Centrality
- Eigenvector Centrality
- Network Density

---

## 4. Run Explainable Machine Learning Experiments

```bash
python src/03_ml_experiments.py
```

Prediction tasks:

- Bottleneck Detection
- Delay Prediction
- Defect Prediction
- Rework Prediction

Explainability:

- SHAP Global Explanations
- SHAP Local Explanations
- Feature Importance Analysis

---

## 5. Evaluate Digital Twin Scenarios

```bash
python src/04_digital_twin_scenarios.py
```

Evaluates alternative future-state Value Stream Designs through Discrete-Event Simulation.

KPIs:

- Lead Time
- Throughput
- Work-In-Process
- Defect Rate
- Rework Rate

---

## 6. Run Ablation Study and Statistical Validation

```bash
python src/05_ablation_and_statistics.py
```

Includes:

- ANOVA
- Kruskal-Wallis Tests
- Tukey HSD Post-Hoc Analysis
- Confidence Intervals
- Scenario Comparisons

---

# Real-World Dataset Experiments

## Dataset Preparation

```bash
python src/07_prepare_real_datasets.py
```

## Dataset Execution

```bash
python src/08_run_dataset_experiment.py
```

## Real-World Machine Learning Evaluation

```bash
python src/09_real_dataset_ml.py
```

Datasets:

- BPI2017
- BPI2019
- Data Quality Issues (DQI)

Outputs:

```text
results/bpi2017/
results/bpi2019/
results/data_quality_issues/
```

---


# Main Outputs

## Process Mining and VSM

Generated KPIs:

- Lead Time (LT)
- Cycle Time (CT)
- Work-In-Process (WIP)
- Defect Rate (DR)
- Rework Rate (RR)
- Throughput (TH)

Files:

```text
results/vsm_stage_kpis.csv
results/case_summary.csv
```

---

## Complex Network Analysis

Generated Networks:

- Activity Network
- Resource Network
- Value-Flow Network

Metrics:

- Degree Centrality
- Betweenness Centrality
- Closeness Centrality
- Eigenvector Centrality
- Network Density

Files:

```text
results/network_activity_features.csv
```

---

## Explainable Machine Learning

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Files:

```text
results/ml_results.csv
results/real_datasets_ml_results.csv
```

---

## Digital Twin

Files:

```text
results/scenario_results.csv
results/scenario_replications.csv
```

---

## Statistical Validation

Files:

```text
results/statistical_tests.txt
results/ablation_results.csv
```

---

# Experimental Configurations

The ablation study evaluates the incremental contribution of each analytical layer.

| Configuration | Description |
|--------------|-------------|
| Baseline | Original Process |
| Manual VSM | Traditional Value Stream Mapping |
| PM-VSM | Process Mining + Automated VSM |
| PM-VSM-XGB | PM-VSM + Explainable Machine Learning |
| PM-VSM-XGB-DT | PM-VSM + Explainable Machine Learning + Digital Twin |
| Full Framework | PM + XML + DT + Complex Networks |

---

# Key Findings

Compared with the baseline process, the complete framework achieved:

| Indicator | Improvement |
|-----------|-------------|
| Lead Time Reduction | 32.40% |
| Throughput Increase | 71.73% |
| WIP Reduction | 45.09% |
| Defect Rate Reduction | 27.63% |
| Rework Rate Reduction | 45.52% |

Additional findings include:

- Consistent improvements in bottleneck prediction using network-derived features.
- Significant gains in rework prediction performance across heterogeneous datasets.
- Improved future-state process evaluation through Digital Twin simulation.
- Identification of structurally critical process activities through Complex Network Analysis.
- Demonstrated generalization across manufacturing and business process domains.

---

# Full Paper Reproducibility

This repository contains all scripts required to reproduce:

- Synthetic manufacturing experiments
- BPI2017 experiments
- BPI2019 experiments
- Data Quality Issues experiments
- Process Mining analyses
- Complex Network Analysis
- Explainable Machine Learning results
- Digital Twin scenarios
- Ablation study
- Statistical validation
- Paper tables
- Paper figures
- Results dashboard

---

# Contributors

This research was developed through a collaborative effort between:

- Universidad de las Américas Puebla (UDLAP), Mexico
- Hochschule Bielefeld (HSBI), Germany
- Universidad Autónoma Metropolitana Azcapotzalco (UAM-A), Mexico

---

# Authors

### Edwin Montes-Orozco
Department of Computing, Electronics and Mechatronics  
Universidad de las Américas Puebla (UDLAP), Mexico

### Mariam Dopslaf
Department of Industrial Engineering  
Hochschule Bielefeld (HSBI), Germany

### Roman Anselmo Mora-Gutiérrez
Department of Systems  
Universidad Autónoma Metropolitana Azcapotzalco (UAM-A), Mexico

### Sergio Gerardo de-los-Cobos-Silva
Department of Electrical Engineering  
Universidad Autónoma Metropolitana Azcapotzalco (UAM-A), Mexico

### Eric Alfredo Rincón-García
Department of Electrical Engineering  
Universidad Autónoma Metropolitana Azcapotzalco (UAM-A), Mexico

### Pedro Lara-Velázquez
Department of Electrical Engineering  
Universidad Autónoma Metropolitana Azcapotzalco (UAM-A), Mexico

### Miguel Ángel Gutiérrez-Andrade
Department of Electrical Engineering  
Universidad Autónoma Metropolitana Azcapotzalco (UAM-A), Mexico

---

# Citation

```bibtex
@article{MontesOrozco2026VSM,
  title={A Data-Driven Value Stream Mapping and Design Framework Integrating Process Mining, Explainable Machine Learning, Digital Twins, and Complex Networks},
  author={
    Montes-Orozco, Edwin and
    Dopslaf, Mariam and
    Mora-Gutiérrez, Roman Anselmo and
    de-los-Cobos-Silva, Sergio Gerardo and
    Rincón-García, Eric Alfredo and
    Lara-Velázquez, Pedro and
    Gutiérrez-Andrade, Miguel Ángel
  },
  year={2026}
}
```

---

# License

This repository is released for academic and research purposes.
