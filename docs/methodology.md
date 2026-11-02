# Methods and interpretation

## Event-log analysis

Operational summaries and directly-follows graphs are calculated from the included event logs. NetworkX computes activity centralities. The full-log descriptive network is separate from the training-derived graph used by the classifiers. Validation and test rows receive the frozen training graph descriptors, joined by activity; unseen activities receive zero-valued descriptors.

## Diagnostic classification

Random Forest and XGBoost use fixed settings recorded in source. The chronological row split assigns 70% to training, 15% to a reserved validation slice, and 15% to testing. The validation slice is not used for hyperparameter selection. Mixed-format timestamps are parsed in UTC and ordering ties are stable.

The delay proxy excludes waiting time from its predictors. The activity-level bottleneck proxy uses training-derived mean cycle or waiting times and excludes both durations from the model predictors. Controls compare operational attributes, network attributes, activity-majority rules, and one-hot activity classifiers.

BPI2017 and BPI2019 labels are activity-defined in the prepared data. Their activity-only controls attain F1 = 1.0000. Graph centralities can encode activity identity; high diagnostic accuracy is not proof of prospective prediction. Completion or next-event durations may also be unavailable at an earlier decision time. Cases can overlap across temporal row partitions, so unseen-case generalization is not established.

## Monte Carlo scenarios

The sampler draws independent orders using NumPy. It does not implement resource contention, a shared queue, or a discrete-event scheduling engine. Six configurations apply the following predefined factors:

| ID | Stored label | Cycle | Waiting | Defect | Additional factor |
|---|---|---:|---:|---:|---:|
| G0 | Baseline | 1.00 | 1.00 | 1.00 | 0.00 |
| G1 | Manual VSM | 0.98 | 0.92 | 0.97 | 0.00 |
| G2 | PM-VSM | 0.96 | 0.85 | 0.95 | 0.00 |
| G3 | PM-VSM-XGB | 0.92 | 0.78 | 0.88 | 0.00 |
| G4 | PM-VSM-XGB-DT | 0.88 | 0.68 | 0.82 | 0.02 |
| G5 | Full Framework | 0.84 | 0.58 | 0.72 | 0.06 |

Stored labels preserve compatibility with the numerical outputs. They describe scenario assumptions, not empirical estimates of a module's contribution. The sampler does not estimate factors from process-mining, classification, SHAP, or network outputs.

The workload proxy is a nonnegative random quantity centered on sampled waiting time divided by eight and scaled by the additional factor. The throughput proxy divides 500 orders by a constructed calendar based on 7.5 minutes per order times the waiting factor plus mean lead time. Neither quantity is a physical queue count from a resource-constrained simulator.

## Statistical comparisons

Replication numbers form shared seed blocks across scenarios. Friedman tests compare configurations. Two-sided Wilcoxon comparisons against baseline use Holm adjustment across five comparisons within each KPI. Paired-block bootstrap intervals use 10,000 resamples with seed 20261024. The correction is within each KPI family, not global across all indicators.

These procedures characterize stochastic variation conditional on the sampler and predefined factors. They do not quantify plant-calibration uncertainty, physical-system fidelity, causal intervention effects, or environmental gains.

## Architecture extensions

The evaluated code provides activity-transition analysis, diagnostic classifiers, and scenario sampling. Explanation-driven intervention selection, resource and value-flow networks, conformance checking, resource-constrained simulation, physical implementation, and live synchronization require additional implementation and evidence.
