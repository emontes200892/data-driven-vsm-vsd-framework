# Reproducing the experiments

## Environment

The numerical workflow uses Python 3.12 and the pinned packages in `requirements.txt`. Install them in a virtual environment. The commands in the README use a POSIX shell and also work on Debian. XGBoost is configured for CPU execution; no CUDA device is required.

## Stages

`python reproduce.py` executes all stages in dependency order. Every invocation checks the five input files against `datasets/input_manifest.json`. Individual stages use existing upstream outputs; execute `all` when rebuilding from inputs.

| Stage | Commands invoked | Main outputs |
|---|---|---|
| `verify` | SHA-256 and byte-count checks | Console verification |
| `synthetic` | KPI extraction, descriptive activity graph, synthetic ML | `results/ml_results.csv`, synthetic predictions and protocol |
| `scenarios` | Six Monte Carlo scenarios | `scenario_results.csv`, `scenario_replications.csv` |
| `real` | Event-log indicators and classifiers | Per-dataset result folders and `real_datasets_ml_results.csv` |
| `controls` | Operational, network, and activity-only diagnostic controls | `results/controls/<dataset>/` |
| `statistics` | Friedman, paired Wilcoxon, Holm, paired bootstrap | `results/paired_statistics/` |
| `tables` | CSV and LaTeX export | `results/tables/` |
| `figures` | CSV-derived figure inputs and rendering | `results/figure_data.json`, `results/figures/` |

`--dry-run` verifies inputs and prints the commands without running the calculations. Subprocess errors stop the workflow immediately.

Outputs are recalculated in place. Historical numerical comparisons in `results/reference/` are not used as inputs to the active workflow and are not overwritten by the runner.

## Randomness and statistics

Scenario replication `r` uses seed `1000 + r` across configurations. There are 30 replications and 500 independently sampled orders per configuration and replication. The paired bootstrap uses 10,000 resamples and seed 20261024. Classification parameters and random seeds are explicit in the source. The validation slice is reserved but is not used for model tuning.

## Optional data generation and import

`python src/generate_synthetic_data.py` recreates the synthetic input files and overwrites `data/`. It is excluded from the standard reproduction command because delivered inputs are verified rather than silently regenerated.

The generic XES adapter is a separate optional tool. Its import dependency is listed in `requirements-xes.txt`. Read its source configuration and the data documentation before using it: inter-event gaps do not establish measured service or queue durations, and this adapter is not the documented origin of the included BPI prepared inputs.

## Validation

```bash
python -m compileall -q src scripts reproduce.py
python -m unittest discover -s tests -v
```

Continuous integration checks input integrity, reruns the scenario sampler and seed-blocked statistics, and tests mixed-format timestamp handling, scenario seed blocks, numerical aggregation, and the headline scenario percentages. Full model fitting can be run locally through `python reproduce.py`.
