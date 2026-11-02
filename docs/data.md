# Data and operational variables

## Inputs

| File | Description |
|---|---|
| `data/synthetic_event_log.csv` | Synthetic manufacturing events and flags |
| `data/route_parameters.csv` | Seven route stations and sampling parameters |
| `datasets/prepared/bpi2017_prepared.csv` | Prepared loan-application events |
| `datasets/prepared/bpi2019_prepared.csv` | Prepared purchase-order line-item events |
| `datasets/prepared/data_quality_issues_prepared.csv` | Prepared event log with quality issues |

The five exact input files are recorded with byte counts and SHA-256 digests in `datasets/input_manifest.json`. The synthetic log contains eight observed activities because rework may be inserted in addition to the seven route stations.

The BPI benchmark names identify the underlying business-process domains. The exact raw-to-prepared transformations for the two included BPI CSVs are not established by the generic XES adapter in this repository. Reproduction of the included numerical analyses starts from the prepared CSV files; it does not establish independent reconstruction from raw benchmark downloads.

The original public BPI2017 ZIP is retained in `datasets/raw/`. It is not read by the standard reproduction workflow. The prepared DQI file does not include an independently verified source record identifying its original release; interpret it as the included prepared input, not as a complete raw-data provenance chain.

## Common fields

| Field | Interpretation |
|---|---|
| `case_id`, `event_id` | Process-case and event identifiers |
| `activity`, `resource_id` | Recorded activity and resource identifiers |
| `start_ts`, `end_ts` | Timestamp fields parsed in UTC with mixed-format support |
| `cycle_time_min`, `waiting_time_min` | Supplied durations or derived operational proxies |
| `queue_before`, `wip_before` | Supplied queue/workload fields; constant values do not establish physical absence |
| `defect_flag`, `rework_flag` | Supplied diagnostic labels |

The analysis reader orders events stably by case and start time for activity transitions. The model reader then uses a stable chronological row order for a 70/15/15 split. A row split is not a partition of complete cases.

## Labels and domain interpretation

- In synthetic inputs, `rework_flag` identifies the `Rework` activity.
- In BPI2017, positive prepared rework labels identify `O_Returned`.
- In BPI2019, prepared rework labels identify a set of change activities.
- In DQI, the prepared rework flag corresponds to a repeated activity within a case.

An administrative active-case workload is not physical inventory. Administrative changes and repetition are not automatically defective products or confirmed corrective work. Inter-event gaps can include multiple process effects and should not be interpreted as measured machine service without lifecycle evidence.

BPI2019 contains unusually early timestamp values in the included prepared file. They remain in the input rather than being silently removed; derived calendar spans are not treated as validated industrial observation horizons.

## Benchmark attribution

- van Dongen, B. (2017). *BPI Challenge 2017*. 4TU.Centre for Research Data. [Dataset DOI](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b).
- van Dongen, B. (2019). *BPI Challenge 2019*. 4TU.Centre for Research Data. [Dataset DOI](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1).

These records describe the original benchmarks. The prepared subsets and diagnostic labels used here are identified by the repository input manifest and the scope described above.
