# A Closed-Loop Data-Driven Framework for Value Stream Mapping and Future-State Design

Research materials for the manuscript submitted to *The International Journal of Advanced Manufacturing Technology* (JAMT-D-26-06306).

## Revision materials

The materials supporting the revised analyses are in [`revision_2026_11_02/`](revision_2026_11_02/). Legacy files at the repository root are retained as the original implementation. Use the revision directory for corrected analyses and the evidence boundaries reported in the revised manuscript.

- [`README.md`](revision_2026_11_02/README.md): package contents and reproduction instructions.
- [`AUDIT_REPORT.md`](revision_2026_11_02/AUDIT_REPORT.md): executed checks and limitations.
- `code/scripts/`: corrected implementation and diagnostic controls.
- `code/original_scripts/`: original scripts retained for comparison.
- `code/data/` and `code/datasets/prepared/`: synthetic inputs and the exact prepared CSVs used in the analyses.
- `code/results/` and `results/`: corrected outputs, original reproductions, activity controls, predictions, and scenario statistics.
- `input_manifest.json` and `source_hashes.json`: input and source hashes from the audit.
- `manuscript/`: editable manuscript sources and figures. Its Data Availability statement points to this repository directory.

## What was evaluated

The numerical demonstration evaluates activity-transition networks, retrospective diagnostic-label classification, and offline Monte Carlo scenarios with predefined factors. The synthetic scenario comparison reproduces reductions of 32.40% in sampled lead time, 45.09% in a workload proxy, 27.63% in defect rate, and 45.52% in rework rate, with a 71.73% increase in a calendar-based throughput proxy. These are responses to specified simulation factors, not measured factory improvements or causal estimates of module effects.

BPI2017 perfect rework classification is also reproduced by activity-only controls because the prepared target identifies `O_Returned`. These results do not demonstrate prospective prediction or incremental structural information beyond activity identity. Temporal row splitting does not establish unseen-case generalization.

SHAP-driven intervention selection, resource/value-flow networks, resource-constrained simulation, live digital-twin synchronization, and physical feedback remain architectural or deployment extensions rather than evaluated capabilities.

## Reproduce the corrected analyses

From the repository root, with a Python version compatible with the pinned packages:

```bash
python3 -m venv .venv-ijamt
source .venv-ijamt/bin/activate
python -m pip install -r revision_2026_11_02/code/requirements-tested.txt
cd revision_2026_11_02/code
python run_audit.py
```

The runner uses the included prepared inputs and recalculates result files. See the revision README and audit report for the original and corrected protocols, input provenance limitations, and the separate optional XES conversion path.

## Citation

Montes-Orozco, E., Dopslaf, M., Mora-Gutiérrez, R. A., de-los-Cobos-Silva, S. G., Rincón-García, E. A., Lara-Velázquez, P., and Gutiérrez-Andrade, M. Á. (2026). *A Closed-Loop Data-Driven Framework for Value Stream Mapping and Future-State Design*. Revised manuscript.
