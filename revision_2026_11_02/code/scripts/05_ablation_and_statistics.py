"""Preserve scenario summaries and run the seed-blocked statistical analysis."""
from pathlib import Path
import importlib.util,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
shutil.copyfile(OUT/'scenario_results.csv',OUT/'ablation_results.csv')
spec=importlib.util.spec_from_file_location('paired_statistics',Path(__file__).with_name('15_paired_scenario_statistics.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.analyze(OUT/'scenario_replications.csv',OUT/'paired_statistics')
