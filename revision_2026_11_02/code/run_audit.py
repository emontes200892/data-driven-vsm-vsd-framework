"""Reproduce the corrected diagnostics and scenario analysis from included inputs."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
commands=[[f'scripts/{n}'] for n in ['01_vsm_kpis.py','02_network_features.py','03_ml_experiments.py','04_digital_twin_scenarios.py','05_ablation_and_statistics.py','08_run_dataset_experiment.py','09_real_dataset_ml.py','10_generate_paper_tables.py']]
commands += [['scripts/14_audit_ml.py','--input','data/synthetic_event_log.csv','--output','results/synthetic_audit','--timestamp-parser','mixed']]
for name in ['bpi2017','bpi2019','data_quality_issues']:
 commands.append(['scripts/14_audit_ml.py','--input',f'datasets/prepared/{name}_prepared.csv','--output',f'results/{name}_audit','--timestamp-parser','mixed','--class-weight','balanced'])
commands.append(['scripts/16_generate_audited_figures.py'])
for command in commands:subprocess.run([sys.executable,*command],cwd=root,check=True)
