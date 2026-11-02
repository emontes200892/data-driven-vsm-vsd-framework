from pathlib import Path
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'
raw = pd.read_csv(RESULTS/'scenario_replications.csv')
summary = pd.read_csv(RESULTS/'scenario_results.csv')
summary.to_csv(RESULTS/'ablation_results.csv', index=False)

lines = []
for metric in ['lead_time_mean','throughput_per_hour','wip_mean','defect_rate','rework_rate']:
    groups = [g[metric].values for _, g in raw.groupby('scenario')]
    shapiro_p = min(stats.shapiro(g[:min(len(g),5000)]).pvalue for g in groups)
    if shapiro_p > 0.05:
        stat, p = stats.f_oneway(*groups)
        lines.append(f'{metric}: one-way ANOVA F={stat:.4f}, p={p:.6f}')
        tukey = pairwise_tukeyhsd(endog=raw[metric], groups=raw['scenario'], alpha=0.05)
        lines.append(str(tukey))
    else:
        stat, p = stats.kruskal(*groups)
        lines.append(f'{metric}: Kruskal-Wallis H={stat:.4f}, p={p:.6f}')
    lines.append('')

(RESULTS/'statistical_tests.txt').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines[:12]))
