"""Seed-blocked analysis of scenario_replications.csv, not module attribution.
Each replication number is a shared initial seed across all six scenarios.
Bootstrap resamples replication blocks and computes ratios of scenario means.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ORDER=['Baseline','Manual VSM','PM-VSM','PM-VSM-XGB','PM-VSM-XGB-DT','Full Framework']
METRICS={'lead_time_mean':-1,'throughput_per_hour':1,'wip_mean':-1,'defect_rate':-1,'rework_rate':-1}

def analyze(path,out,bootstrap=10000):
 out.mkdir(parents=True,exist_ok=True);d=pd.read_csv(path)
 if d.duplicated(['scenario','replication']).any():raise ValueError('Duplicate seed/scenario rows.')
 metrics=[];comparisons=[];global_tests=[];summary=[]
 rng=np.random.default_rng(20261024)
 for metric,direction in METRICS.items():
  p=d.pivot(index='replication',columns='scenario',values=metric).reindex(columns=ORDER)
  if p.isna().any().any():raise ValueError('All scenarios must have matched replication blocks and finite metric values.')
  if len(p)<3:raise ValueError('At least three replication blocks are required.')
  for name in ORDER:
   v=p[name].to_numpy();sd=np.std(v,ddof=1);margin=stats.t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v))
   summary.append({'metric':metric,'scenario':name,'n':len(v),'mean':v.mean(),'sd':sd,'ci95_lower':v.mean()-margin,'ci95_upper':v.mean()+margin})
  stat,pvalue=stats.friedmanchisquare(*[p[name].to_numpy() for name in ORDER])
  global_tests.append({'metric':metric,'test':'Friedman, replication/seed blocks','statistic':stat,'pvalue':pvalue,'n_blocks':len(p)})
  base=p['Baseline'].to_numpy();boot_idx=rng.integers(0,len(p),size=(bootstrap,len(p)))
  local=[]
  for name in ORDER[1:]:
   value=p[name].to_numpy();delta=direction*(value-base)
   test=stats.wilcoxon(value,base,alternative='two-sided',zero_method='wilcox',method='auto')
   ratio=direction*(value.mean()/base.mean()-1)*100
   br=direction*(value[boot_idx].mean(axis=1)/base[boot_idx].mean(axis=1)-1)*100
   lo,hi=np.quantile(br,[.025,.975])
   md=delta.mean();margin=stats.t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
   local.append({'metric':metric,'scenario':name,'baseline':'Baseline','paired_blocks':len(p),'mean_favorable_difference':md,'difference_ci95_lower':md-margin,'difference_ci95_upper':md+margin,'relative_improvement_percent':ratio,'relative_ci95_lower':lo,'relative_ci95_upper':hi,'wilcoxon_statistic':test.statistic,'pvalue':test.pvalue})
  corrected=multipletests([x['pvalue'] for x in local],method='holm')[1]
  for row,padj in zip(local,corrected):row['pvalue_holm_within_metric']=padj
  comparisons.extend(local)
 pd.DataFrame(summary).to_csv(out/'scenario_kpis_with_ci.csv',index=False)
 pd.DataFrame(global_tests).to_csv(out/'friedman_tests.csv',index=False)
 c=pd.DataFrame(comparisons);c.to_csv(out/'paired_comparisons.csv',index=False)
 c[c.scenario.eq('Full Framework')].to_csv(out/'G5_vs_G0_verified.csv',index=False)
 metadata={'input':str(path),'model':'Original parameter-factor Monte Carlo scenario sampler',
 'pairing':'Shared initial seed 1000+replication; random draw paths may diverge across parameter settings.',
 'bootstrap_blocks':bootstrap,'bootstrap_seed':20261024,
 'multiplicity':'Holm correction across five baseline comparisons within each KPI; not across all KPI families.',
 'interpretation':'Conditional scenario differences, not empirically estimated effects of PM, ML, XAI, or CNA.'}
 (out/'statistics_protocol.json').write_text(json.dumps(metadata,indent=2))
 print(c[c.scenario.eq('Full Framework')][['metric','relative_improvement_percent','relative_ci95_lower','relative_ci95_upper','pvalue_holm_within_metric']].to_string(index=False))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.input,a.output)
