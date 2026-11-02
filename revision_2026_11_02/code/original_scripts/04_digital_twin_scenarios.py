from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT/'data', ROOT/'results'
RESULTS.mkdir(exist_ok=True)
route = pd.read_csv(DATA/'route_parameters.csv')

SCENARIOS = {
    'Baseline':        {'ct_factor':1.00,'wait_factor':1.00,'defect_factor':1.00,'network_gain':0.00},
    'Manual VSM':      {'ct_factor':0.98,'wait_factor':0.92,'defect_factor':0.97,'network_gain':0.00},
    'PM-VSM':          {'ct_factor':0.96,'wait_factor':0.85,'defect_factor':0.95,'network_gain':0.00},
    'PM-VSM-XGB':      {'ct_factor':0.92,'wait_factor':0.78,'defect_factor':0.88,'network_gain':0.00},
    'PM-VSM-XGB-DT':   {'ct_factor':0.88,'wait_factor':0.68,'defect_factor':0.82,'network_gain':0.02},
    'Full Framework':  {'ct_factor':0.84,'wait_factor':0.58,'defect_factor':0.72,'network_gain':0.06},
}

def simulate_scenario(name, p, n_orders=500, seed=42):
    rng = np.random.default_rng(seed)
    lead_times=[]; defects=[]; reworks=[]; wips=[]
    for _ in range(n_orders):
        ct = np.maximum(0.1, rng.normal(route.mean_cycle_time_min, route.sd_cycle_time_min)).sum() * p['ct_factor']
        wait = rng.gamma(2.2, route.base_wait_min.mean()/2.2, size=len(route)).sum() * p['wait_factor']
        defect_count = rng.binomial(1, np.clip(route.defect_probability.values * p['defect_factor'], 0, 1)).sum()
        rework = int(defect_count > 0 and rng.random() < 0.42 * p['defect_factor'])
        rework_time = rework * max(0, rng.normal(8, 1.5))
        lt = ct + wait + rework_time
        # full framework network gain reduces disruption propagation
        lt *= (1 - p['network_gain'])
        lead_times.append(lt)
        defects.append(defect_count / len(route))
        reworks.append(rework)
        wips.append(max(0, rng.normal(wait/8, 1.5)) * (1 - p['network_gain']))
    total_calendar_min = n_orders * 7.5 * p['wait_factor'] + np.mean(lead_times)
    return {
        'scenario': name,
        'lead_time_mean': np.mean(lead_times),
        'lead_time_sd': np.std(lead_times, ddof=1),
        'throughput_per_hour': n_orders/(total_calendar_min/60),
        'wip_mean': np.mean(wips),
        'defect_rate': np.mean(defects),
        'rework_rate': np.mean(reworks),
    }

rows=[]
for scenario, params in SCENARIOS.items():
    for rep in range(1, 31):
        r = simulate_scenario(scenario, params, seed=1000+rep)
        r['replication'] = rep
        rows.append(r)
raw = pd.DataFrame(rows)
raw.to_csv(RESULTS/'scenario_replications.csv', index=False)
summary = raw.groupby('scenario').agg(
    Lead_Time=('lead_time_mean','mean'), Lead_Time_SD=('lead_time_mean','std'),
    Throughput=('throughput_per_hour','mean'), WIP=('wip_mean','mean'),
    Defect_Rate=('defect_rate','mean'), Rework_Rate=('rework_rate','mean')
).reset_index()
summary['Lead_Time_CI95'] = 1.96 * raw.groupby('scenario')['lead_time_mean'].std().values / np.sqrt(30)
summary.to_csv(RESULTS/'scenario_results.csv', index=False)
print(summary.round(3))
