import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT/'data', ROOT/'results'
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA/'synthetic_event_log.csv', parse_dates=['start_ts','end_ts'])
obs_hours = (df.end_ts.max() - df.start_ts.min()).total_seconds()/3600
stage = df.groupby('activity').agg(
    events=('event_id','count'),
    CT_min=('cycle_time_min','mean'),
    WT_min=('waiting_time_min','mean'),
    WIP=('wip_before','mean'),
    Queue=('queue_before','mean'),
    Defect_Rate=('defect_flag','mean'),
    Rework_Rate=('rework_flag','mean'),
    Bottleneck_Rate=('bottleneck_flag','mean'),
).reset_index()
stage['Throughput_events_per_hour'] = stage['events']/obs_hours
stage.to_csv(RESULTS/'vsm_stage_kpis.csv', index=False)

case = df.groupby('case_id').agg(
    first_start=('start_ts','min'), last_end=('end_ts','max'),
    total_processing_min=('cycle_time_min','sum'), total_waiting_min=('waiting_time_min','sum'),
    rework_events=('rework_flag','sum'), defects=('defect_flag','sum')
).reset_index()
case['lead_time_min'] = (case.last_end-case.first_start).dt.total_seconds()/60
case['value_added_ratio'] = case.total_processing_min / case.lead_time_min
case.to_csv(RESULTS/'case_summary.csv', index=False)
print(stage.round(3))
print('\nCase-level summary saved.')
