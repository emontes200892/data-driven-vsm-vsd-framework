from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

stations = pd.DataFrame([
    # activity, resource, machine, mean_ct, sd_ct, base_wait, defect_p, rework_p, capacity_weight
    ('Cutting', 'Operator_A', 'CNC_01', 6.0, 1.1, 2.0, 0.020, 0.005, 1.00),
    ('Machining', 'Operator_B', 'CNC_02', 9.0, 1.8, 5.0, 0.030, 0.010, 0.92),
    ('Initial Inspection', 'Inspector_1', 'INS_01', 4.5, 0.8, 8.0, 0.060, 0.020, 0.72),
    ('Assembly', 'Operator_C', 'ASM_01', 12.0, 2.2, 10.0, 0.040, 0.025, 0.65),
    ('Functional Testing', 'Tester_1', 'TEST_01', 10.5, 2.5, 12.0, 0.070, 0.050, 0.58),
    ('Packaging', 'Operator_D', 'PACK_01', 5.5, 1.0, 4.0, 0.015, 0.005, 0.90),
    ('Release', 'Clerk_1', 'REL_01', 2.5, 0.5, 1.0, 0.005, 0.000, 1.00),
], columns=['activity','resource_id','machine_id','mean_cycle_time_min','sd_cycle_time_min','base_wait_min','defect_probability','rework_probability','capacity_weight'])
stations.to_csv(DATA / 'route_parameters.csv', index=False)

families = ['A', 'B', 'C']
family_multiplier = {'A': 0.95, 'B': 1.10, 'C': 1.25}
shift_multiplier = {'Morning': 0.95, 'Afternoon': 1.05, 'Night': 1.18}
waste_labels = ['Waiting', 'Overprocessing', 'Defects', 'Motion', 'Inventory', 'Transportation']

n_cases = 1200
start_date = pd.Timestamp('2026-01-01 06:00:00')
rows = []

def positive_normal(mean: float, sd: float, min_val: float = 0.1) -> float:
    return float(max(min_val, rng.normal(mean, sd)))

for case_idx in range(1, n_cases + 1):
    family = rng.choice(families, p=[0.45, 0.35, 0.20])
    shift = rng.choice(list(shift_multiplier), p=[0.45, 0.40, 0.15])
    release_offset_min = int(rng.integers(0, 60 * 24 * 45))
    current_time = start_date + pd.Timedelta(minutes=release_offset_min)
    cumulative_defect = False
    rework_inserted = False
    queue_pressure = rng.gamma(2.0, 2.0)

    for event_pos, st in stations.iterrows():
        mean_ct = st.mean_cycle_time_min * family_multiplier[family] * shift_multiplier[shift]
        cycle_time = positive_normal(mean_ct, st.sd_cycle_time_min)

        # Deliberately make Assembly and Testing more likely to become bottlenecks.
        structural_penalty = {'Assembly': 1.7, 'Functional Testing': 2.0, 'Initial Inspection': 1.3}.get(st.activity, 1.0)
        waiting_time = rng.gamma(2.0, st.base_wait_min * structural_penalty / 2.0) + queue_pressure
        if shift == 'Night':
            waiting_time *= 1.25
        queue_before = int(max(0, rng.poisson(waiting_time / 4.0)))
        wip_before = int(max(queue_before, rng.poisson(waiting_time / 3.0 + 1)))

        defect_flag = int(rng.random() < st.defect_probability * family_multiplier[family] * shift_multiplier[shift])
        rework_flag = 0
        cumulative_defect = cumulative_defect or bool(defect_flag)
        bottleneck_flag = int((waiting_time > 18) or (cycle_time > mean_ct * 1.35) or st.activity in ['Assembly','Functional Testing'] and rng.random() < 0.35)
        delay_flag = int(waiting_time > 15)
        waste_label = 'Defects' if defect_flag else ('Waiting' if delay_flag else rng.choice(waste_labels, p=[.30,.14,.08,.12,.22,.14]))

        start_ts = current_time + pd.Timedelta(minutes=float(waiting_time))
        end_ts = start_ts + pd.Timedelta(minutes=float(cycle_time))
        rows.append({
            'case_id': f'ORD-{case_idx:05d}',
            'event_id': f'EV-{case_idx:05d}-{event_pos+1:02d}',
            'activity': st.activity,
            'resource_id': st.resource_id,
            'machine_id': st.machine_id,
            'product_family': family,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'cycle_time_min': round(cycle_time, 3),
            'waiting_time_min': round(waiting_time, 3),
            'queue_before': queue_before,
            'wip_before': wip_before,
            'shift': shift,
            'defect_flag': defect_flag,
            'rework_flag': rework_flag,
            'bottleneck_flag': bottleneck_flag,
            'delay_flag': delay_flag,
            'waste_label': waste_label,
        })
        current_time = end_ts

        # Rework loop after functional testing for selected cases.
        if st.activity == 'Functional Testing' and cumulative_defect and (rng.random() < 0.55) and not rework_inserted:
            wait = rng.gamma(2.5, 4.5)
            ct = positive_normal(8.0 * family_multiplier[family], 1.8)
            start_rw = current_time + pd.Timedelta(minutes=float(wait))
            end_rw = start_rw + pd.Timedelta(minutes=float(ct))
            rows.append({
                'case_id': f'ORD-{case_idx:05d}',
                'event_id': f'EV-{case_idx:05d}-RW',
                'activity': 'Rework',
                'resource_id': 'Rework_Team',
                'machine_id': 'RW_01',
                'product_family': family,
                'start_ts': start_rw,
                'end_ts': end_rw,
                'cycle_time_min': round(ct, 3),
                'waiting_time_min': round(wait, 3),
                'queue_before': int(rng.poisson(wait/3+1)),
                'wip_before': int(rng.poisson(wait/2+1)),
                'shift': shift,
                'defect_flag': 0,
                'rework_flag': 1,
                'bottleneck_flag': int(wait > 15),
                'delay_flag': int(wait > 15),
                'waste_label': 'Defects',
            })
            current_time = end_rw
            rework_inserted = True

df = pd.DataFrame(rows).sort_values(['case_id','start_ts']).reset_index(drop=True)
df.to_csv(DATA / 'synthetic_event_log.csv', index=False)
print(f'Generated {len(df):,} events for {df.case_id.nunique():,} cases')
print(DATA / 'synthetic_event_log.csv')
