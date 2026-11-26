import sys
from pathlib import Path
import pandas as pd
r=Path(sys.argv[1]); keys=['return','cost','route_progress_m','mean_planar_jerk_mps3','collision','offroad','mean_speed_kmh','distance_m','near_stop_fraction','executed_overlap_fraction','mean_abs_delta_steer','mean_abs_delta_longitudinal','mean_inference_ms']
result={}
for mode in ('uspi','rule','dql'):
    f=r/f'eval_{mode}'
    if (f/'INVALID.txt').exists():raise RuntimeError(f'Invalid evaluation: {f}')
    d=pd.read_csv(f/'episodes.csv')
    if len(d)!=100 or set(d.seed)!=set(range(1000,1100)):raise ValueError(f'Incomplete cohort: {mode}')
    result[mode]=d[keys].mean()
print(pd.DataFrame(result).round(6));pd.DataFrame(result).to_csv(r/'comparison_100.csv')
