"""Read the exact RMS stored in a locally trusted FQE checkpoint; no retraining."""
import argparse,json,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('checkpoint');a=p.parse_args()
c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
assert c['format']=='easycarla-rule-fqe-v1'; scale=float(c['model']['outcome_scale'][0]);assert 0<scale<float('inf')
Path('monitor_reward_scale_v5.json').write_text(json.dumps({'reward_rms':scale,'source':'FQE checkpoint outcome_scale[0]'},indent=2)+'\n')
