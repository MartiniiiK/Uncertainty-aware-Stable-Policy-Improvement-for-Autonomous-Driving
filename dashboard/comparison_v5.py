"""Read-only V1 comparison adapter. USPI demo status is independent of V1 results."""
import csv, math, time
from functools import lru_cache
import research_monitor as monitor
ROOT=monitor.ROOT
RUN=ROOT/'da_runs/rcdpi42'; DQL=ROOT/'da_runs/strong42_eta1'
KEYS=('rcdpi','dql','rule')

def episodes(key):
    current=RUN/f'eval_{key}'
    old={'dql':DQL/'dev_baseline_100','rule':ROOT/'da_runs/pilot42_town03/eval_rule'}.get(key)
    folder=current if (current/'episodes.csv').exists() or (current/'INVALID.txt').exists() or old is None else old
    invalid=(folder/'INVALID.txt').exists()
    rows=monitor.rows(folder/'episodes.csv') if not invalid else []
    by_seed={int(r['seed']):r for r in rows if all(k in r for k in ('seed','episode','return','cost')) and 1000<=r['seed']<=1099}
    return dict(folder=str(folder.relative_to(ROOT)),historical=folder!=current,invalid=invalid,rows=[by_seed[k] for k in sorted(by_seed)],count=len(by_seed),target=100)

@lru_cache(maxsize=18)
def trajectory(path,seed,episode,bucket):
    points=[]
    try:
        with open(path,newline='') as f:
            for r in csv.DictReader(f):
                try:
                    if int(r['seed'])!=seed or int(r['episode'])!=episode:continue
                    p=[int(r['step']),float(r['x']),float(r['y'])]
                    if all(math.isfinite(x) for x in p):points.append(p)
                except (ValueError,KeyError,TypeError):continue
    except OSError:pass
    return points

def comparison(seed,port):
    r=monitor.rcdpi_state('rcdpi42'); scale=monitor.config(ROOT/'monitor_reward_scale_v5.json').get('reward_rms')
    if not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0:scale=None
    ev={k:episodes(k) for k in KEYS}; paths={}
    for k,e in ev.items():
        hit=next((p for p in e['rows'] if p['seed']==seed),None)
        paths[k]=trajectory(str(ROOT/e['folder']/'actions.csv'),seed,int(hit['episode']),int(time.monotonic()//20)) if hit else []
    return dict(timestamp=time.time(),fqe=r['series']['fqe'],stage=r['stages'][0],status=r['status'],pipeline=r['rcdpi']['pipeline'],log=r['log'],source=r['rcdpi']['source'],dql=[dict(update=p['update'],reward_raw=p['critic_loss']/2,reward=p['critic_loss']/(2*scale**2) if scale else None) for p in monitor.rows(DQL/'dql_training.csv') if 'update' in p and 'critic_loss' in p],scale=scale,evals=ev,paths=paths,seed=seed,gpu=monitor.gpu(),live=monitor.live(port))
