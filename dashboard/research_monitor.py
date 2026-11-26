#!/usr/bin/env python3
"""Read-only research monitor. Python standard library; no CARLA or torch imports."""
import argparse, ast, csv, json, math, os, re, shutil, subprocess, time
from collections import Counter
from threading import Lock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'da_runs'; LOGS=ROOT/'logs'

def tail(p,n=20):
    try:
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-65536)); return f.read().decode(errors='replace').replace('\r','\n').splitlines()[-n:]
    except OSError: return []

def config(p):
    try: return json.loads(p.read_text())
    except (OSError,ValueError): return {}

def rows(p):
    result=[]
    try:
        with p.open(newline='') as f:
            for row in csv.DictReader(f):
                try:
                    if None in row or any(v is None for v in row.values()): continue
                    d={k:float(v) for k,v in row.items() if v!=''}
                    if all(math.isfinite(v) for v in d.values()): result.append(d)
                except (ValueError,TypeError): continue
    except OSError: pass
    return result

def sample(r,limit=2000):
    if len(r)<=limit: return r
    return [r[round(i*(len(r)-1)/(limit-1))] for i in range(limit)]

def processes(run):
    found=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if not any(Path(a).name in ('train_da_dql.py','train_dql_v2.py','train_da_control.py') for a in args): continue
            o=args[args.index('--out')+1] if '--out' in args else next(a.split('=',1)[1] for a in args if a.startswith('--out='))
            if (Path(os.readlink(p/'cwd'))/o).resolve()==run.resolve(): found.append(int(p.name))
        except (OSError,ValueError,IndexError,StopIteration): continue
    return found

def legacy_state(name):
    run=RUNS/name; cfg=config(run/'config.json'); log=tail(LOGS/f'{name}_train.log',25)
    if (LOGS/f'{name}_aware_train.log').is_file(): log=tail(LOGS/f'{name}_aware_train.log',25) or log
    if not log: log=tail(RUNS/f'{name}.log',25)
    pids=processes(run); data={}; stages=[]; mode=cfg.get('mode'); expected={'evaluator':mode in ('both','aware','evaluator') and not cfg.get('evaluator_file'),'dql':mode in ('both','dql'),'aware':mode in ('both','aware')}
    for key in ('evaluator','dql','aware'):
        r=rows(run/f'{key}_training.csv'); data[key]=sample(r)
        target=cfg.get('evaluator_steps' if key=='evaluator' else 'steps'); done=int(r[-1].get('update',0)) if r else 0
        saved=(run/f'{key}.pth').is_file(); calibrated=(run/'evaluator_report.json').is_file()
        complete=bool(target and done>=target and saved and (key!='evaluator' or calibrated))
        if key=='evaluator' and cfg.get('evaluator_file') and saved: state='Loaded'
        elif complete: state='Complete'
        elif cfg and not expected[key]: state='Not scheduled'
        elif r and pids: state='Calibrating / saving' if target and done>=target else 'Running'
        elif r: state='Stopped / incomplete'
        else: state='Waiting'
        stages.append(dict(key=key,done=done,target=target,state=state,checkpoint=saved))
    # Only a currently active process is called running. A checkpoint may be intermediate.
    completed=any(x.startswith('COMPLETE:') for x in log)
    if pids: status='Training' if any(s['state'] in ('Running','Calibrating / saving') for s in stages) else 'Preparing / transitioning'
    elif completed and all(s['state'] in ('Complete','Loaded','Not scheduled') for s in stages): status='Complete'
    elif any('Traceback' in x or 'Error:' in x for x in log): status='Failed / see log'
    elif cfg: status='Not running / inspect log'
    else: status='Awaiting training'
    evaluations={}; eval_meta={}
    for key, folders in {'dql':('dev_baseline_100','dev_baseline','eval_dql'), 'full':('dev_aware_100','dev_aware','eval_full')}.items():
        folder=next((run/f for f in folders if (run/f/'episodes.csv').exists()),None)
        r=rows(folder/'episodes.csv') if folder else []
        invalid=bool(folder and (folder/'INVALID.txt').exists())
        evaluations[key]=[] if invalid else sample(r)
        if folder: eval_meta[key]=dict(folder=folder.name,count=len(r),invalid=invalid)
    if not pids and any(s['state']=='Complete' for s in stages) and not any(s['state']=='Stopped / incomplete' for s in stages):
        status='Training checkpoint saved' if status!='Complete' else status
    return dict(name=name,kind="legacy",config=cfg,status=status,pids=pids,stages=stages,series=data,evaluations=evaluations,eval_meta=eval_meta,log=log)


FQE_KEYS=('reward','progress','cost','jerk')
EVAL_KEYS=('rcdpi','rule','dql')
METRICS=('return','cost','collision','offroad','mean_speed_kmh','distance_m','near_stop_fraction','executed_overlap_fraction','mean_abs_delta_steer','mean_abs_delta_longitudinal','mean_planar_jerk_mps3','route_progress_m','mean_inference_ms','rule_selection_fraction')
_source_cache={}
_source_lock=Lock()

def textfile(p):
    try: return p.read_text(errors='replace')
    except OSError: return ''

def fqe_points(log, heads, target):
    points={}
    for line in log.splitlines():
        match=re.search(r'step=(\d+) normalized_td_mse=(\{.*\})',line)
        if not match: continue
        try:
            step=int(match[1]); raw=ast.literal_eval(match[2]); values={k:float(raw[k]) for k in FQE_KEYS}
            if 0<step<=target and all(math.isfinite(v) and v>=0 for v in values.values()):
                points[step]=dict(step=step,update=step*heads,**values)
        except (ValueError,SyntaxError,KeyError,TypeError,OverflowError): continue
    return [points[k] for k in sorted(points)]

def pipeline_pid(name):
    try:
        pid=int((LOGS/f'{name}_pipeline.pid').read_text().strip())
        proc=Path(f'/proc/{pid}')
        args=proc.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        if any(re.fullmatch(rf'run_{re.escape(name)}(?:_fix[0-9]+)?\.sh',Path(a).name) for a in args) and proc.joinpath('cwd').resolve()==ROOT: return pid
    except (OSError,ValueError): pass
    return None

def sources(p, completed):
    # Only finished episodes. Cache scans so the UI never re-reads a growing trace on every refresh.
    key=str(p); now=time.monotonic(); ids=frozenset(completed)
    try: signature=(ids,p.stat().st_size,p.stat().st_mtime_ns)
    except OSError: signature=(ids,0,0)
    with _source_lock:
        old=_source_cache.get(key)
        if old and (old[1]==signature or now-old[0]<15): return old[2]
        counts=Counter()
        try:
            with p.open(newline='') as f:
                for row in csv.DictReader(f):
                    try:
                        if int(row['episode']) in ids and row.get('selected_source'):
                            counts[row['selected_source']]+=1
                    except (TypeError,ValueError,KeyError): continue
        except OSError: pass
        result=dict(counts=dict(counts),total=sum(counts.values()),episodes=len(ids))
        _source_cache[key]=(now,signature,result)
        return result

def rcdpi_state(name):
    run=RUNS/name; cfg=config(run/'experiment_config.json'); pipeline=textfile(LOGS/f'{name}_pipeline.log'); train=textfile(LOGS/f'{name}_train.log')
    heads=int(cfg.get('heads',3)); target=int(cfg.get('steps_per_head',66667)); n=int(cfg.get('episodes_per_policy',100)); seed=int(cfg.get('evaluation_seed_start',1000))
    points=fqe_points(train,heads,target); step=points[-1]['step'] if points else 0
    pid=pipeline_pid(name); pids=processes(run); active=bool(pid or pids)
    checkpoint=(run/'critics.pt').is_file()
    trained=checkpoint and step==target and f'RCDPI FQE TRAINING COMPLETE: {heads*target} member updates' in pipeline
    failed='PIPELINE FAILED' in pipeline or 'Traceback (most recent call last)' in train
    evaluations={}; meta={}
    for k in EVAL_KEYS:
        folder=run/f'eval_{k}'; invalid=(folder/'INVALID.txt').exists(); failed=failed or invalid
        byep={int(r['episode']):r for r in rows(folder/'episodes.csv') if 'episode' in r and 'return' in r and 'cost' in r and 1<=r['episode']<=n and r.get('seed')==seed+r['episode']-1}
        rr=[byep[i] for i in sorted(byep)]
        verified=not invalid and len(rr)==n and f'EVALUATION VERIFIED: {folder.relative_to(ROOT)} ' in pipeline
        metrics={m:sum(r[m] for r in rr if m in r)/sum(m in r for r in rr) for m in METRICS if any(m in r for r in rr)} if not invalid else {}
        evaluations[k]=[] if invalid else rr
        started=f'STARTING {k} EVALUATION:' in pipeline
        meta[k]=dict(count=len(rr),target=n,invalid=invalid,verified=verified,metrics=metrics,state='INVALID' if invalid else 'Verified' if verified else 'Running / unverified' if active and started else 'Stopped / unverified' if started or rr else 'Waiting',log=tail(LOGS/f'{name}_eval_{k}.log',5))
    finished='ALL RCDPI EXPERIMENTS COMPLETE' in pipeline and (ROOT/f'{name}_analysis.zip').is_file() and all(m['verified'] for m in meta.values()) and trained
    if finished: status='Complete'
    elif failed: status='Failed / see log'
    elif active: status='FQE training / preparing' if not trained else 'Evaluating / packaging'
    elif trained: status='FQE saved / pipeline stopped'
    else: status='Not running / inspect log' if cfg else 'Awaiting training'
    stages=[dict(key='fqe',done=step*heads,target=target*heads,state='Complete' if trained else 'Calibrating / saving' if active and step==target else 'Running' if active and points else 'Preparing' if active else 'Stopped / incomplete' if points else 'Waiting',checkpoint=checkpoint)]
    sc=sources(run/'eval_rcdpi/actions.csv',[int(r['episode']) for r in evaluations['rcdpi']]) if evaluations['rcdpi'] else dict(counts={},total=0,episodes=0)
    return dict(name=name,kind='rcdpi',config={**cfg,'steps':heads*target},status=status,pids=sorted(set(pids+([pid] if pid else []))),stages=stages,series={'fqe':sample(points)},evaluations=evaluations,log=tail(LOGS/f'{name}_train.log',25),rcdpi=dict(step=step,steps_per_head=target,heads=heads,trained=trained,evals=meta,source=sc,pipeline=tail(LOGS/f'{name}_pipeline.log',35)))

def run_state(name):
    cfg=config(RUNS/name/'experiment_config.json')
    if cfg.get('method','').upper()=='RCDPI' or name=='rcdpi42': return rcdpi_state(name)
    return legacy_state(name)

def gpu():
    try:
        result=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],timeout=3,text=True,stderr=subprocess.DEVNULL)
        return [dict(name=a.strip(),util=float(b),used=float(c),total=float(d)) for a,b,c,d in (x.split(',') for x in result.strip().splitlines())]
    except (OSError,ValueError,subprocess.SubprocessError): return []

def live(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/stats',timeout=1.5) as f: stats=json.load(f)
        top={}
        if stats.get('status')=='Running':
            try:
                with urlopen(f'http://127.0.0.1:{port}/dual_camera_status',timeout=.6) as f: top=json.load(f)
            except Exception: top={'ready':False,'error':'Dual camera endpoint unavailable'}
        return dict(available=True,stats=stats,topdown=top)
    except Exception: return dict(available=False,stats={})

def download():
    file=ROOT/'data/easycarla_offline_dataset.hdf5'; logs=[LOGS/'dataset_download.log',LOGS/'dataset_hf.log']
    existing=[p for p in logs if p.is_file()]; latest=max(existing,key=lambda p:p.stat().st_mtime) if existing else None
    return dict(file_bytes=file.stat().st_size if file.is_file() else 0,partial=Path(str(file)+'.aria2').exists(),log=tail(latest,8) if latest else [],disk_free=shutil.disk_usage(ROOT).free)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def send(self,data,ctype='application/json',status=200):
        if not isinstance(data,bytes): data=json.dumps(data,allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type',ctype); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=='/': self.send((ROOT/'research_monitor.html').read_bytes(),'text/html; charset=utf-8'); return
        if u.path=='/api/comparison':
            from comparison_v5 import comparison
            try: seed=int(parse_qs(u.query).get('seed',['1000'])[0])
            except ValueError: self.send({'error':'Invalid seed'},status=400); return
            if not 1000<=seed<=1099: self.send({'error':'Invalid seed'},status=400); return
            self.send(comparison(seed,self.server.video_port)); return
        if u.path=='/api/state':
            names=sorted(p.name for p in RUNS.iterdir() if p.is_dir() and re.fullmatch(r'[\w.-]+',p.name) and not p.name.startswith('.')) if RUNS.exists() else []
            name=parse_qs(u.query).get('run',['rcdpi42' if 'rcdpi42' in names else 'pilot42' if 'pilot42' in names else names[-1] if names else 'pilot42'])[0]
            if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*',name) or '..' in name: self.send({'error':'Invalid run'},status=400); return
            self.send(dict(run=run_state(name),runs=sorted(set(names+[name])),gpu=gpu(),download=download(),live=live(self.server.video_port),timestamp=time.time())); return
        if u.path in ('/video_feed','/video_feed_topdown'):
            try: upstream=urlopen(f'http://127.0.0.1:{self.server.video_port}{u.path}',timeout=10)
            except Exception: self.send({'error':'Video unavailable'},status=503); return
            with upstream:
                self.send_response(200); self.send_header('Content-Type',upstream.headers.get('Content-Type','multipart/x-mixed-replace; boundary=frame')); self.send_header('Cache-Control','no-store'); self.end_headers()
                try:
                    while True:
                        chunk=upstream.read1(65536)
                        if not chunk: break
                        self.wfile.write(chunk); self.wfile.flush()
                except (OSError,TimeoutError): pass
            return
        self.send({'error':'Not found'},status=404)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=8081); p.add_argument('--video-port',type=int,default=8080); a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler); server.video_port=a.video_port
    print(f'Research monitor: http://127.0.0.1:{a.port} (video upstream {a.video_port})',flush=True); server.serve_forever()
if __name__=='__main__': main()
