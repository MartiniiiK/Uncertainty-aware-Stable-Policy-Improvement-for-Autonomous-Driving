"""Faithful EasyCarla DQL core + explicit DA actor-only extension. Offline training."""
import argparse, csv, hashlib, inspect, json, random, textwrap, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from da_dql import build_dql, project, Evaluator
from train_da_dql import Data, seed_all, train_evaluator
import torch.nn.functional as F

class Replay:
    def __init__(self, data, seed):
        self.data=data; self.rng=np.random.default_rng(seed)
    def sample(self, batch_size):
        return self.data.batch(self.rng.choice(self.data.train,batch_size,replace=True))

def attach_da(model, ev, cfg):
    # Use exact installed DQL train body, altering only the actor objective.
    import types
    original=type(model).train
    src=textwrap.dedent(inspect.getsource(original))
    anchor='actor_loss = bc_loss + self.eta * q_loss'
    if src.count(anchor)!=1: raise RuntimeError('Unexpected upstream DQL implementation; refusing patch')
    src=src.replace(anchor,anchor+' + self.da_penalty(state, new_action)',1)
    scope=dict(original.__globals__); exec(compile(src,'<DA actor extension>','exec'),scope)
    model.train=types.MethodType(scope['train'],model)
    model.da_rows=[]
    def penalty(s, raw):
        a=project(raw); p=ev.scores(s,a,cfg.beta)
        dyn=(F.relu(p['u']/ev.u_limit-1)+F.relu(p['k']/ev.k_limit-1)).mean()
        safe=F.relu(p['risk']/cfg.budget-1).mean()
        event=p['events'].mean(0).sum()
        # Penalize simultaneous activation; do not change offline action labels.
        conflict=(a[:,0]*a[:,2]).mean()
        total=cfg.lambda_dyn*dyn+cfg.lambda_safe*safe+cfg.lambda_event*event+cfg.lambda_control*conflict
        model.da_rows.append(torch.stack([dyn.detach(),safe.detach(),event.detach(),conflict.detach()]))
        return total
    model.da_penalty=penalty

def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); os.replace(tmp,path)

def train(data,cfg,root,mode,ev,epochs,resume):
    seed_all(cfg.seed); replay=Replay(data,cfg.seed); m=build_dql(data.device,cfg.eta)
    if mode=='aware': attach_da(m,ev,cfg)
    last=root/f'{mode}_resume.pth'; log=root/f'{mode}_training.csv'; start=0
    if last.exists():
        if not resume: raise RuntimeError(f'{last} exists; use --resume')
        # Local, self-produced full training checkpoint only.
        c=torch.load(last,map_location='cpu',weights_only=False)
        for k in ('actor','critic','critic_target','ema_model'): getattr(m,k).load_state_dict(c[k])
        m.actor_optimizer.load_state_dict(c['actor_opt']); m.critic_optimizer.load_state_dict(c['critic_opt'])
        m.step=c['update']; start=c['epoch']; replay.rng.bit_generator.state=c['replay_rng']; random.setstate(c['python_rng']); np.random.set_state(c['numpy_rng']); torch.set_rng_state(c['torch_rng'])
        if torch.cuda.is_available() and c['cuda_rng'] is not None: torch.cuda.set_rng_state_all(c['cuda_rng'])
    fields=['epoch','update','actor_loss','bc_loss','ql_loss','critic_loss','dynamics_loss','safety_loss','event_loss','control_loss']
    if log.exists():
        if not resume: raise RuntimeError(f'{log} exists; use --resume')
        with log.open() as f: old=[r for r in csv.DictReader(f) if int(r['epoch'])<=start]
    else: old=[]
    with log.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(old); f.flush()
        for ep in range(start+1,epochs+1):
            if mode=='aware': m.da_rows.clear()
            metrics=m.train(replay,iterations=cfg.updates_per_epoch,batch_size=cfg.batch)
            row={k:float(np.mean(v)) for k,v in metrics.items()}; row.update(epoch=ep,update=m.step)
            if mode=='aware': row.update(zip(('dynamics_loss','safety_loss','event_loss','control_loss'),torch.stack(m.da_rows).mean(0).cpu().tolist()))
            if not all(np.isfinite(v) for v in row.values()): raise FloatingPointError(f'Nonfinite epoch {ep}')
            weights={k:getattr(m,k).state_dict() for k in ('actor','critic','critic_target','ema_model')}
            policy={**weights,'mode':mode,'format':'official_raw_v2','eta':cfg.eta,'beta':cfg.beta,'budget':cfg.budget,'epoch':ep,'step':m.step,'selection':cfg.selection}
            if ep%cfg.save_epochs==0 or ep==epochs: atomic_save(policy,root/f'{mode}_epoch{ep:04d}.pth')
            atomic_save(policy,root/f'{mode}.pth')
            w.writerow(row); f.flush()
            atomic_save({**weights,'actor_opt':m.actor_optimizer.state_dict(),'critic_opt':m.critic_optimizer.state_dict(),'update':m.step,'epoch':ep,'replay_rng':replay.rng.bit_generator.state,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},last)
            print(mode,row,flush=True)
    print(f'COMPLETE: {mode}, epoch={epochs}, updates={m.step}',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/easycarla_offline_dataset.hdf5'); p.add_argument('--out',required=True); p.add_argument('--phase',choices=['dql','aware'],required=True); p.add_argument('--epochs',type=int,default=200); p.add_argument('--updates-per-epoch',type=int,default=1000); p.add_argument('--batch',type=int,default=256); p.add_argument('--save-epochs',type=int,default=10); p.add_argument('--evaluator-steps',type=int,default=20000); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='cuda'); p.add_argument('--resume',action='store_true'); p.add_argument('--eta',type=float,default=0.01); a=p.parse_args()
    if min(a.epochs,a.updates_per_epoch,a.batch,a.save_epochs,a.evaluator_steps)<1: p.error('positive budgets required')
    root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
    cfg=dict(data=str(Path(a.data).resolve()),epochs=a.epochs,updates_per_epoch=a.updates_per_epoch,steps=a.epochs*a.updates_per_epoch,batch=a.batch,save_epochs=a.save_epochs,evaluator_steps=a.evaluator_steps,seed=a.seed,split_seed=2026,device=a.device,mode='both',eta=a.eta,heads=5,horizon=10,gamma_cost=.99,beta=1.,quantile=.99,budget=.25,cost_weight=2.,event_weight=1.,log_every=250,lambda_dyn=.05,lambda_safe=.05,lambda_event=.05,lambda_control=.1,selection={'dyn':1.,'risk':1.,'event':1.,'control':2.,'smooth':2.,'temperature':.5},protocol='official raw state/action, upstream EMA and Q loss, no LR decay')
    m=build_dql(torch.device(a.device),cfg['eta']); cfg['upstream_sha256']=hashlib.sha256(Path(inspect.getfile(type(m))).read_bytes()).hexdigest(); del m
    path=root/'config.json'
    if path.exists():
        old=json.loads(path.read_text()); immutable=[k for k in cfg if k not in ('epochs','steps')]
        if any(old.get(k)!=cfg[k] for k in immutable): raise RuntimeError('Run configuration mismatch; use a new directory')
        if a.epochs<old['epochs']: raise RuntimeError('Cannot decrease epoch budget')
    elif a.phase=='aware': raise RuntimeError('Train baseline first in this run')
    if a.phase=='aware':
        ck=root/'dql.pth'
        if not ck.exists() or torch.load(ck,map_location='cpu',weights_only=True)['epoch']<a.epochs: raise RuntimeError('Complete DQL at matching epoch budget first')
    path.write_text(json.dumps(cfg,indent=2)); cfg=SimpleNamespace(**cfg)
    data=Data(cfg.data,cfg.split_seed,cfg.horizon,cfg.gamma_cost,torch.device(cfg.device)); ev=None
    if a.phase=='aware':
        file=root/'evaluator.pth'
        if file.exists():
            e=torch.load(file,map_location=data.device,weights_only=True); ev=Evaluator(e['heads'],e['dt']).to(data.device); ev.load_state_dict(e['state']); ev.freeze()
        else: ev=train_evaluator(data,cfg,root)
    train(data,cfg,root,a.phase,ev,a.epochs,a.resume)
if __name__=='__main__': main()
