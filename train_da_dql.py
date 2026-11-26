"""Offline training: shared DQL baseline and dynamics-aware DQL."""
import argparse, csv, json, random
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn.functional as F
from da_dql import (Evaluator, Norm, build_dql, update, motion, features,
                    kinematic_residual, project, COST_SCALE, DT)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def horizon_targets(c,events,d,h,gamma):
    cost=c.copy(); e=events.copy()
    for _ in range(1,h):
        cc=c.copy(); ee=events.copy()
        cc[:-1]+=gamma*(~d[:-1])*cost[1:]
        ee[:-1]=np.maximum(ee[:-1], (~d[:-1,None])*e[1:])
        cost,e=cc,ee
    return cost,e

class Data:
    def __init__(self,path,split_seed,horizon,gamma,device):
        print('Reading dataset into CPU RAM...',flush=True)
        with h5py.File(path,'r') as f:
            self.s=np.asarray(f['observations'],dtype=np.float32); self.a=np.asarray(f['actions'],dtype=np.float32)
            self.ns=np.asarray(f['next_observations'],dtype=np.float32); self.r=np.asarray(f['rewards'],dtype=np.float32).reshape(-1)
            self.c=np.asarray(f['costs'],dtype=np.float32).reshape(-1); rawd=np.asarray(f['done']).reshape(-1)
            raw_e=np.stack([np.asarray(f['info/'+key]).reshape(-1) for key in ('is_collision','is_off_road')],-1)
        n=len(self.s)
        if self.s.shape!=(n,307) or self.ns.shape!=(n,307) or self.a.shape!=(n,3) or self.r.shape!=(n,) or self.c.shape!=(n,) or rawd.shape!=(n,) or raw_e.shape!=(n,2): raise ValueError('Unexpected dataset shapes')
        if not all(np.isfinite(x).all() for x in (self.s,self.a,self.ns,self.r,self.c,rawd,raw_e)): raise ValueError('Nonfinite dataset values')
        if (self.c<0).any() or not np.isin(rawd,[0,1]).all() or not np.isin(raw_e,[0,1]).all(): raise ValueError('Invalid cost or event flags')
        self.d=rawd.astype(bool); events=raw_e.astype(np.float32)
        ends=np.flatnonzero(self.d)+1
        if len(ends)<3: raise ValueError('Need >=3 complete episodes')
        end=ends[-1]; starts=np.r_[0,ends[:-1]]; rng=np.random.default_rng(split_seed)
        if horizon>1:
            links=np.flatnonzero(~self.d[:end-1]); probe=rng.choice(links,size=min(len(links),8192),replace=False)
            if len(probe) and not np.allclose(self.ns[probe],self.s[probe+1],atol=1e-4,rtol=1e-4):
                raise ValueError('Rows are not temporally adjacent; use --horizon 1 or repair dataset order.')
        ids=rng.permutation(len(ends)); k=max(1,int(.1*len(ids)))
        def indices(ii): return np.concatenate([np.arange(starts[i],ends[i],dtype=np.int64) for i in ii])
        self.train=indices(ids[2*k:]); self.cal=indices(ids[:k]); self.test=indices(ids[k:2*k]); self.device=device
        self.hcost,self.hevent=horizon_targets(self.c[:end],events[:end],self.d[:end],horizon,gamma)
        self.nd=(~self.d).astype(np.float32)
        print(f'Rows train/cal/test: {len(self.train)}/{len(self.cal)}/{len(self.test)}; dropped tail={n-end}; horizon={horizon}; out-of-bounds action fraction={np.mean(np.any((self.a<np.array([0,-1,0]))|(self.a>1),axis=1)):.6f}',flush=True)
        print('Train horizon event prevalence:',self.hevent[self.train].mean(0).tolist(),flush=True)
    def t(self,x,i): return torch.as_tensor(x[i],device=self.device,dtype=torch.float32)
    def batch(self,i):
        s,a,n,r,nd=(self.t(x,i) for x in (self.s,self.a,self.ns,self.r,self.nd)); return s,a,n,r[:,None],nd[:,None]
    def eval_batch(self,i):
        s,a,n,_,_=self.batch(i); return s,a,n,self.t(self.hcost,i)/COST_SCALE,self.t(self.hevent,i)

def log(path,fields):
    fp=path.open('w',newline=''); w=csv.DictWriter(fp,fieldnames=fields); w.writeheader(); fp.flush(); return fp,w

def sample(rng,idx,size): return rng.choice(idx,size=size,replace=True)

def train_evaluator(data,args,out):
    seed_all(args.seed+100); rng=np.random.default_rng(args.seed+100)
    ev=Evaluator(args.heads,DT).to(data.device)
    ref=rng.choice(data.train,size=min(50000,len(data.train)),replace=False)
    s,a,n,_,_=data.eval_batch(ref); ev.xnorm.fit(features(s,a)); ev.ynorm.fit(motion(s,n)); del s,a,n
    opts=[torch.optim.Adam(h.parameters(),lr=3e-4) for h in ev.heads]
    # Fixed bootstrap resamples of complete training rows, separate for each head.
    boots=[sample(rng,data.train,len(data.train)) for _ in ev.heads]
    fp,w=log(out/'evaluator_training.csv',['update','motion_loss','cost_loss','event_loss'])
    with fp:
        for step in range(1,args.evaluator_steps+1):
            losses=[]
            for head,opt,boot in zip(ev.heads,opts,boots):
                s,a,n,c,e=data.eval_batch(sample(rng,boot,args.batch)); m,cp,ep=head(ev.xnorm(features(s,a)))
                lm=F.mse_loss(m,ev.ynorm(motion(s,n))); lc=F.mse_loss(cp,c); le=F.binary_cross_entropy_with_logits(ep,e)
                loss=lm+args.cost_weight*lc+args.event_weight*le
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite evaluator loss')
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),5.); opt.step()
                losses.append([float(lm.detach()),float(lc.detach()),float(le.detach())])
            if step%args.log_every==0 or step==args.evaluator_steps:
                row=dict(zip(('motion_loss','cost_loss','event_loss'),np.mean(losses,axis=0).tolist()),update=step); w.writerow(row); fp.flush(); print('Evaluator',row,flush=True)
    ev.freeze()
    @torch.no_grad()
    def metrics(idx,calibrate=False):
        # Limit reporting to a fixed random subset; calibration and test remain disjoint.
        idx=rng.choice(idx,size=min(len(idx),20000),replace=False)
        allp={k:[] for k in ('u','k','risk','cost_mean','events','motion','truth','cost','event','real_k')}
        for start in range(0,len(idx),1024):
            s,a,n,c,e=data.eval_batch(idx[start:start+1024]); p=ev.predict(s,a,args.beta); true=motion(s,n)
            vals={**p,'truth':true,'cost':c,'event':e,'real_k':kinematic_residual(s,true,ev.dt)}
            for k in allp: allp[k].append(vals[k].cpu())
        x={k:torch.cat(v) for k,v in allp.items()}
        if calibrate:
            ev.u_limit.copy_(torch.quantile(x['u'],args.quantile).clamp_min(1e-4))
            ev.k_limit.copy_(torch.maximum(torch.quantile(x['k'],args.quantile),torch.quantile(x['real_k'],args.quantile)).clamp_min(1e-4))
        ml=(ev.ynorm.inverse(x['motion'].to(data.device)).cpu()-x['truth']).abs().mean(0).tolist()
        return {'n':len(idx),'motion_mae_physical':ml,'cost_mae':float((x['cost_mean']-x['cost']).abs().mean()*COST_SCALE),'cost_mean_pred':float(x['cost_mean'].mean()*COST_SCALE),'cost_mean_real':float(x['cost'].mean()*COST_SCALE),'event_brier':(x['events']-x['event']).square().mean(0).tolist(),'event_prevalence':x['event'].mean(0).tolist(),'event_predicted_mean':x['events'].mean(0).tolist(),'validity_acceptance':float(((x['u']<=float(ev.u_limit))&(x['k']<=float(ev.k_limit))).float().mean()),'risk_acceptance':float((x['risk']<=args.budget).float().mean()),'u_limit':float(ev.u_limit),'k_limit_m':float(ev.k_limit)}
    report={'calibration':metrics(data.cal,True),'heldout_test':metrics(data.test)}
    (out/'evaluator_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2),flush=True)
    torch.save({'heads':args.heads,'dt':DT,'horizon':args.horizon,'gamma':args.gamma_cost,'beta':args.beta,'state':ev.state_dict()},out/'evaluator.pth')
    return ev

def train_policy(data,args,out,mode,ev,norm):
    seed_all(args.seed); rng=np.random.default_rng(args.seed); m=build_dql(data.device,args.eta)
    fields=['update','actor_loss','bc_loss','ql_loss','critic_loss','dynamics_loss','safety_loss']
    fp,w=log(out/f'{mode}_training.csv',fields); accum=[]
    def save():
        path=out/f'{mode}.pth'; temp=path.with_suffix('.tmp')
        torch.save({'actor':m.actor.state_dict(),'critic':m.critic.state_dict(),'critic_target':m.critic_target.state_dict(),'norm':norm.state_dict(),'mode':mode,'eta':args.eta,'budget':args.budget,'beta':args.beta,'step':m.step},temp); temp.replace(path)
    with fp:
        for step in range(1,args.steps+1):
            b=data.batch(sample(rng,data.train,args.batch)); row=update(m,norm,b,ev if mode=='aware' else None,args.lambda_dyn if mode=='aware' else 0.,args.lambda_safe if mode=='aware' else 0.,args.budget,args.beta); accum.append(row)
            if step%args.log_every==0 or step==args.steps:
                summary={k:float(np.mean([v[k] for v in accum])) for k in row}; summary['update']=step; w.writerow(summary); fp.flush(); accum=[]; print(mode,summary,flush=True)
            if step%args.save_every==0 or step==args.steps: save()
    print('Saved',out/f'{mode}.pth',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--mode',choices=['both','dql','aware','evaluator'],default='both'); p.add_argument('--evaluator-file'); p.add_argument('--seed',type=int,default=42); p.add_argument('--split-seed',type=int,default=2026); p.add_argument('--steps',type=int,default=20000); p.add_argument('--evaluator-steps',type=int,default=5000); p.add_argument('--batch',type=int,default=256); p.add_argument('--heads',type=int,default=5); p.add_argument('--horizon',type=int,default=10); p.add_argument('--gamma-cost',type=float,default=.99); p.add_argument('--beta',type=float,default=1.); p.add_argument('--quantile',type=float,default=.99); p.add_argument('--budget',type=float,default=.25,help='Normalized discounted cost; .25 = 5 cost units'); p.add_argument('--lambda-dyn',type=float,default=.05); p.add_argument('--lambda-safe',type=float,default=.1); p.add_argument('--cost-weight',type=float,default=2.); p.add_argument('--event-weight',type=float,default=1.); p.add_argument('--eta',type=float,default=.01); p.add_argument('--log-every',type=int,default=250); p.add_argument('--save-every',type=int,default=5000); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); args=p.parse_args()
    if min(args.steps,args.evaluator_steps,args.batch,args.horizon,args.log_every,args.save_every)<1 or args.heads<2 or args.budget<=0 or args.beta<0 or not .5<args.quantile<1 or not 0<=args.gamma_cost<=1 or min(args.lambda_dyn,args.lambda_safe,args.cost_weight,args.event_weight,args.eta)<0: p.error('Invalid arguments')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(f'Output directory is nonempty: {out}; use a new directory.')
    (out/'config.json').write_text(json.dumps(vars(args),indent=2)); seed_all(args.seed); device=torch.device(args.device)
    data=Data(args.data,args.split_seed,args.horizon,args.gamma_cost,device)
    rng=np.random.default_rng(args.split_seed); norm=Norm(307).to(device); norm.fit(data.t(data.s,rng.choice(data.train,min(50000,len(data.train)),replace=False)))
    ev=None
    if args.mode in ('both','aware','evaluator'):
        if args.evaluator_file:
            e=torch.load(args.evaluator_file,map_location=device,weights_only=True)
            if e['horizon']!=args.horizon or abs(e['beta']-args.beta)>1e-8 or abs(e['gamma']-args.gamma_cost)>1e-8: raise ValueError('Evaluator configuration mismatch')
            ev=Evaluator(e['heads'],e['dt']).to(device); ev.load_state_dict(e['state']); ev.freeze(); torch.save(e,out/'evaluator.pth')
        else: ev=train_evaluator(data,args,out)
    for mode in ('dql','aware'):
        if args.mode in ('both',mode): train_policy(data,args,out,mode,ev,norm)
    print('COMPLETE:',out,flush=True)
if __name__=='__main__': main()
