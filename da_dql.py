"""Dynamics-aware DQL, independent extension to EasyCarla-RL.
No synthetic replay. All Bellman targets use logged transitions.
"""
from pathlib import Path
import sys, math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

KEYS = ('ego_state','lane_info','lidar','nearby_vehicles','waypoints')
STATE_DIM, ACTION_DIM = 307, 3
DT, DESIRED_SPEED, COST_SCALE = .1, 8., 20.

def flatten(obs):
    x = np.concatenate([np.asarray(obs[k]).reshape(-1) for k in KEYS]).astype(np.float32)
    if x.shape != (307,) or not np.isfinite(x).all():
        raise ValueError(f'Invalid EasyCarla observation {x.shape}')
    return x

def project(a):
    return torch.maximum(torch.minimum(a, a.new_tensor([1.,1.,1.])), a.new_tensor([0.,-1.,0.]))

def wrap(x): return torch.atan2(torch.sin(x),torch.cos(x))

def features(s,a):
    # Local state: remove absolute map pose; rotate world-frame accelerations.
    x=s[...,3:].clone()
    c,si=torch.cos(s[...,2]),torch.sin(s[...,2])
    x[...,2]=c*s[...,5]+si*s[...,6]
    x[...,3]=-si*s[...,5]+c*s[...,6]
    # Nearby vehicles and waypoints include relative angles.
    for j in [253+4*i for i in range(5)]: # global index 253 = first nearby yaw
        x[...,j-3]=wrap(s[...,j])
    for j in [273+3*i for i in range(12)]: # global index 273 = first route yaw
        x[...,j-3]=wrap(s[...,j])
    return torch.cat((x,project(a)),dim=-1)

def motion(s,n):
    dx,dy=n[...,0]-s[...,0],n[...,1]-s[...,1]
    c,si=torch.cos(s[...,2]),torch.sin(s[...,2])
    return torch.stack((c*dx+si*dy,-si*dx+c*dy,wrap(n[...,2]-s[...,2]),n[...,3]-s[...,3],n[...,10]),dim=-1)

def kinematic_residual(s,m,dt=DT):
    # Trapezoidal planar displacement, approximate no-slip relation.
    v=s[...,3]; v1=v+m[...,3]; yaw=m[...,2]
    x=.5*dt*(v+v1*torch.cos(yaw)); y=.5*dt*v1*torch.sin(yaw)
    return torch.sqrt((m[...,0]-x).square()+(m[...,1]-y).square()+1e-10)

class Norm(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.register_buffer('mean',torch.zeros(dim)); self.register_buffer('std',torch.ones(dim))
    @torch.no_grad()
    def fit(self,x):
        self.mean.copy_(x.mean(0)); self.std.copy_(x.std(0,unbiased=False).clamp_min(.01))
    def forward(self,x): return (x-self.mean)/self.std
    def inverse(self,x): return x*self.std+self.mean

class DynamicsHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(307,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.motion=nn.Linear(256,5)
        self.risk=nn.Sequential(nn.Linear(261,128),nn.SiLU(),nn.Linear(128,3))
    def forward(self,x):
        h=self.body(x); m=self.motion(h); y=self.risk(torch.cat((h,m),-1))
        # Normalized cumulative cost and logits for collision and off-road events.
        return m,F.softplus(y[...,0]),y[...,1:]

class Evaluator(nn.Module):
    def __init__(self,heads=5,dt=DT):
        super().__init__(); self.dt=dt
        self.xnorm=Norm(307); self.ynorm=Norm(5)
        self.heads=nn.ModuleList(DynamicsHead() for _ in range(heads))
        self.register_buffer('u_limit',torch.tensor(1.))
        self.register_buffer('k_limit',torch.tensor(1.))
    def predict(self,s,a,beta=1.):
        x=self.xnorm(features(s,a)); preds=[h(x) for h in self.heads]
        m=torch.stack([p[0] for p in preds]); c=torch.stack([p[1] for p in preds]); e=torch.stack([p[2].sigmoid() for p in preds])
        # Epistemic proxy: ensemble disagreement in normalized motion space.
        u=torch.sqrt(m.var(0,unbiased=False).mean(-1)+1e-10)
        # Mean+beta*std of per-head kinematic residual and predicted cost.
        k=kinematic_residual(s.unsqueeze(0),self.ynorm.inverse(m),self.dt)
        km=k.mean(0)+beta*torch.sqrt(k.var(0,unbiased=False)+1e-10)
        risk=c.mean(0)+beta*torch.sqrt(c.var(0,unbiased=False)+1e-10)
        return dict(motion=m.mean(0),u=u,k=km,risk=risk,cost_mean=c.mean(0),events=e.mean(0))
    def scores(self,s,a,beta=1.):
        p=self.predict(s,a,beta); p['validity']=torch.maximum(p['u']/self.u_limit,p['k']/self.k_limit); return p
    def freeze(self):
        self.eval(); self.requires_grad_(False); return self

def build_dql(device,eta=.01):
    # Original example package remains untouched.
    root=Path(__file__).resolve().parent
    for d in (root/'example',root.parent/'example'):
        if (d/'agents'/'ql_diffusion.py').is_file():
            sys.path.insert(0,str(d)); break
    from agents.ql_diffusion import Diffusion_QL
    return Diffusion_QL(state_dim=307,action_dim=3,max_action=1.,device=device,discount=.99,tau=.005,eta=eta,beta_schedule='vp',n_timesteps=5)

def update(model,norm,batch,evaluator=None,lam_dyn=0.,lam_safe=0.,budget=.25,beta=1.):
    s,a,ns,r,nd=batch; a=project(a); z,nz=norm(s),norm(ns)
    # Shared real-data Bellman update; use executed-action bounds consistently.
    with torch.no_grad():
        na=project(model.ema_model(nz)); tq1,tq2=model.critic_target(nz,na)
        target=r+nd*model.discount*torch.minimum(tq1,tq2)
    q1,q2=model.critic(z,a); cl=F.mse_loss(q1,target)+F.mse_loss(q2,target)
    model.critic_optimizer.zero_grad(set_to_none=True); cl.backward()
    nn.utils.clip_grad_norm_(model.critic.parameters(),model.grad_norm); model.critic_optimizer.step()
    # Freeze critic weights, NOT input gradients. Same for the evaluator.
    model.critic.requires_grad_(False)
    try:
        bc=model.actor.loss(a,z); act=project(model.actor(z)); q1,q2=model.critic(z,act)
        ql= -q1.mean()/q2.abs().mean().detach().clamp_min(1e-3) if np.random.uniform()>.5 else -q2.mean()/q1.abs().mean().detach().clamp_min(1e-3)
        dyn=bc.new_zeros(()); safe=bc.new_zeros(())
        if evaluator is not None and (lam_dyn or lam_safe):
            p=evaluator.scores(s,act,beta)
            dyn=(F.relu(p['u']/evaluator.u_limit-1)+F.relu(p['k']/evaluator.k_limit-1)).mean()
            safe=F.relu(p['risk']/max(budget,1e-6)-1).mean()
        loss=bc+model.eta*ql+lam_dyn*dyn+lam_safe*safe
        if not torch.isfinite(loss+cl): raise FloatingPointError('Non-finite loss')
        model.actor_optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.actor.parameters(),model.grad_norm); model.actor_optimizer.step()
    finally: model.critic.requires_grad_(True)
    with torch.no_grad():
        # Shared warm-up correction: EMA tracks current actor until warm-up ends.
        if model.step<model.step_start_ema: model.ema_model.load_state_dict(model.actor.state_dict())
        elif model.step%model.update_ema_every==0: model.ema.update_model_average(model.ema_model,model.actor)
        for p,t in zip(model.critic.parameters(),model.critic_target.parameters()): t.lerp_(p,model.tau)
    model.step+=1
    return {k:float(v.detach()) for k,v in dict(actor_loss=loss,bc_loss=bc,ql_loss=ql,critic_loss=cl,dynamics_loss=dyn,safety_loss=safe).items()}

class Policy:
    def __init__(self,checkpoint,mode='dql',device=None,evaluator_file=None,budget=None,beta=None,candidates=50):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        data=torch.load(checkpoint,map_location=self.device,weights_only=True)
        expected='aware' if mode in ('actor','full') else 'dql'
        if data.get('mode')!=expected: raise ValueError(f'{mode} requires a {expected} training checkpoint')
        self.model=build_dql(self.device,float(data.get('eta',.01)))
        for k in ('actor','critic','critic_target'):
            getattr(self.model,k).load_state_dict(data[k]); getattr(self.model,k).eval()
        self.norm=Norm(307).to(self.device); self.norm.load_state_dict(data['norm']); self.mode=mode
        self.budget=float(data.get('budget',.25) if budget is None else budget); self.beta=float(data.get('beta',1.) if beta is None else beta); self.candidates=candidates
        if self.budget<=0 or self.beta<0 or candidates<1: raise ValueError('Invalid selection configuration')
        self.ev=None
        if mode in ('gate','full','dyn_gate','risk_gate') or evaluator_file:
            file=Path(evaluator_file) if evaluator_file else Path(checkpoint).parent/'evaluator.pth'
            e=torch.load(file,map_location=self.device,weights_only=True); self.ev=Evaluator(e['heads'],e['dt']).to(self.device); self.ev.load_state_dict(e['state']); self.ev.freeze()
            if abs(float(e.get('beta',self.beta))-self.beta)>1e-8: raise ValueError('Use the beta used during evaluator calibration')
            self.horizon=e['horizon']
        self.last={}
    @torch.no_grad()
    def __call__(self,obs):
        s=torch.as_tensor(flatten(obs) if isinstance(obs,dict) else np.asarray(obs,dtype=np.float32),device=self.device).view(1,307).repeat(self.candidates,1)
        z=self.norm(s); a=project(self.model.actor.sample(z)); q=self.model.critic_target.q_min(z,a).flatten()
        if not torch.isfinite(a).all() or not torch.isfinite(q).all(): raise RuntimeError('Nonfinite actor/critic output')
        mask=torch.ones_like(q,dtype=torch.bool); p=None; fallback=False
        if self.ev is not None:
            p=self.ev.scores(s,a,self.beta)
            if not all(torch.isfinite(p[k]).all() for k in ('validity','risk')): raise RuntimeError('Nonfinite evaluator output')
            dyn=p['validity']<=1; safe=p['risk']<=self.budget
            if self.mode in ('gate','full','dyn_gate'): mask &= dyn
            if self.mode in ('gate','full','risk_gate'): mask &= safe
        if mask.any(): idx=int(torch.multinomial(torch.softmax(q.masked_fill(~mask,-torch.inf),dim=0),1).item())
        else:
            # Explicit least-violation fallback within the proposed set; no guarantee.
            fallback=True; violation=torch.zeros_like(q)
            if self.mode in ('gate','full','dyn_gate'): violation+=F.relu(p['validity']-1)
            if self.mode in ('gate','full','risk_gate'): violation+=F.relu(p['risk']/self.budget-1)
            idx=int(violation.argmin())
        self.last={'candidate_count':self.candidates,'accepted':int(mask.sum()),'fallback':fallback,'q':float(q[idx])}
        if p is not None:
            self.last.update(validity=float(p['validity'][idx]),predicted_cost=float(p['risk'][idx]*COST_SCALE),risk_budget=self.budget*COST_SCALE,collision_probability=float(p['events'][idx,0]),offroad_probability=float(p['events'][idx,1]),dynamics_accepted=int(dyn.sum()),safety_accepted=int(safe.sum()),horizon=self.horizon)
        return a[idx].cpu().numpy()
