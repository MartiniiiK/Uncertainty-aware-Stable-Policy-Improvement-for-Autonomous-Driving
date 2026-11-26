"""Inference on raw upstream states; DA ranks executable candidates with continuous penalties."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from da_dql import build_dql, flatten, project, Evaluator, COST_SCALE

class Policy:
    def __init__(self,checkpoint,mode='dql',device=None,evaluator_file=None,budget=None):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu')); self.mode=mode
        if mode not in ('dql','actor','full'): raise ValueError('V2 supports dql, actor, full')
        c=torch.load(checkpoint,map_location=self.device,weights_only=True)
        if c.get('format')!='official_raw_v2' or c['mode']!=('dql' if mode=='dql' else 'aware'): raise ValueError('Wrong V2 checkpoint')
        self.model=build_dql(self.device,c['eta']); self.model.actor.load_state_dict(c['actor']); self.model.critic_target.load_state_dict(c['critic_target']); self.model.actor.eval(); self.model.critic_target.eval()
        self.budget=float(c['budget'] if budget is None else budget); self.beta=c['beta']; self.cfg=c['selection']; self.ev=None; self.last={}; self.prev=None
        if mode=='full':
            e=torch.load(evaluator_file or Path(checkpoint).parent/'evaluator.pth',map_location=self.device,weights_only=True); self.ev=Evaluator(e['heads'],e['dt']).to(self.device); self.ev.load_state_dict(e['state']); self.ev.freeze(); self.horizon=e['horizon']
    def reset(self): self.prev=None; self.last={}
    @torch.no_grad()
    def __call__(self,obs):
        s=torch.as_tensor(flatten(obs) if isinstance(obs,dict) else np.asarray(obs,dtype=np.float32),device=self.device).reshape(1,307).repeat(50,1)
        raw=self.model.actor.sample(s); q=self.model.critic_target.q_min(s,raw).flatten(); a=project(raw)
        if not torch.isfinite(raw).all() or not torch.isfinite(q).all(): raise FloatingPointError('Nonfinite action/value')
        if self.ev is None:
            i=int(torch.multinomial(torch.softmax(q,dim=0),1)); self.last={'candidate_count':50,'accepted':50,'fallback':False,'q':float(q[i])}
        else:
            p=self.ev.scores(s,a,self.beta); cfg=self.cfg
            # Normalize Q differences per candidate set, avoid arbitrary dependence on Q offset.
            z=(q-q.mean())/q.std(unbiased=False).clamp_min(1.)
            dyn=F.relu(p['u']/self.ev.u_limit-1)+F.relu(p['k']/self.ev.k_limit-1)
            risk=p['risk']/self.budget; event=p['events'].sum(-1); conflict=a[:,0]*a[:,2]
            smooth=torch.zeros_like(q)
            if self.prev is not None:
                smooth=(a[:,1]-self.prev[1]).square()+((a[:,0]-a[:,2])-(self.prev[0]-self.prev[2])).square()
                # Do not penalize increased braking for a potential emergency.
                smooth=torch.where(a[:,2]>self.prev[2]+.1,(a[:,1]-self.prev[1]).square(),smooth)
            score=z-cfg['dyn']*dyn-cfg['risk']*risk-cfg['event']*event-cfg['control']*conflict-cfg['smooth']*smooth
            if not torch.isfinite(score).all(): raise FloatingPointError('Nonfinite DA score')
            i=int(torch.multinomial(torch.softmax(score/cfg['temperature'],dim=0),1)); dm=p['validity']<=1; sm=p['risk']<=self.budget; mask=dm&sm
            # Acceptance is diagnostic, not a hard gate or safety guarantee.
            self.last={'candidate_count':50,'accepted':int(mask.sum()),'dynamics_accepted':int(dm.sum()),'safety_accepted':int(sm.sum()),'fallback':False,'no_feasible':not bool(mask.any()),'q':float(q[i]),'validity':float(p['validity'][i]),'predicted_cost':float(p['risk'][i])*COST_SCALE,'risk_budget':self.budget*COST_SCALE,'collision_probability':float(p['events'][i,0]),'offroad_probability':float(p['events'][i,1]),'horizon':self.horizon,'selection':'soft DA ranking','selected_passes':bool(mask[i])}
        self.prev=a[i].clone(); return a[i].cpu().numpy()
