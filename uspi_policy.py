"""Uncertainty-aware Stable Policy Improvement (USPI).
Inference-only extension of the archived RCDPI actor and reference FQE ensemble.
All thresholds are provisional research defaults, not calibrated safety guarantees.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from da_dql import flatten, project
from da_control_policy import ControlPolicy, _candidate_actions, SOURCE_NAMES, DIFFUSION_DRAWS
from da_control_rule import rule_actions
from da_control_support import candidate_support


@dataclass(frozen=True)
class StableConfig:
    beta: float = 1.0
    min_support: int = 5
    # Normalized by the FQE per-step RMS (not a statistical confidence level).
    margin: float = 0.0
    min_improvement: float = 1e-6
    steer_step: float = 0.08
    pedal_step: float = 0.12
    steer_ref_slack: float = 0.02
    pedal_ref_slack: float = 0.04
    dt: float = 0.1
    horizon: int = 5
    wheelbase: float = 2.875
    max_wheel_angle: float = 0.5
    accel_gain: float = 3.0
    brake_gain: float = 6.0
    lane_buffer: float = 0.25
    lateral_slack: float = 0.05
    heading_slack: float = 0.03

    def __post_init__(self):
        for name, v in asdict(self).items():
            if not math.isfinite(v) or v < 0:
                raise ValueError(f'Invalid {name}: {v}')
        if type(self.min_support) is not int or type(self.horizon) is not int:
            raise ValueError("min_support and horizon must be integers")
        if self.min_support < 1 or self.horizon < 1 or self.dt <= 0 or self.wheelbase <= 0:
            raise ValueError('Positive support, horizon, dt and wheelbase required')
        if self.max_wheel_angle >= math.pi / 2:
            raise ValueError('Wheel angle must be < pi/2')


def paired_lower_bound(values, scales, beta):
    """[M,N,4] physical values -> [N,4] normalized pessimistic gains.
    Positive means improvement for all four channels. Pair within each head.
    min - beta*population_std is a heuristic, not a calibrated interval.
    """
    if values.ndim != 3 or values.shape[-1] != 4 or values.shape[0] < 2:
        raise ValueError('Expected at least two heads and four outcomes')
    if scales.shape != (4,) or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError('Invalid FQE scales')
    gain = (values - values[:, :1]) * values.new_tensor([1, 1, -1, -1]) / scales
    return gain.amin(0) - beta * gain.std(0, unbiased=False)


def continuity_mask(actions, previous, cfg):
    # At episode start use the current reference as the anchor, not a stale action.
    anchor = actions[0] if previous is None else previous
    steer = (actions[:, 1] - anchor[1]).abs()
    pedal = actions[:, 0] - actions[:, 2]
    du = (pedal - (anchor[0] - anchor[2])).abs()
    return ((steer <= cfg.steer_step) & (du <= cfg.pedal_step)
            & (steer <= steer[0] + cfg.steer_ref_slack)
            & (du <= du[0] + cfg.pedal_ref_slack))


def local_trajectory(state, actions, cfg):
    """Constant-command bicycle proxy, in the current ego frame.
    Returns per-candidate max centerline distance, max heading error, validity.
    Zero-padded route tails and rollout past route end are not trusted.
    This is not a CARLA forward model, and cannot certify collision avoidance.
    """
    wp = state.reshape(-1)[-36:].reshape(12, 3)
    # Keep the first point even if it is exactly the origin with zero yaw.
    valid = torch.nonzero((wp[1:] != 0).any(-1), as_tuple=True)[0]
    n = len(actions)
    inf = actions.new_full((n,), float('inf'))
    bad = torch.zeros(n, dtype=torch.bool, device=actions.device)
    if len(valid) == 0:
        return inf, inf, bad
    end = int(valid[-1]) + 2
    if not bool((wp[1:end] != 0).any(-1).all()):
        return inf, inf, bad
    a, b = wp[:end-1,:2], wp[1:end,:2]
    seg = b-a
    lengths = seg.square().sum(-1)
    keep = lengths > 1e-6
    if not bool(keep.any()):
        return inf, inf, bad
    a, seg, lengths = a[keep], seg[keep], lengths[keep]
    angles = torch.atan2(seg[:,1],seg[:,0])
    v = actions.new_full((n,), max(0., float(state.reshape(-1)[3])))
    xy = actions.new_zeros((n,2)); yaw = actions.new_zeros(n)
    peak_dist = actions.new_zeros(n); peak_heading = actions.new_zeros(n)
    ok = torch.ones_like(bad)
    accel = cfg.accel_gain*actions[:,0] - cfg.brake_gain*actions[:,2]
    wheel = cfg.max_wheel_angle*actions[:,1]
    for _ in range(cfg.horizon):
        v1 = (v + cfg.dt*accel).clamp_min(0.)
        vbar = .5*(v+v1)
        dyaw = cfg.dt*vbar*torch.tan(wheel)/cfg.wheelbase
        mid = yaw + .5*dyaw
        xy = xy + cfg.dt*vbar[:,None]*torch.stack((mid.cos(),mid.sin()),-1)
        yaw = yaw+dyaw; v=v1
        delta = xy[:,None,:]-a[None,:,:]
        t = (delta*seg[None,:,:]).sum(-1)/lengths
        residual = delta-t.clamp(0,1)[...,None]*seg
        dist, idx = residual.square().sum(-1).min(-1)
        peak_dist = torch.maximum(peak_dist,dist.sqrt())
        err = yaw-angles[idx]
        peak_heading = torch.maximum(peak_heading,torch.atan2(err.sin(),err.cos()).abs())
        # Reject trajectories that extend beyond the last available segment.
        ok &= ~((idx == len(a)-1) & (t[:,-1] > 1.0))
    return peak_dist,peak_heading,ok & torch.isfinite(peak_dist) & torch.isfinite(peak_heading)


def stable_select(state, actions, values, counts, scales, previous, cfg):
    gains = paired_lower_bound(values, scales, cfg.beta)
    supported = counts >= cfg.min_support
    finite = torch.isfinite(gains).all(-1)
    advantage = finite & (gains >= cfg.margin).all(-1) & (gains > cfg.min_improvement).any(-1)
    continuity = continuity_mask(actions, previous, cfg)
    dist, heading, route_valid = local_trajectory(state, actions, cfg)
    lane_width = state.reshape(-1)[9]
    lane_ok = (route_valid & route_valid[0] & (lane_width > 2*cfg.lane_buffer)
               & (dist <= lane_width/2-cfg.lane_buffer)
               & (dist <= dist[0]+cfg.lateral_slack)
               & (heading <= heading[0]+cfg.heading_slack))
    physical = torch.isfinite(actions).all(-1) & (actions == project(actions)).all(-1)
    physical &= ~((actions[:,0]>0)&(actions[:,2]>0))
    # Preserve the reference emergency response; no candidate may weaken it.
    front = state.reshape(-1)[7]
    if bool((front > 0) & (front < 7)):
        physical &= (actions[:,0]==0) & (actions[:,2]>=actions[0,2])
    feasible = physical & supported & advantage & continuity & lane_ok
    feasible[0] = False  # reference is a fallback, not an accepted improvement
    candidates = torch.nonzero(feasible, as_tuple=True)[0].tolist()
    # Conservative lexicographic ranking: cost, jerk, progress, reward.
    # Stable index tie-break; no unsafe post-selection action smoothing.
    idx = max(candidates,key=lambda i:(float(gains[i,2]),float(gains[i,3]),float(gains[i,1]),float(gains[i,0]),-i)) if candidates else 0
    finite_value=bool(torch.isfinite(values[:,idx]).all())
    selected_gain=gains[idx]*scales
    diagnostics = dict(
        accepted=len(candidates)+1, fallback=not bool(candidates),
        no_feasible=not bool(candidates), selected_passes=bool(candidates),
        q=float(values[:,idx,0].mean()) if finite_value else math.nan,
        predicted_cost=float(values[:,idx,2].mean()) if finite_value else math.nan,
        **{name+'_improvement':float(selected_gain[k]) for k,name in enumerate(('reward','progress','cost','jerk'))},
        best_cost_improvement=float(selected_gain[2]),
        gate_support_count=int(supported[1:].sum()), gate_advantage_count=int((supported&advantage)[1:].sum()),
        gate_continuity_count=int((supported&advantage&continuity)[1:].sum()), gate_trajectory_count=len(candidates),
        selected_lateral_proxy_m=float(dist[idx]),selected_heading_proxy_rad=float(heading[idx]),
        fallback_reason='no_admissible_improvement' if not candidates else '',
    )
    return idx, diagnostics


class USPIPolicy(ControlPolicy):
    def __init__(self, checkpoint, control_checkpoint, device=None, config=None):
        self.config = config or StableConfig()
        super().__init__(checkpoint, control_checkpoint, device, epsilon_cost=0.)
        metadata=torch.load(control_checkpoint,map_location='cpu',weights_only=True)
        if not math.isclose(float(metadata['dt']),self.config.dt):
            raise ValueError('FQE and rollout dt differ')
        if len(self.critics.heads)<2:
            raise ValueError('USPI requires at least two FQE ensemble members')

    @torch.no_grad()
    def act(self, obs):
        raw=flatten(obs) if isinstance(obs,dict) else np.asarray(obs,dtype=np.float32)
        state=torch.as_tensor(raw,dtype=torch.float32,device=self.device).reshape(1,307)
        if not bool(torch.isfinite(state).all()): raise ValueError('Nonfinite observation')
        rule=rule_actions(state)[0]
        draws=self.actor.sample(state.expand(DIFFUSION_DRAWS,-1))
        actions,sources=_candidate_actions(rule,draws,self.previous)
        states=state.expand(len(actions),-1)
        counts=candidate_support(states,actions,self.critics.support_counts)
        values=self.critics.values(states,actions)
        idx,self.last=stable_select(state,actions,values,counts,self.critics.outcome_scale,self.previous,self.config)
        self.last.update(candidate_count=len(actions),supported_count=int((counts>=self.config.min_support).sum()),
                         selected_support_count=int(counts[idx]),selected_rule=(idx==0),selected_source=SOURCE_NAMES[int(sources[idx])])
        # The evaluator replaces this proposal with actual CARLA readback.
        self.previous=actions[idx].clone()
        return self.previous.cpu().numpy().copy()
    __call__=act
