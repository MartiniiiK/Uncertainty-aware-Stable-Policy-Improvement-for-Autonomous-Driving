"""One offline policy-improvement step over the EasyCarla rule controller."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

from da_dql import flatten, project
from da_control_model import ControlCritics
from da_control_rule import rule_actions
from da_control_support import MIN_SUPPORT_COUNT, candidate_support


SOURCE_NAMES = ("rule", "coast", "full_brake", "hold", "diffusion_drive", "diffusion_brake")
DIFFUSION_DRAWS = 23


def _candidate_actions(rule, draws, previous):
    """At most 50 physical actions. Keep the rule at index zero after deduplication."""
    fixed = rule.reshape(1, 3).repeat(4, 1)
    fixed[1, 0] = fixed[1, 2] = 0
    fixed[2, 0], fixed[2, 2] = 0, 1
    fixed[3] = rule if previous is None else previous
    # A read-back command can differ from the previous proposal. Only hold an
    # actually executable, exclusive command; external overrides remain history.
    if previous is not None and bool((previous[0] > 0) & (previous[2] > 0)):
        fixed[3] = rule

    physical = project(draws[torch.isfinite(draws).all(dim=-1)])
    drive = physical.clone()
    drive[:, 2] = 0
    brake = physical.clone()
    brake[:, 0] = 0
    actions = torch.cat((fixed, drive, brake))
    sources = torch.cat((
        torch.arange(4, device=actions.device),
        torch.full((len(drive),), 4, device=actions.device, dtype=torch.long),
        torch.full((len(brake),), 5, device=actions.device, dtype=torch.long),
    ))
    equal = (actions[:, None, :] == actions[None, :, :]).all(dim=2)
    first = ~torch.tril(equal, diagonal=-1).any(dim=1)
    return actions[first], sources[first]


def _choose(actions, values, support_counts, previous, epsilon_cost):
    """Constrain each ensemble member's paired difference to the rule action."""
    if (values.ndim != 3 or values.shape[0] < 1
            or tuple(values.shape[1:]) != (len(actions), 4)):
        raise ValueError("Control critics must return [ensemble, candidates, 4]")
    if tuple(support_counts.shape) != (len(actions),):
        raise ValueError("Expected one training support count per candidate")
    finite = torch.isfinite(values).all(dim=0).all(dim=1)
    if not bool(finite[0]):
        return 0, {
            "accepted": 0, "fallback": True, "no_feasible": True,
            "selected_passes": False, "fallback_reason": "nonfinite_rule_value",
            "q": math.nan, "predicted_cost": math.nan,
            "reward_improvement": math.nan, "progress_improvement": math.nan,
            "cost_improvement": math.nan, "jerk_improvement": math.nan,
            "best_cost_improvement": math.nan,
        }

    # Positive always means better: reward, route progress, cost reduction,
    # jerk reduction. Pair within each member before taking the minimum.
    signs = values.new_tensor([1, 1, -1, -1])
    gains = ((values - values[:, :1]) * signs).amin(dim=0)
    supported = support_counts >= MIN_SUPPORT_COUNT
    supported[0] = True  # The reference is always an available action.
    feasible = supported & finite & (gains[:, :3] >= 0).all(dim=1)
    best_cost_gain = gains[feasible, 2].max()
    shortlist = feasible & (gains[:, 2] >= (best_cost_gain - epsilon_cost))
    shortlist &= gains[:, 3] == gains[shortlist, 3].max()
    shortlist &= gains[:, 1] == gains[shortlist, 1].max()
    shortlist &= gains[:, 0] == gains[shortlist, 0].max()
    # Preserve the current reference on a value tie. A previous throttle
    # command must not displace a newly braking rule on an uninformative Q.
    if bool(shortlist[0]):
        index = 0
    elif previous is not None:
        steer_change = actions[:, 1] - previous[1]
        pedal_change = (actions[:, 0] - actions[:, 2]) - (previous[0] - previous[2])
        change = steer_change.square() + pedal_change.square()
        index = int(change.masked_fill(~shortlist, float("inf")).argmin())
    else:
        index = int(torch.nonzero(shortlist, as_tuple=True)[0][0])

    accepted = int(feasible.sum())
    chosen_gains = gains[index].tolist()
    reward, _, cost, _ = values[:, index].mean(dim=0).tolist()
    return index, {
        "q": reward, "predicted_cost": cost, "accepted": accepted,
        "fallback": accepted == 1, "no_feasible": False, "selected_passes": True,
        "reward_improvement": chosen_gains[0], "progress_improvement": chosen_gains[1],
        "cost_improvement": chosen_gains[2], "jerk_improvement": chosen_gains[3],
        "best_cost_improvement": float(best_cost_gain),
    }


class ControlPolicy:
    def __init__(self, checkpoint, control_checkpoint, device=None, epsilon_cost=0.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.epsilon_cost = float(epsilon_cost)
        if not math.isfinite(self.epsilon_cost) or self.epsilon_cost < 0:
            raise ValueError("epsilon_cost must be finite and nonnegative")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ckpt.get("format") != "official_raw_v2" or ckpt.get("mode") != "dql":
            raise ValueError("Expected an official_raw_v2 DQL checkpoint for diffusion proposals")
        # The V2 checkpoint stores the 5-step VP diffusion actor. Its optimizer,
        # target actor and original reward critics are unnecessary for proposal sampling.
        example_dir = Path(__file__).resolve().parent / "example"
        if example_dir.is_dir() and str(example_dir) not in sys.path:
            sys.path.insert(0, str(example_dir))
        from agents.diffusion import Diffusion
        from agents.model import MLP
        denoiser = MLP(state_dim=307, action_dim=3, device=self.device)
        self.actor = Diffusion(307, 3, denoiser, 1.0, beta_schedule="vp", n_timesteps=5).to(self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval().requires_grad_(False)
        self.critics = ControlCritics.from_checkpoint(control_checkpoint, self.device)
        self.critics.eval().requires_grad_(False)
        self.reset()

    def reset(self):
        self.previous = None
        self.last = {}

    def observe_executed(self, actual):
        """Use CARLA's control readback when supplied by the evaluator."""
        action = torch.as_tensor(actual, dtype=torch.float32, device=self.device)
        if action.shape != (3,) or not torch.isfinite(action).all() or not torch.equal(action, project(action)):
            raise ValueError("Invalid physical control readback")
        self.previous = action.clone()

    @torch.no_grad()
    def act(self, obs):
        raw = flatten(obs) if isinstance(obs, dict) else np.asarray(obs, dtype=np.float32)
        state = torch.as_tensor(raw, dtype=torch.float32, device=self.device).reshape(1, 307)
        if not torch.isfinite(state).all():
            raise ValueError("Nonfinite driving observation")
        rule = rule_actions(state)[0]
        # The strong V2 actor consumed raw states; the FQE ensemble applies its
        # own fitted transformation inside values().
        draws = self.actor.sample(state.expand(DIFFUSION_DRAWS, -1))
        actions, sources = _candidate_actions(rule, draws, self.previous)
        states = state.expand(len(actions), -1)
        counts = candidate_support(states, actions, self.critics.support_counts)
        values = self.critics.values(states, actions)
        index, self.last = _choose(actions, values, counts, self.previous, self.epsilon_cost)
        self.last.update(
            candidate_count=len(actions),
            supported_count=int((counts >= MIN_SUPPORT_COUNT).sum()),
            selected_support_count=int(counts[index]),
            selected_rule=(index == 0),
            selected_source=SOURCE_NAMES[int(sources[index])],
        )
        self.previous = actions[index].clone()
        return self.previous.cpu().numpy().copy()

    __call__ = act
