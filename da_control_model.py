"""Offline estimates of four discounted returns under EasyCarla's rule policy."""

import torch
from torch import nn

from da_control_support import NUM_SUPPORT_BINS


OUTCOMES = ("reward", "progress", "cost", "jerk")
FORMAT = "easycarla-rule-fqe-v1"


def physical_signals(s, ns, dt=0.1):
    """Signed progress (m) and planar jerk magnitude (m/s^3) on a real transition."""
    # Waypoint 0 is the nearest lane center. Its yaw is relative to ego, in radians.
    lane_yaw = s[..., 2] + s[..., 273]
    delta_xy = ns[..., :2] - s[..., :2]
    progress = lane_yaw.cos() * delta_xy[..., 0] + lane_yaw.sin() * delta_xy[..., 1]
    has_route = (s[..., 271:307] != 0).any(dim=-1)
    progress = torch.where(has_route, progress, torch.zeros_like(progress))
    delta_accel = ns[..., 5:7] - s[..., 5:7]
    jerk = torch.linalg.vector_norm(delta_accel, dim=-1) / dt
    return progress, jerk


def state_features(s):
    return torch.cat((s[..., :2], s[..., 2:3].cos(), s[..., 2:3].sin(), s[..., 3:]), dim=-1)


def executed_actions(a):
    """The component-wise saturation used by carla_env.step; preserve pedal overlap."""
    return torch.maximum(torch.minimum(a, a.new_tensor([1, 1, 1])), a.new_tensor([0, -1, 0]))


class ValueHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(308, width), nn.LayerNorm(width), nn.SiLU())
        self.state_action = nn.Sequential(
            nn.Linear(width + 5, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, len(OUTCOMES)),
        )

    def forward(self, state, action):
        # Explicit zero-pedal indicators distinguish coasting and brake release.
        pedals_on = (action[..., [0, 2]] > 0).to(action.dtype)
        features = torch.cat((self.state_encoder(state), action, pedals_on), dim=-1)
        return self.state_action(features)


class ControlCritics(nn.Module):
    def __init__(self, heads=3, width=256):
        super().__init__()
        self.width = width
        self.heads = nn.ModuleList(ValueHead(width) for _ in range(heads))
        self.register_buffer("state_mean", torch.zeros(308))
        self.register_buffer("state_scale", torch.ones(308))
        self.register_buffer("outcome_scale", torch.ones(len(OUTCOMES)))
        self.register_buffer("support_counts", torch.zeros(NUM_SUPPORT_BINS, dtype=torch.long))

    def normalized_state(self, s):
        return (state_features(s) - self.state_mean) / self.state_scale

    def values(self, s, a):
        """Return [ensemble, batch, 4] in physical/reporting units."""
        x = self.normalized_state(s)
        a = executed_actions(a)
        return torch.stack([head(x, a) for head in self.heads]) * self.outcome_scale

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format") != FORMAT or tuple(checkpoint.get("outcomes", ())) != OUTCOMES:
            raise ValueError(f"Not a DA control rule-policy FQE checkpoint: {path}")
        model = cls(checkpoint["heads"], checkpoint["width"]).to(device)
        model.load_state_dict(checkpoint["model"])
        return model.eval().requires_grad_(False)
