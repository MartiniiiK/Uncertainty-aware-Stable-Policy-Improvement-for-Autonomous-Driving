"""Tensor version of the Town03 evaluator's rule, shared by FQE and deployment."""
import torch


def rule_actions(states: torch.Tensor) -> torch.Tensor:
    """Map unnormalized EasyCarla observations [..., 307] to (throttle, steer, brake)."""
    if states.shape[-1] != 307:
        raise ValueError("Expected a raw EasyCarla state with 307 features")
    # The last 36 entries are twelve ego-frame waypoints (x, y, relative yaw).
    target = states[..., -36:].reshape(*states.shape[:-1], 12, 3)[..., 3, :]
    heading = torch.atan2(target[..., 1], target[..., 0].clamp_min(0.1))
    steer = (1.5 * heading).clamp(-0.8, 0.8)
    speed = states[..., 3]
    front = states[..., 7]
    close = (front > 0) & (front < 7)
    throttle = (0.25 * (8.0 - speed)).clamp(0.0, 0.8)
    brake = (0.35 * (speed - 8.0)).clamp(0.0, 1.0)
    throttle = torch.where(close, torch.zeros_like(throttle), throttle)
    brake = torch.where(close, brake.clamp_min(0.6), brake)
    return torch.stack((throttle, steer, brake), dim=-1)
