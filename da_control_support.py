"""Counts of observed pedal commands in coarse Town03 driving contexts."""
import numpy as np
import torch

from da_control_rule import rule_actions


NUM_SUPPORT_BINS = 6 * 4 * 3 * 4 * 4 * 5
MIN_SUPPORT_COUNT = 5


def _bucket(value, edges):
    return torch.bucketize(value.contiguous(), value.new_tensor(edges), right=True)


def _bin_ids(states, actions):
    # Zero front range is the environment's no-detection sentinel.
    speed = _bucket(states[:, 3], [0.5, 3.0, 6.0, 9.0, 14.0])
    front = states[:, 7]
    gap = torch.where(front > 0, 1 + _bucket(front, [7.0, 15.0]), 0)
    route = _bucket(rule_actions(states)[:, 1], [-0.2, 0.2])
    throttle = actions[:, 0].clamp(0, 1)
    brake = actions[:, 2].clamp(0, 1)
    # Coast, drive, brake, and simultaneous pedals stay in different bins.
    mode = (throttle > 0).long() + 2 * (brake > 0).long()
    strength = _bucket(torch.maximum(throttle, brake), [0.25, 0.65, 0.9])
    steer = _bucket(actions[:, 1].clamp(-1, 1), [-0.55, -0.15, 0.15, 0.55])
    return ((((speed * 4 + gap) * 3 + route) * 4 + mode) * 4 + strength) * 5 + steer


def build_support_counts(states, actions, train_indices):
    """Build the table on CPU without copying the full observation matrix."""
    counts = torch.zeros(NUM_SUPPORT_BINS, dtype=torch.long)
    for start in range(0, len(train_indices), 8192):
        rows = train_indices[start:start + 8192]
        state_batch = torch.as_tensor(np.asarray(states[rows]), dtype=torch.float32)
        action_batch = torch.as_tensor(np.asarray(actions[rows]), dtype=torch.float32)
        counts += torch.bincount(_bin_ids(state_batch, action_batch), minlength=NUM_SUPPORT_BINS)
    return counts


def candidate_support(states, actions, counts):
    return counts[_bin_ids(states.expand(len(actions), -1), actions)]
