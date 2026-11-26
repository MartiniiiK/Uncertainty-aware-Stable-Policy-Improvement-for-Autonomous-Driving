"""Fit the reference-rule critics on the strong42 training trajectory split."""

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.nn import functional as F

from train_da_dql import Data, seed_all
from da_control_model import ControlCritics, FORMAT, OUTCOMES, executed_actions, physical_signals, state_features
from da_control_rule import rule_actions
from da_control_support import build_support_counts


class EpisodeBootstrap:
    """Draw complete train episodes with replacement, sample their rows uniformly."""

    def __init__(self, starts, ends, train_ids, rng):
        sampled = rng.choice(train_ids, size=len(train_ids), replace=True)
        self.starts = starts[sampled]
        lengths = ends[sampled] - self.starts
        self.offsets = np.r_[0, np.cumsum(lengths)]
        self.rng = rng

    def rows(self, batch_size):
        positions = self.rng.integers(self.offsets[-1], size=batch_size)
        episodes = np.searchsorted(self.offsets, positions, side="right") - 1
        return self.starts[episodes] + positions - self.offsets[episodes]


def episode_bootstraps(data, heads, seed):
    ends = np.flatnonzero(data.d) + 1
    starts = np.r_[0, ends[:-1]]
    train_mask = np.zeros(len(data.d), dtype=bool)
    train_mask[data.train] = True
    train_ids = np.flatnonzero(train_mask[starts])
    sizes = ends[train_ids] - starts[train_ids]
    if sizes.sum() != len(data.train) or not np.all(train_mask[ends[train_ids] - 1]):
        raise ValueError("Expected the offline split to contain complete training episodes")
    seeds = np.random.SeedSequence(seed).spawn(heads)
    return [EpisodeBootstrap(starts, ends, train_ids, np.random.default_rng(s)) for s in seeds]


def batch(data, rows, dt):
    s, a, ns, r, nd = data.batch(rows)
    progress, jerk = physical_signals(s, ns, dt)
    signals = torch.stack((r[:, 0], progress, data.t(data.c, rows), jerk), dim=-1)
    return s, executed_actions(a), ns, signals, nd


@torch.no_grad()
def fit_scales(model, data, dt, rng):
    # All statistics used at deployment are fixed from the actor/critic training split.
    rows = rng.choice(data.train, size=min(20000, len(data.train)), replace=False)
    s, _, _, signals, _ = batch(data, rows, dt)
    features = state_features(s)
    model.state_mean.copy_(features.mean(0))
    model.state_scale.copy_(features.std(0, unbiased=False).clamp_min(1e-2))
    # Scale only, without centering. This keeps terminal Bellman targets exact.
    model.outcome_scale.copy_(signals.square().mean(0).sqrt().clamp_min(0.1))
    model.support_counts.copy_(build_support_counts(data.s, data.a, data.train).to(model.support_counts.device))


def train(args):
    device = torch.device(args.device)
    seed_all(args.seed)
    # The original loader draws an adjacency probe before its episode permutation when
    # horizon > 1. A horizon of 1 would silently change strong42's split at the same seed.
    data = Data(args.data, args.split_seed, 10, args.gamma, device)
    logged = data.a[data.train]
    out_of_bounds = ((logged < [0, -1, 0]) | (logged > 1)).any(axis=1)
    overlap = (logged[:, 0] > 0) & (logged[:, 2] > 0)
    print(f"Train actions: {len(logged)} total, {out_of_bounds.sum()} saturated by env bounds, "
          f"{overlap.sum()} simultaneous positive pedals retained", flush=True)
    del logged

    model = ControlCritics(args.heads, args.width).to(device)
    fit_scales(model, data, args.dt, np.random.default_rng(args.seed))
    target = deepcopy(model).eval().requires_grad_(False)
    samplers = episode_bootstraps(data, args.heads, args.seed)
    optimizers = [torch.optim.Adam(head.parameters(), lr=args.lr) for head in model.heads]
    train_loss = torch.zeros(len(OUTCOMES), device=device)
    since_report = 0
    for step in range(1, args.steps + 1):
        for head, frozen, opt, sampler in zip(model.heads, target.heads, optimizers, samplers):
            s, a, ns, signals, nd = batch(data, sampler.rows(args.batch), args.dt)
            with torch.no_grad():
                next_q = frozen(model.normalized_state(ns), rule_actions(ns))
                target_q = signals / model.outcome_scale + args.gamma * nd * next_q
            q = head(model.normalized_state(s), a)
            losses = F.mse_loss(q, target_q, reduction="none").mean(0)
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite critic loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0, error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for target_param, param in zip(frozen.parameters(), head.parameters()):
                    target_param.lerp_(param, args.tau)
                train_loss.add_(losses.detach())
            since_report += 1
        if step % args.log_every == 0 or step == args.steps:
            averages = (train_loss / since_report).tolist()
            print(f"step={step} normalized_td_mse={dict(zip(OUTCOMES, averages))}", flush=True)
            train_loss.zero_()
            since_report = 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "critics.pt"
    checkpoint = {
        "format": FORMAT, "outcomes": list(OUTCOMES), "model": model.state_dict(),
        "heads": args.heads, "width": args.width, "dt": args.dt, "gamma": args.gamma,
        "seed": args.seed, "split_seed": args.split_seed, "steps_per_head": args.steps,
        "terminal_mask": "logged done", "continuation": "easycarla_town03_rule",
    }
    # Only complete checkpoints become visible to the evaluator.
    with NamedTemporaryFile(dir=out, prefix="critics-", suffix=".tmp", delete=False) as fp:
        temporary = Path(fp.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/easycarla_offline_dataset.hdf5")
    parser.add_argument("--out", default="da_runs/da_control", help="output directory for critics.pt")
    parser.add_argument("--steps", type=int, default=20000, help="gradient updates per ensemble member")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005, help="target parameter update rate")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.heads, args.width, args.log_every) < 1:
        parser.error("steps, batch, heads, width and log-every must be positive")
    if not (0 <= args.gamma < 1 and 0 < args.tau <= 1 and args.dt > 0 and args.lr > 0):
        parser.error("expected 0 <= gamma < 1, 0 < tau <= 1, dt > 0, lr > 0")
    train(args)


if __name__ == "__main__":
    main()
