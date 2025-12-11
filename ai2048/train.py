from dataclasses import dataclass
from datetime import datetime
import statistics
import os
import random
from collections import deque
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer

from ai2048.game import Game
from ai2048.vec_game import VectorizedGame
from ai2048.model import Policy, Value


@dataclass
class GameResult:
    states: torch.Tensor
    valid_actions: torch.Tensor
    actions: torch.Tensor
    probabilities: torch.Tensor
    rewards: torch.Tensor


def play_games(
    policy: Policy, device: torch.device, value_net: Value | None = None
) -> tuple[list[GameResult], dict[str, torch.Tensor | float | int]]:
    rollout_cfg = {
        "num_envs": 256,
        "target_episodes": 256,
        "overshoot_factor": 1.5,
        "min_training_length": 100,
        "short_fraction": 0.45,
        "late_tile_threshold": 512,
        "curriculum_cap": 0.25,
    }

    num_envs = rollout_cfg["num_envs"]
    target_episodes = rollout_cfg["target_episodes"]
    overshoot_factor = rollout_cfg["overshoot_factor"]
    min_training_length = rollout_cfg["min_training_length"]
    short_fraction = rollout_cfg["short_fraction"]
    late_tile_threshold = rollout_cfg["late_tile_threshold"]
    curriculum_cap = rollout_cfg["curriculum_cap"]

    vec_game = VectorizedGame(num_envs, device)
    policy_device = next(policy.parameters()).device

    # Storage for all steps
    episode_states = []
    episode_valid_actions = []
    episode_actions = []
    episode_probs = []

    # Track start index for each env
    env_start_indices = torch.zeros(num_envs, dtype=torch.long, device=device)

    completed_results = []
    late_state_buffer = deque(maxlen=4096)

    target_with_overshoot = target_episodes * overshoot_factor

    current_step = 0

    while len(completed_results) < target_with_overshoot:
        states = vec_game.board.float()

        # Compute moves once
        all_next_states = vec_game.get_moves()
        valid_actions = vec_game.get_valid_actions(all_next_states)

        # Fix for finished games (avoid NaN)
        valid_actions[valid_actions.sum(dim=1) == 0] = 1.0

        states_for_policy = states.to(policy_device)
        valid_for_policy = valid_actions.to(policy_device)
        with torch.no_grad():
            policy_outputs = policy(states_for_policy, valid_for_policy)
        actions = torch.multinomial(policy_outputs, 1).squeeze(1)
        probabilities = policy_outputs[
            torch.arange(num_envs, device=policy_device), actions
        ].detach()

        actions_cpu = actions.to(device)
        probabilities_cpu = probabilities.to(device)

        vec_game.step(actions_cpu, all_next_states)
        dones = vec_game.get_done()

        # collect late-game states for curriculum restarts
        with torch.no_grad():
            max_tiles = vec_game.board.view(num_envs, -1).amax(dim=1)
            late_mask = max_tiles >= late_tile_threshold
            if late_mask.any():
                for state in vec_game.board[late_mask]:
                    late_state_buffer.append(state.clone())

        episode_states.append(states)
        episode_valid_actions.append(valid_actions)
        episode_actions.append(actions_cpu)
        episode_probs.append(probabilities_cpu)

        # Handle done envs
        done_indices = torch.nonzero(dones).squeeze(-1)
        if len(done_indices) > 0:
            for idx in done_indices:
                env_idx = idx.item()
                start_step = env_start_indices[env_idx].item()
                end_step = current_step
                completed_results.append(
                    {
                        "env_idx": env_idx,
                        "start_step": start_step,
                        "end_step": end_step,
                        "length": end_step - start_step + 1,
                    }
                )
                env_start_indices[env_idx] = current_step + 1
            vec_game.reset(done_indices)
            if late_state_buffer:
                dynamic_prob = min(curriculum_cap, len(late_state_buffer) / 2000)
                if dynamic_prob > 0:
                    num_curriculum = int(len(done_indices) * dynamic_prob)
                    num_curriculum = min(num_curriculum, len(late_state_buffer))
                    if num_curriculum > 0:
                        perm = torch.randperm(len(done_indices))[:num_curriculum]
                        chosen_envs = done_indices[perm]
                        sampled_states = torch.stack(
                            [
                                late_state_buffer[
                                    int(
                                        torch.randint(
                                            0, len(late_state_buffer), (1,)
                                        ).item()
                                    )
                                ].clone()
                                for _ in range(num_curriculum)
                            ]
                        )
                        vec_game.set_states(chosen_envs, sampled_states)

        current_step += 1

    # Stack to (T, N, ...)
    all_states = torch.stack(episode_states)
    all_valid = torch.stack(episode_valid_actions)
    all_actions = torch.stack(episode_actions)
    all_probs = torch.stack(episode_probs)

    lengths = torch.tensor(
        [meta["length"] for meta in completed_results], dtype=torch.long
    )

    eligible_indices = (
        torch.nonzero(lengths >= min_training_length).squeeze(-1).tolist()
    )
    short_indices = torch.nonzero(lengths < min_training_length).squeeze(-1).tolist()

    selected: list[dict] = []
    short_selected_count = 0

    short_quota = int(target_episodes * short_fraction)
    if short_quota > 0 and short_indices:
        short_pick = min(short_quota, len(short_indices))
        chosen = random.sample(short_indices, short_pick)
        short_selected_count = len(chosen)
        selected.extend([completed_results[idx] for idx in chosen])

    remaining_quota = target_episodes - len(selected)
    if remaining_quota > 0 and eligible_indices:
        if len(eligible_indices) >= remaining_quota:
            perm = torch.randperm(len(eligible_indices))[:remaining_quota]
            selected.extend(
                [completed_results[eligible_indices[i]] for i in perm.tolist()]
            )
        else:
            selected.extend([completed_results[i] for i in eligible_indices])
            remaining_needed = remaining_quota - len(eligible_indices)
            if remaining_needed > 0:
                remaining_indices = [
                    i
                    for i in range(len(completed_results))
                    if i not in eligible_indices
                ]
                if remaining_indices:
                    perm = torch.randperm(len(remaining_indices))[
                        : min(remaining_needed, len(remaining_indices))
                    ]
                    selected.extend(
                        [completed_results[remaining_indices[i]] for i in perm.tolist()]
                    )

    if len(selected) < target_episodes and len(completed_results) > 0:
        while len(selected) < target_episodes:
            selected.append(
                completed_results[
                    int(torch.randint(0, len(completed_results), (1,)).item())
                ]
            )

    results = []
    selected_lengths: list[int] = []
    for meta in selected:
        env_idx = meta["env_idx"]
        start = meta["start_step"]
        end = meta["end_step"]
        length = end - start + 1
        if length <= 0:
            continue
        selected_lengths.append(length)
        results.append(
            GameResult(
                states=all_states[start : end + 1, env_idx],
                valid_actions=all_valid[start : end + 1, env_idx],
                actions=all_actions[start : end + 1, env_idx],
                probabilities=all_probs[start : end + 1, env_idx],
                rewards=torch.ones(length, device=device),
            )
        )

    rollout_stats = {
        "raw_lengths": lengths.cpu(),
        "selected_lengths": torch.tensor(selected_lengths, dtype=torch.long).cpu(),
        "min_training_length": min_training_length,
        "num_completed": int(len(completed_results)),
        "num_selected": int(len(results)),
        "num_short_selected": short_selected_count,
        "short_fraction": short_fraction,
        "target_episodes": target_episodes,
        "num_envs": num_envs,
    }

    return results, rollout_stats


@dataclass
class Trajectory:
    states: torch.Tensor
    valid_actions: torch.Tensor
    actions: torch.Tensor
    probabilities: torch.Tensor
    advantage_estimates: torch.Tensor
    rewards_to_go: torch.Tensor


def flipped_cumsum(t: torch.Tensor, dim: int) -> torch.Tensor:
    return t.flip(dim).cumsum(dim).flip(dim)


def compute_trajectory(
    value: Value, result: GameResult, device: torch.device
) -> Trajectory | None:
    discount = 0.99
    gae_decay = 0.95

    states = result.states
    valid_actions = result.valid_actions
    actions = result.actions
    probabilities = result.probabilities
    rewards = result.rewards

    if states.shape[0] == 0:
        return None

    # we'll take the value of a state as a measure of the maximum number of
    # moves we can make starting from the state
    value_device = next(value.parameters()).device
    with torch.no_grad():
        values = value(states.to(value_device)).squeeze(-1).to(states.device)

    # compute the TD residuals using tensor arithmetic
    td_residuals = (
        rewards
        + torch.cat((discount * values[1:], torch.tensor([0.0], device=states.device)))
        - values
    )
    # compute the advantage estimates using GAE (Generalized Advantage Estimation)
    advantage_estimates = []
    advantage = 0.0
    for t in reversed(range(len(result.rewards))):
        delta = td_residuals[t]
        advantage = delta + discount * gae_decay * advantage
        advantage_estimates.insert(0, advantage)
    advantage_estimates = torch.tensor(advantage_estimates, device=states.device)

    rewards_to_go = flipped_cumsum(rewards, 0)

    return Trajectory(
        states=states,
        valid_actions=valid_actions,
        actions=actions,
        probabilities=probabilities,
        advantage_estimates=advantage_estimates,
        rewards_to_go=rewards_to_go,
    )


def _save_sample_rollouts(
    trajs: list[Trajectory], out_prefix: str, max_samples: int = 4
):
    """Save up to `max_samples` trajectories: .npz of states and a PNG visualizing boards.

    `out_prefix` should be a path prefix (without extension).
    """
    out_dir = os.path.dirname(out_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save raw numpy data for inspection
    save_data = {}
    for i, traj in enumerate(trajs[:max_samples]):
        save_data[f"traj_{i}_states"] = traj.states.cpu().numpy()
        save_data[f"traj_{i}_actions"] = traj.actions.cpu().numpy()
    np.savez_compressed(out_prefix + ".npz", **save_data)

    # Create a PNG visual: each trajectory in a row, time along columns
    n = min(len(trajs), max_samples)
    max_len = max([len(trajs[i].states) for i in range(n)]) if n > 0 else 0
    if max_len == 0:
        return

    fig, axes = plt.subplots(n, max_len, figsize=(max_len * 1.2, n * 1.2))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(n):
        states = trajs[i].states.cpu().numpy()
        for t in range(max_len):
            ax = axes[i, t]
            ax.axis("off")
            if t < len(states):
                board = states[t]
                # display numbers; blank zeros
                ax.imshow(np.zeros((4, 4)), cmap="gray")
                for r in range(4):
                    for c in range(4):
                        v = int(board[r, c])
                        txt = "" if v == 0 else str(v)
                        ax.text(c, r, txt, ha="center", va="center", fontsize=10)
            else:
                ax.set_facecolor((0.9, 0.9, 0.9))
    plt.tight_layout()
    fig.savefig(out_prefix + ".png")
    plt.close(fig)


def _update_bc_buffer(
    trajs: list[Trajectory],
    bc_buffer: deque,
    config: dict,
):
    if bc_buffer is None:
        return

    min_length = config.get("min_length", 200)
    min_tile = config.get("min_tile", 512)
    stride = max(1, config.get("sample_stride", 4))

    for traj in trajs:
        states = traj.states.detach()
        valid = traj.valid_actions.detach()
        actions = traj.actions.detach()

        if states.shape[0] < min_length:
            continue

        max_tiles = states.view(states.shape[0], -1).amax(dim=1)
        tile_mask = max_tiles >= min_tile
        if not tile_mask.any():
            continue

        indices = torch.nonzero(tile_mask).squeeze(-1)
        indices = indices[::stride]
        for idx in indices:
            bc_buffer.append(
                (
                    states[idx].cpu(),
                    valid[idx].cpu(),
                    int(actions[idx].item()),
                )
            )


def _sample_bc_batch(
    bc_buffer: deque,
    batch_size: int,
    device: torch.device,
):
    if bc_buffer is None or len(bc_buffer) == 0:
        return None

    sample_size = min(batch_size, len(bc_buffer))
    samples = random.sample(bc_buffer, sample_size)
    states, valid_actions, actions = zip(*samples)
    bc_states = torch.stack([s.to(device) for s in states])
    bc_valid = torch.stack([v.to(device) for v in valid_actions])
    bc_actions = torch.tensor(actions, device=device, dtype=torch.long)
    return bc_states, bc_valid, bc_actions


def train_iteration(
    policy: Policy,
    value: Value,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    device: torch.device,
    bc_buffer: deque,
) -> tuple[float, list[Trajectory], dict[str, torch.Tensor | float | int]]:
    """Train the model with one iteration of PPO."""

    eps = 0.2
    max_grad_norm = 1.0
    step_count = 5
    bc_config = {
        "min_length": 300,
        "min_tile": 512,
        "sample_stride": 4,
        "batch_size": 256,
        "loss_weight": 0.2,
    }

    rollout_device = torch.device("cpu")
    policy.eval()
    value.eval()

    games, rollout_stats = play_games(policy, rollout_device, value)
    trajs: list[Trajectory] = []
    for game in games:
        traj = compute_trajectory(value, game, rollout_device)
        if traj is not None:
            trajs.append(traj)

    if len(trajs) == 0:
        return 0.0, [], rollout_stats

    _update_bc_buffer(trajs, bc_buffer, bc_config)

    states = torch.cat([traj.states for traj in trajs]).to(device)
    valid_actions = torch.cat([traj.valid_actions for traj in trajs]).to(device)
    actions = torch.cat([traj.actions for traj in trajs]).to(device)
    probabilities = torch.cat([traj.probabilities for traj in trajs]).to(device)
    advantage_estimates = torch.cat([traj.advantage_estimates for traj in trajs]).to(
        device
    )
    rewards_to_go = torch.cat([traj.rewards_to_go for traj in trajs]).to(device)

    # improve policy

    policy.train(True)

    for _ in range(step_count):

        policy_outputs = policy(states, valid_actions)

        new_probabilities = policy_outputs[
            torch.arange(len(policy_outputs), device=device), actions
        ]
        ratios = new_probabilities / probabilities
        clipped_ratios = torch.clamp(ratios, 1 - eps, 1 + eps)
        policy_losses = -torch.min(
            ratios * advantage_estimates, clipped_ratios * advantage_estimates
        )
        bc_loss = torch.tensor(0.0, device=device)
        bc_batch = _sample_bc_batch(bc_buffer, bc_config["batch_size"], device)
        if bc_batch is not None:
            bc_states, bc_valid, bc_actions = bc_batch
            bc_outputs = policy(bc_states, bc_valid)
            bc_loss = torch.nn.functional.nll_loss(
                torch.log(bc_outputs + 1e-8), bc_actions
            )

        total_policy_loss = policy_losses.mean() + bc_config["loss_weight"] * bc_loss

        policy_optimizer.zero_grad()
        total_policy_loss.backward()
        # nn_utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        policy_optimizer.step()

    policy.train(False)

    # improve value

    value.train(True)

    for _ in range(step_count):
        new_values = value(states).squeeze()
        total_value_loss = torch.nn.functional.mse_loss(new_values, rewards_to_go)

        value_optimizer.zero_grad()
        total_value_loss.backward()
        # nn_utils.clip_grad_norm_(value.parameters(), max_grad_norm)
        value_optimizer.step()

    value.train(False)

    return statistics.mean([len(traj.states) for traj in trajs]), trajs, rollout_stats


def train(
    policy: Policy,
    value: Value,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    device: torch.device,
):
    counts = []
    bc_buffer = deque(maxlen=20000)

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax_curve = axes[0, 0]
    ax_hist = axes[0, 1]
    ax_sample = axes[1, 0]
    ax_stats = axes[1, 1]

    (line,) = ax_curve.plot(counts, label="moves")
    ax_curve.set_xlabel("Iteration")
    ax_curve.set_ylabel("Moves")
    ax_curve.legend()

    ax_hist.set_title("Episode length distribution")
    ax_sample.set_title("Sample rollouts")
    ax_stats.axis("off")

    t = datetime.now().isoformat()

    for i in range(100001):
        # with torch.autograd.detect_anomaly():
        count, trajs, rollout_stats = train_iteration(
            policy, value, policy_optimizer, value_optimizer, device, bc_buffer
        )

        if i % 100 == 0:
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "value": value.state_dict(),
                    "policy_optimizer": policy_optimizer.state_dict(),
                    "value_optimizer": value_optimizer.state_dict(),
                },
                f"./ckpt-{t}-{i}.pt",
            )
            # save sample rollouts (raw + PNG)
            out_prefix = f"images/rollouts/rollout-{t}-{i}"
            _save_sample_rollouts(trajs, out_prefix)
            try:
                img = plt.imread(out_prefix + ".png")
                ax_sample.clear()
                ax_sample.imshow(img)
                ax_sample.axis("off")
            except Exception:
                pass

        counts.append(count)
        line.set_ydata(counts)
        line.set_xdata(range(len(counts)))
        ax_curve.relim()
        ax_curve.autoscale_view()

        # update histogram of episode lengths (raw vs. selected)
        ax_hist.clear()
        raw_lengths = rollout_stats.get("raw_lengths") if rollout_stats else None
        selected_lengths = (
            rollout_stats.get("selected_lengths") if rollout_stats else None
        )
        bins = 20
        plotted = False
        if raw_lengths is not None and len(raw_lengths) > 0:
            ax_hist.hist(
                raw_lengths.cpu().numpy(),
                bins=bins,
                alpha=0.5,
                label="all rollouts",
                color="#1f77b4",
            )
            plotted = True
        if selected_lengths is not None and len(selected_lengths) > 0:
            ax_hist.hist(
                selected_lengths.cpu().numpy(),
                bins=bins,
                alpha=0.6,
                label="training batch",
                color="#ff7f0e",
            )
            plotted = True
        min_train = rollout_stats.get("min_training_length") if rollout_stats else None
        if min_train is not None:
            ax_hist.axvline(
                min_train,
                color="red",
                linestyle="--",
                linewidth=1,
                label="min length filter",
            )
        if plotted:
            ax_hist.set_title("Episode length distribution")
            ax_hist.set_xlabel("moves per episode")
            ax_hist.set_ylabel("count")
            ax_hist.legend(loc="upper right", fontsize=8)
        else:
            ax_hist.set_title("Episode length distribution")
            ax_hist.text(0.5, 0.5, "no data", ha="center", va="center")

        # update stats panel
        ax_stats.clear()
        if len(counts) > 0:
            avg = float(np.mean(counts))
            med = float(np.median(counts))
            best = float(np.max(counts))
            raw_avg = raw_med = raw_best = None
            if raw_lengths is not None and len(raw_lengths) > 0:
                raw_avg = float(raw_lengths.float().mean().item())
                raw_med = float(raw_lengths.float().median().item())
                raw_best = float(raw_lengths.max().item())
            ax_stats.text(0.05, 0.8, f"train latest: {count:.1f}", fontsize=10)
            ax_stats.text(0.05, 0.6, f"train avg: {avg:.1f}", fontsize=10)
            ax_stats.text(0.05, 0.4, f"train median: {med:.1f}", fontsize=10)
            ax_stats.text(0.05, 0.2, f"train best: {best:.1f}", fontsize=10)
            if raw_avg is not None:
                ax_stats.text(0.55, 0.8, f"raw avg: {raw_avg:.1f}", fontsize=10)
                ax_stats.text(0.55, 0.6, f"raw median: {raw_med:.1f}", fontsize=10)
                ax_stats.text(0.55, 0.4, f"raw best: {raw_best:.1f}", fontsize=10)
            short_mix = (
                rollout_stats.get("num_short_selected") if rollout_stats else None
            )
            target_eps = rollout_stats.get("target_episodes") if rollout_stats else None
            if short_mix is not None and target_eps:
                frac = short_mix / target_eps if target_eps else 0.0
                ax_stats.text(
                    0.55,
                    0.2,
                    f"short mix: {short_mix}/{target_eps} ({frac:.0%})",
                    fontsize=10,
                )
        ax_stats.axis("off")
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    plt.show()
