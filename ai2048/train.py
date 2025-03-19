from dataclasses import dataclass
from datetime import datetime
import statistics

import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer

from ai2048.game import Game
from ai2048.model import Policy, Value


@dataclass
class GameResult:
    states_and_valid_actions: list[tuple[torch.Tensor, torch.Tensor]]
    actions: list[int]
    probabilities: list[float]
    rewards: list[float]


def play_games(policy: Policy) -> list[GameResult]:
    not_done = [
        (
            Game(),
            GameResult(
                states_and_valid_actions=[],
                actions=[],
                probabilities=[],
                rewards=[],
            ),
        )
        for _ in range(64)
    ]
    done = []

    while not_done != []:
        states = torch.stack([torch.tensor(game.state, dtype=torch.float32) for game, _ in not_done])
        valid_actions = torch.tensor(
            [[float(game.valid(i)) for i in range(4)] for game, _ in not_done]
        )
        policy_outputs = policy(states, valid_actions)
        actions = torch.multinomial(policy_outputs, 1).squeeze(1)
        probabilities = policy_outputs[torch.arange(len(policy_outputs)), actions]

        for state, valid_actions_mask, action, probability, (game, result) in zip(
            states, valid_actions, actions, probabilities, not_done
        ):
            game.move(action.item())

            result.states_and_valid_actions.append((state, valid_actions_mask))
            result.actions.append(action.item())
            result.probabilities.append(probability.item())
            # reward of 1 just for making any move
            result.rewards.append(1.0)

        done.extend([result for game, result in not_done if not game.alive()])
        not_done = [(game, result) for game, result in not_done if game.alive()]

    return done


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


def compute_trajectory(value: Value, result: GameResult) -> Trajectory:
    discount = 0.99
    gae_decay = 0.95

    states, valid_actions = tuple(zip(*result.states_and_valid_actions))
    states = torch.stack(states)
    valid_actions = torch.stack(valid_actions)
    actions = torch.tensor(result.actions)
    probabilities = torch.tensor(result.probabilities)
    rewards = torch.tensor(result.rewards)

    # we'll take the value of a state as a measure of the maximum number of
    # moves we can make starting from the state
    values = value(states).squeeze()

    # compute the TD residuals using tensor arithmetic
    td_residuals = (
        rewards + torch.cat((discount * values[1:], torch.tensor([0.0]))) - values
    )
    # compute the advantage estimates using GAE (Generalized Advantage Estimation)
    advantage_estimates = []
    advantage = 0.0
    for t in reversed(range(len(result.rewards))):
        delta = td_residuals[t]
        advantage = delta + discount * gae_decay * advantage
        advantage_estimates.insert(0, advantage)
    advantage_estimates = torch.tensor(advantage_estimates)

    rewards_to_go = flipped_cumsum(rewards, 0)

    return Trajectory(
        states=states,
        valid_actions=valid_actions,
        actions=actions,
        probabilities=probabilities,
        advantage_estimates=advantage_estimates,
        rewards_to_go=rewards_to_go,
    )


def train_iteration(
    policy: Policy,
    value: Value,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
) -> float:
    """Train the model with one iteration of PPO."""

    eps = 0.2
    max_grad_norm = 1.0
    step_count = 5

    games = play_games(policy)
    trajs = [compute_trajectory(value, game) for game in games]

    states = torch.cat([traj.states for traj in trajs])
    valid_actions = torch.cat([traj.valid_actions for traj in trajs])
    actions = torch.cat([traj.actions for traj in trajs])
    probabilities = torch.cat([traj.probabilities for traj in trajs])
    advantage_estimates = torch.cat([traj.advantage_estimates for traj in trajs])
    rewards_to_go = torch.cat([traj.rewards_to_go for traj in trajs])

    # improve policy

    policy.train(True)

    for _ in range(step_count):

        policy_outputs = policy(states, valid_actions)

        new_probabilities = policy_outputs[torch.arange(len(policy_outputs)), actions]
        ratios = new_probabilities / probabilities
        clipped_ratios = torch.clamp(ratios, 1 - eps, 1 + eps)
        policy_losses = -torch.min(
            ratios * advantage_estimates, clipped_ratios * advantage_estimates
        )
        total_policy_loss = policy_losses.mean()

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

    return statistics.mean([len(traj.states) for traj in trajs])


def train(
    policy: Policy,
    value: Value,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
):
    counts = []

    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot(counts)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Moves")

    t = datetime.now().isoformat()

    for i in range(100001):
        # with torch.autograd.detect_anomaly():
        count = train_iteration(policy, value, policy_optimizer, value_optimizer)

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

        counts.append(count)
        line.set_ydata(counts)
        line.set_xdata(range(len(counts)))
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.0001)

    plt.ioff()
    plt.show()
