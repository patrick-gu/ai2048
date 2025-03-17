from dataclasses import dataclass
from datetime import datetime
import statistics

import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer

from ai2048.game import Game
from ai2048.model import Policy, Value


def serialize_game(game: Game) -> torch.Tensor:
    """Serialize game as input to NNs."""

    flat = torch.tensor(game.state).flatten()
    flat[flat == 0] = 1
    return torch.log2(flat)


@dataclass
class GameResult:
    states: list[torch.Tensor]
    valid_actions: list[list[int]]
    action_idxs: list[int]
    probabilities: list[float]
    rewards: list[float]


def play_games(policy: Policy) -> list[GameResult]:
    not_done = [
        (
            Game(),
            GameResult(
                states=[],
                valid_actions=[],
                action_idxs=[],
                probabilities=[],
                rewards=[],
            ),
        )
        for _ in range(16)
    ]
    done = []

    while not_done != []:
        states = torch.stack([serialize_game(game) for game, _ in not_done])
        policy_outputs = policy(states)
        for state, policy_output, (game, result) in zip(
            states, policy_outputs, not_done
        ):
            valid_actions_list = [i for i in range(4) if game.valid(i)]
            valid_probs = torch.softmax(policy_output[valid_actions_list], 0)
            action_idx = torch.multinomial(valid_probs, 1).item()
            action = valid_actions_list[action_idx]

            game.move(action)

            result.states.append(state)
            result.valid_actions.append(valid_actions_list)
            result.action_idxs.append(action_idx)
            result.probabilities.append(valid_probs[action_idx])
            # reward of 1 just for making any move
            result.rewards.append(1.0)

        done.extend([result for game, result in not_done if not game.alive()])
        not_done = [(game, result) for game, result in not_done if game.alive()]

    return done


@dataclass
class Trajectory:
    states: torch.Tensor

    valid_actions: list[list[int]]
    action_idxs: list[int]

    probabilities: torch.Tensor
    advantage_estimates: torch.Tensor

    rewards_to_go: torch.Tensor


def flipped_cumsum(t: torch.Tensor, dim: int) -> torch.Tensor:
    return t.flip(dim).cumsum(dim).flip(dim)


def compute_trajectory(value: Value, result: GameResult) -> Trajectory:
    discount = 0.99
    gae_decay = 0.95

    states = torch.stack(result.states)
    probabilities = torch.tensor(result.probabilities)

    # we'll take the value of a state as a measure of the maximum number of
    # moves we can make starting from the state
    values = value(states).squeeze()

    # compute the TD residuals using tensor arithmetic
    rewards_tensor = torch.tensor(result.rewards)
    td_residuals = (
        rewards_tensor
        + torch.cat((discount * values[1:], torch.tensor([0.0])))
        - values
    )
    # compute the advantage estimates using GAE (Generalized Advantage Estimation)
    advantage_estimates = []
    advantage = 0.0
    for t in reversed(range(len(result.rewards))):
        delta = td_residuals[t]
        advantage = delta + discount * gae_decay * advantage
        advantage_estimates.insert(0, advantage)
    advantage_estimates = torch.tensor(advantage_estimates)

    rewards_to_go = flipped_cumsum(rewards_tensor, 0)

    return Trajectory(
        states=states,
        valid_actions=result.valid_actions,
        action_idxs=result.action_idxs,
        probabilities=probabilities,
        advantage_estimates=advantage_estimates,
        rewards_to_go=rewards_to_go,
    )


def produce_trajectory(policy: Policy, value: Value) -> Trajectory:
    discount = 0.99
    gae_decay = 0.95

    game = Game()

    states = []
    valid_actions = []
    action_idxs = []
    probabilities = []
    rewards = []

    while game.alive():
        state = serialize_game(game)
        policy_output = policy(state)
        valid_actions_list = [i for i in range(4) if game.valid(i)]
        valid_probs = torch.softmax(policy_output[valid_actions_list], 0)
        action_idx = torch.multinomial(valid_probs, 1).item()
        action = valid_actions_list[action_idx]
        game.move(action)

        states.append(state)
        valid_actions.append(valid_actions_list)
        action_idxs.append(action_idx)
        probabilities.append(valid_probs[action_idx])
        # reward of 1 just for making any move
        rewards.append(1.0)

    states = torch.stack(states)
    probabilities = torch.tensor(probabilities)

    # we'll take the value of a state as a measure of the maximum number of
    # moves we can make starting from the state
    values = value(states).squeeze()

    # compute the TD residuals using tensor arithmetic
    rewards_tensor = torch.tensor(rewards)
    td_residuals = (
        rewards_tensor
        + torch.cat((discount * values[1:], torch.tensor([0.0])))
        - values
    )
    # compute the advantage estimates using GAE (Generalized Advantage Estimation)
    advantage_estimates = []
    advantage = 0.0
    for t in reversed(range(len(rewards))):
        delta = td_residuals[t]
        advantage = delta + discount * gae_decay * advantage
        advantage_estimates.insert(0, advantage)
    advantage_estimates = torch.tensor(advantage_estimates)

    rewards_to_go = flipped_cumsum(rewards_tensor, 0)

    return Trajectory(
        states=states,
        valid_actions=valid_actions,
        action_idxs=action_idxs,
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
    valid_actions = [action for traj in trajs for action in traj.valid_actions]
    action_idxs = [idx for traj in trajs for idx in traj.action_idxs]
    probabilities = torch.cat([traj.probabilities for traj in trajs])
    advantage_estimates = torch.cat([traj.advantage_estimates for traj in trajs])
    rewards_to_go = torch.cat([traj.rewards_to_go for traj in trajs])

    # improve policy

    policy.train(True)

    for _ in range(step_count):
        policy_outputs = policy(states)
        valid_probs = [
            torch.softmax(policy_outputs[i][valid_actions[i]], 0)
            for i in range(len(valid_actions))
        ]
        new_probabilities = torch.stack(
            [valid_probs[i][action_idxs[i]] for i in range(len(action_idxs))]
        )

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

        if i % 500 == 0:
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
