from dataclasses import dataclass
from datetime import datetime

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


def postfix_sum(arr):
    """Compute the postfix sum of an array."""

    postfix = [0] * len(arr)
    postfix[-1] = arr[-1]
    for i in range(len(arr) - 2, -1, -1):
        postfix[i] = postfix[i + 1] + arr[i]
    return postfix


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


def play(policy: Policy, value: Value) -> Trajectory:
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
    td_residuals = rewards_tensor + torch.cat((discount * values[1:], torch.tensor([0.0]))) - values
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

    traj = play(policy, value)
    states = traj.states
    valid_actions = traj.valid_actions
    action_idxs = traj.action_idxs
    probabilities = traj.probabilities
    advantage_estimates = traj.advantage_estimates
    rewards_to_go = traj.rewards_to_go

    # improve policy

    policy.train(True)

    for _ in range(step_count):
        policy_outputs = policy(states)
        valid_probs = [torch.softmax(policy_outputs[i][valid_actions[i]], 0) for i in range(len(valid_actions))]
        new_probabilities = torch.stack([valid_probs[i][action_idxs[i]] for i in range(len(action_idxs))])

        ratios = new_probabilities / probabilities
        clipped_ratios = torch.clamp(ratios, 1 - eps, 1 + eps)
        policy_losses = -torch.min(ratios * advantage_estimates, clipped_ratios * advantage_estimates)

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

    return len(states)


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

    for i in range(100000):
        # with torch.autograd.detect_anomaly():
        count = train_iteration(policy, value, policy_optimizer, value_optimizer)

        if i % 2500 == 0:
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
