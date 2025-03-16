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


def train_iteration(
    policy: Policy,
    value: Value,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
) -> float:
    """Train the model with one iteration of PPO."""

    eps = 0.2
    max_grad_norm = 1.0

    game = Game()
    game_states = []
    actions = []
    valid_moves = []
    action_idxs = []
    probabilities = []
    rewards = []
    values = []

    while game.alive():
        state = serialize_game(game)
        policy_output = policy(state)
        valid_actions = [i for i in range(4) if game.valid(i)]
        valid_probs = torch.softmax(policy_output[valid_actions], 0)
        action_idx = torch.multinomial(valid_probs, 1).item()
        action = valid_actions[action_idx]
        game.move(action)

        valid_moves.append(valid_actions)
        game_states.append(state)
        actions.append(action)
        action_idxs.append(action_idx)
        probabilities.append(valid_probs[action_idx])

        # reward of 1 just for making any move
        rewards.append(1.0)
        # we'll take the value of a state as a measure of the maximum number of
        # moves we can make starting from the state
        values.append(value(state).item())

    rewards_to_go = postfix_sum(rewards)

    # check if this is right?
    # advantage estimate is difference between what we were able to achieve and
    # the value function
    advantage_estimates = [r - v for r, v in zip(rewards_to_go, values)]

    count = len(game_states)

    # improve policy

    policy.train(True)

    total_policy_loss = torch.Tensor([0.0])
    for (
        state,
        action,
        action_idx,
        probability,
        reward,
        advantage_estimate,
        valid_actions,
    ) in zip(
        game_states,
        actions,
        action_idxs,
        probabilities,
        rewards,
        advantage_estimates,
        valid_moves,
    ):
        policy_output = policy(state)
        valid_probs = torch.softmax(policy_output[valid_actions], 0)
        new_probability = valid_probs[action_idx]

        ratio = new_probability / probability
        clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
        policy_loss = -torch.min(
            ratio * advantage_estimate, clipped_ratio * advantage_estimate
        )
        total_policy_loss += policy_loss
    total_policy_loss /= count

    policy_optimizer.zero_grad()
    total_policy_loss.backward()
    # nn_utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    policy_optimizer.step()

    policy.train(False)

    # improve value

    value.train(True)

    total_value_loss = torch.Tensor([0.0])
    for state, reward in zip(game_states, rewards_to_go):
        # value loss
        value_loss = (value(state) - reward) ** 2
        total_value_loss += value_loss
    total_value_loss /= count

    value_optimizer.zero_grad()
    total_value_loss.backward()
    # nn_utils.clip_grad_norm_(value.parameters(), max_grad_norm)
    value_optimizer.step()

    value.train(False)

    return count


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
