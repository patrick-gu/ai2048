import torch
from ai2048.model import Policy, Value
from ai2048.train import train

from torch.optim import Adam


if __name__ == "__main__":
    policy = Policy()
    value = Value()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    policy.to(device)
    value.to(device)

    policy_optimizer = Adam(policy.parameters(), lr=1e-3)
    value_optimizer = Adam(value.parameters(), lr=1e-3)

    # checkpoint = torch.load("./ckpt-0.pt")
    # policy.load_state_dict(checkpoint['policy'])
    # value.load_state_dict(checkpoint['value'])
    # policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
    # value_optimizer.load_state_dict(checkpoint['value_optimizer'])

    train(policy, value, policy_optimizer, value_optimizer, device)
