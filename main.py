from ai2048.model import Policy, Value
from ai2048.train import train

from torch.optim import Adam


if __name__ == "__main__":
    policy = Policy()
    value = Value()
    policy_optimizer = Adam(policy.parameters(), lr=1e-3)
    value_optimizer = Adam(value.parameters(), lr=1e-3)

    # checkpoint = torch.load("./ckpt-0.pt")
    # policy.load_state_dict(checkpoint['policy'])
    # value.load_state_dict(checkpoint['value'])
    # policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
    # value_optimizer.load_state_dict(checkpoint['value_optimizer'])

    train(policy, value, policy_optimizer, value_optimizer)
