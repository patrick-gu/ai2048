import torch
import torch.nn as nn


def _prep_board(board: torch.Tensor) -> torch.Tensor:
    flat = board.flatten(-2).float()
    flat = torch.where(flat == 0, torch.ones_like(flat), flat)
    return torch.log2(flat)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        input_dim = 4 * 4
        hidden_dims = [512, 512, 512, 256]
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            prev = dim
        layers.append(nn.Linear(prev, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, board: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        assert board.shape[-2:] == (4, 4)
        assert valid.shape[-1] == 4
        assert board.shape[:-2] == valid.shape[:-1]

        input = _prep_board(board)
        logits: torch.Tensor = self.network(input)
        valid_mask = torch.where(valid == 1, 0.0, -torch.inf)
        return (logits + valid_mask).softmax(-1)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        input_dim = 4 * 4
        hidden_dims = [512, 512, 512, 256]
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        assert board.shape[-2:] == (4, 4)

        input = _prep_board(board)
        return self.network(input)
