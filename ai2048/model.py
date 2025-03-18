import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = nn.Sequential(
            nn.Linear(4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

    def forward(self, board: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        assert board.shape[-2:] == (4, 4)
        assert valid.shape[-1] == 4
        assert board.shape[:-2] == valid.shape[:-1]

        flat = board.flatten(-2)
        flat[flat == 0] = 1
        input = flat.log2()
        valid = torch.where(valid == 1, 0.0, -torch.inf)
        logits: torch.Tensor = self.network(input) + valid
        return logits.softmax(-1)
        


class Value(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = nn.Sequential(
            nn.Linear(4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        assert board.shape[-2:] == (4, 4)

        flat = board.flatten(-2)
        flat[flat == 0] = 1
        input = flat.log2()
        return self.network(input)
