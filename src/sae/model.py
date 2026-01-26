import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, dict_size: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        h = self.act(self.encoder(x))
        x_hat = self.decoder(h)
        return x_hat, h
