from __future__ import annotations

import torch
from torch import nn


class ValueFunctionHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 1024, num_bins: int = 201):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)

    @torch.no_grad()
    def predict(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(embedding)
        probs = torch.softmax(logits, dim=-1)
        values = torch.linspace(-1.0, 0.0, steps=self.num_bins, device=embedding.device)
        expected_value = (probs * values.unsqueeze(0)).sum(dim=-1)
        return expected_value, probs
