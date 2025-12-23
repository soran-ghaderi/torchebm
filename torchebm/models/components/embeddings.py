from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class MLPTimestepEmbedder(nn.Module):
    """Embed a scalar timestep into a vector via sinusoid + MLP.

    This is a generic block (useful for EqM, diffusion, flows, etc.).
    """

    def __init__(self, out_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # t: (B,)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.reshape(t.shape[0])
        freq = self.sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(freq)


class LabelEmbedder(nn.Module):
    """Label embedding with optional classifier-free guidance token dropping.

    If `dropout_prob>0`, one extra embedding row is allocated to represent the
    *null/unconditional* label.

    Note: this module does *not* assume any specific loss/sampler; it only
    produces vectors.
    """

    def __init__(self, num_classes: int, out_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dropout_prob = float(dropout_prob)
        use_null = self.dropout_prob > 0
        self.null_label_id = self.num_classes if use_null else None
        self.embedding = nn.Embedding(self.num_classes + (1 if use_null else 0), out_dim)

    def maybe_drop_labels(
        self,
        labels: torch.Tensor,
        *,
        force_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dropout_prob <= 0:
            return labels
        if self.null_label_id is None:
            raise RuntimeError("LabelEmbedder configured without null label.")

        if force_drop_mask is None:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_mask = force_drop_mask.to(device=labels.device, dtype=torch.bool)

        return torch.where(drop_mask, torch.full_like(labels, self.null_label_id), labels)

    def forward(
        self,
        labels: torch.Tensor,
        *,
        training: bool,
        force_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if training or (force_drop_mask is not None):
            labels = self.maybe_drop_labels(labels, force_drop_mask=force_drop_mask)
        return self.embedding(labels)
