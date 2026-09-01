"""Internal machinery shared by the conditioned backbones. Not public API."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from torchebm.models.components import LabelEmbedder, MLPTimestepEmbedder


def _xavier_linear_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _ConditionalBackbone(nn.Module):
    r"""Base for backbones conditioned on summed t/y/cond embeddings.

    Owns the pluggable timestep and label embedders, their validation, the
    conditioning-vector assembly, and the reference initialization of the
    default embedders, so every backbone exposes the same
    ``forward(x, t=None, y=None, *, cond=None)`` contract with identical
    semantics. ``cond_dim=0`` builds no embedders (unconditional backbone);
    subclasses decide whether that is allowed.
    """

    def __init__(
        self,
        *,
        cond_dim: int,
        num_classes: Optional[int] = None,
        class_dropout_prob: float = 0.0,
        t_embedder: Optional[nn.Module] = None,
        y_embedder: Optional[nn.Module] = None,
    ):
        super().__init__()
        if cond_dim < 0:
            raise ValueError(f"cond_dim must be non-negative, got {cond_dim}")
        if not 0.0 <= class_dropout_prob <= 1.0:
            raise ValueError(f"class_dropout_prob must be in [0, 1], got {class_dropout_prob}")
        if y_embedder is not None and (num_classes is not None or class_dropout_prob > 0):
            raise ValueError(
                "pass either y_embedder or num_classes/class_dropout_prob, not both"
            )
        if class_dropout_prob > 0 and num_classes is None:
            raise ValueError("class_dropout_prob > 0 requires num_classes")
        if num_classes is not None and num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if cond_dim == 0 and (
            num_classes is not None or t_embedder is not None or y_embedder is not None
        ):
            raise ValueError(
                "cond_dim=0 builds an unconditional backbone; num_classes, "
                "t_embedder and y_embedder are not accepted"
            )

        self.cond_dim = int(cond_dim)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self._default_t_embedder = t_embedder is None and self.cond_dim > 0
        self._default_y_embedder = y_embedder is None and self.num_classes is not None

        if self.cond_dim == 0:
            self.t_embedder = None
            self.y_embedder = None
        else:
            self.t_embedder = (
                t_embedder
                if t_embedder is not None
                else MLPTimestepEmbedder(self.cond_dim)
            )
            if y_embedder is not None:
                self.y_embedder = y_embedder
            elif self.num_classes is not None:
                self.y_embedder = LabelEmbedder(
                    self.num_classes,
                    self.cond_dim,
                    dropout_prob=class_dropout_prob,
                    null_token=True,
                )
            else:
                self.y_embedder = None

    def _condition(
        self,
        t: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        c = None
        if t is not None:
            c = self.t_embedder(t)
        if y is not None:
            if self.y_embedder is None:
                raise ValueError(
                    f"y was given but this {type(self).__name__} has no label "
                    "embedder; construct it with num_classes= or y_embedder="
                )
            if isinstance(self.y_embedder, LabelEmbedder):
                emb = self.y_embedder(y, training=self.training)
            else:
                emb = self.y_embedder(y)
            c = emb if c is None else c + emb
        if cond is not None:
            if cond.shape[-1] != self.cond_dim:
                raise ValueError(
                    f"cond has width {cond.shape[-1]}, expected cond_dim={self.cond_dim}"
                )
            c = cond if c is None else c + cond
        if c is None:
            raise ValueError(
                f"{type(self).__name__}.forward requires at least one of t, y, or cond"
            )
        return c

    def _init_default_embedder_weights(self) -> None:
        r"""Reference init for default-built embedders; user modules untouched."""
        if self._default_t_embedder:
            self.t_embedder.apply(_xavier_linear_init)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self._default_y_embedder:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
