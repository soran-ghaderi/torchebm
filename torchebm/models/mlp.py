from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from torchebm.models.backbone import _ConditionalBackbone, _xavier_linear_init
from torchebm.models.components import AdaLNZeroLinearHead, AdaLNZeroMLPBlock

_CONDITIONING_KINDS = ("adaln_zero", "concat")


class TimeConditionedMLP(_ConditionalBackbone):
    r"""Conditioned MLP backbone for low-dimensional data.

    Maps ``(B, in_dim)`` to ``(B, out_dim)`` under the library's
    ``model(x, t, y=)`` convention, sharing `DiT`'s conditioning semantics:
    the vector \(c\) is the sum of the embeddings of whichever of ``t``,
    ``y``, and ``cond`` are given, both embedders are pluggable, and
    ``num_classes`` always allocates the null label row (id ``num_classes``)
    for classifier-free guidance.

    Two conditioning mechanisms:

    - ``"adaln_zero"`` (default): a stem Linear into ``hidden_dim``,
      ``depth`` residual blocks
      \(h \leftarrow h + g \odot \mathrm{MLP}(\mathrm{LN}(h)(1+s) + b)\)
      with shift/scale/gate derived from \(c\), and an adaLN-Zero linear
      head. This is the DiT conditioning mechanism and its initialization
      scheme on a vector stream: Xavier-uniform Linears with zero biases,
      std-0.02 default embedders, zero-initialized modulations and head, so
      the output is exactly zero at initialization.
    - ``"concat"``: \([x \,\Vert\, c]\) through ``depth`` hidden layers, the
      hand-rolled tutorial baseline. Weights keep PyTorch's default
      initialization (only the default embedders get the std-0.02 scheme).

    ``cond_dim=0`` (concat only) builds an unconditional, time-invariant
    field: ``t`` is accepted and ignored, which is the Equilibrium Matching
    model convention, while explicit ``y``/``cond`` raise.

    Args:
        in_dim: Input feature width.
        out_dim: Output width. ``None`` (default) matches ``in_dim`` (the
            vector-field convention).
        hidden_dim: Width of the hidden stream.
        depth: Number of residual blocks (``"adaln_zero"``) or hidden layers
            (``"concat"``).
        conditioning: ``"adaln_zero"`` or ``"concat"``.
        mlp_ratio: Expansion ratio inside each residual block. ``None``
            (default) means 4.0; setting it with ``"concat"`` raises, the
            knob does not apply there.
        activation: Callable producing an activation module (e.g.
            ``nn.SiLU``). ``None`` (default) selects the mode default:
            tanh-approximated GELU for ``"adaln_zero"`` (DiT parity), SiLU
            for ``"concat"``.
        cond_dim: Width of the conditioning vector. ``None`` (default) uses
            ``hidden_dim``; 0 disables conditioning (``"concat"`` only).
        num_classes: Enables the built-in label embedder (plus null row).
        class_dropout_prob: Model-side classifier-free label dropout.
            Requires ``num_classes``.
        t_embedder: Module mapping ``(B,)`` timesteps to ``(B, cond_dim)``.
            ``None`` builds an `MLPTimestepEmbedder`.
        y_embedder: Module mapping labels to ``(B, cond_dim)``. Mutually
            exclusive with ``num_classes``/``class_dropout_prob``.

    Example:
        ```python
        from torchebm.losses import FlowMatchingLoss
        from torchebm.models import TimeConditionedMLP

        field = TimeConditionedMLP(in_dim=2)
        loss = FlowMatchingLoss(model=field)(x_batch)
        ```
    """

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: Optional[int] = None,
        hidden_dim: int = 256,
        depth: int = 3,
        conditioning: str = "adaln_zero",
        mlp_ratio: Optional[float] = None,
        activation: Optional[Callable[[], nn.Module]] = None,
        cond_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        class_dropout_prob: float = 0.0,
        t_embedder: Optional[nn.Module] = None,
        y_embedder: Optional[nn.Module] = None,
    ):
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if conditioning not in _CONDITIONING_KINDS:
            raise ValueError(
                f"conditioning must be one of {_CONDITIONING_KINDS}, got {conditioning!r}"
            )
        resolved_cond = int(cond_dim) if cond_dim is not None else int(hidden_dim)
        if conditioning == "adaln_zero" and resolved_cond == 0:
            raise ValueError("adaln_zero conditioning requires cond_dim > 0")
        if conditioning == "concat" and mlp_ratio is not None:
            raise ValueError("mlp_ratio applies only to adaln_zero conditioning")

        super().__init__(
            cond_dim=resolved_cond,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            t_embedder=t_embedder,
            y_embedder=y_embedder,
        )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim) if out_dim is not None else self.in_dim
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.conditioning = conditioning

        if conditioning == "adaln_zero":
            act = activation if activation is not None else _tanh_gelu
            self.stem = nn.Linear(self.in_dim, self.hidden_dim)
            self.blocks = nn.ModuleList(
                AdaLNZeroMLPBlock(
                    embed_dim=self.hidden_dim,
                    cond_dim=self.cond_dim,
                    mlp_ratio=mlp_ratio if mlp_ratio is not None else 4.0,
                    act_layer=act,
                )
                for _ in range(self.depth)
            )
            self.head = AdaLNZeroLinearHead(
                embed_dim=self.hidden_dim,
                cond_dim=self.cond_dim,
                out_dim=self.out_dim,
            )
            self._initialize_weights()
        else:
            act = activation if activation is not None else nn.SiLU
            width_in = self.in_dim + self.cond_dim
            layers: list[nn.Module] = []
            for _ in range(self.depth):
                layers.append(nn.Linear(width_in, self.hidden_dim))
                layers.append(act())
                width_in = self.hidden_dim
            layers.append(nn.Linear(self.hidden_dim, self.out_dim))
            self.net = nn.Sequential(*layers)
            self._init_default_embedder_weights()

    def _initialize_weights(self) -> None:
        r"""Reference DiT initialization on the vector stream."""
        _xavier_linear_init(self.stem)
        for module in (self.blocks, self.head):
            module.apply(_xavier_linear_init)
        self._init_default_embedder_weights()
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.head.modulation[-1].weight)
        nn.init.zeros_(self.head.modulation[-1].bias)
        nn.init.zeros_(self.head.proj.weight)
        nn.init.zeros_(self.head.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        *,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Map ``(B, in_dim)`` to ``(B, out_dim)``.

        Args:
            x: Input of shape ``(B, in_dim)``.
            t: Timesteps of shape ``(B,)``. Ignored when the model is
                unconditional (``cond_dim=0``), so library losses that always
                pass ``t`` drive a time-invariant field unchanged.
            y: Integer class labels of shape ``(B,)``.
            cond: Pre-built conditioning vector of shape ``(B, cond_dim)``.

        Returns:
            torch.Tensor: Output of shape ``(B, out_dim)``.
        """
        if self.cond_dim == 0:
            if y is not None or cond is not None:
                raise ValueError(
                    f"{type(self).__name__} was built unconditional (cond_dim=0); "
                    "y and cond are not accepted"
                )
            return self.net(x)
        c = self._condition(t, y, cond)
        if self.conditioning == "adaln_zero":
            h = self.stem(x)
            for block in self.blocks:
                h = block(h, c)
            return self.head(h, c)
        return self.net(torch.cat([x, c], dim=-1))


def _tanh_gelu() -> nn.Module:
    return nn.GELU(approximate="tanh")
