"""Internal machinery shared by the conditioned backbones. Not public API."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from torchebm.models.components import LabelEmbedder, MLPTimestepEmbedder

_TWO_TIME_PREDICTION_TYPES = ("average_velocity", "flow_map")
_PREDICTION_TYPES = ("velocity", "score", "epsilon", *_TWO_TIME_PREDICTION_TYPES)
_GUIDANCE_CONDITIONING = (None, "scale", "scale_and_interval")


def _xavier_linear_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _GuidanceScaleEmbedder(MLPTimestepEmbedder):
    r"""Embeds a guidance scale \(\omega \geq 1\) through \(1 - 1/\omega\).

    The embedding of Improved MeanFlow (Lyy-iiis/imeanflow, ``imfDiT``:
    ``omega_embedder(1 - 1 / w)``): \(1 - 1/\omega\) maps \([1, \infty)\)
    onto \([0, 1)\), and the guided velocity
    \(v + (1 - 1/\omega)(v_c - v_u)\) is affine in it.
    """

    def forward(self, guidance_scale: torch.Tensor) -> torch.Tensor:
        return super().forward(1 - 1 / guidance_scale)


class _ConditionalBackbone(nn.Module):
    r"""Base for backbones conditioned on summed t/y/cond embeddings.

    Owns the pluggable timestep and label embedders, their validation, the
    conditioning-vector assembly, and the reference initialization of the
    default embedders, so every backbone exposes the same
    ``forward(x, t=None, y=None, *, cond=None)`` contract with identical
    semantics. ``cond_dim=0`` builds no embedders (unconditional backbone);
    subclasses decide whether that is allowed. ``time_invariant=True`` pins
    ``t`` to zero on every call, the Equilibrium Matching model convention,
    so the backbone defines one field for all times. ``prediction_type``
    declares what ``forward`` returns (``"velocity"``, ``"score"``, or
    ``"epsilon"``); it does not change the computation.

    The two-time prediction types ``"average_velocity"`` and ``"flow_map"``
    condition on the time \(s\) of the input and the interval \(t - s\), as
    in MeanFlow (Geng et al., 2025): ``t_embedder`` embeds \(s\), a second
    embedder ``dt_embedder`` embeds \(t - s\), and both are summed into the
    conditioning vector, so the network computes the average velocity
    \(u(x_s, s, t)\) of the flow from time \(s\) to time \(t\).
    ``"average_velocity"`` returns \(u\); ``"flow_map"`` returns the flow map
    \(F(x_s, s, t) = x_s + (t - s)\, u(x_s, s, t)\), which satisfies
    \(F(x_s, s, s) = x_s\) by construction. With ``time_invariant=True`` a
    two-time model embeds \(t - s\) only and builds no ``t_embedder``, as
    the Improved MeanFlow model conditions on \(t - r\) only (Lyy-iiis/
    imeanflow, ``imfDiT``, following Sun et al., 2025).

    ``guidance_conditioning`` adds the classifier-free guidance scale, and
    with ``"scale_and_interval"`` the guidance interval, as inputs, as in
    Improved MeanFlow (Geng et al., 2025b, Sec. 4.2): ``guidance_embedder``
    embeds the scale, ``interval_start_embedder`` and
    ``interval_end_embedder`` the two ends of the interval, and all are
    summed into the conditioning vector.
    """

    prediction_type: str = "velocity"

    def __init__(
        self,
        *,
        cond_dim: int,
        num_classes: Optional[int] = None,
        class_dropout_prob: float = 0.0,
        t_embedder: Optional[nn.Module] = None,
        y_embedder: Optional[nn.Module] = None,
        dt_embedder: Optional[nn.Module] = None,
        guidance_conditioning: Optional[str] = None,
        guidance_embedder: Optional[nn.Module] = None,
        time_invariant: bool = False,
        prediction_type: str = "velocity",
    ):
        super().__init__()
        if prediction_type not in _PREDICTION_TYPES:
            raise ValueError(
                f"prediction_type must be one of {_PREDICTION_TYPES}, "
                f"got {prediction_type!r}"
            )
        if guidance_conditioning not in _GUIDANCE_CONDITIONING:
            raise ValueError(
                f"guidance_conditioning must be one of {_GUIDANCE_CONDITIONING}, "
                f"got {guidance_conditioning!r}"
            )
        if guidance_embedder is not None and guidance_conditioning is None:
            raise ValueError("guidance_embedder requires guidance_conditioning")
        if guidance_conditioning is not None and cond_dim <= 0:
            raise ValueError("guidance_conditioning requires cond_dim > 0")
        two_time = prediction_type in _TWO_TIME_PREDICTION_TYPES
        if dt_embedder is not None and not two_time:
            raise ValueError(
                "dt_embedder requires prediction_type 'average_velocity' or 'flow_map'"
            )
        if two_time and cond_dim <= 0:
            raise ValueError(
                f"prediction_type={prediction_type!r} conditions on s and t; it "
                "requires cond_dim > 0"
            )
        no_t_embedder = two_time and time_invariant
        if no_t_embedder and t_embedder is not None:
            raise ValueError(
                "a two-time model with time_invariant=True embeds t - s only and "
                "takes no t_embedder"
            )
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
        self.time_invariant = bool(time_invariant)
        self.prediction_type = prediction_type
        self._default_t_embedder = t_embedder is None and self.cond_dim > 0 and not no_t_embedder
        self._default_y_embedder = y_embedder is None and self.num_classes is not None
        self._default_dt_embedder = dt_embedder is None and two_time
        self._default_guidance_embedder = (
            guidance_embedder is None and guidance_conditioning is not None
        )
        self.guidance_conditioning = guidance_conditioning

        if self.cond_dim == 0:
            self.t_embedder = None
            self.y_embedder = None
            self.dt_embedder = None
        else:
            if no_t_embedder:
                self.t_embedder = None
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
            if two_time:
                self.dt_embedder = (
                    dt_embedder
                    if dt_embedder is not None
                    else MLPTimestepEmbedder(self.cond_dim)
                )
            else:
                self.dt_embedder = None

        if guidance_conditioning is None:
            self.guidance_embedder = None
        else:
            self.guidance_embedder = (
                guidance_embedder
                if guidance_embedder is not None
                else _GuidanceScaleEmbedder(self.cond_dim)
            )
        if guidance_conditioning == "scale_and_interval":
            self.interval_start_embedder = MLPTimestepEmbedder(self.cond_dim)
            self.interval_end_embedder = MLPTimestepEmbedder(self.cond_dim)
        else:
            self.interval_start_embedder = self.interval_end_embedder = None

    def _embeddings(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        cond: Optional[torch.Tensor],
        s: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
        guidance_interval: Optional[torch.Tensor] = None,
    ) -> List[Tuple[str, torch.Tensor]]:
        r"""Embeddings of the conditioning inputs by name, in summation order."""
        embeddings = []
        if self.dt_embedder is not None:
            if s is None or t is None:
                raise ValueError(
                    f"prediction_type={self.prediction_type!r} requires both s and t"
                )
            if self.t_embedder is not None:
                embeddings.append(("t", self.t_embedder(s)))
            embeddings.append(("dt", self.dt_embedder(t - s)))
        elif s is not None:
            raise ValueError(
                f"s was given but this {type(self).__name__} is a one-time model; "
                "construct it with prediction_type='average_velocity' or 'flow_map'"
            )
        elif self.time_invariant:
            embeddings.append(("t", self.t_embedder(x.new_zeros(x.shape[0]))))
        elif t is not None:
            embeddings.append(("t", self.t_embedder(t)))
        if y is not None:
            if self.y_embedder is None:
                raise ValueError(
                    f"y was given but this {type(self).__name__} has no label "
                    "embedder; construct it with num_classes= or y_embedder="
                )
            if isinstance(self.y_embedder, LabelEmbedder):
                embeddings.append(("y", self.y_embedder(y, training=self.training)))
            else:
                embeddings.append(("y", self.y_embedder(y)))
        if cond is not None:
            if cond.shape[-1] != self.cond_dim:
                raise ValueError(
                    f"cond has width {cond.shape[-1]}, expected cond_dim={self.cond_dim}"
                )
            embeddings.append(("cond", cond))
        if not embeddings:
            raise ValueError(
                f"{type(self).__name__}.forward requires at least one of t, y, or cond"
            )
        if self.guidance_embedder is None:
            if guidance_scale is not None or guidance_interval is not None:
                raise ValueError(
                    f"guidance inputs were given but this {type(self).__name__} has no "
                    "guidance conditioning; construct it with guidance_conditioning="
                )
            return embeddings
        if guidance_scale is None:
            raise ValueError(
                f"guidance_conditioning={self.guidance_conditioning!r} requires guidance_scale"
            )
        embeddings.append(("guidance", self.guidance_embedder(guidance_scale)))
        if self.interval_start_embedder is None:
            if guidance_interval is not None:
                raise ValueError(
                    "guidance_interval requires guidance_conditioning='scale_and_interval'"
                )
            return embeddings
        if guidance_interval is None:
            raise ValueError(
                "guidance_conditioning='scale_and_interval' requires guidance_interval"
            )
        start, end = guidance_interval.unbind(-1)
        embeddings.append(("interval_start", self.interval_start_embedder(start)))
        embeddings.append(("interval_end", self.interval_end_embedder(end)))
        return embeddings

    def _condition(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        cond: Optional[torch.Tensor],
        s: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
        guidance_interval: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self._embeddings(x, t, y, cond, s, guidance_scale, guidance_interval)
        c = embeddings[0][1]
        for _, emb in embeddings[1:]:
            c = c + emb
        return c

    def _output(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        s: Optional[torch.Tensor],
        t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.prediction_type != "flow_map":
            return u
        return x + (t - s).to(x.dtype).reshape(-1, *(1,) * (x.ndim - 1)) * u

    def _default_timestep_embedders(self) -> List[MLPTimestepEmbedder]:
        r"""The default-built timestep embedders, in a fixed order."""
        return [
            embedder
            for embedder, is_default in (
                (self.t_embedder, self._default_t_embedder),
                (self.dt_embedder, self._default_dt_embedder),
                (self.guidance_embedder, self._default_guidance_embedder),
                (self.interval_start_embedder, self.interval_start_embedder is not None),
                (self.interval_end_embedder, self.interval_end_embedder is not None),
            )
            if is_default
        ]

    def _init_default_embedder_weights(self) -> None:
        r"""Reference init for default-built embedders; user modules untouched."""
        for embedder in self._default_timestep_embedders():
            embedder.apply(_xavier_linear_init)
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)
        if self._default_y_embedder:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
