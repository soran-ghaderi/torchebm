import torch

from torchebm.core import BaseModel, BaseSampler
from torchebm.losses import ContrastiveDivergence


class _QuadraticEnergy(BaseModel):
    def __init__(self, scale: float = 0.5) -> None:
        super().__init__(device="cpu")
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x.square().sum(dim=-1)


class _SeededOffsetSampler(BaseSampler):
    def sample(
        self,
        x: torch.Tensor | None = None,
        dim: int | tuple[int, ...] | None = None,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        reset_schedulers: bool = True,
        *,
        generator: torch.Generator | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del dim, n_steps, n_samples, thin, return_trajectory, return_diagnostics
        del reset_schedulers, kwargs
        if x is None:
            raise ValueError("x is required for this test sampler")
        offset = torch.rand(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        return x + offset


class _ScaledContrastiveDivergence(ContrastiveDivergence):
    def compute_loss(
        self,
        x: torch.Tensor,
        pred_x: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        return 2.0 * super().compute_loss(x, pred_x, *args, **kwargs)


def _make_cd(
    loss_cls: type[ContrastiveDivergence] = ContrastiveDivergence,
) -> ContrastiveDivergence:
    model = _QuadraticEnergy()
    sampler = _SeededOffsetSampler(model=model, device="cpu")
    return loss_cls(
        model=model,
        sampler=sampler,
        k_steps=3,
        energy_reg_weight=0.0,
        device="cpu",
    )


def test_contrastive_divergence_pins_value_grad_and_seeded_sample() -> None:
    loss_fn = _make_cd()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    generator = torch.Generator(device="cpu").manual_seed(1234)

    loss, samples = loss_fn(x, generator=generator)

    expected_samples = torch.tensor(
        [
            [1.0289793015, 2.4018986225],
            [3.2598443031, 4.3666415215],
        ]
    )
    torch.testing.assert_close(samples, expected_samples, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        loss,
        torch.tensor(-1.6305141449),
        rtol=0.0,
        atol=1e-6,
    )

    loss.backward()
    torch.testing.assert_close(
        loss_fn.model.scale.grad,
        torch.tensor(-3.2610282898),
        rtol=0.0,
        atol=1e-6,
    )


def test_contrastive_divergence_repr_and_strict_state_dict_roundtrip() -> None:
    loss_fn = _make_cd()
    assert repr(loss_fn) == (
        "ContrastiveDivergence(model=_QuadraticEnergy(), "
        "sampler=_SeededOffsetSampler(model=_QuadraticEnergy))"
    )

    clone = _make_cd()
    result = clone.load_state_dict(loss_fn.state_dict(), strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    torch.testing.assert_close(clone.model.scale, loss_fn.model.scale)


def test_contrastive_divergence_user_compute_loss_override_is_honored() -> None:
    loss_fn = _make_cd(_ScaledContrastiveDivergence)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    generator = torch.Generator(device="cpu").manual_seed(1234)

    loss, _ = loss_fn(x, generator=generator)

    torch.testing.assert_close(
        loss,
        torch.tensor(-3.2610282898),
        rtol=0.0,
        atol=1e-6,
    )
