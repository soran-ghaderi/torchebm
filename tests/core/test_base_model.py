import pytest
import torch

from torchebm.core.base_model import (
    AckleyModel,
    BaseModel,
    BoothModel,
    DoubleWellModel,
    GaussianModel,
    HarmonicModel,
    HimmelblauModel,
    RastriginModel,
    RosenbrockModel,
    StyblinskiTangModel,
)


def test_base_model_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseModel()


@pytest.mark.parametrize("shape", [(3,), (4, 3)])
def test_harmonic_model_output_shape(shape):
    m = HarmonicModel(k=2.0)
    x = torch.randn(*shape)
    y = m(x)
    assert y.shape == (shape[0] if len(shape) == 2 else 1,)
    assert torch.isfinite(y).all()


def test_harmonic_energy_zero_at_origin():
    m = HarmonicModel(k=1.0)
    x = torch.zeros(2, 3)
    assert torch.allclose(m(x), torch.zeros(2))


def test_double_well_energy_zero_at_wells():
    m = DoubleWellModel(barrier_height=2.0, b=1.0)
    x = torch.tensor([[1.0], [-1.0]])
    assert torch.allclose(m(x), torch.zeros(2), atol=1e-6)


def test_double_well_gradient_finite():
    m = DoubleWellModel()
    x = torch.randn(4, 3)
    g = m.gradient(x)
    assert g.shape == x.shape
    assert torch.isfinite(g).all()


def test_gaussian_model_rejects_bad_mean():
    with pytest.raises(ValueError):
        GaussianModel(mean=torch.randn(2, 2), cov=torch.eye(2))


def test_gaussian_model_rejects_non_square_cov():
    with pytest.raises(ValueError):
        GaussianModel(mean=torch.zeros(2), cov=torch.randn(2, 3))


def test_gaussian_model_rejects_singular_cov():
    with pytest.raises(ValueError):
        GaussianModel(mean=torch.zeros(2), cov=torch.zeros(2, 2))


def test_gaussian_model_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        GaussianModel(mean=torch.zeros(3), cov=torch.eye(2))


def test_gaussian_model_energy_zero_at_mean():
    mean = torch.zeros(3)
    cov = torch.eye(3)
    m = GaussianModel(mean=mean, cov=cov)
    assert torch.allclose(m(mean), torch.zeros(1), atol=1e-6)


def test_gaussian_model_batch_forward():
    m = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    x = torch.randn(5, 2)
    y = m(x)
    assert y.shape == (5,)


def test_gaussian_model_rejects_wrong_input_shape():
    m = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    with pytest.raises(ValueError):
        m(torch.randn(3, 4))


def test_rosenbrock_energy_zero_at_minimum():
    m = RosenbrockModel(a=1.0, b=100.0)
    x = torch.ones(1, 3)
    assert torch.allclose(m(x), torch.zeros(1), atol=1e-5)


def test_rosenbrock_requires_min_two_dims():
    m = RosenbrockModel()
    with pytest.raises(ValueError):
        m(torch.randn(2, 1))


def test_ackley_energy_zero_at_origin():
    m = AckleyModel()
    x = torch.zeros(1, 3)
    assert torch.allclose(m(x), torch.zeros(1), atol=1e-5)


def test_rastrigin_energy_zero_at_origin():
    m = RastriginModel()
    x = torch.zeros(1, 3)
    assert torch.allclose(m(x), torch.zeros(1), atol=1e-5)


@pytest.mark.parametrize(
    "model,minimum,origin_energy",
    [
        (HimmelblauModel(), [3.0, 2.0], 170.0),
        (BoothModel(), [1.0, 3.0], 74.0),
    ],
)
def test_two_dimensional_landscape_batch_values(model, minimum, origin_energy):
    x = torch.tensor([minimum, [0.0, 0.0]])
    assert torch.allclose(model(x), torch.tensor([0.0, origin_energy]), atol=1e-6)


@pytest.mark.parametrize("model", [HimmelblauModel(), BoothModel()])
def test_two_dimensional_landscape_rejects_wrong_dimension(model):
    with pytest.raises(ValueError, match="requires 2 dimensions"):
        model(torch.randn(3, 4))


def test_styblinski_tang_is_vectorized_over_batches():
    model = StyblinskiTangModel()
    x = torch.tensor([[0.0, 0.0], [1.0, -2.0]])
    assert torch.allclose(model(x), torch.tensor([0.0, -34.0]))


@pytest.mark.parametrize(
    "model,x,expected_gradient",
    [
        (HimmelblauModel(), [1.0, -1.0], [-54.0, -2.0]),
        (BoothModel(), [-1.0, 2.0], [-28.0, -26.0]),
        (StyblinskiTangModel(), [1.0, -2.0], [-11.5, 18.5]),
    ],
)
def test_analytic_landscape_gradient(model, x, expected_gradient):
    x = torch.tensor([x])
    expected = torch.tensor([expected_gradient])
    assert torch.allclose(model.gradient(x), expected)


def test_harmonic_gradient_linear_in_x():
    m = HarmonicModel(k=2.0)
    x = torch.randn(4, 3)
    g = m.gradient(x)
    assert torch.allclose(g, 2.0 * x, atol=1e-5)


class TrainableQuadratic(BaseModel):
    r"""E(x) = 0.5 * scale * w * ||x||^2, so the score is -scale * w * x."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x, scale=1.0):
        return 0.5 * scale * self.w * x.square().sum(dim=-1)


def test_prediction_type_defaults_to_energy():
    m = TrainableQuadratic()
    x = torch.randn(4, 3)
    assert m.prediction_type == "energy"
    assert torch.equal(m(x), m.forward(x))


def test_score_prediction_is_negative_energy_gradient():
    x = torch.randn(4, 3)
    score = HarmonicModel(k=2.0, prediction_type="score")(x)
    assert torch.allclose(score, -HarmonicModel(k=2.0).gradient(x))


def test_energy_and_score_prediction_returns_both():
    m = TrainableQuadratic(prediction_type=("energy", "score"))
    x = torch.randn(4, 3)
    energy, score = m(x)
    assert torch.allclose(energy, x.square().sum(dim=-1))
    assert torch.allclose(score, -2.0 * x)


def test_score_forwards_model_kwargs():
    m = TrainableQuadratic(prediction_type="score")
    x = torch.randn(4, 3)
    assert torch.allclose(m(x, scale=3.0), -6.0 * x)


def test_score_trains_the_model_under_grad_mode():
    m = TrainableQuadratic(prediction_type="score")
    x = torch.randn(4, 3)
    m(x).sum().backward()
    assert torch.allclose(m.w.grad, -x.sum())
    assert not x.requires_grad


def test_score_keeps_graph_to_input_that_requires_grad():
    m = TrainableQuadratic(prediction_type="score")
    x = torch.randn(4, 3, requires_grad=True)
    (divergence,) = torch.autograd.grad(m(x).sum(), x)
    assert torch.allclose(divergence, torch.full_like(x, -2.0))


def test_score_is_detached_under_no_grad():
    m = TrainableQuadratic(prediction_type=("energy", "score"))
    with torch.no_grad():
        energy, score = m(torch.randn(4, 3))
    assert not energy.requires_grad
    assert not score.requires_grad


def test_score_prediction_rejects_inference_mode():
    m = TrainableQuadratic(prediction_type="score")
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no_grad"):
        m(torch.randn(4, 3))


@pytest.mark.parametrize("prediction_type", ["velocity", ("score", "energy"), "bogus"])
def test_invalid_prediction_type_raises(prediction_type):
    with pytest.raises(ValueError, match="prediction_type"):
        TrainableQuadratic(prediction_type=prediction_type)


def test_functional_model_is_held_as_structure_only():
    with torch.device("meta"):
        template = TrainableQuadratic()
    m = TrainableQuadratic(prediction_type="score", functional_model=template)
    assert list(m.children()) == []
    assert len(list(m.parameters())) == 1
    assert m.state_dict().keys() == {"w"}


def test_functional_model_must_be_an_energy_model():
    with pytest.raises(ValueError, match="functional_model"):
        TrainableQuadratic(
            prediction_type="score",
            functional_model=TrainableQuadratic(prediction_type="score"),
        )


def test_model_without_prediction_type_attribute_is_energy_model():
    m = TrainableQuadratic(prediction_type="score")
    del m.prediction_type
    x = torch.randn(4, 3)
    assert torch.equal(m(x), m.forward(x))
