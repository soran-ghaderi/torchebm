r"""Parity with the official Equilibrium Matching code (github.com/raywang4/EqM).

`tests/fixtures/eqm_reference.pt` holds a small EqM with perturbed weights,
its inputs, and the outputs of the official model, loss and `sample_gd.py`
loop in float64, written by `tests/fixtures/generate_eqm_reference.py`.
"""

import pathlib

import pytest
import torch
from torch.testing import assert_close

from torchebm.interpolants import LinearInterpolant
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.models import ClassifierFreeGuidance, DiT
from torchebm.samplers import GradientDescentSampler, NesterovSampler

REFERENCE = torch.load(
    pathlib.Path(__file__).parent / "fixtures" / "eqm_reference.pt", weights_only=True
)
RENAMES = (
    ("x_embedder.", "patch_embed."),
    ("y_embedder.embedding_table.", "y_embedder.embedding."),
    ("attn.proj.", "attn.out_proj."),
    ("mlp.fc1.", "mlp.net.0."),
    ("mlp.fc2.", "mlp.net.3."),
    ("adaLN_modulation.", "modulation."),
    ("final_layer.linear.", "head.proj."),
    ("final_layer.", "head."),
)


def _rename(key):
    for old, new in RENAMES:
        key = key.replace(old, new)
    return key


@pytest.fixture(scope="module")
def model():
    model = DiT(
        input_size=8, in_channels=4, embed_dim=32, depth=2, num_heads=2,
        num_classes=10, time_invariant=True,
    )
    state_dict = REFERENCE["state_dict"]
    assert_close(model.pos_embed, state_dict["pos_embed"], rtol=0, atol=0)
    model.load_state_dict({_rename(k): v for k, v in state_dict.items() if k != "pos_embed"})
    return model.double().eval()


def test_model(model):
    with torch.no_grad():
        output = model(REFERENCE["x"], REFERENCE["t"], y=REFERENCE["y"])
    assert_close(output, REFERENCE["output"], rtol=0, atol=1e-13)


def test_loss(model):
    x0, x1, t = REFERENCE["x0"], REFERENCE["x1"], REFERENCE["t_loss"]
    x_t, u_t = LinearInterpolant()(x0, x1, t)
    with torch.no_grad():
        loss = EquilibriumMatchingLoss(reduction="none")(model(x_t, y=REFERENCE["y"]), u_t, t)
    assert_close(loss, REFERENCE["loss"], rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    "name, sampler, guidance_scale",
    [
        ("gd", GradientDescentSampler(step_size=0.0017, prediction_type="velocity"), None),
        (
            "nag_gd",
            NesterovSampler(
                step_size=0.0017, momentum=0.3, lookahead_only=True, prediction_type="velocity"
            ),
            None,
        ),
        (
            "nag_gd_cfg",
            NesterovSampler(
                step_size=0.0017, momentum=0.3, lookahead_only=True, prediction_type="velocity"
            ),
            1.5,
        ),
    ],
)
def test_sampler(model, name, sampler, guidance_scale):
    if guidance_scale is not None:
        model = ClassifierFreeGuidance(
            model, guidance_scale=guidance_scale, null_condition=10, guide_channels=3
        )
    samples = sampler(
        model, REFERENCE["z"], n_steps=249, model_kwargs={"y": REFERENCE["y_sample"]}
    )
    assert_close(samples, REFERENCE[name], rtol=0, atol=1e-12)
