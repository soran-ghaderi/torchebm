r"""Write `eqm_reference.pt` from the official Equilibrium Matching code.

Runs the model, loss and `sample_gd.py` loop of github.com/raywang4/EqM at
commit fe3b6897 on a small EqM in float64, and saves the weights, inputs and
outputs that `tests/test_eqm_reference.py` checks torchebm against. The
official timestep embedding is computed in float32; with `uncond=True` the
timestep is set to zero, so it is exact, and a forward pre-hook casts it to
float64. Run from a checkout of that commit (needs timm):

    PYTHONPATH=/path/to/EqM python tests/fixtures/generate_eqm_reference.py
"""

import pathlib

import torch
from models import EqM
from transport import create_transport

COMMIT = "fe3b6897feb43f5af9624903c81ea5bcd41ee6bc"
STEP_SIZE, MU, CFG_SCALE, NUM_SAMPLING_STEPS = 0.0017, 0.3, 1.5, 250


def sample_gd(model_fn, z, y, sampler):
    xt, m = z, torch.zeros_like(z)
    t = torch.ones(len(z), dtype=z.dtype)
    for _ in range(NUM_SAMPLING_STEPS - 1):
        if sampler == "gd":
            out = model_fn(xt, t, y)
        if sampler == "ngd":
            x_ = xt + STEP_SIZE * m * MU
            out = model_fn(x_, t, y)
            m = out
        xt = xt + out * STEP_SIZE
        t += STEP_SIZE
    return xt


torch.manual_seed(0)
model = EqM(
    input_size=8, patch_size=2, in_channels=4, hidden_size=32, depth=2, num_heads=2,
    num_classes=10, learn_sigma=False, uncond=True, ebm="none",
)
with torch.no_grad():
    for name, param in model.named_parameters():
        if name != "pos_embed":
            param.add_(0.1 * torch.randn_like(param))
state_dict = {k: v.clone() for k, v in model.state_dict().items()}
model = model.double().eval()
model.t_embedder.mlp.register_forward_pre_hook(lambda module, args: (args[0].double(),))

x = torch.randn(4, 4, 8, 8, dtype=torch.float64)
t = torch.rand(4, dtype=torch.float64)
y = torch.tensor([1, 4, 7, 10])
x1 = torch.randn(4, 4, 8, 8, dtype=torch.float64)
z = torch.randn(4, 4, 8, 8, dtype=torch.float64)
y_sample = torch.tensor([2, 5, 8, 9])

transport = create_transport("Linear", "velocity")
with torch.no_grad():
    output = model(x, t, y)
    torch.manual_seed(1)
    t_loss, x0, _ = transport.sample(x1)
    torch.manual_seed(1)
    loss = transport.training_losses(model, x1, dict(y=y))["loss"]
    gd = sample_gd(model.forward, z, y_sample, "gd")
    nag_gd = sample_gd(model.forward, z, y_sample, "ngd")
    y_cfg = torch.cat([y_sample, torch.full_like(y_sample, 10)])
    nag_gd_cfg = sample_gd(
        lambda x, t, y: model.forward_with_cfg(x, t, y, CFG_SCALE),
        torch.cat([z, z]), y_cfg, "ngd",
    )[: len(z)]

torch.save(
    {
        "commit": COMMIT, "state_dict": state_dict,
        "x": x, "t": t, "y": y, "output": output,
        "x1": x1, "x0": x0, "t_loss": t_loss, "loss": loss,
        "z": z, "y_sample": y_sample, "gd": gd, "nag_gd": nag_gd, "nag_gd_cfg": nag_gd_cfg,
    },
    pathlib.Path(__file__).with_name("eqm_reference.pt"),
)
