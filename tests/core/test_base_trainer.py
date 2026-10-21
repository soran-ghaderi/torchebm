"""BaseTrainer with the model-centric criteria."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchebm._deprecation import TorchEBMDeprecationWarning
from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.core.base_trainer import BaseTrainer, ContrastiveDivergenceTrainer
from torchebm.losses import ContrastiveDivergenceLoss
from torchebm.samplers import LangevinDynamics
from torchebm.utils import ReplayBuffer


class Energy(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1, bias=False)
        )
        self.seen_labels = []

    def forward(self, x, y=None):
        self.seen_labels.append(y)
        energy = self.net(x).squeeze(-1)
        return energy if y is None else energy + y


class ScheduledLoss(ContrastiveDivergenceLoss):
    def __init__(self):
        super().__init__()
        self.schedule = TemperatureScheduler(
            epsilon_max=0.1, tau_star=0.5, n_steps=10, sqrt=False
        )
        self._register_param("weight", self.schedule)


def trainer(loss_fn=None, **kwargs):
    torch.manual_seed(0)
    model = Energy()
    return BaseTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        ContrastiveDivergenceLoss() if loss_fn is None else loss_fn,
        device="cpu",
        sampler=LangevinDynamics(step_size=0.1),
        n_steps=3,
        **kwargs,
    )


def parameters(model):
    return [p.detach().clone() for p in model.parameters()]


def changed(before, model):
    return any(not torch.equal(b, p) for b, p in zip(before, parameters(model)))


def test_train_step_draws_negatives_and_updates_the_model():
    t = trainer(buffer=ReplayBuffer(64, (2,)))
    before = parameters(t.model)
    metrics = t.train_step(torch.randn(8, 2))
    assert metrics["loss"].dim() == 0 and torch.isfinite(metrics["loss"])
    assert changed(before, t.model)
    assert t.buffer._position == 8


def test_without_a_buffer_the_negatives_start_from_noise():
    t = trainer()
    before = parameters(t.model)
    t.train_step(torch.randn(8, 2))
    assert changed(before, t.model)


def test_train_step_matches_the_manual_loop():
    t = trainer(buffer=ReplayBuffer(64, (2,)))
    model, buffer = copy.deepcopy(t.model), copy.deepcopy(t.buffer)
    x = torch.randn(8, 2)

    torch.manual_seed(1)
    loss = t.train_step(x)["loss"]

    torch.manual_seed(1)
    negative = LangevinDynamics(step_size=0.1)(model, buffer.sample(8), n_steps=3)
    buffer.add(negative)
    expected = ContrastiveDivergenceLoss()(model(x), model(negative))
    assert torch.equal(loss, expected.detach())
    assert torch.equal(t.buffer.storage, buffer.storage)


def test_conditional_negatives_share_the_labels():
    t = trainer()
    y = torch.arange(8, dtype=torch.float32)
    t.train_step((torch.randn(8, 2), y))
    assert len(t.model.seen_labels) == 3 + 2
    assert all(torch.equal(seen, y) for seen in t.model.seen_labels)


def test_gradient_accumulation_steps_on_the_boundary():
    t = trainer(grad_accum_steps=2)
    before = parameters(t.model)
    t.train_step(torch.randn(8, 2))
    assert not changed(before, t.model)
    t.train_step(torch.randn(8, 2))
    assert changed(before, t.model)


def test_criterion_schedulers_advance_once_per_optimizer_step():
    loss_fn = ScheduledLoss()
    t = trainer(loss_fn, grad_accum_steps=2)
    for _ in range(4):
        t.train_step(torch.randn(8, 2))
    assert loss_fn.schedule.step_count == 2


def test_compute_loss_override_drives_the_step():
    class Descent(BaseTrainer):
        def compute_loss(self, batch, model_kwargs):
            return self.model(batch, **model_kwargs).mean()

    torch.manual_seed(0)
    model = Energy()
    t = Descent(model, torch.optim.SGD(model.parameters(), lr=0.1), None, device="cpu")
    before = parameters(model)
    t.train_step(torch.randn(8, 2))
    assert changed(before, model)


def test_checkpoint_round_trip_restores_the_buffer(tmp_path):
    t = trainer(buffer=ReplayBuffer(64, (2,)))
    t.train_step(torch.randn(8, 2))
    path = tmp_path / "trainer.pt"
    t.save_checkpoint(str(path))

    restored = trainer(buffer=ReplayBuffer(64, (2,)))
    restored.load_checkpoint(str(path))
    assert torch.equal(restored.buffer.storage, t.buffer.storage)
    assert restored.buffer._position == t.buffer._position
    assert all(
        torch.equal(a, b)
        for a, b in zip(parameters(restored.model), parameters(t.model))
    )


def test_train_runs_the_epochs_and_the_callbacks():
    class Recorder:
        def __init__(self):
            self.events = []

        def on_train_start(self, trainer):
            self.events.append("train_start")

        def on_epoch_start(self, trainer, epoch):
            self.events.append("epoch_start")

        def on_batch_start(self, trainer, batch):
            self.events.append("batch_start")

        def on_batch_end(self, trainer, batch, metrics):
            self.events.append("batch_end")

        def on_epoch_end(self, trainer, epoch, metrics):
            self.events.append("epoch_end")

        def on_train_end(self, trainer, history):
            self.events.append("train_end")

    recorder = Recorder()
    t = trainer(buffer=ReplayBuffer(64, (2,)), callbacks=[recorder])
    loader = DataLoader(TensorDataset(torch.randn(32, 2)), batch_size=8)
    history = t.train(loader, num_epochs=2)
    assert len(history["loss"]) == 2
    assert all(isinstance(value, float) for value in history["loss"])
    assert recorder.events.count("batch_end") == 8
    assert recorder.events[0] == "train_start" and recorder.events[-1] == "train_end"


def test_contrastive_divergence_trainer_is_deprecated():
    with pytest.warns(TorchEBMDeprecationWarning, match="ContrastiveDivergenceTrainer"):
        with pytest.raises(TypeError):
            ContrastiveDivergenceTrainer(Energy(), LangevinDynamics(step_size=0.1))
