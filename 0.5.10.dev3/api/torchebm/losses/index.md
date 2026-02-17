# `torchebm.losses`

Loss functions for training energy-based models and generative models.

## `ContrastiveDivergence`

Bases: `BaseContrastiveDivergence`

Standard Contrastive Divergence (CD-k) loss.

CD approximates the log-likelihood gradient by running an MCMC sampler for `k_steps` to generate negative samples.

Parameters:

| Name                        | Type | Description                                       | Default         |
| --------------------------- | ---- | ------------------------------------------------- | --------------- |
| `model`                     |      | The energy-based model to train.                  | *required*      |
| `sampler`                   |      | The MCMC sampler for generating negative samples. | *required*      |
| `k_steps`                   |      | The number of MCMC steps (k in CD-k).             | `10`            |
| `persistent`                |      | If True, uses Persistent CD with a replay buffer. | `False`         |
| `buffer_size`               |      | Size of the replay buffer for PCD.                | `10000`         |
| `init_steps`                |      | Number of MCMC steps to warm up the buffer.       | `100`           |
| `new_sample_ratio`          |      | Fraction of new random samples for PCD chains.    | `0.05`          |
| `energy_reg_weight`         |      | Weight for energy regularization term.            | `0.001`         |
| `use_temperature_annealing` |      | Whether to use temperature annealing.             | `False`         |
| `min_temp`                  |      | Minimum temperature for annealing.                | `0.01`          |
| `max_temp`                  |      | Maximum temperature for annealing.                | `2.0`           |
| `temp_decay`                |      | Decay rate for temperature annealing.             | `0.999`         |
| `dtype`                     |      | Data type for computations.                       | `float32`       |
| `device`                    |      | Device for computations.                          | `device('cpu')` |

Example

```python
from torchebm.losses import ContrastiveDivergence
from torchebm.samplers import LangevinDynamics
from torchebm.core import DoubleWellEnergy

energy = DoubleWellEnergy()
sampler = LangevinDynamics(energy, step_size=0.01)
cd_loss = ContrastiveDivergence(model=energy, sampler=sampler, k_steps=10)
x = torch.randn(32, 2)
loss, neg_samples = cd_loss(x)
```

Source code in `torchebm/losses/contrastive_divergence.py`

````python
class ContrastiveDivergence(BaseContrastiveDivergence):
    r"""
    Standard Contrastive Divergence (CD-k) loss.

    CD approximates the log-likelihood gradient by running an MCMC sampler
    for `k_steps` to generate negative samples.

    Args:
        model: The energy-based model to train.
        sampler: The MCMC sampler for generating negative samples.
        k_steps: The number of MCMC steps (k in CD-k).
        persistent: If True, uses Persistent CD with a replay buffer.
        buffer_size: Size of the replay buffer for PCD.
        init_steps: Number of MCMC steps to warm up the buffer.
        new_sample_ratio: Fraction of new random samples for PCD chains.
        energy_reg_weight: Weight for energy regularization term.
        use_temperature_annealing: Whether to use temperature annealing.
        min_temp: Minimum temperature for annealing.
        max_temp: Maximum temperature for annealing.
        temp_decay: Decay rate for temperature annealing.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import ContrastiveDivergence
        from torchebm.samplers import LangevinDynamics
        from torchebm.core import DoubleWellEnergy

        energy = DoubleWellEnergy()
        sampler = LangevinDynamics(energy, step_size=0.01)
        cd_loss = ContrastiveDivergence(model=energy, sampler=sampler, k_steps=10)
        x = torch.randn(32, 2)
        loss, neg_samples = cd_loss(x)
        ```
    """

    def __init__(
        self,
        model,
        sampler,
        k_steps=10,
        persistent=False,
        buffer_size=10000,
        init_steps=100,
        new_sample_ratio=0.05,
        energy_reg_weight=0.001,
        use_temperature_annealing=False,
        min_temp=0.01,
        max_temp=2.0,
        temp_decay=0.999,
        dtype=torch.float32,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            sampler=sampler,
            k_steps=k_steps,
            persistent=persistent,
            buffer_size=buffer_size,
            new_sample_ratio=new_sample_ratio,
            init_steps=init_steps,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )
        # Additional parameters for improved stability
        self.energy_reg_weight = energy_reg_weight
        self.use_temperature_annealing = use_temperature_annealing
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temp_decay = temp_decay
        self.current_temp = max_temp

        # Register temperature as buffer for persistence
        self.register_buffer(
            "temperature", torch.tensor(max_temp, dtype=self.dtype, device=self.device)
        )

    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Contrastive Divergence loss and generates negative samples.

        Args:
            x (torch.Tensor): A batch of real data samples (positive samples).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The scalar CD loss value.
                - The generated negative samples.
        """

        batch_size = x.shape[0]
        data_shape = x.shape[1:]

        # Update temperature if annealing is enabled
        if self.use_temperature_annealing and self.training:
            self.current_temp = max(self.min_temp, self.current_temp * self.temp_decay)
            self.temperature[...] = self.current_temp  # Use ellipsis instead of index

            # If sampler has a temperature parameter, update it
            if hasattr(self.sampler, "temperature"):
                self.sampler.temperature = self.current_temp
            elif hasattr(self.sampler, "noise_scale"):
                # For samplers like Langevin, adjust noise scale based on temperature
                original_noise = getattr(self.sampler, "_original_noise_scale", None)
                if original_noise is None:
                    setattr(
                        self.sampler, "_original_noise_scale", self.sampler.noise_scale
                    )
                    original_noise = self.sampler.noise_scale

                self.sampler.noise_scale = original_noise * math.sqrt(self.current_temp)

        # Get starting points for chains (either from buffer or data)
        start_points = self.get_start_points(x)

        # Run MCMC chains to get negative samples
        pred_samples = self.sampler.sample(
            x=start_points,
            n_steps=self.k_steps,
        )

        # Update persistent buffer if using PCD
        if self.persistent:
            with torch.no_grad():
                self.update_buffer(pred_samples.detach())

        # Add energy regularization to kwargs for compute_loss
        kwargs["energy_reg_weight"] = kwargs.get(
            "energy_reg_weight", self.energy_reg_weight
        )

        # Compute contrastive divergence loss
        loss = self.compute_loss(x, pred_samples, *args, **kwargs)

        return loss, pred_samples

    def compute_loss(
        self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Computes the Contrastive Divergence loss from positive and negative samples.

        The loss is the difference between the mean energy of positive samples
        and the mean energy of negative samples.

        Args:
            x (torch.Tensor): Real data samples (positive samples).
            pred_x (torch.Tensor): Generated negative samples.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Ensure inputs are on the correct device and dtype
        x = x.to(self.device, self.dtype)
        pred_x = pred_x.to(self.device, self.dtype)

        # Compute energy of real and generated samples
        with torch.set_grad_enabled(True):
            # Add small noise to real data for stability (optional)
            if kwargs.get("add_noise_to_real", False):
                noise_scale = kwargs.get("noise_scale", 1e-4)
                x_noisy = x + noise_scale * torch.randn_like(x)
                x_energy = self.model(x_noisy)
            else:
                x_energy = self.model(x)

            pred_x_energy = self.model(pred_x)

        # Compute mean energies with improved numerical stability
        mean_x_energy = torch.mean(x_energy)
        mean_pred_energy = torch.mean(pred_x_energy)

        # Basic contrastive divergence loss: E[data] - E[model]
        loss = mean_x_energy - mean_pred_energy

        # Optional: Regularization to prevent energies from becoming too large
        # This helps with stability especially in the early phases of training
        energy_reg_weight = kwargs.get("energy_reg_weight", 0.001)
        if energy_reg_weight > 0:
            energy_reg = energy_reg_weight * (
                torch.mean(x_energy**2) + torch.mean(pred_x_energy**2)
            )
            loss = loss + energy_reg

        # Prevent extremely large gradients with a safety check
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(
                f"NaN or Inf detected in CD loss. x_energy: {mean_x_energy}, pred_energy: {mean_pred_energy}",
                RuntimeWarning,
            )
            # Return a small positive constant instead of NaN/Inf to prevent training collapse
            return torch.tensor(0.1, device=self.device, dtype=self.dtype)

        return loss
````

### `compute_loss(x, pred_x, *args, **kwargs)`

Computes the Contrastive Divergence loss from positive and negative samples.

The loss is the difference between the mean energy of positive samples and the mean energy of negative samples.

Parameters:

| Name       | Type     | Description                           | Default    |
| ---------- | -------- | ------------------------------------- | ---------- |
| `x`        | `Tensor` | Real data samples (positive samples). | *required* |
| `pred_x`   | `Tensor` | Generated negative samples.           | *required* |
| `*args`    |          | Additional positional arguments.      | `()`       |
| `**kwargs` |          | Additional keyword arguments.         | `{}`       |

Returns:

| Type     | Description                          |
| -------- | ------------------------------------ |
| `Tensor` | torch.Tensor: The scalar loss value. |

Source code in `torchebm/losses/contrastive_divergence.py`

```python
def compute_loss(
    self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """
    Computes the Contrastive Divergence loss from positive and negative samples.

    The loss is the difference between the mean energy of positive samples
    and the mean energy of negative samples.

    Args:
        x (torch.Tensor): Real data samples (positive samples).
        pred_x (torch.Tensor): Generated negative samples.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The scalar loss value.
    """
    # Ensure inputs are on the correct device and dtype
    x = x.to(self.device, self.dtype)
    pred_x = pred_x.to(self.device, self.dtype)

    # Compute energy of real and generated samples
    with torch.set_grad_enabled(True):
        # Add small noise to real data for stability (optional)
        if kwargs.get("add_noise_to_real", False):
            noise_scale = kwargs.get("noise_scale", 1e-4)
            x_noisy = x + noise_scale * torch.randn_like(x)
            x_energy = self.model(x_noisy)
        else:
            x_energy = self.model(x)

        pred_x_energy = self.model(pred_x)

    # Compute mean energies with improved numerical stability
    mean_x_energy = torch.mean(x_energy)
    mean_pred_energy = torch.mean(pred_x_energy)

    # Basic contrastive divergence loss: E[data] - E[model]
    loss = mean_x_energy - mean_pred_energy

    # Optional: Regularization to prevent energies from becoming too large
    # This helps with stability especially in the early phases of training
    energy_reg_weight = kwargs.get("energy_reg_weight", 0.001)
    if energy_reg_weight > 0:
        energy_reg = energy_reg_weight * (
            torch.mean(x_energy**2) + torch.mean(pred_x_energy**2)
        )
        loss = loss + energy_reg

    # Prevent extremely large gradients with a safety check
    if torch.isnan(loss) or torch.isinf(loss):
        warnings.warn(
            f"NaN or Inf detected in CD loss. x_energy: {mean_x_energy}, pred_energy: {mean_pred_energy}",
            RuntimeWarning,
        )
        # Return a small positive constant instead of NaN/Inf to prevent training collapse
        return torch.tensor(0.1, device=self.device, dtype=self.dtype)

    return loss
```

### `forward(x, *args, **kwargs)`

Computes the Contrastive Divergence loss and generates negative samples.

Parameters:

| Name       | Type     | Description                                      | Default    |
| ---------- | -------- | ------------------------------------------------ | ---------- |
| `x`        | `Tensor` | A batch of real data samples (positive samples). | *required* |
| `*args`    |          | Additional positional arguments.                 | `()`       |
| `**kwargs` |          | Additional keyword arguments.                    | `{}`       |

Returns:

| Type                    | Description                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `Tuple[Tensor, Tensor]` | Tuple\[torch.Tensor, torch.Tensor\]: - The scalar CD loss value. - The generated negative samples. |

Source code in `torchebm/losses/contrastive_divergence.py`

```python
def forward(
    self, x: torch.Tensor, *args, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Contrastive Divergence loss and generates negative samples.

    Args:
        x (torch.Tensor): A batch of real data samples (positive samples).
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - The scalar CD loss value.
            - The generated negative samples.
    """

    batch_size = x.shape[0]
    data_shape = x.shape[1:]

    # Update temperature if annealing is enabled
    if self.use_temperature_annealing and self.training:
        self.current_temp = max(self.min_temp, self.current_temp * self.temp_decay)
        self.temperature[...] = self.current_temp  # Use ellipsis instead of index

        # If sampler has a temperature parameter, update it
        if hasattr(self.sampler, "temperature"):
            self.sampler.temperature = self.current_temp
        elif hasattr(self.sampler, "noise_scale"):
            # For samplers like Langevin, adjust noise scale based on temperature
            original_noise = getattr(self.sampler, "_original_noise_scale", None)
            if original_noise is None:
                setattr(
                    self.sampler, "_original_noise_scale", self.sampler.noise_scale
                )
                original_noise = self.sampler.noise_scale

            self.sampler.noise_scale = original_noise * math.sqrt(self.current_temp)

    # Get starting points for chains (either from buffer or data)
    start_points = self.get_start_points(x)

    # Run MCMC chains to get negative samples
    pred_samples = self.sampler.sample(
        x=start_points,
        n_steps=self.k_steps,
    )

    # Update persistent buffer if using PCD
    if self.persistent:
        with torch.no_grad():
            self.update_buffer(pred_samples.detach())

    # Add energy regularization to kwargs for compute_loss
    kwargs["energy_reg_weight"] = kwargs.get(
        "energy_reg_weight", self.energy_reg_weight
    )

    # Compute contrastive divergence loss
    loss = self.compute_loss(x, pred_samples, *args, **kwargs)

    return loss, pred_samples
```

## `DenoisingScoreMatching`

Bases: `BaseScoreMatching`

Denoising Score Matching (DSM) from Vincent (2011).

Avoids computing the Hessian trace by matching the score of noise-perturbed data. More computationally efficient and often more stable than standard Score Matching.

Parameters:

| Name                      | Type                           | Description                                  | Default    |
| ------------------------- | ------------------------------ | -------------------------------------------- | ---------- |
| `model`                   | `BaseModel`                    | The energy-based model to train.             | *required* |
| `noise_scale`             | `float`                        | Standard deviation of Gaussian noise to add. | `0.01`     |
| `regularization_strength` | `float`                        | Coefficient for regularization.              | `0.0`      |
| `custom_regularization`   | `Optional[Callable]`           | A custom regularization function.            | `None`     |
| `use_mixed_precision`     | `bool`                         | Whether to use mixed-precision training.     | `False`    |
| `dtype`                   | `dtype`                        | Data type for computations.                  | `float32`  |
| `device`                  | `Optional[Union[str, device]]` | Device for computations.                     | `None`     |

Example

```python
from torchebm.losses import DenoisingScoreMatching
from torchebm.core import DoubleWellEnergy
import torch

energy = DoubleWellEnergy()
loss_fn = DenoisingScoreMatching(model=energy, noise_scale=0.1)
x = torch.randn(32, 2)
loss = loss_fn(x)
```

Source code in `torchebm/losses/score_matching.py`

````python
class DenoisingScoreMatching(BaseScoreMatching):
    r"""
    Denoising Score Matching (DSM) from Vincent (2011).

    Avoids computing the Hessian trace by matching the score of noise-perturbed
    data. More computationally efficient and often more stable than standard
    Score Matching.

    Args:
        model: The energy-based model to train.
        noise_scale: Standard deviation of Gaussian noise to add.
        regularization_strength: Coefficient for regularization.
        custom_regularization: A custom regularization function.
        use_mixed_precision: Whether to use mixed-precision training.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import DenoisingScoreMatching
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        loss_fn = DenoisingScoreMatching(model=energy, noise_scale=0.1)
        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        noise_scale: float = 0.01,
        regularization_strength: float = 0.0,
        custom_regularization: Optional[Callable] = None,
        use_mixed_precision: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            noise_scale=noise_scale,
            regularization_strength=regularization_strength,
            use_autograd=True,
            custom_regularization=custom_regularization,
            use_mixed_precision=use_mixed_precision,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the denoising score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar denoising score matching loss.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(x, *args, **kwargs)

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the denoising score matching loss.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar denoising score matching loss.
        """
        x_perturbed, noise = self.perturb_data(x)

        score = self.compute_score(x_perturbed)

        target_score = -noise / (self.noise_scale**2)

        loss = (
            0.5
            * torch.sum(
                (score - target_score) ** 2, dim=list(range(1, len(x.shape)))
            ).mean()
        )

        return loss
````

### `compute_loss(x, *args, **kwargs)`

Computes the denoising score matching loss.

Parameters:

| Name       | Type     | Description                                           | Default    |
| ---------- | -------- | ----------------------------------------------------- | ---------- |
| `x`        | `Tensor` | Input data tensor of shape (batch_size, \*data_dims). | *required* |
| `*args`    |          | Additional arguments.                                 | `()`       |
| `**kwargs` |          | Additional keyword arguments.                         | `{}`       |

Returns:

| Type     | Description                                             |
| -------- | ------------------------------------------------------- |
| `Tensor` | torch.Tensor: The scalar denoising score matching loss. |

Source code in `torchebm/losses/score_matching.py`

```python
def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    r"""
    Computes the denoising score matching loss.

    Args:
        x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The scalar denoising score matching loss.
    """
    x_perturbed, noise = self.perturb_data(x)

    score = self.compute_score(x_perturbed)

    target_score = -noise / (self.noise_scale**2)

    loss = (
        0.5
        * torch.sum(
            (score - target_score) ** 2, dim=list(range(1, len(x.shape)))
        ).mean()
    )

    return loss
```

### `forward(x, *args, **kwargs)`

Computes the denoising score matching loss for a batch of data.

Parameters:

| Name       | Type     | Description                                           | Default    |
| ---------- | -------- | ----------------------------------------------------- | ---------- |
| `x`        | `Tensor` | Input data tensor of shape (batch_size, \*data_dims). | *required* |
| `*args`    |          | Additional positional arguments.                      | `()`       |
| `**kwargs` |          | Additional keyword arguments.                         | `{}`       |

Returns:

| Type     | Description                                             |
| -------- | ------------------------------------------------------- |
| `Tensor` | torch.Tensor: The scalar denoising score matching loss. |

Source code in `torchebm/losses/score_matching.py`

```python
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    r"""
    Computes the denoising score matching loss for a batch of data.

    Args:
        x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The scalar denoising score matching loss.
    """
    if (x.device != self.device) or (x.dtype != self.dtype):
        x = x.to(device=self.device, dtype=self.dtype)

    with self.autocast_context():
        loss = self.compute_loss(x, *args, **kwargs)

    if self.regularization_strength > 0 or self.custom_regularization is not None:
        loss = self.add_regularization(loss, x)

    return loss
```

## `EquilibriumMatchingLoss`

Bases: `BaseLoss`

Equilibrium Matching (EqM) training loss.

Implements gradient matching for learning equilibrium energy landscapes. Supports both implicit (vector field) and explicit (energy-based) formulations, with multiple prediction types and loss weighting schemes.

The target is ((\\epsilon - x) \\cdot c(\\gamma)) where:

- (\\epsilon) is noise (x0), (x) is data (x1)
- For linear interpolant: target is ((x_0 - x_1) \\cdot c(t)) (noise - data)
- (c(\\gamma) = \\lambda \\cdot \\min(1, (1-\\gamma)/(1-a))) is truncated decay

For ODE sampling, use `negate_velocity=True` in FlowSampler since velocity (v = -f(x) = x - \\epsilon).

Parameters:

| Name                  | Type                                          | Description                                                                                                                                                                                                                             | Default      |
| --------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `model`               | `Module`                                      | Neural network predicting velocity/score/noise.                                                                                                                                                                                         | *required*   |
| `prediction`          | `Literal['velocity', 'score', 'noise']`       | Network prediction type ('velocity', 'score', or 'noise').                                                                                                                                                                              | `'velocity'` |
| `energy_type`         | `Literal['none', 'dot', 'l2', 'mean']`        | Energy formulation type: - 'none': Implicit EqM, model predicts gradient directly - 'dot': (g(x) = x \\cdot f(x)), dot product energy formulation - 'l2': (g(x) = -\\frac{1}{2}\|f(x)\|^2) (experimental) - 'mean': Same as dot (alias) | `'none'`     |
| `interpolant`         | `Union[str, BaseInterpolant]`                 | Interpolant name (e.g. 'linear', 'cosine', 'vp') or BaseInterpolant instance.                                                                                                                                                           | `'linear'`   |
| `loss_weight`         | `Optional[Literal['velocity', 'likelihood']]` | Loss weighting scheme ('velocity', 'likelihood', or None).                                                                                                                                                                              | `None`       |
| `train_eps`           | `float`                                       | Epsilon for training time interval stability.                                                                                                                                                                                           | `0.0`        |
| `ct_threshold`        | `float`                                       | Decay threshold (a) for (c(t)). Decay starts after (t > a). Default: 0.8.                                                                                                                                                               | `0.8`        |
| `ct_multiplier`       | `float`                                       | Gradient multiplier (\\lambda) for (c(t)). Default: 4.0.                                                                                                                                                                                | `4.0`        |
| `apply_dispersion`    | `bool`                                        | Whether to apply dispersive regularization.                                                                                                                                                                                             | `False`      |
| `dispersion_weight`   | `float`                                       | Weight for dispersive loss term.                                                                                                                                                                                                        | `0.5`        |
| `time_invariant`      | `bool`                                        | If True, pass zeros for time to model (EqM default).                                                                                                                                                                                    | `True`       |
| `dtype`               | `dtype`                                       | Data type for computations.                                                                                                                                                                                                             | `float32`    |
| `device`              | `Optional[Union[str, device]]`                | Device for computations.                                                                                                                                                                                                                | `None`       |
| `use_mixed_precision` | `bool`                                        | Whether to use mixed precision.                                                                                                                                                                                                         | `False`      |
| `clip_value`          | `Optional[float]`                             | Optional value to clamp the loss.                                                                                                                                                                                                       | `None`       |

Example

```python
from torchebm.losses import EquilibriumMatchingLoss
import torch.nn as nn
import torch

# Implicit EqM with velocity prediction (default)
model = MyTimeConditionedModel()
loss_fn = EquilibriumMatchingLoss(
    model=model,
    prediction="velocity",
    energy_type="none",
)

# Explicit EqM-E with dot product (for OOD detection)
loss_fn_explicit = EquilibriumMatchingLoss(
    model=model,
    prediction="velocity",
    energy_type="dot",
)

x = torch.randn(32, 2)
loss = loss_fn(x)
```

Source code in `torchebm/losses/equilibrium_matching.py`

````python
class EquilibriumMatchingLoss(BaseLoss):
    r"""Equilibrium Matching (EqM) training loss.

    Implements gradient matching for learning equilibrium energy landscapes.
    Supports both implicit (vector field) and explicit (energy-based) formulations,
    with multiple prediction types and loss weighting schemes.

    The target is $(\epsilon - x) \cdot c(\gamma)$ where:
    - $\epsilon$ is noise (x0), $x$ is data (x1)
    - For linear interpolant: target is $(x_0 - x_1) \cdot c(t)$ (noise - data)
    - $c(\gamma) = \lambda \cdot \min(1, (1-\gamma)/(1-a))$ is truncated decay

    For ODE sampling, use ``negate_velocity=True`` in FlowSampler since
    velocity $v = -f(x) = x - \epsilon$.

    Args:
        model: Neural network predicting velocity/score/noise.
        prediction: Network prediction type ('velocity', 'score', or 'noise').
        energy_type: Energy formulation type:
            - 'none': Implicit EqM, model predicts gradient directly
            - 'dot': $g(x) = x \cdot f(x)$, dot product energy formulation
            - 'l2': $g(x) = -\frac{1}{2}\|f(x)\|^2$ (experimental)
            - 'mean': Same as dot (alias)
        interpolant: Interpolant name (e.g. 'linear', 'cosine', 'vp') or BaseInterpolant instance.
        loss_weight: Loss weighting scheme ('velocity', 'likelihood', or None).
        train_eps: Epsilon for training time interval stability.
        ct_threshold: Decay threshold $a$ for $c(t)$. Decay starts after $t > a$. Default: 0.8.
        ct_multiplier: Gradient multiplier $\lambda$ for $c(t)$. Default: 4.0.
        apply_dispersion: Whether to apply dispersive regularization.
        dispersion_weight: Weight for dispersive loss term.
        time_invariant: If True, pass zeros for time to model (EqM default).
        dtype: Data type for computations.
        device: Device for computations.
        use_mixed_precision: Whether to use mixed precision.
        clip_value: Optional value to clamp the loss.

    Example:
        ```python
        from torchebm.losses import EquilibriumMatchingLoss
        import torch.nn as nn
        import torch

        # Implicit EqM with velocity prediction (default)
        model = MyTimeConditionedModel()
        loss_fn = EquilibriumMatchingLoss(
            model=model,
            prediction="velocity",
            energy_type="none",
        )

        # Explicit EqM-E with dot product (for OOD detection)
        loss_fn_explicit = EquilibriumMatchingLoss(
            model=model,
            prediction="velocity",
            energy_type="dot",
        )

        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        prediction: Literal["velocity", "score", "noise"] = "velocity",
        energy_type: Literal["none", "dot", "l2", "mean"] = "none",
        interpolant: Union[str, BaseInterpolant] = "linear",
        loss_weight: Optional[Literal["velocity", "likelihood"]] = None,
        train_eps: float = 0.0,
        ct_threshold: float = 0.8,
        ct_multiplier: float = 4.0,
        apply_dispersion: bool = False,
        dispersion_weight: float = 0.5,
        time_invariant: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = False,
        clip_value: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
            clip_value=clip_value,
            *args,
            **kwargs,
        )
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.prediction = prediction
        self.energy_type = energy_type
        self.loss_weight = loss_weight
        self.train_eps = train_eps
        self.ct_threshold = ct_threshold
        self.ct_multiplier = ct_multiplier
        self.apply_dispersion = apply_dispersion
        self.dispersion_weight = dispersion_weight
        self.time_invariant = time_invariant
        if isinstance(interpolant, str):
            self.interpolant = get_interpolant(interpolant)
        else:
            self.interpolant = interpolant

    def _check_interval(self) -> tuple[float, float]:
        r"""Get training time interval respecting epsilon."""
        t0 = self.train_eps
        t1 = 1.0 - self.train_eps
        return t0, t1

    def _compute_explicit_energy_gradient(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute explicit energy and its gradient.

        Args:
            xt: Interpolated samples with requires_grad=True.
            model_output: Raw model output (vector field).
            training: Whether to create computation graph.

        Returns:
            Tuple of (gradient field, energy scalar per sample).
        """
        if self.energy_type == "dot" or self.energy_type == "mean":
            # g(x) = x · f(x)
            energy = (xt * model_output).sum(dim=tuple(range(1, xt.ndim)))
        elif self.energy_type == "l2":
            # g(x) = -0.5 ||f(x)||^2
            energy = -0.5 * (model_output**2).sum(dim=tuple(range(1, model_output.ndim)))
        else:
            raise ValueError(f"Unknown energy type: {self.energy_type}")

        # Compute gradient of energy w.r.t. input
        if xt.requires_grad:
            grad = torch.autograd.grad(
                energy.sum(),
                xt,
                create_graph=training,
            )[0]
        else:
            grad = model_output  # Fallback if no grad required

        return grad, energy

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute EqM loss (nn.Module interface).

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            **kwargs: Additional model arguments.

        Returns:
            Scalar loss value.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(x, *args, **kwargs)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute the equilibrium matching loss.

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            **kwargs: Additional model arguments passed to the network.

        Returns:
            Scalar loss value.
        """
        terms = self.training_losses(x, model_kwargs=kwargs)
        return terms["loss"].mean()

    def training_losses(
        self,
        x1: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute training losses with detailed outputs.

        Implements gradient matching with EqM target:
        - Target: $(\epsilon - x) \cdot c(t) = (x_0 - x_1) \cdot c(t)$
        - Time-invariant: zeros out time if time_invariant=True

        Args:
            x1: Data samples of shape (batch_size, ...).
            model_kwargs: Additional model arguments.

        Returns:
            Dictionary with 'loss' (per-sample), 'pred', and optionally 'energy'.
        """
        if model_kwargs is None:
            model_kwargs = {}

        x1 = x1.to(device=self.device, dtype=self.dtype)
        batch = x1.shape[0]

        # Sample noise and time
        x0 = torch.randn_like(x1)
        t0, t1 = self._check_interval()
        t = torch.rand(batch, device=self.device, dtype=self.dtype) * (t1 - t0) + t0

        # Interpolate: xt between x0 (noise) and x1 (data)
        xt, ut = self.interpolant.interpolate(x0, x1, t)

        # EqM target: -ut * c(t) where ut = d_alpha*x1 + d_sigma*x0
        # For linear interpolant, -ut = x0 - x1 (equivalent to original formulation).
        # For VP/cosine, ut encodes the schedule-specific velocity coefficients.
        # Sampling with negate_velocity=True recovers the positive velocity ut*c(t).
        ct = compute_eqm_ct(t, threshold=self.ct_threshold, multiplier=self.ct_multiplier)
        ct = ct.view(batch, *([1] * (xt.ndim - 1)))
        target = -ut * ct

        # For explicit energy, we need gradients w.r.t. xt
        if self.energy_type != "none":
            xt = xt.detach().requires_grad_(True)

        # EqM: zero out time for time-invariance (model still receives t for API compat)
        t_model = torch.zeros_like(t) if self.time_invariant else t

        with self.autocast_context():
            model_output = self.model(xt, t_model, **model_kwargs)

        if isinstance(model_output, tuple):
            model_output, act = model_output
        else:
            act = []

        # Compute dispersive loss if enabled
        disp_loss = 0.0
        if self.apply_dispersion and len(act) > 0:
            if isinstance(act, list):
                disp_loss = dispersive_loss(act[-1])
            else:
                disp_loss = dispersive_loss(act)

        terms = {"pred": model_output}

        # Compute loss based on prediction type
        if self.prediction == "velocity":
            if self.energy_type == "none":
                # Implicit EqM: model directly predicts gradient field
                terms["loss"] = mean_flat((model_output - target) ** 2)
            else:
                # Explicit EqM-E: compute gradient of energy function
                grad, energy = self._compute_explicit_energy_gradient(
                    xt, model_output, training=self.model.training
                )
                terms["loss"] = mean_flat((grad - target) ** 2)
                terms["energy"] = energy
        else:
            # Score or noise prediction with optional weighting
            t_expanded = expand_t_like_x(t, xt)
            _, drift_var = self.interpolant.compute_drift(xt, t)
            sigma_t, _ = self.interpolant.compute_sigma_t(t_expanded)

            if self.loss_weight == "velocity":
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_weight == "likelihood":
                weight = drift_var / (sigma_t**2)
            else:
                weight = 1.0

            if self.prediction == "noise":
                terms["loss"] = mean_flat(weight * (model_output - x0) ** 2)
            elif self.prediction == "score":
                terms["loss"] = mean_flat(weight * (model_output * sigma_t + x0) ** 2)
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction}")

        # Add dispersive regularization
        if self.apply_dispersion:
            terms["loss"] = terms["loss"] + self.dispersion_weight * disp_loss

        return terms

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prediction={self.prediction!r}, "
            f"energy_type={self.energy_type!r}, "
            f"interpolant={type(self.interpolant).__name__})"
        )
````

### `compute_loss(x, *args, **kwargs)`

Compute the equilibrium matching loss.

Parameters:

| Name       | Type     | Description                                       | Default    |
| ---------- | -------- | ------------------------------------------------- | ---------- |
| `x`        | `Tensor` | Data samples of shape (batch_size, ...).          | *required* |
| `*args`    |          | Additional positional arguments.                  | `()`       |
| `**kwargs` |          | Additional model arguments passed to the network. | `{}`       |

Returns:

| Type     | Description        |
| -------- | ------------------ |
| `Tensor` | Scalar loss value. |

Source code in `torchebm/losses/equilibrium_matching.py`

```python
def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    r"""
    Compute the equilibrium matching loss.

    Args:
        x: Data samples of shape (batch_size, ...).
        *args: Additional positional arguments.
        **kwargs: Additional model arguments passed to the network.

    Returns:
        Scalar loss value.
    """
    terms = self.training_losses(x, model_kwargs=kwargs)
    return terms["loss"].mean()
```

### `forward(x, *args, **kwargs)`

Compute EqM loss (nn.Module interface).

Parameters:

| Name       | Type     | Description                              | Default    |
| ---------- | -------- | ---------------------------------------- | ---------- |
| `x`        | `Tensor` | Data samples of shape (batch_size, ...). | *required* |
| `*args`    |          | Additional positional arguments.         | `()`       |
| `**kwargs` |          | Additional model arguments.              | `{}`       |

Returns:

| Type     | Description        |
| -------- | ------------------ |
| `Tensor` | Scalar loss value. |

Source code in `torchebm/losses/equilibrium_matching.py`

```python
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    r"""
    Compute EqM loss (nn.Module interface).

    Args:
        x: Data samples of shape (batch_size, ...).
        *args: Additional positional arguments.
        **kwargs: Additional model arguments.

    Returns:
        Scalar loss value.
    """
    if (x.device != self.device) or (x.dtype != self.dtype):
        x = x.to(device=self.device, dtype=self.dtype)

    with self.autocast_context():
        loss = self.compute_loss(x, *args, **kwargs)

    return loss
```

### `training_losses(x1, model_kwargs=None)`

Compute training losses with detailed outputs.

Implements gradient matching with EqM target:

- Target: ((\\epsilon - x) \\cdot c(t) = (x_0 - x_1) \\cdot c(t))
- Time-invariant: zeros out time if time_invariant=True

Parameters:

| Name           | Type                       | Description                              | Default    |
| -------------- | -------------------------- | ---------------------------------------- | ---------- |
| `x1`           | `Tensor`                   | Data samples of shape (batch_size, ...). | *required* |
| `model_kwargs` | `Optional[Dict[str, Any]]` | Additional model arguments.              | `None`     |

Returns:

| Type                | Description                                                           |
| ------------------- | --------------------------------------------------------------------- |
| `Dict[str, Tensor]` | Dictionary with 'loss' (per-sample), 'pred', and optionally 'energy'. |

Source code in `torchebm/losses/equilibrium_matching.py`

```python
def training_losses(
    self,
    x1: torch.Tensor,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    r"""Compute training losses with detailed outputs.

    Implements gradient matching with EqM target:
    - Target: $(\epsilon - x) \cdot c(t) = (x_0 - x_1) \cdot c(t)$
    - Time-invariant: zeros out time if time_invariant=True

    Args:
        x1: Data samples of shape (batch_size, ...).
        model_kwargs: Additional model arguments.

    Returns:
        Dictionary with 'loss' (per-sample), 'pred', and optionally 'energy'.
    """
    if model_kwargs is None:
        model_kwargs = {}

    x1 = x1.to(device=self.device, dtype=self.dtype)
    batch = x1.shape[0]

    # Sample noise and time
    x0 = torch.randn_like(x1)
    t0, t1 = self._check_interval()
    t = torch.rand(batch, device=self.device, dtype=self.dtype) * (t1 - t0) + t0

    # Interpolate: xt between x0 (noise) and x1 (data)
    xt, ut = self.interpolant.interpolate(x0, x1, t)

    # EqM target: -ut * c(t) where ut = d_alpha*x1 + d_sigma*x0
    # For linear interpolant, -ut = x0 - x1 (equivalent to original formulation).
    # For VP/cosine, ut encodes the schedule-specific velocity coefficients.
    # Sampling with negate_velocity=True recovers the positive velocity ut*c(t).
    ct = compute_eqm_ct(t, threshold=self.ct_threshold, multiplier=self.ct_multiplier)
    ct = ct.view(batch, *([1] * (xt.ndim - 1)))
    target = -ut * ct

    # For explicit energy, we need gradients w.r.t. xt
    if self.energy_type != "none":
        xt = xt.detach().requires_grad_(True)

    # EqM: zero out time for time-invariance (model still receives t for API compat)
    t_model = torch.zeros_like(t) if self.time_invariant else t

    with self.autocast_context():
        model_output = self.model(xt, t_model, **model_kwargs)

    if isinstance(model_output, tuple):
        model_output, act = model_output
    else:
        act = []

    # Compute dispersive loss if enabled
    disp_loss = 0.0
    if self.apply_dispersion and len(act) > 0:
        if isinstance(act, list):
            disp_loss = dispersive_loss(act[-1])
        else:
            disp_loss = dispersive_loss(act)

    terms = {"pred": model_output}

    # Compute loss based on prediction type
    if self.prediction == "velocity":
        if self.energy_type == "none":
            # Implicit EqM: model directly predicts gradient field
            terms["loss"] = mean_flat((model_output - target) ** 2)
        else:
            # Explicit EqM-E: compute gradient of energy function
            grad, energy = self._compute_explicit_energy_gradient(
                xt, model_output, training=self.model.training
            )
            terms["loss"] = mean_flat((grad - target) ** 2)
            terms["energy"] = energy
    else:
        # Score or noise prediction with optional weighting
        t_expanded = expand_t_like_x(t, xt)
        _, drift_var = self.interpolant.compute_drift(xt, t)
        sigma_t, _ = self.interpolant.compute_sigma_t(t_expanded)

        if self.loss_weight == "velocity":
            weight = (drift_var / sigma_t) ** 2
        elif self.loss_weight == "likelihood":
            weight = drift_var / (sigma_t**2)
        else:
            weight = 1.0

        if self.prediction == "noise":
            terms["loss"] = mean_flat(weight * (model_output - x0) ** 2)
        elif self.prediction == "score":
            terms["loss"] = mean_flat(weight * (model_output * sigma_t + x0) ** 2)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction}")

    # Add dispersive regularization
    if self.apply_dispersion:
        terms["loss"] = terms["loss"] + self.dispersion_weight * disp_loss

    return terms
```

## `ScoreMatching`

Bases: `BaseScoreMatching`

Original Score Matching loss from Hyvärinen (2005).

Trains an energy-based model by matching the model's score function (\\nabla_x \\log p\_\\theta(x)) to the data's score. Avoids MCMC sampling but requires computing the trace of the Hessian.

Parameters:

| Name                      | Type                           | Description                                     | Default    |
| ------------------------- | ------------------------------ | ----------------------------------------------- | ---------- |
| `model`                   | `BaseModel`                    | The energy-based model to train.                | *required* |
| `hessian_method`          | `str`                          | Method for Hessian trace ('exact' or 'approx'). | `'exact'`  |
| `regularization_strength` | `float`                        | Coefficient for regularization.                 | `0.0`      |
| `custom_regularization`   | `Optional[Callable]`           | A custom regularization function.               | `None`     |
| `use_mixed_precision`     | `bool`                         | Whether to use mixed-precision training.        | `False`    |
| `dtype`                   | `dtype`                        | Data type for computations.                     | `float32`  |
| `device`                  | `Optional[Union[str, device]]` | Device for computations.                        | `None`     |

Example

```python
from torchebm.losses import ScoreMatching
from torchebm.core import DoubleWellEnergy
import torch

energy = DoubleWellEnergy()
loss_fn = ScoreMatching(model=energy, hessian_method="exact")
x = torch.randn(32, 2)
loss = loss_fn(x)
```

Source code in `torchebm/losses/score_matching.py`

````python
class ScoreMatching(BaseScoreMatching):
    r"""
    Original Score Matching loss from Hyvärinen (2005).

    Trains an energy-based model by matching the model's score function
    \(\nabla_x \log p_\theta(x)\) to the data's score. Avoids MCMC sampling
    but requires computing the trace of the Hessian.

    Args:
        model: The energy-based model to train.
        hessian_method: Method for Hessian trace ('exact' or 'approx').
        regularization_strength: Coefficient for regularization.
        custom_regularization: A custom regularization function.
        use_mixed_precision: Whether to use mixed-precision training.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import ScoreMatching
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        loss_fn = ScoreMatching(model=energy, hessian_method="exact")
        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        hessian_method: str = "exact",
        regularization_strength: float = 0.0,
        custom_regularization: Optional[Callable] = None,
        use_mixed_precision: bool = False,
        is_training=True,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            regularization_strength=regularization_strength,
            use_autograd=True,
            custom_regularization=custom_regularization,
            use_mixed_precision=use_mixed_precision,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

        self.hessian_method = hessian_method
        self.training = is_training
        valid_methods = ["exact", "approx"]
        if self.hessian_method not in valid_methods:
            warnings.warn(
                f"Invalid hessian_method '{self.hessian_method}'. "
                f"Using 'exact' instead. Valid options are: {valid_methods}",
                UserWarning,
            )
            self.hessian_method = "exact"

        if self.use_mixed_precision and self.hessian_method == "exact":
            warnings.warn(
                "Using 'exact' Hessian method with mixed precision may be unstable. "
                "Consider using SlicedScoreMatching for better numerical stability.",
                UserWarning,
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar score matching loss.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(x, *args, **kwargs)

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the score matching loss using the specified Hessian computation method.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar score matching loss.
        """

        if self.hessian_method == "approx":
            return self._approx_score_matching(x)
        else:
            return self._exact_score_matching(x)

    def _exact_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes score matching loss with an exact Hessian trace.

        This method is computationally expensive for high-dimensional data.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: The score matching loss.
        """
        batch_size = x.shape[0]
        feature_dim = x.numel() // batch_size

        x_leaf = x.detach().clone()
        x_leaf.requires_grad_(True)

        energy = self.model(x_leaf)
        logp_sum = (-energy).sum()
        grad1 = torch.autograd.grad(
            logp_sum, x_leaf, create_graph=True, retain_graph=True
        )[0]

        grad1_flat = grad1.view(batch_size, -1)
        term1 = 0.5 * grad1_flat.pow(2).sum(dim=1)

        laplacian = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        for i in range(feature_dim):
            comp_sum = grad1_flat[:, i].sum()
            grad2_full = torch.autograd.grad(
                comp_sum,
                x_leaf,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad2_full is None:
                grad2_comp = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
            else:
                grad2_comp = grad2_full.view(batch_size, -1)[:, i]
            laplacian += grad2_comp

        loss_per_sample = term1 + laplacian
        return loss_per_sample.mean()

    def _approx_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes score matching loss using a finite-difference approximation for the Hessian trace.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: The score matching loss.
        """

        batch_size = x.shape[0]
        data_dim = x.numel() // batch_size

        x_detached = x.detach().clone()
        x_detached.requires_grad_(True)

        score = self.compute_score(x_detached)
        score_square_term = (
            0.5 * torch.sum(score**2, dim=list(range(1, len(x.shape)))).mean()
        )

        epsilon = 1e-5
        x_noise = x_detached + epsilon * torch.randn_like(x_detached)

        score_x = self.compute_score(x_detached)
        score_x_noise = self.compute_score(x_noise)

        hessian_trace = torch.sum(
            (score_x_noise - score_x) * (x_noise - x_detached),
            dim=list(range(1, len(x.shape))),
        ).mean() / (epsilon**2 * data_dim)

        loss = score_square_term - hessian_trace

        return loss

    def _hutchinson_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        DEPRECATED: Use SlicedScoreMatching for efficient trace estimation.

        This method has been deprecated in favor of SlicedScoreMatching which provides
        a more efficient and theoretically sound implementation of Hutchinson's estimator.
        """
        warnings.warn(
            "ScoreMatching._hutchinson_score_matching is deprecated. "
            "Use SlicedScoreMatching for efficient trace estimation instead.",
            DeprecationWarning,
        )
        return self._exact_score_matching(x)
````

### `compute_loss(x, *args, **kwargs)`

Computes the score matching loss using the specified Hessian computation method.

Parameters:

| Name       | Type     | Description                                           | Default    |
| ---------- | -------- | ----------------------------------------------------- | ---------- |
| `x`        | `Tensor` | Input data tensor of shape (batch_size, \*data_dims). | *required* |
| `*args`    |          | Additional arguments.                                 | `()`       |
| `**kwargs` |          | Additional keyword arguments.                         | `{}`       |

Returns:

| Type     | Description                                   |
| -------- | --------------------------------------------- |
| `Tensor` | torch.Tensor: The scalar score matching loss. |

Source code in `torchebm/losses/score_matching.py`

```python
def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    r"""
    Computes the score matching loss using the specified Hessian computation method.

    Args:
        x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The scalar score matching loss.
    """

    if self.hessian_method == "approx":
        return self._approx_score_matching(x)
    else:
        return self._exact_score_matching(x)
```

### `forward(x, *args, **kwargs)`

Computes the score matching loss for a batch of data.

Parameters:

| Name       | Type     | Description                                           | Default    |
| ---------- | -------- | ----------------------------------------------------- | ---------- |
| `x`        | `Tensor` | Input data tensor of shape (batch_size, \*data_dims). | *required* |
| `*args`    |          | Additional positional arguments.                      | `()`       |
| `**kwargs` |          | Additional keyword arguments.                         | `{}`       |

Returns:

| Type     | Description                                   |
| -------- | --------------------------------------------- |
| `Tensor` | torch.Tensor: The scalar score matching loss. |

Source code in `torchebm/losses/score_matching.py`

```python
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    r"""
    Computes the score matching loss for a batch of data.

    Args:
        x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The scalar score matching loss.
    """
    if (x.device != self.device) or (x.dtype != self.dtype):
        x = x.to(device=self.device, dtype=self.dtype)

    with self.autocast_context():
        loss = self.compute_loss(x, *args, **kwargs)

    if self.regularization_strength > 0 or self.custom_regularization is not None:
        loss = self.add_regularization(loss, x)

    return loss
```

## `SlicedScoreMatching`

Bases: `BaseScoreMatching`

Sliced Score Matching (SSM) from Song et al. (2019).

A scalable variant that uses random projections to efficiently approximate the score matching objective, avoiding expensive Hessian trace computation.

Parameters:

| Name                      | Type                           | Description                                               | Default        |
| ------------------------- | ------------------------------ | --------------------------------------------------------- | -------------- |
| `model`                   | `BaseModel`                    | The energy-based model to train.                          | *required*     |
| `n_projections`           | `int`                          | Number of random projections to use.                      | `5`            |
| `projection_type`         | `str`                          | Type of projections ('rademacher', 'sphere', 'gaussian'). | `'rademacher'` |
| `regularization_strength` | `float`                        | Coefficient for regularization.                           | `0.0`          |
| `custom_regularization`   | `Optional[Callable]`           | A custom regularization function.                         | `None`         |
| `use_mixed_precision`     | `bool`                         | Whether to use mixed-precision training.                  | `False`        |
| `dtype`                   | `dtype`                        | Data type for computations.                               | `float32`      |
| `device`                  | `Optional[Union[str, device]]` | Device for computations.                                  | `None`         |

Example

```python
from torchebm.losses import SlicedScoreMatching
from torchebm.core import DoubleWellEnergy
import torch

energy = DoubleWellEnergy()
loss_fn = SlicedScoreMatching(model=energy, n_projections=5)
x = torch.randn(32, 2)
loss = loss_fn(x)
```

Source code in `torchebm/losses/score_matching.py`

````python
class SlicedScoreMatching(BaseScoreMatching):
    r"""
    Sliced Score Matching (SSM) from Song et al. (2019).

    A scalable variant that uses random projections to efficiently approximate
    the score matching objective, avoiding expensive Hessian trace computation.

    Args:
        model: The energy-based model to train.
        n_projections: Number of random projections to use.
        projection_type: Type of projections ('rademacher', 'sphere', 'gaussian').
        regularization_strength: Coefficient for regularization.
        custom_regularization: A custom regularization function.
        use_mixed_precision: Whether to use mixed-precision training.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import SlicedScoreMatching
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        loss_fn = SlicedScoreMatching(model=energy, n_projections=5)
        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        n_projections: int = 5,
        projection_type: str = "rademacher",
        regularization_strength: float = 0.0,
        custom_regularization: Optional[Callable] = None,
        use_mixed_precision: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            regularization_strength=regularization_strength,
            use_autograd=True,
            custom_regularization=custom_regularization,
            use_mixed_precision=use_mixed_precision,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

        self.n_projections = n_projections
        self.projection_type = projection_type

        # Validate projection_type
        valid_types = ["rademacher", "sphere", "gaussian"]
        if self.projection_type not in valid_types:
            warnings.warn(
                f"Invalid projection_type '{self.projection_type}'. "
                f"Using 'rademacher' instead. Valid options are: {valid_types}",
                UserWarning,
            )
            self.projection_type = "rademacher"

    def _get_random_projections(self, shape: torch.Size) -> torch.Tensor:
        r"""
        Generates random vectors for projections.

        Args:
            shape (torch.Size): The shape of the vectors to generate.

        Returns:
            torch.Tensor: A tensor of random projection vectors.
        """
        vectors = torch.randn_like(shape)
        if self.projection_type == "rademacher":
            return vectors.sign()
        elif self.projection_type == "sphere":
            return (
                vectors
                / torch.norm(vectors, dim=-1, keepdim=True)
                * torch.sqrt(vectors.shape[-1])
            )
        else:
            return vectors

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the sliced score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar sliced score matching loss.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(x, *args, **kwargs)

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the sliced score matching loss using random projections.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar sliced score matching loss.
        """

        dup_x = (
            x.unsqueeze(0)
            .expand(self.n_projections, *x.shape)
            .contiguous()
            .view(-1, *x.shape[1:])
        ).requires_grad_(
            True
        )  # final shape: (n_particles * batch_size, d). tracing the shape: (batch_size, d) -> (1, batch_size, d)
        # -> (n_particles, batch_size, d) -> (n_particles, batch_size, d) -> (n_particles * batch_size, d)

        n_vectors = self._get_random_projections(dup_x)

        logp = (-self.model(dup_x)).sum()
        grad1 = torch.autograd.grad(logp, dup_x, create_graph=True)[0]
        v_score = torch.sum(grad1 * n_vectors, dim=-1)
        term1 = 0.5 * (v_score**2)

        grad_v = torch.autograd.grad(v_score.sum(), dup_x, create_graph=True)[0]
        term2 = torch.sum(n_vectors * grad_v, dim=-1)

        term1 = term1.view(self.n_projections, -1).mean(dim=0)
        term2 = term2.view(self.n_projections, -1).mean(dim=0)

        loss = term2 + term1

        return loss.mean()
````

### `compute_loss(x, *args, **kwargs)`

Computes the sliced score matching loss using random projections.

Parameters:

| Name       | Type     | Description                                           | Default    |
| ---------- | -------- | ----------------------------------------------------- | ---------- |
| `x`        | `Tensor` | Input data tensor of shape (batch_size, \*data_dims). | *required* |
| `*args`    |          | Additional arguments.                                 | `()`       |
| `**kwargs` |          | Additional keyword arguments.                         | `{}`       |

Returns:

| Type     | Description                                          |
| -------- | ---------------------------------------------------- |
| `Tensor` | torch.Tensor: The scalar sliced score matching loss. |

Source code in `torchebm/losses/score_matching.py`

```python
def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    r"""
    Computes the sliced score matching loss using random projections.

    Args:
        x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The scalar sliced score matching loss.
    """

    dup_x = (
        x.unsqueeze(0)
        .expand(self.n_projections, *x.shape)
        .contiguous()
        .view(-1, *x.shape[1:])
    ).requires_grad_(
        True
    )  # final shape: (n_particles * batch_size, d). tracing the shape: (batch_size, d) -> (1, batch_size, d)
    # -> (n_particles, batch_size, d) -> (n_particles, batch_size, d) -> (n_particles * batch_size, d)

    n_vectors = self._get_random_projections(dup_x)

    logp = (-self.model(dup_x)).sum()
    grad1 = torch.autograd.grad(logp, dup_x, create_graph=True)[0]
    v_score = torch.sum(grad1 * n_vectors, dim=-1)
    term1 = 0.5 * (v_score**2)

    grad_v = torch.autograd.grad(v_score.sum(), dup_x, create_graph=True)[0]
    term2 = torch.sum(n_vectors * grad_v, dim=-1)

    term1 = term1.view(self.n_projections, -1).mean(dim=0)
    term2 = term2.view(self.n_projections, -1).mean(dim=0)

    loss = term2 + term1

    return loss.mean()
```

### `forward(x, *args, **kwargs)`

Computes the sliced score matching loss for a batch of data.

Parameters:

| Name       | Type     | Description                                           | Default    |
| ---------- | -------- | ----------------------------------------------------- | ---------- |
| `x`        | `Tensor` | Input data tensor of shape (batch_size, \*data_dims). | *required* |
| `*args`    |          | Additional positional arguments.                      | `()`       |
| `**kwargs` |          | Additional keyword arguments.                         | `{}`       |

Returns:

| Type     | Description                                          |
| -------- | ---------------------------------------------------- |
| `Tensor` | torch.Tensor: The scalar sliced score matching loss. |

Source code in `torchebm/losses/score_matching.py`

```python
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    r"""
    Computes the sliced score matching loss for a batch of data.

    Args:
        x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The scalar sliced score matching loss.
    """
    if (x.device != self.device) or (x.dtype != self.dtype):
        x = x.to(device=self.device, dtype=self.dtype)

    with self.autocast_context():
        loss = self.compute_loss(x, *args, **kwargs)

    if self.regularization_strength > 0 or self.custom_regularization is not None:
        loss = self.add_regularization(loss, x)

    return loss
```

## `compute_eqm_ct(t, threshold=0.8, multiplier=4.0)`

Energy-compatible target scaling c(t) used in EqM.

The scaling function (truncated decay with gradient multiplier) is:

[ c(t) = \\lambda \\cdot \\min\\left(1, \\frac{1 - t}{1 - a}\\right) ]

where (a) is the threshold and (\\lambda) is the multiplier.

Parameters:

| Name         | Type     | Description                                                    | Default    |
| ------------ | -------- | -------------------------------------------------------------- | ---------- |
| `t`          | `Tensor` | Time tensor of shape (batch_size,).                            | *required* |
| `threshold`  | `float`  | Decay threshold (a), decay starts after (t > a). Default: 0.8. | `0.8`      |
| `multiplier` | `float`  | Gradient multiplier (\\lambda). Default: 4.0.                  | `4.0`      |

Returns:

| Type     | Description                             |
| -------- | --------------------------------------- |
| `Tensor` | Scaling factor c(t) of same shape as t. |

Source code in `torchebm/losses/loss_utils.py`

```python
def compute_eqm_ct(
    t: torch.Tensor,
    threshold: float = 0.8,
    multiplier: float = 4.0,
) -> torch.Tensor:
    r"""Energy-compatible target scaling c(t) used in EqM.

    The scaling function (truncated decay with gradient multiplier) is:

    \[
    c(t) = \lambda \cdot \min\left(1, \frac{1 - t}{1 - a}\right)
    \]

    where \(a\) is the threshold and \(\lambda\) is the multiplier.

    Args:
        t: Time tensor of shape (batch_size,).
        threshold: Decay threshold \(a\), decay starts after \(t > a\). Default: 0.8.
        multiplier: Gradient multiplier \(\lambda\). Default: 4.0.

    Returns:
        Scaling factor c(t) of same shape as t.
    """
    start = 1.0
    ct = (
        torch.minimum(
            start - (start - 1) / threshold * t,
            1 / (1 - threshold) - 1 / (1 - threshold) * t,
        )
        * multiplier
    )
    return ct
```

## `dispersive_loss(z)`

Dispersive loss (InfoNCE-L2 variant) for regularization.

Encourages diversity in generated samples by penalizing samples that are too close to each other in feature space.

Parameters:

| Name | Type     | Description                                | Default    |
| ---- | -------- | ------------------------------------------ | ---------- |
| `z`  | `Tensor` | Feature tensor of shape (batch_size, ...). | *required* |

Returns:

| Type     | Description             |
| -------- | ----------------------- |
| `Tensor` | Scalar dispersive loss. |

Source code in `torchebm/losses/loss_utils.py`

```python
def dispersive_loss(z: torch.Tensor) -> torch.Tensor:
    r"""Dispersive loss (InfoNCE-L2 variant) for regularization.

    Encourages diversity in generated samples by penalizing samples
    that are too close to each other in feature space.

    Args:
        z: Feature tensor of shape (batch_size, ...).

    Returns:
        Scalar dispersive loss.
    """
    z = z.reshape((z.shape[0], -1))
    diff = torch.nn.functional.pdist(z).pow(2) / z.shape[1]
    diff = torch.concat((diff, diff, torch.zeros(z.shape[0], device=z.device)))
    return torch.log(torch.exp(-diff).mean())
```

## `get_interpolant(interpolant_type)`

Get interpolant instance by name.

Parameters:

| Name               | Type  | Description                         | Default    |
| ------------------ | ----- | ----------------------------------- | ---------- |
| `interpolant_type` | `str` | One of 'linear', 'cosine', or 'vp'. | *required* |

Returns:

| Type              | Description           |
| ----------------- | --------------------- |
| `BaseInterpolant` | Interpolant instance. |

Raises:

| Type         | Description                            |
| ------------ | -------------------------------------- |
| `ValueError` | If interpolant_type is not recognized. |

Source code in `torchebm/losses/loss_utils.py`

```python
def get_interpolant(interpolant_type: str) -> BaseInterpolant:
    r"""Get interpolant instance by name.

    Args:
        interpolant_type: One of 'linear', 'cosine', or 'vp'.

    Returns:
        Interpolant instance.

    Raises:
        ValueError: If interpolant_type is not recognized.
    """
    interpolants = {
        "linear": LinearInterpolant,
        "cosine": CosineInterpolant,
        "vp": VariancePreservingInterpolant,
    }
    if interpolant_type not in interpolants:
        raise ValueError(
            f"Unknown interpolant: {interpolant_type}. "
            f"Choose from {list(interpolants.keys())}"
        )
    return interpolants[interpolant_type]()
```

## `mean_flat(tensor)`

Take mean over all non-batch dimensions.

Parameters:

| Name     | Type     | Description                              | Default    |
| -------- | -------- | ---------------------------------------- | ---------- |
| `tensor` | `Tensor` | Input tensor of shape (batch_size, ...). | *required* |

Returns:

| Type     | Description                                                |
| -------- | ---------------------------------------------------------- |
| `Tensor` | Tensor of shape (batch_size,) with mean over spatial dims. |

Source code in `torchebm/losses/loss_utils.py`

```python
def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    r"""Take mean over all non-batch dimensions.

    Args:
        tensor: Input tensor of shape (batch_size, ...).

    Returns:
        Tensor of shape (batch_size,) with mean over spatial dims.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
```
