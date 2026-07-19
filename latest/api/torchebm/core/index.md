# `torchebm.core`

Core functionality for energy-based models, including energy functions, base sampler class, and training utilities.

## `AckleyModel`

Bases: `BaseModel`

Energy-based model for the Ackley function.

Parameters:

| Name | Type    | Description                             | Default  |
| ---- | ------- | --------------------------------------- | -------- |
| `a`  | `float` | The a parameter of the Ackley function. | `20.0`   |
| `b`  | `float` | The b parameter of the Ackley function. | `0.2`    |
| `c`  | `float` | The c parameter of the Ackley function. | `2 * pi` |

Source code in `torchebm/core/base_model.py`

```python
class AckleyModel(BaseModel):
    r"""
    Energy-based model for the Ackley function.

    Args:
        a (float): The `a` parameter of the Ackley function.
        b (float): The `b` parameter of the Ackley function.
        c (float): The `c` parameter of the Ackley function.
    """
    def __init__(
        self, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Ackley energy."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        n = x.shape[-1]
        sum1 = torch.sum(x**2, dim=-1)
        sum2 = torch.sum(torch.cos(self.c * x), dim=-1)
        term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum1 / n))
        term2 = -torch.exp(sum2 / n)
        return term1 + term2 + self.a + math.e
```

### `forward(x)`

Computes the Ackley energy.

Source code in `torchebm/core/base_model.py`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    r"""Computes the Ackley energy."""
    if x.ndim == 1:
        x = x.unsqueeze(0)

    n = x.shape[-1]
    sum1 = torch.sum(x**2, dim=-1)
    sum2 = torch.sum(torch.cos(self.c * x), dim=-1)
    term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum1 / n))
    term2 = -torch.exp(sum2 / n)
    return term1 + term2 + self.a + math.e
```

## `BaseContrastiveDivergence`

Bases: `BaseLoss`

Abstract base class for Contrastive Divergence (CD) based loss functions.

Parameters:

| Name               | Type                           | Description                                                           | Default    |
| ------------------ | ------------------------------ | --------------------------------------------------------------------- | ---------- |
| `model`            | `BaseModel`                    | The energy-based model to be trained.                                 | *required* |
| `sampler`          | `BaseSampler`                  | The MCMC sampler for generating negative samples.                     | *required* |
| `k_steps`          | `int`                          | The number of MCMC steps to perform for each update.                  | `1`        |
| `persistent`       | `bool`                         | If True, uses a replay buffer for Persistent CD (PCD).                | `False`    |
| `buffer_size`      | `int`                          | The size of the replay buffer for PCD.                                | `100`      |
| `new_sample_ratio` | `float`                        | The ratio of new random samples to introduce into the MCMC chain.     | `0.0`      |
| `init_steps`       | `int`                          | The number of MCMC steps to run when initializing new chain elements. | `0`        |
| `dtype`            | `dtype`                        | Data type for computations.                                           | `float32`  |
| `device`           | `Optional[Union[str, device]]` | Device for computations.                                              | `None`     |

Source code in `torchebm/core/base_loss.py`

```python
class BaseContrastiveDivergence(BaseLoss):
    """
    Abstract base class for Contrastive Divergence (CD) based loss functions.

    Args:
        model (BaseModel): The energy-based model to be trained.
        sampler (BaseSampler): The MCMC sampler for generating negative samples.
        k_steps (int): The number of MCMC steps to perform for each update.
        persistent (bool): If `True`, uses a replay buffer for Persistent CD (PCD).
        buffer_size (int): The size of the replay buffer for PCD.
        new_sample_ratio (float): The ratio of new random samples to introduce into the MCMC chain.
        init_steps (int): The number of MCMC steps to run when initializing new chain elements.
        dtype (torch.dtype): Data type for computations.
        device (Optional[Union[str, torch.device]]): Device for computations.
    """

    def __init__(
        self,
        model: BaseModel,
        sampler: BaseSampler,
        k_steps: int = 1,
        persistent: bool = False,
        buffer_size: int = 100,
        new_sample_ratio: float = 0.0,
        init_steps: int = 0,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )
        self.model = model
        self.sampler = sampler
        self.k_steps = k_steps
        self.persistent = persistent
        self.buffer_size = buffer_size
        self.new_sample_ratio = new_sample_ratio
        self.init_steps = init_steps

        self.register_buffer("replay_buffer", None)
        self.register_buffer(
            "buffer_ptr", torch.tensor(0, dtype=torch.long, device=self.device)
        )
        self._buffer_ptr_int: int = 0
        self.buffer_initialized = False

    def initialize_buffer(
        self,
        data_shape_no_batch: Tuple[int, ...],
        buffer_chunk_size: int = 1024,
        init_noise_scale: float = 0.01,
    ) -> torch.Tensor:
        """
        Initializes the replay buffer with random noise for PCD.

        Args:
            data_shape_no_batch (Tuple[int, ...]): The shape of the data excluding the batch dimension.
            buffer_chunk_size (int): The size of chunks to process during initialization.
            init_noise_scale (float): The scale of the initial noise.

        Returns:
            torch.Tensor: The initialized replay buffer.
        """
        if not self.persistent or self.buffer_initialized:
            return

        if self.buffer_size <= 0:
            raise ValueError(
                f"Replay buffer size must be positive, got {self.buffer_size}"
            )

        buffer_shape = (
            self.buffer_size,
        ) + data_shape_no_batch  # shape: [buffer_size, *data_shape]
        logger.info("Initializing replay buffer with shape %s...", buffer_shape)

        self.replay_buffer = (
            torch.randn(buffer_shape, dtype=self.dtype, device=self.device)
            * init_noise_scale
        )

        if self.init_steps > 0:
            logger.info("Running %d MCMC steps to populate buffer...", self.init_steps)
            with torch.no_grad():
                chunk_size = min(self.buffer_size, buffer_chunk_size)
                for i in range(0, self.buffer_size, chunk_size):
                    end = min(i + chunk_size, self.buffer_size)
                    current_chunk = self.replay_buffer[i:end].clone()
                    try:
                        with self.autocast_context():
                            updated_chunk = self.sampler.sample(
                                x=current_chunk, n_steps=self.init_steps
                            ).detach()

                        if updated_chunk.shape == current_chunk.shape:
                            self.replay_buffer[i:end] = updated_chunk
                        else:
                            warnings.warn(
                                f"Sampler output shape mismatch during buffer init. Expected {current_chunk.shape}, got {updated_chunk.shape}. Skipping update for chunk {i}-{end}."
                            )
                    except Exception as e:
                        warnings.warn(
                            f"Error during buffer initialization sampling for chunk {i}-{end}: {e}. Keeping noise for this chunk."
                        )

        self.buffer_ptr.zero_()
        self._buffer_ptr_int = 0
        self.buffer_initialized = True
        logger.info("Replay buffer initialized.")

        return self.replay_buffer

    def get_start_points(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the starting points for the MCMC sampler.

        For standard CD, this is the input data. For PCD, it's samples from the replay buffer.

        Args:
            x (torch.Tensor): The input data batch.

        Returns:
            torch.Tensor: The tensor of starting points for the sampler.
        """
        x = x.to(device=self.device, dtype=self.dtype)

        batch_size = x.shape[0]
        data_shape_no_batch = x.shape[1:]

        if self.persistent:
            if not self.buffer_initialized:
                self.initialize_buffer(data_shape_no_batch)
                if not self.buffer_initialized:
                    raise RuntimeError("Buffer initialization failed.")

            if self.buffer_size < batch_size:
                warnings.warn(
                    f"Buffer size ({self.buffer_size}) is smaller than batch size ({batch_size}). Sampling with replacement.",
                    UserWarning,
                )
                indices = torch.randint(
                    0, self.buffer_size, (batch_size,), device=self.device
                )
            else:
                # stratified sampling for better buffer coverage
                stride = self.buffer_size // batch_size
                base_indices = torch.arange(0, batch_size, device=self.device) * stride
                offset = torch.randint(0, stride, (batch_size,), device=self.device)
                indices = (base_indices + offset) % self.buffer_size

            start_points = self.replay_buffer[indices]

            # add some noise for exploration
            if self.new_sample_ratio > 0.0:
                n_new = max(1, int(batch_size * self.new_sample_ratio))
                noise_indices = torch.randperm(batch_size, device=self.device)[:n_new]
                noise_scale = 0.01
                start_points[noise_indices] = (
                    start_points[noise_indices]
                    + torch.randn_like(
                        start_points[noise_indices],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * noise_scale
                )
        else:
            # standard CD-k uses data as starting points
            start_points = x.detach().clone()

        return start_points

    def get_negative_samples(self, x, batch_size, data_shape) -> torch.Tensor:
        """
        Gets negative samples using the replay buffer strategy.

        Args:
            x: (Unused) The input data tensor.
            batch_size (int): The number of samples to generate.
            data_shape (Tuple[int, ...]): The shape of the data samples (excluding batch size).

        Returns:
            torch.Tensor: Negative samples.
        """
        if not self.persistent or not self.buffer_initialized:
            # For non-persistent CD, just return random noise
            return torch.randn(
                (batch_size,) + data_shape, dtype=self.dtype, device=self.device
            )

        n_new = max(1, int(batch_size * self.new_sample_ratio))
        n_old = batch_size - n_new

        all_samples = torch.empty(
            (batch_size,) + data_shape, dtype=self.dtype, device=self.device
        )

        # new random samples
        if n_new > 0:
            all_samples[:n_new] = torch.randn(
                (n_new,) + data_shape, dtype=self.dtype, device=self.device
            )

        # samples from buffer
        if n_old > 0:

            indices = torch.randint(0, self.buffer_size, (n_old,), device=self.device)
            all_samples[n_new:] = self.replay_buffer[indices]

        return all_samples

    def update_buffer(self, samples: torch.Tensor) -> None:
        """
        Updates the replay buffer with new samples using a FIFO strategy.

        Args:
            samples (torch.Tensor): New samples to add to the buffer.
        """
        if not self.persistent or not self.buffer_initialized:
            return

        # Ensure samples are on the correct device and dtype
        samples = samples.to(device=self.device, dtype=self.dtype).detach()

        batch_size = samples.shape[0]

        # FIFO strategy — use cached Python int to avoid GPU sync every step
        ptr = self._buffer_ptr_int

        if batch_size >= self.buffer_size:
            # batch larger than buffer, use latest samples
            self.replay_buffer[:] = samples[-self.buffer_size :]
            self._buffer_ptr_int = 0
            self.buffer_ptr.zero_()
        else:
            # handle buffer wraparound
            end_ptr = (ptr + batch_size) % self.buffer_size

            if end_ptr > ptr:
                self.replay_buffer[ptr:end_ptr] = samples
            else:
                # wraparound case - split update
                first_part = self.buffer_size - ptr
                self.replay_buffer[ptr:] = samples[:first_part]
                self.replay_buffer[:end_ptr] = samples[first_part:]

            self._buffer_ptr_int = end_ptr
            self.buffer_ptr.fill_(end_ptr)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # sync cached int with loaded tensor buffer_ptr
        self._buffer_ptr_int = int(self.buffer_ptr.item())

    @abstractmethod
    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the CD loss given real data samples.

        Args:
            x (torch.Tensor): Real data samples (positive samples).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The contrastive divergence loss.
                - The generated negative samples.
        """
        pass

    @abstractmethod
    def compute_loss(
        self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Computes the contrastive divergence loss from positive and negative samples.

        Args:
            x (torch.Tensor): Real data samples (positive samples).
            pred_x (torch.Tensor): Generated negative samples.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The contrastive divergence loss.
        """
        pass

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}(model={self.model}, sampler={self.sampler})"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()
```

### `__repr__()`

Return a string representation of the loss function.

Source code in `torchebm/core/base_loss.py`

```python
def __repr__(self):
    """Return a string representation of the loss function."""
    return f"{self.__class__.__name__}(model={self.model}, sampler={self.sampler})"
```

### `__str__()`

Return a string representation of the loss function.

Source code in `torchebm/core/base_loss.py`

```python
def __str__(self):
    """Return a string representation of the loss function."""
    return self.__repr__()
```

### `compute_loss(x, pred_x, *args, **kwargs)`

Computes the contrastive divergence loss from positive and negative samples.

Parameters:

| Name       | Type     | Description                           | Default    |
| ---------- | -------- | ------------------------------------- | ---------- |
| `x`        | `Tensor` | Real data samples (positive samples). | *required* |
| `pred_x`   | `Tensor` | Generated negative samples.           | *required* |
| `*args`    |          | Additional positional arguments.      | `()`       |
| `**kwargs` |          | Additional keyword arguments.         | `{}`       |

Returns:

| Type     | Description                                    |
| -------- | ---------------------------------------------- |
| `Tensor` | torch.Tensor: The contrastive divergence loss. |

Source code in `torchebm/core/base_loss.py`

```python
@abstractmethod
def compute_loss(
    self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """
    Computes the contrastive divergence loss from positive and negative samples.

    Args:
        x (torch.Tensor): Real data samples (positive samples).
        pred_x (torch.Tensor): Generated negative samples.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The contrastive divergence loss.
    """
    pass
```

### `forward(x, *args, **kwargs)`

Computes the CD loss given real data samples.

Parameters:

| Name | Type     | Description                           | Default    |
| ---- | -------- | ------------------------------------- | ---------- |
| `x`  | `Tensor` | Real data samples (positive samples). | *required* |

Returns:

| Type                    | Description                                                                                               |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `Tuple[Tensor, Tensor]` | Tuple\[torch.Tensor, torch.Tensor\]: - The contrastive divergence loss. - The generated negative samples. |

Source code in `torchebm/core/base_loss.py`

```python
@abstractmethod
def forward(
    self, x: torch.Tensor, *args, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the CD loss given real data samples.

    Args:
        x (torch.Tensor): Real data samples (positive samples).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - The contrastive divergence loss.
            - The generated negative samples.
    """
    pass
```

### `get_negative_samples(x, batch_size, data_shape)`

Gets negative samples using the replay buffer strategy.

Parameters:

| Name         | Type              | Description                                           | Default    |
| ------------ | ----------------- | ----------------------------------------------------- | ---------- |
| `x`          |                   | (Unused) The input data tensor.                       | *required* |
| `batch_size` | `int`             | The number of samples to generate.                    | *required* |
| `data_shape` | `Tuple[int, ...]` | The shape of the data samples (excluding batch size). | *required* |

Returns:

| Type     | Description                     |
| -------- | ------------------------------- |
| `Tensor` | torch.Tensor: Negative samples. |

Source code in `torchebm/core/base_loss.py`

```python
def get_negative_samples(self, x, batch_size, data_shape) -> torch.Tensor:
    """
    Gets negative samples using the replay buffer strategy.

    Args:
        x: (Unused) The input data tensor.
        batch_size (int): The number of samples to generate.
        data_shape (Tuple[int, ...]): The shape of the data samples (excluding batch size).

    Returns:
        torch.Tensor: Negative samples.
    """
    if not self.persistent or not self.buffer_initialized:
        # For non-persistent CD, just return random noise
        return torch.randn(
            (batch_size,) + data_shape, dtype=self.dtype, device=self.device
        )

    n_new = max(1, int(batch_size * self.new_sample_ratio))
    n_old = batch_size - n_new

    all_samples = torch.empty(
        (batch_size,) + data_shape, dtype=self.dtype, device=self.device
    )

    # new random samples
    if n_new > 0:
        all_samples[:n_new] = torch.randn(
            (n_new,) + data_shape, dtype=self.dtype, device=self.device
        )

    # samples from buffer
    if n_old > 0:

        indices = torch.randint(0, self.buffer_size, (n_old,), device=self.device)
        all_samples[n_new:] = self.replay_buffer[indices]

    return all_samples
```

### `get_start_points(x)`

Gets the starting points for the MCMC sampler.

For standard CD, this is the input data. For PCD, it's samples from the replay buffer.

Parameters:

| Name | Type     | Description           | Default    |
| ---- | -------- | --------------------- | ---------- |
| `x`  | `Tensor` | The input data batch. | *required* |

Returns:

| Type     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| `Tensor` | torch.Tensor: The tensor of starting points for the sampler. |

Source code in `torchebm/core/base_loss.py`

```python
def get_start_points(self, x: torch.Tensor) -> torch.Tensor:
    """
    Gets the starting points for the MCMC sampler.

    For standard CD, this is the input data. For PCD, it's samples from the replay buffer.

    Args:
        x (torch.Tensor): The input data batch.

    Returns:
        torch.Tensor: The tensor of starting points for the sampler.
    """
    x = x.to(device=self.device, dtype=self.dtype)

    batch_size = x.shape[0]
    data_shape_no_batch = x.shape[1:]

    if self.persistent:
        if not self.buffer_initialized:
            self.initialize_buffer(data_shape_no_batch)
            if not self.buffer_initialized:
                raise RuntimeError("Buffer initialization failed.")

        if self.buffer_size < batch_size:
            warnings.warn(
                f"Buffer size ({self.buffer_size}) is smaller than batch size ({batch_size}). Sampling with replacement.",
                UserWarning,
            )
            indices = torch.randint(
                0, self.buffer_size, (batch_size,), device=self.device
            )
        else:
            # stratified sampling for better buffer coverage
            stride = self.buffer_size // batch_size
            base_indices = torch.arange(0, batch_size, device=self.device) * stride
            offset = torch.randint(0, stride, (batch_size,), device=self.device)
            indices = (base_indices + offset) % self.buffer_size

        start_points = self.replay_buffer[indices]

        # add some noise for exploration
        if self.new_sample_ratio > 0.0:
            n_new = max(1, int(batch_size * self.new_sample_ratio))
            noise_indices = torch.randperm(batch_size, device=self.device)[:n_new]
            noise_scale = 0.01
            start_points[noise_indices] = (
                start_points[noise_indices]
                + torch.randn_like(
                    start_points[noise_indices],
                    device=self.device,
                    dtype=self.dtype,
                )
                * noise_scale
            )
    else:
        # standard CD-k uses data as starting points
        start_points = x.detach().clone()

    return start_points
```

### `initialize_buffer(data_shape_no_batch, buffer_chunk_size=1024, init_noise_scale=0.01)`

Initializes the replay buffer with random noise for PCD.

Parameters:

| Name                  | Type              | Description                                          | Default    |
| --------------------- | ----------------- | ---------------------------------------------------- | ---------- |
| `data_shape_no_batch` | `Tuple[int, ...]` | The shape of the data excluding the batch dimension. | *required* |
| `buffer_chunk_size`   | `int`             | The size of chunks to process during initialization. | `1024`     |
| `init_noise_scale`    | `float`           | The scale of the initial noise.                      | `0.01`     |

Returns:

| Type     | Description                                  |
| -------- | -------------------------------------------- |
| `Tensor` | torch.Tensor: The initialized replay buffer. |

Source code in `torchebm/core/base_loss.py`

```python
def initialize_buffer(
    self,
    data_shape_no_batch: Tuple[int, ...],
    buffer_chunk_size: int = 1024,
    init_noise_scale: float = 0.01,
) -> torch.Tensor:
    """
    Initializes the replay buffer with random noise for PCD.

    Args:
        data_shape_no_batch (Tuple[int, ...]): The shape of the data excluding the batch dimension.
        buffer_chunk_size (int): The size of chunks to process during initialization.
        init_noise_scale (float): The scale of the initial noise.

    Returns:
        torch.Tensor: The initialized replay buffer.
    """
    if not self.persistent or self.buffer_initialized:
        return

    if self.buffer_size <= 0:
        raise ValueError(
            f"Replay buffer size must be positive, got {self.buffer_size}"
        )

    buffer_shape = (
        self.buffer_size,
    ) + data_shape_no_batch  # shape: [buffer_size, *data_shape]
    logger.info("Initializing replay buffer with shape %s...", buffer_shape)

    self.replay_buffer = (
        torch.randn(buffer_shape, dtype=self.dtype, device=self.device)
        * init_noise_scale
    )

    if self.init_steps > 0:
        logger.info("Running %d MCMC steps to populate buffer...", self.init_steps)
        with torch.no_grad():
            chunk_size = min(self.buffer_size, buffer_chunk_size)
            for i in range(0, self.buffer_size, chunk_size):
                end = min(i + chunk_size, self.buffer_size)
                current_chunk = self.replay_buffer[i:end].clone()
                try:
                    with self.autocast_context():
                        updated_chunk = self.sampler.sample(
                            x=current_chunk, n_steps=self.init_steps
                        ).detach()

                    if updated_chunk.shape == current_chunk.shape:
                        self.replay_buffer[i:end] = updated_chunk
                    else:
                        warnings.warn(
                            f"Sampler output shape mismatch during buffer init. Expected {current_chunk.shape}, got {updated_chunk.shape}. Skipping update for chunk {i}-{end}."
                        )
                except Exception as e:
                    warnings.warn(
                        f"Error during buffer initialization sampling for chunk {i}-{end}: {e}. Keeping noise for this chunk."
                    )

    self.buffer_ptr.zero_()
    self._buffer_ptr_int = 0
    self.buffer_initialized = True
    logger.info("Replay buffer initialized.")

    return self.replay_buffer
```

### `update_buffer(samples)`

Updates the replay buffer with new samples using a FIFO strategy.

Parameters:

| Name      | Type     | Description                       | Default    |
| --------- | -------- | --------------------------------- | ---------- |
| `samples` | `Tensor` | New samples to add to the buffer. | *required* |

Source code in `torchebm/core/base_loss.py`

```python
def update_buffer(self, samples: torch.Tensor) -> None:
    """
    Updates the replay buffer with new samples using a FIFO strategy.

    Args:
        samples (torch.Tensor): New samples to add to the buffer.
    """
    if not self.persistent or not self.buffer_initialized:
        return

    # Ensure samples are on the correct device and dtype
    samples = samples.to(device=self.device, dtype=self.dtype).detach()

    batch_size = samples.shape[0]

    # FIFO strategy — use cached Python int to avoid GPU sync every step
    ptr = self._buffer_ptr_int

    if batch_size >= self.buffer_size:
        # batch larger than buffer, use latest samples
        self.replay_buffer[:] = samples[-self.buffer_size :]
        self._buffer_ptr_int = 0
        self.buffer_ptr.zero_()
    else:
        # handle buffer wraparound
        end_ptr = (ptr + batch_size) % self.buffer_size

        if end_ptr > ptr:
            self.replay_buffer[ptr:end_ptr] = samples
        else:
            # wraparound case - split update
            first_part = self.buffer_size - ptr
            self.replay_buffer[ptr:] = samples[:first_part]
            self.replay_buffer[:end_ptr] = samples[first_part:]

        self._buffer_ptr_int = end_ptr
        self.buffer_ptr.fill_(end_ptr)
```

## `BaseCostCoupling`

Bases: `BaseCoupling`

Family base for couplings that pair minibatches by minimizing a cost.

This implements the shared pipeline once as a template method and asks concretes for a single piece, the assignment solver, playing the role Butcher tableaus play for `BaseRungeKuttaIntegrator`. `couple`:

1. runs under `torch.no_grad()`,
1. validates that a target batch exists and batch sizes match,
1. passes a single-sample batch through unchanged,
1. builds a pairwise cost matrix via `compute_cost` (overridable), and
1. delegates pairing to the abstract `_solve`.

`x0` order and marginal are always preserved; only `x1` is reindexed. Subclasses supply `_solve`; they may also override `compute_cost` for a non-Euclidean or conditioning-aware ground cost. Weighted cost variants (unbalanced OT) override `couple` itself, reusing `compute_cost`, to attach per-pair `weights` to the result.

Source code in `torchebm/core/base_coupling.py`

```python
class BaseCostCoupling(BaseCoupling):
    r"""
    Family base for couplings that pair minibatches by minimizing a cost.

    This implements the shared pipeline once as a template method and asks
    concretes for a single piece, the assignment solver, playing the role
    Butcher tableaus play for `BaseRungeKuttaIntegrator`. `couple`:

    1. runs under `torch.no_grad()`,
    2. validates that a target batch exists and batch sizes match,
    3. passes a single-sample batch through unchanged,
    4. builds a pairwise cost matrix via `compute_cost` (overridable), and
    5. delegates pairing to the abstract `_solve`.

    `x0` order and marginal are always preserved; only `x1` is reindexed.
    Subclasses supply `_solve`; they may also override `compute_cost` for a
    non-Euclidean or conditioning-aware ground cost. Weighted cost variants
    (unbalanced OT) override `couple` itself, reusing `compute_cost`, to
    attach per-pair `weights` to the result.
    """

    @torch.no_grad()
    def couple(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> CouplingResult:
        x1 = self._require_x1(x1)
        self._check_batch(x0, x1)
        if x0.shape[0] == 1:
            return CouplingResult(x0, x1)
        cost = self.compute_cost(x0, x1, **kwargs)
        idx = self._solve(cost)
        return CouplingResult(x0, x1[idx])

    def compute_cost(
        self, x0: torch.Tensor, x1: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        r"""
        Pairwise ground-cost matrix between source and target samples.

        The default is the max-normalized squared Euclidean cost on flattened
        samples:

        \[
        C_{ij} = \frac{\|x_0^{(i)} - x_1^{(j)}\|^2}{\max_{kl} C_{kl}}
        \]

        Args:
            x0: Source samples of shape (batch_size, ...).
            x1: Target samples of shape (batch_size, ...).
            **kwargs: Optional conditioning (unused by the default cost).

        Returns:
            Cost matrix of shape (batch_size, batch_size).
        """
        batch = x0.shape[0]
        cost = torch.cdist(x0.reshape(batch, -1), x1.reshape(batch, -1)).square()
        return cost / cost.max().clamp(min=1e-12)

    @abstractmethod
    def _solve(self, cost: torch.Tensor) -> torch.Tensor:
        r"""
        Solve the pairing problem on a square cost matrix.

        Args:
            cost: Cost matrix of shape (n, n).

        Returns:
            Long tensor `idx` of shape (n,) pairing `x0[i]` with `x1[idx[i]]`.
            Assignment solvers return a permutation; entropic solvers return
            row-conditional draws (the `x0` marginal is preserved exactly).
        """
        raise NotImplementedError
```

### `compute_cost(x0, x1, **kwargs)`

Pairwise ground-cost matrix between source and target samples.

The default is the max-normalized squared Euclidean cost on flattened samples:

[ C\_{ij} = \\frac{|x_0^{(i)} - x_1^{(j)}|^2}{\\max\_{kl} C\_{kl}} ]

Parameters:

| Name       | Type     | Description                                         | Default    |
| ---------- | -------- | --------------------------------------------------- | ---------- |
| `x0`       | `Tensor` | Source samples of shape (batch_size, ...).          | *required* |
| `x1`       | `Tensor` | Target samples of shape (batch_size, ...).          | *required* |
| `**kwargs` | `Any`    | Optional conditioning (unused by the default cost). | `{}`       |

Returns:

| Type     | Description                                    |
| -------- | ---------------------------------------------- |
| `Tensor` | Cost matrix of shape (batch_size, batch_size). |

Source code in `torchebm/core/base_coupling.py`

```python
def compute_cost(
    self, x0: torch.Tensor, x1: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    r"""
    Pairwise ground-cost matrix between source and target samples.

    The default is the max-normalized squared Euclidean cost on flattened
    samples:

    \[
    C_{ij} = \frac{\|x_0^{(i)} - x_1^{(j)}\|^2}{\max_{kl} C_{kl}}
    \]

    Args:
        x0: Source samples of shape (batch_size, ...).
        x1: Target samples of shape (batch_size, ...).
        **kwargs: Optional conditioning (unused by the default cost).

    Returns:
        Cost matrix of shape (batch_size, batch_size).
    """
    batch = x0.shape[0]
    cost = torch.cdist(x0.reshape(batch, -1), x1.reshape(batch, -1)).square()
    return cost / cost.max().clamp(min=1e-12)
```

## `BaseCoupling`

Bases: `ABC`

Abstract base class for couplings.

A coupling is a rule that pairs a batch of source samples (x_0) (typically noise) with target samples (x_1) (data) before interpolation. Depending on the family it may **reorder** or **resample** an incoming target batch (independent, minibatch OT), **transform** it (equivariant alignment, closed-form maps), or **generate** it from the source (model-induced couplings such as reflow and DSBM, where (x_1 = \\Phi(x_0))).

Couplings are computed under `torch.no_grad()` and never propagate gradients. Equal batch sizes are not enforced at this level; families that require them call the `_check_batch` helper (all cost-based couplings do), and families that require a target batch call `_require_x1`.

Two extension channels keep this contract closed against future families:

- **Conditioning in:** `couple`/`__call__` accept `**kwargs` (class labels, prompts, geometry) that conditional and structure-aware couplings consume; unconditional couplings ignore them.
- **Extras out:** results are `CouplingResult` objects that unpack as the `(x0, x1)` pair while carrying optional extras as attributes (per-pair `weights` for unbalanced/reweighted OT today; more may be appended without breaking any consumer).

Subclasses must implement `couple`. Cost-based couplings should instead subclass `BaseCostCoupling`, which supplies the template and asks only for a solver; model-induced couplings subclass `BaseModelCoupling`.

Source code in `torchebm/core/base_coupling.py`

```python
class BaseCoupling(ABC):
    r"""
    Abstract base class for couplings.

    A coupling is a rule that pairs a batch of source samples \(x_0\)
    (typically noise) with target samples \(x_1\) (data) before
    interpolation. Depending on the family it may **reorder** or **resample**
    an incoming target batch (independent, minibatch OT), **transform** it
    (equivariant alignment, closed-form maps), or **generate** it from the
    source (model-induced couplings such as reflow and DSBM, where
    \(x_1 = \Phi(x_0)\)).

    Couplings are computed under `torch.no_grad()` and never propagate
    gradients. Equal batch sizes are not enforced at this level; families
    that require them call the `_check_batch` helper (all cost-based
    couplings do), and families that require a target batch call
    `_require_x1`.

    Two extension channels keep this contract closed against future families:

    - **Conditioning in:** `couple`/`__call__` accept `**kwargs` (class
      labels, prompts, geometry) that conditional and structure-aware
      couplings consume; unconditional couplings ignore them.
    - **Extras out:** results are `CouplingResult` objects that unpack as the
      `(x0, x1)` pair while carrying optional extras as attributes (per-pair
      `weights` for unbalanced/reweighted OT today; more may be appended
      without breaking any consumer).

    Subclasses must implement `couple`. Cost-based couplings should instead
    subclass `BaseCostCoupling`, which supplies the template and asks only
    for a solver; model-induced couplings subclass `BaseModelCoupling`.
    """

    @abstractmethod
    def couple(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> CouplingResult:
        r"""
        Pair source and target samples.

        Args:
            x0: Source samples of shape (batch_size, ...).
            x1: Target samples of shape (batch_size, ...). Optional at this
                level: generate-family couplings produce the target from the
                source and ignore (or do not need) an incoming batch;
                pairing families require it via `_require_x1`.
            **kwargs: Optional conditioning forwarded by the caller; ignored
                by unconditional couplings.

        Returns:
            A `CouplingResult` that unpacks as the aligned pair
            ``(x0, x1)``; weighted couplings also set its ``weights``.
        """
        raise NotImplementedError

    def _check_batch(self, x0: torch.Tensor, x1: torch.Tensor) -> None:
        r"""Validate that both batches have the same leading dimension."""
        if x0.shape[0] != x1.shape[0]:
            raise ValueError(
                f"Coupling requires equal batch sizes, got {x0.shape[0]} and {x1.shape[0]}"
            )

    def _require_x1(self, x1: Optional[torch.Tensor]) -> torch.Tensor:
        r"""Validate that a target batch was provided (pairing families)."""
        if x1 is None:
            raise ValueError(
                f"{self.__class__.__name__} pairs against an existing target "
                f"batch; x1 must not be None"
            )
        return x1

    def __call__(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> CouplingResult:
        return self.couple(x0, x1, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
```

### `couple(x0, x1=None, **kwargs)`

Pair source and target samples.

Parameters:

| Name       | Type               | Description                                                                                                                                                                                                                  | Default    |
| ---------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `x0`       | `Tensor`           | Source samples of shape (batch_size, ...).                                                                                                                                                                                   | *required* |
| `x1`       | `Optional[Tensor]` | Target samples of shape (batch_size, ...). Optional at this level: generate-family couplings produce the target from the source and ignore (or do not need) an incoming batch; pairing families require it via \_require_x1. | `None`     |
| `**kwargs` | `Any`              | Optional conditioning forwarded by the caller; ignored by unconditional couplings.                                                                                                                                           | `{}`       |

Returns:

| Type             | Description                                        |
| ---------------- | -------------------------------------------------- |
| `CouplingResult` | A CouplingResult that unpacks as the aligned pair  |
| `CouplingResult` | (x0, x1); weighted couplings also set its weights. |

Source code in `torchebm/core/base_coupling.py`

```python
@abstractmethod
def couple(
    self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
) -> CouplingResult:
    r"""
    Pair source and target samples.

    Args:
        x0: Source samples of shape (batch_size, ...).
        x1: Target samples of shape (batch_size, ...). Optional at this
            level: generate-family couplings produce the target from the
            source and ignore (or do not need) an incoming batch;
            pairing families require it via `_require_x1`.
        **kwargs: Optional conditioning forwarded by the caller; ignored
            by unconditional couplings.

    Returns:
        A `CouplingResult` that unpacks as the aligned pair
        ``(x0, x1)``; weighted couplings also set its ``weights``.
    """
    raise NotImplementedError
```

## `BaseIntegrator`

Bases: `TorchEBMModule`, `ABC`

Abstract integrator that advances a sampler state according to dynamics.

The integrator operates on a generic state dict to remain reusable across samplers (e.g., Langevin uses only position `x`, HMC uses position `x` and momentum `p`).

Methods follow PyTorch conventions and inherit `device`/`dtype` from `TorchEBMModule`.

Source code in `torchebm/core/base_integrator.py`

```python
class BaseIntegrator(TorchEBMModule, ABC):
    """
    Abstract integrator that advances a sampler state according to dynamics.

    The integrator operates on a generic state dict to remain reusable across
    samplers (e.g., Langevin uses only position `x`, HMC uses position `x` and
    momentum `p`).

    Methods follow PyTorch conventions and inherit `device`/`dtype` from
    `TorchEBMModule`.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *args,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)

    @staticmethod
    def _resolve_drift(
        drift: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""Return the drift callable after validation.

        Raises:
            ValueError: If ``drift`` is ``None``.
        """
        if drift is not None:
            return drift
        raise ValueError(
            "drift must be provided explicitly. For EBM sampling, pass "
            "drift=lambda x, t: -model.gradient(x) from the caller."
        )

    @abstractmethod
    def step(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Advance the dynamical state by one integrator application.

        Args:
            state: Mapping containing required tensors (e.g., {'x': ..., 'p': ...}).
            step_size: Step size for the integration.
            *args: Additional positional arguments specific to the integrator.
            **kwargs: Additional keyword arguments specific to the integrator.

        Returns:
            Updated state dict with the same keys as the input `state`.
        """
        raise NotImplementedError

    @abstractmethod
    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor,
        n_steps: int,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Advance the dynamical state by `n_steps` integrator applications.

        Args:
            state: Mapping containing required tensors (e.g., {'x': ..., 'p': ...}).
            step_size: Step size for the integration.
            n_steps: The number of integration steps to perform.
            *args: Additional positional arguments specific to the integrator.
            **kwargs: Additional keyword arguments specific to the integrator.

        Returns:
            Updated state dict with the same keys as the input `state`.
        """
        raise NotImplementedError
```

### `integrate(state, step_size, n_steps, *args, **kwargs)`

Advance the dynamical state by `n_steps` integrator applications.

Parameters:

| Name        | Type                | Description                                                       | Default    |
| ----------- | ------------------- | ----------------------------------------------------------------- | ---------- |
| `state`     | `Dict[str, Tensor]` | Mapping containing required tensors (e.g., {'x': ..., 'p': ...}). | *required* |
| `step_size` | `Tensor`            | Step size for the integration.                                    | *required* |
| `n_steps`   | `int`               | The number of integration steps to perform.                       | *required* |
| `*args`     |                     | Additional positional arguments specific to the integrator.       | `()`       |
| `**kwargs`  |                     | Additional keyword arguments specific to the integrator.          | `{}`       |

Returns:

| Type                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `Dict[str, Tensor]` | Updated state dict with the same keys as the input state. |

Source code in `torchebm/core/base_integrator.py`

```python
@abstractmethod
def integrate(
    self,
    state: Dict[str, torch.Tensor],
    step_size: torch.Tensor,
    n_steps: int,
    *args,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Advance the dynamical state by `n_steps` integrator applications.

    Args:
        state: Mapping containing required tensors (e.g., {'x': ..., 'p': ...}).
        step_size: Step size for the integration.
        n_steps: The number of integration steps to perform.
        *args: Additional positional arguments specific to the integrator.
        **kwargs: Additional keyword arguments specific to the integrator.

    Returns:
        Updated state dict with the same keys as the input `state`.
    """
    raise NotImplementedError
```

### `step(state, step_size, *args, **kwargs)`

Advance the dynamical state by one integrator application.

Parameters:

| Name        | Type                | Description                                                       | Default    |
| ----------- | ------------------- | ----------------------------------------------------------------- | ---------- |
| `state`     | `Dict[str, Tensor]` | Mapping containing required tensors (e.g., {'x': ..., 'p': ...}). | *required* |
| `step_size` | `Tensor`            | Step size for the integration.                                    | *required* |
| `*args`     |                     | Additional positional arguments specific to the integrator.       | `()`       |
| `**kwargs`  |                     | Additional keyword arguments specific to the integrator.          | `{}`       |

Returns:

| Type                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `Dict[str, Tensor]` | Updated state dict with the same keys as the input state. |

Source code in `torchebm/core/base_integrator.py`

```python
@abstractmethod
def step(
    self,
    state: Dict[str, torch.Tensor],
    step_size: torch.Tensor,
    *args,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Advance the dynamical state by one integrator application.

    Args:
        state: Mapping containing required tensors (e.g., {'x': ..., 'p': ...}).
        step_size: Step size for the integration.
        *args: Additional positional arguments specific to the integrator.
        **kwargs: Additional keyword arguments specific to the integrator.

    Returns:
        Updated state dict with the same keys as the input `state`.
    """
    raise NotImplementedError
```

## `BaseInterpolant`

Bases: `ABC`

Abstract base class for stochastic interpolants.

An interpolant defines a conditional probability path between a source distribution (typically Gaussian noise) and a target distribution (data).

The interpolation is parameterized as:

[ x_t = \\alpha(t) x_1 + \\sigma(t) x_0 ]

where (x_0 \\sim \\mathcal{N}(0, I)) and (x_1 \\sim p\_{\\text{data}}).

Subclasses must implement `compute_alpha_t` and `compute_sigma_t`.

Source code in `torchebm/core/base_interpolant.py`

```python
class BaseInterpolant(ABC):
    r"""
    Abstract base class for stochastic interpolants.

    An interpolant defines a conditional probability path between a source
    distribution (typically Gaussian noise) and a target distribution (data).

    The interpolation is parameterized as:

    \[
    x_t = \alpha(t) x_1 + \sigma(t) x_0
    \]

    where \(x_0 \sim \mathcal{N}(0, I)\) and \(x_1 \sim p_{\text{data}}\).

    Subclasses must implement `compute_alpha_t` and `compute_sigma_t`.
    """

    @abstractmethod
    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the data coefficient \(\alpha(t)\) and its time derivative.

        Args:
            t: Time tensor of shape (batch_size, ...).

        Returns:
            Tuple of (\(\alpha(t)\), \(\dot{\alpha}(t)\)).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the noise coefficient \(\sigma(t)\) and its time derivative.

        Args:
            t: Time tensor of shape (batch_size, ...).

        Returns:
            Tuple of (\(\sigma(t)\), \(\dot{\sigma}(t)\)).
        """
        raise NotImplementedError

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the ratio \(\dot{\alpha}(t) / \alpha(t)\) for numerical stability.

        This method can be overridden for better numerical precision.

        Args:
            t: Time tensor.

        Returns:
            The ratio tensor.
        """
        alpha, d_alpha = self.compute_alpha_t(t)
        return d_alpha / torch.clamp(alpha, min=1e-8)

    def interpolate(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the interpolated sample \(x_t\) and conditional velocity \(u_t\).

        Args:
            x0: Noise samples of shape (batch_size, ...).
            x1: Data samples of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Tuple of (x_t, u_t) where:
                - x_t = α(t) x₁ + σ(t) x₀
                - u_t = α̇(t) x₁ + σ̇(t) x₀
        """
        t_expanded = expand_t_like_x(t, x0)
        alpha, d_alpha = self.compute_alpha_t(t_expanded)
        sigma, d_sigma = self.compute_sigma_t(t_expanded)

        xt = alpha * x1 + sigma * x0
        ut = d_alpha * x1 + d_sigma * x0

        return xt, ut

    def compute_drift(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute drift coefficients for score-based parameterization.

        For the probability flow ODE in score parameterization:
        dx = [-drift_mean + drift_var * score] dt

        Args:
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Tuple of (drift_mean, drift_var) for score parameterization.
        """
        t_expanded = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t_expanded)
        sigma, d_sigma = self.compute_sigma_t(t_expanded)

        drift_mean = alpha_ratio * x
        drift_var = alpha_ratio * (sigma**2) - sigma * d_sigma

        return -drift_mean, drift_var

    def compute_diffusion(
        self, x: torch.Tensor, t: torch.Tensor, form: str = "SBDM", norm: float = 1.0
    ) -> torch.Tensor:
        r"""
        Compute diffusion coefficient for SDE sampling.

        Args:
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).
            form: Diffusion form. Choices:
                - 'constant': Constant diffusion
                - 'SBDM': Score-based diffusion matching
                - 'sigma': Proportional to noise schedule
                - 'linear': Linear decay
                - 'decreasing': Faster decay towards t=1
                - 'increasing-decreasing': Peak at midpoint
            norm: Scaling factor for diffusion.

        Returns:
            Diffusion coefficient tensor.
        """
        t_expanded = expand_t_like_x(t, x)
        sigma, _ = self.compute_sigma_t(t_expanded)
        _, drift_var = self.compute_drift(x, t)

        if form == "constant":
            return norm * torch.ones_like(drift_var)
        elif form == "SBDM":
            return norm * drift_var / (sigma + 1e-8)
        elif form == "sigma":
            return norm * sigma
        elif form == "linear":
            return norm * (1 - t_expanded)
        elif form == "decreasing":
            # Faster decay: (1-t)^2
            return norm * (1 - t_expanded) ** 2
        elif form == "increasing-decreasing":
            # Peak at t=0.5: 4*t*(1-t)
            return norm * 4 * t_expanded * (1 - t_expanded)
        else:
            raise ValueError(
                f"Unknown diffusion form '{form}'. "
                f"Choose from: constant, SBDM, sigma, linear, decreasing, increasing-decreasing"
            )

    def velocity_to_score(
        self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Convert velocity prediction to score.

        Args:
            velocity: Predicted velocity of shape (batch_size, ...).
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Score tensor of shape (batch_size, ...).
        """
        t_expanded = expand_t_like_x(t, x)
        alpha, d_alpha = self.compute_alpha_t(t_expanded)
        sigma, d_sigma = self.compute_sigma_t(t_expanded)

        alpha = torch.clamp(alpha, min=1e-8)
        reverse_alpha_ratio = alpha / d_alpha
        var = sigma**2 - reverse_alpha_ratio * d_sigma * sigma
        score = (reverse_alpha_ratio * velocity - x) / torch.clamp(var, min=1e-12)

        return score

    def velocity_to_noise(
        self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Convert velocity prediction to noise prediction.

        Args:
            velocity: Predicted velocity of shape (batch_size, ...).
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Noise tensor of shape (batch_size, ...).
        """
        t_expanded = expand_t_like_x(t, x)
        alpha, d_alpha = self.compute_alpha_t(t_expanded)
        sigma, d_sigma = self.compute_sigma_t(t_expanded)

        d_alpha = torch.where(d_alpha.abs() < 1e-8, torch.ones_like(d_alpha) * 1e-8, d_alpha)
        reverse_alpha_ratio = alpha / d_alpha
        var = sigma - reverse_alpha_ratio * d_sigma
        var = torch.where(var.abs() < 1e-12, torch.sign(var) * 1e-12 + (var == 0) * 1e-12, var)
        noise = (x - reverse_alpha_ratio * velocity) / var

        return noise

    def score_to_velocity(
        self, score: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Convert score prediction to velocity.

        Args:
            score: Predicted score of shape (batch_size, ...).
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Velocity tensor of shape (batch_size, ...).
        """
        drift_mean, drift_var = self.compute_drift(x, t)
        velocity = drift_var * score - drift_mean
        return velocity
```

### `compute_alpha_t(t)`

Compute the data coefficient (\\alpha(t)) and its time derivative.

Parameters:

| Name | Type     | Description                             | Default    |
| ---- | -------- | --------------------------------------- | ---------- |
| `t`  | `Tensor` | Time tensor of shape (batch_size, ...). | *required* |

Returns:

| Type                    | Description                                   |
| ----------------------- | --------------------------------------------- |
| `Tuple[Tensor, Tensor]` | Tuple of ((\\alpha(t)), (\\dot{\\alpha}(t))). |

Source code in `torchebm/core/base_interpolant.py`

```python
@abstractmethod
def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute the data coefficient \(\alpha(t)\) and its time derivative.

    Args:
        t: Time tensor of shape (batch_size, ...).

    Returns:
        Tuple of (\(\alpha(t)\), \(\dot{\alpha}(t)\)).
    """
    raise NotImplementedError
```

### `compute_d_alpha_alpha_ratio_t(t)`

Compute the ratio (\\dot{\\alpha}(t) / \\alpha(t)) for numerical stability.

This method can be overridden for better numerical precision.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type     | Description       |
| -------- | ----------------- |
| `Tensor` | The ratio tensor. |

Source code in `torchebm/core/base_interpolant.py`

```python
def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the ratio \(\dot{\alpha}(t) / \alpha(t)\) for numerical stability.

    This method can be overridden for better numerical precision.

    Args:
        t: Time tensor.

    Returns:
        The ratio tensor.
    """
    alpha, d_alpha = self.compute_alpha_t(t)
    return d_alpha / torch.clamp(alpha, min=1e-8)
```

### `compute_diffusion(x, t, form='SBDM', norm=1.0)`

Compute diffusion coefficient for SDE sampling.

Parameters:

| Name   | Type     | Description                                                                                                                                                                                                                                                | Default    |
| ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `x`    | `Tensor` | Current state of shape (batch_size, ...).                                                                                                                                                                                                                  | *required* |
| `t`    | `Tensor` | Time values of shape (batch_size,).                                                                                                                                                                                                                        | *required* |
| `form` | `str`    | Diffusion form. Choices: - 'constant': Constant diffusion - 'SBDM': Score-based diffusion matching - 'sigma': Proportional to noise schedule - 'linear': Linear decay - 'decreasing': Faster decay towards t=1 - 'increasing-decreasing': Peak at midpoint | `'SBDM'`   |
| `norm` | `float`  | Scaling factor for diffusion.                                                                                                                                                                                                                              | `1.0`      |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Tensor` | Diffusion coefficient tensor. |

Source code in `torchebm/core/base_interpolant.py`

```python
def compute_diffusion(
    self, x: torch.Tensor, t: torch.Tensor, form: str = "SBDM", norm: float = 1.0
) -> torch.Tensor:
    r"""
    Compute diffusion coefficient for SDE sampling.

    Args:
        x: Current state of shape (batch_size, ...).
        t: Time values of shape (batch_size,).
        form: Diffusion form. Choices:
            - 'constant': Constant diffusion
            - 'SBDM': Score-based diffusion matching
            - 'sigma': Proportional to noise schedule
            - 'linear': Linear decay
            - 'decreasing': Faster decay towards t=1
            - 'increasing-decreasing': Peak at midpoint
        norm: Scaling factor for diffusion.

    Returns:
        Diffusion coefficient tensor.
    """
    t_expanded = expand_t_like_x(t, x)
    sigma, _ = self.compute_sigma_t(t_expanded)
    _, drift_var = self.compute_drift(x, t)

    if form == "constant":
        return norm * torch.ones_like(drift_var)
    elif form == "SBDM":
        return norm * drift_var / (sigma + 1e-8)
    elif form == "sigma":
        return norm * sigma
    elif form == "linear":
        return norm * (1 - t_expanded)
    elif form == "decreasing":
        # Faster decay: (1-t)^2
        return norm * (1 - t_expanded) ** 2
    elif form == "increasing-decreasing":
        # Peak at t=0.5: 4*t*(1-t)
        return norm * 4 * t_expanded * (1 - t_expanded)
    else:
        raise ValueError(
            f"Unknown diffusion form '{form}'. "
            f"Choose from: constant, SBDM, sigma, linear, decreasing, increasing-decreasing"
        )
```

### `compute_drift(x, t)`

Compute drift coefficients for score-based parameterization.

For the probability flow ODE in score parameterization: dx = [-drift_mean + drift_var * score] dt

Parameters:

| Name | Type     | Description                               | Default    |
| ---- | -------- | ----------------------------------------- | ---------- |
| `x`  | `Tensor` | Current state of shape (batch_size, ...). | *required* |
| `t`  | `Tensor` | Time values of shape (batch_size,).       | *required* |

Returns:

| Type                    | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `Tuple[Tensor, Tensor]` | Tuple of (drift_mean, drift_var) for score parameterization. |

Source code in `torchebm/core/base_interpolant.py`

```python
def compute_drift(
    self, x: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute drift coefficients for score-based parameterization.

    For the probability flow ODE in score parameterization:
    dx = [-drift_mean + drift_var * score] dt

    Args:
        x: Current state of shape (batch_size, ...).
        t: Time values of shape (batch_size,).

    Returns:
        Tuple of (drift_mean, drift_var) for score parameterization.
    """
    t_expanded = expand_t_like_x(t, x)
    alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t_expanded)
    sigma, d_sigma = self.compute_sigma_t(t_expanded)

    drift_mean = alpha_ratio * x
    drift_var = alpha_ratio * (sigma**2) - sigma * d_sigma

    return -drift_mean, drift_var
```

### `compute_sigma_t(t)`

Compute the noise coefficient (\\sigma(t)) and its time derivative.

Parameters:

| Name | Type     | Description                             | Default    |
| ---- | -------- | --------------------------------------- | ---------- |
| `t`  | `Tensor` | Time tensor of shape (batch_size, ...). | *required* |

Returns:

| Type                    | Description                                   |
| ----------------------- | --------------------------------------------- |
| `Tuple[Tensor, Tensor]` | Tuple of ((\\sigma(t)), (\\dot{\\sigma}(t))). |

Source code in `torchebm/core/base_interpolant.py`

```python
@abstractmethod
def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute the noise coefficient \(\sigma(t)\) and its time derivative.

    Args:
        t: Time tensor of shape (batch_size, ...).

    Returns:
        Tuple of (\(\sigma(t)\), \(\dot{\sigma}(t)\)).
    """
    raise NotImplementedError
```

### `interpolate(x0, x1, t)`

Compute the interpolated sample (x_t) and conditional velocity (u_t).

Parameters:

| Name | Type     | Description                               | Default    |
| ---- | -------- | ----------------------------------------- | ---------- |
| `x0` | `Tensor` | Noise samples of shape (batch_size, ...). | *required* |
| `x1` | `Tensor` | Data samples of shape (batch_size, ...).  | *required* |
| `t`  | `Tensor` | Time values of shape (batch_size,).       | *required* |

Returns:

| Type                    | Description                                                                    |
| ----------------------- | ------------------------------------------------------------------------------ |
| `Tuple[Tensor, Tensor]` | Tuple of (x_t, u_t) where: - x_t = α(t) x₁ + σ(t) x₀ - u_t = α̇(t) x₁ + σ̇(t) x₀ |

Source code in `torchebm/core/base_interpolant.py`

```python
def interpolate(
    self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute the interpolated sample \(x_t\) and conditional velocity \(u_t\).

    Args:
        x0: Noise samples of shape (batch_size, ...).
        x1: Data samples of shape (batch_size, ...).
        t: Time values of shape (batch_size,).

    Returns:
        Tuple of (x_t, u_t) where:
            - x_t = α(t) x₁ + σ(t) x₀
            - u_t = α̇(t) x₁ + σ̇(t) x₀
    """
    t_expanded = expand_t_like_x(t, x0)
    alpha, d_alpha = self.compute_alpha_t(t_expanded)
    sigma, d_sigma = self.compute_sigma_t(t_expanded)

    xt = alpha * x1 + sigma * x0
    ut = d_alpha * x1 + d_sigma * x0

    return xt, ut
```

### `score_to_velocity(score, x, t)`

Convert score prediction to velocity.

Parameters:

| Name    | Type     | Description                                 | Default    |
| ------- | -------- | ------------------------------------------- | ---------- |
| `score` | `Tensor` | Predicted score of shape (batch_size, ...). | *required* |
| `x`     | `Tensor` | Current state of shape (batch_size, ...).   | *required* |
| `t`     | `Tensor` | Time values of shape (batch_size,).         | *required* |

Returns:

| Type     | Description                                 |
| -------- | ------------------------------------------- |
| `Tensor` | Velocity tensor of shape (batch_size, ...). |

Source code in `torchebm/core/base_interpolant.py`

```python
def score_to_velocity(
    self, score: torch.Tensor, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    r"""
    Convert score prediction to velocity.

    Args:
        score: Predicted score of shape (batch_size, ...).
        x: Current state of shape (batch_size, ...).
        t: Time values of shape (batch_size,).

    Returns:
        Velocity tensor of shape (batch_size, ...).
    """
    drift_mean, drift_var = self.compute_drift(x, t)
    velocity = drift_var * score - drift_mean
    return velocity
```

### `velocity_to_noise(velocity, x, t)`

Convert velocity prediction to noise prediction.

Parameters:

| Name       | Type     | Description                                    | Default    |
| ---------- | -------- | ---------------------------------------------- | ---------- |
| `velocity` | `Tensor` | Predicted velocity of shape (batch_size, ...). | *required* |
| `x`        | `Tensor` | Current state of shape (batch_size, ...).      | *required* |
| `t`        | `Tensor` | Time values of shape (batch_size,).            | *required* |

Returns:

| Type     | Description                              |
| -------- | ---------------------------------------- |
| `Tensor` | Noise tensor of shape (batch_size, ...). |

Source code in `torchebm/core/base_interpolant.py`

```python
def velocity_to_noise(
    self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    r"""
    Convert velocity prediction to noise prediction.

    Args:
        velocity: Predicted velocity of shape (batch_size, ...).
        x: Current state of shape (batch_size, ...).
        t: Time values of shape (batch_size,).

    Returns:
        Noise tensor of shape (batch_size, ...).
    """
    t_expanded = expand_t_like_x(t, x)
    alpha, d_alpha = self.compute_alpha_t(t_expanded)
    sigma, d_sigma = self.compute_sigma_t(t_expanded)

    d_alpha = torch.where(d_alpha.abs() < 1e-8, torch.ones_like(d_alpha) * 1e-8, d_alpha)
    reverse_alpha_ratio = alpha / d_alpha
    var = sigma - reverse_alpha_ratio * d_sigma
    var = torch.where(var.abs() < 1e-12, torch.sign(var) * 1e-12 + (var == 0) * 1e-12, var)
    noise = (x - reverse_alpha_ratio * velocity) / var

    return noise
```

### `velocity_to_score(velocity, x, t)`

Convert velocity prediction to score.

Parameters:

| Name       | Type     | Description                                    | Default    |
| ---------- | -------- | ---------------------------------------------- | ---------- |
| `velocity` | `Tensor` | Predicted velocity of shape (batch_size, ...). | *required* |
| `x`        | `Tensor` | Current state of shape (batch_size, ...).      | *required* |
| `t`        | `Tensor` | Time values of shape (batch_size,).            | *required* |

Returns:

| Type     | Description                              |
| -------- | ---------------------------------------- |
| `Tensor` | Score tensor of shape (batch_size, ...). |

Source code in `torchebm/core/base_interpolant.py`

```python
def velocity_to_score(
    self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    r"""
    Convert velocity prediction to score.

    Args:
        velocity: Predicted velocity of shape (batch_size, ...).
        x: Current state of shape (batch_size, ...).
        t: Time values of shape (batch_size,).

    Returns:
        Score tensor of shape (batch_size, ...).
    """
    t_expanded = expand_t_like_x(t, x)
    alpha, d_alpha = self.compute_alpha_t(t_expanded)
    sigma, d_sigma = self.compute_sigma_t(t_expanded)

    alpha = torch.clamp(alpha, min=1e-8)
    reverse_alpha_ratio = alpha / d_alpha
    var = sigma**2 - reverse_alpha_ratio * d_sigma * sigma
    score = (reverse_alpha_ratio * velocity - x) / torch.clamp(var, min=1e-12)

    return score
```

## `BaseLoss`

Bases: `Schedulable`, `TorchEBMModule`, `ABC`

Abstract base class for loss functions used in energy-based models.

Parameters:

| Name     | Type                           | Description                 | Default   |
| -------- | ------------------------------ | --------------------------- | --------- |
| `dtype`  | `dtype`                        | Data type for computations. | `float32` |
| `device` | `Optional[Union[str, device]]` | Device for computations.    | `None`    |

Source code in `torchebm/core/base_loss.py`

```python
class BaseLoss(Schedulable, TorchEBMModule, ABC):
    """
    Abstract base class for loss functions used in energy-based models.

    Args:
        dtype (torch.dtype): Data type for computations.
        device (Optional[Union[str, torch.device]]): Device for computations.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the base loss class."""
        super().__init__(device=device, dtype=dtype, *args, **kwargs)

    def _resolve_model_kwargs(
        self,
        model_kwargs: Optional[dict],
        legacy_kwargs: Optional[dict] = None,
        *,
        warn_key: str,
    ) -> dict:
        r"""Merge explicit `model_kwargs` with deprecated bare ``**kwargs``.

        Shared shim for losses whose bare ``**kwargs`` historically meant *model*
        conditioning (EqM, EM, score matching). The explicit dict wins on key
        conflicts; a non-empty legacy mapping triggers a one-time
        ``DeprecationWarning`` keyed by `warn_key`. The result is device-
        normalized once (see `_prepare_model_kwargs`) and is a fresh dict, so it
        never aliases the caller's mapping.
        """
        if legacy_kwargs:
            warn_once(
                warn_key,
                "Passing model conditioning as bare keyword arguments is "
                "deprecated; pass model_kwargs={...} instead.",
            )
            merged = {**legacy_kwargs, **(model_kwargs or {})}
        else:
            merged = model_kwargs
        return self._prepare_model_kwargs(merged)

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the loss value.

        Args:
            x (torch.Tensor): Input data tensor from the target distribution.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        pass

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}()"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()
```

### `__init__(dtype=torch.float32, device=None, *args, **kwargs)`

Initialize the base loss class.

Source code in `torchebm/core/base_loss.py`

```python
def __init__(
    self,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
    *args: Any,
    **kwargs: Any,
):
    """Initialize the base loss class."""
    super().__init__(device=device, dtype=dtype, *args, **kwargs)
```

### `__repr__()`

Return a string representation of the loss function.

Source code in `torchebm/core/base_loss.py`

```python
def __repr__(self):
    """Return a string representation of the loss function."""
    return f"{self.__class__.__name__}()"
```

### `__str__()`

Return a string representation of the loss function.

Source code in `torchebm/core/base_loss.py`

```python
def __str__(self):
    """Return a string representation of the loss function."""
    return self.__repr__()
```

### `forward(x, *args, **kwargs)`

Computes the loss value.

Parameters:

| Name       | Type     | Description                                     | Default    |
| ---------- | -------- | ----------------------------------------------- | ---------- |
| `x`        | `Tensor` | Input data tensor from the target distribution. | *required* |
| `*args`    |          | Additional positional arguments.                | `()`       |
| `**kwargs` |          | Additional keyword arguments.                   | `{}`       |

Returns:

| Type     | Description                                   |
| -------- | --------------------------------------------- |
| `Tensor` | torch.Tensor: The computed scalar loss value. |

Source code in `torchebm/core/base_loss.py`

```python
@abstractmethod
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Computes the loss value.

    Args:
        x (torch.Tensor): Input data tensor from the target distribution.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The computed scalar loss value.
    """
    pass
```

## `BaseModel`

Bases: `TorchEBMModule`, `ABC`

Abstract base class for energy-based models (EBMs).

This class provides a unified interface for defining EBMs, which represent the unnormalized negative log-likelihood of a probability distribution. It supports both analytical models and trainable neural networks.

Subclasses must implement the `forward(x)` method and can optionally override the `gradient(x)` method for analytical gradients.

Source code in `torchebm/core/base_model.py`

```python
class BaseModel(TorchEBMModule, ABC):
    r"""
    Abstract base class for energy-based models (EBMs).

    This class provides a unified interface for defining EBMs, which represent
    the unnormalized negative log-likelihood of a probability distribution.
    It supports both analytical models and trainable neural networks.

    Subclasses must implement the `forward(x)` method and can optionally
    override the `gradient(x)` method for analytical gradients.
    """

    #: When True, the default autograd `gradient` computes in float32 regardless
    #: of the input dtype (the historical behavior). Default False respects the
    #: input dtype - byte-identical for float32, no fp64 downcast, no bf16/fp16
    #: round-trip. Set per instance for fp32-precision gradients under low
    #: precision: ``model.force_fp32_gradient = True``.
    force_fp32_gradient: bool = False

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ):
        """Initializes the BaseModel base class."""
        super().__init__(dtype=dtype, *args, **kwargs)

    # @property
    # def device(self) -> torch.device:
    #     """Returns the device associated with the module's parameters/buffers (if any)."""
    #     try:
    #         return next(self.parameters()).device
    #     except StopIteration:
    #         try:
    #             return next(self.buffers()).device
    #         except StopIteration:
    #             return self._device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the scalar energy value for each input sample.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).

        Returns:
            torch.Tensor: Tensor of scalar energy values with shape (batch_size,).
        """
        pass

    def gradient(
        self, x: torch.Tensor, model_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        r"""
        Computes the gradient of the energy function with respect to the input, \(\nabla_x E(x)\).

        This default implementation uses `torch.autograd`. Subclasses can override it
        for analytical gradients.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).
            model_kwargs (Optional[dict]): Conditioning arguments forwarded to
                `forward(x, **model_kwargs)` (e.g. class labels). ``None`` (the
                default) reproduces the unconditional call exactly, so analytic
                `gradient(self, x)` overrides remain valid. Tensor values should
                already be device-resident (samplers normalize them once at
                `sample()` entry); they are **not** dtype-cast here, unlike `x`.

        Returns:
            torch.Tensor: Gradient tensor of the same shape as `x`.
        """

        original_dtype = x.dtype
        device = x.device

        if self.device and device != self.device:
            x = x.to(self.device)
            device = self.device

        grad_dtype = torch.float32 if self.force_fp32_gradient else original_dtype

        with torch.enable_grad():
            x_for_grad = (
                x.detach().to(dtype=grad_dtype, device=device).requires_grad_(True)
            )

            with self.autocast_context():
                energy = self.forward(x_for_grad, **(model_kwargs or {}))

            if energy.shape != (x_for_grad.shape[0],):
                raise ValueError(
                    f"BaseModel forward() output expected shape ({x_for_grad.shape[0]},), but got {energy.shape}."
                )

            if not energy.grad_fn:
                raise RuntimeError(
                    "Cannot compute gradient: `forward` method did not use the "
                    "input `x` in a differentiable way."
                )

            grad = torch.autograd.grad(
                outputs=energy,
                inputs=x_for_grad,
                grad_outputs=torch.ones_like(energy, device=energy.device),
                create_graph=False,  # false for standard grad computation
                retain_graph=None,  # since create_graph=False, let PyTorch decide
            )[0]

        if grad is None:
            raise RuntimeError(
                "Gradient computation failed unexpectedly. Check the forward pass implementation."
            )

        gradient = grad.to(original_dtype)

        return gradient.detach()
```

### `__init__(dtype=torch.float32, *args, **kwargs)`

Initializes the BaseModel base class.

Source code in `torchebm/core/base_model.py`

```python
def __init__(
    self,
    dtype: torch.dtype = torch.float32,
    *args,
    **kwargs,
):
    """Initializes the BaseModel base class."""
    super().__init__(dtype=dtype, *args, **kwargs)
```

### `forward(x)`

Computes the scalar energy value for each input sample.

Parameters:

| Name | Type     | Description                                       | Default    |
| ---- | -------- | ------------------------------------------------- | ---------- |
| `x`  | `Tensor` | Input tensor of shape (batch_size, \*input_dims). | *required* |

Returns:

| Type     | Description                                                            |
| -------- | ---------------------------------------------------------------------- |
| `Tensor` | torch.Tensor: Tensor of scalar energy values with shape (batch_size,). |

Source code in `torchebm/core/base_model.py`

```python
@abstractmethod
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the scalar energy value for each input sample.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).

    Returns:
        torch.Tensor: Tensor of scalar energy values with shape (batch_size,).
    """
    pass
```

### `gradient(x, model_kwargs=None)`

Computes the gradient of the energy function with respect to the input, (\\nabla_x E(x)).

This default implementation uses `torch.autograd`. Subclasses can override it for analytical gradients.

Parameters:

| Name           | Type             | Description                                                                                                                                                                                                                                                                                                                                         | Default    |
| -------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `x`            | `Tensor`         | Input tensor of shape (batch_size, \*input_dims).                                                                                                                                                                                                                                                                                                   | *required* |
| `model_kwargs` | `Optional[dict]` | Conditioning arguments forwarded to forward(x, \*\*model_kwargs) (e.g. class labels). None (the default) reproduces the unconditional call exactly, so analytic gradient(self, x) overrides remain valid. Tensor values should already be device-resident (samplers normalize them once at sample() entry); they are not dtype-cast here, unlike x. | `None`     |

Returns:

| Type     | Description                                           |
| -------- | ----------------------------------------------------- |
| `Tensor` | torch.Tensor: Gradient tensor of the same shape as x. |

Source code in `torchebm/core/base_model.py`

```python
def gradient(
    self, x: torch.Tensor, model_kwargs: Optional[dict] = None
) -> torch.Tensor:
    r"""
    Computes the gradient of the energy function with respect to the input, \(\nabla_x E(x)\).

    This default implementation uses `torch.autograd`. Subclasses can override it
    for analytical gradients.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).
        model_kwargs (Optional[dict]): Conditioning arguments forwarded to
            `forward(x, **model_kwargs)` (e.g. class labels). ``None`` (the
            default) reproduces the unconditional call exactly, so analytic
            `gradient(self, x)` overrides remain valid. Tensor values should
            already be device-resident (samplers normalize them once at
            `sample()` entry); they are **not** dtype-cast here, unlike `x`.

    Returns:
        torch.Tensor: Gradient tensor of the same shape as `x`.
    """

    original_dtype = x.dtype
    device = x.device

    if self.device and device != self.device:
        x = x.to(self.device)
        device = self.device

    grad_dtype = torch.float32 if self.force_fp32_gradient else original_dtype

    with torch.enable_grad():
        x_for_grad = (
            x.detach().to(dtype=grad_dtype, device=device).requires_grad_(True)
        )

        with self.autocast_context():
            energy = self.forward(x_for_grad, **(model_kwargs or {}))

        if energy.shape != (x_for_grad.shape[0],):
            raise ValueError(
                f"BaseModel forward() output expected shape ({x_for_grad.shape[0]},), but got {energy.shape}."
            )

        if not energy.grad_fn:
            raise RuntimeError(
                "Cannot compute gradient: `forward` method did not use the "
                "input `x` in a differentiable way."
            )

        grad = torch.autograd.grad(
            outputs=energy,
            inputs=x_for_grad,
            grad_outputs=torch.ones_like(energy, device=energy.device),
            create_graph=False,  # false for standard grad computation
            retain_graph=None,  # since create_graph=False, let PyTorch decide
        )[0]

    if grad is None:
        raise RuntimeError(
            "Gradient computation failed unexpectedly. Check the forward pass implementation."
        )

    gradient = grad.to(original_dtype)

    return gradient.detach()
```

## `BaseModelCoupling`

Bases: `BaseCoupling`

Family base for couplings that generate the target from a model map.

The template `couple` runs under `torch.no_grad()` and returns ((x_0, \\Phi(x_0))); any incoming `x1` is ignored (documented family behavior; the argument stays optional for standalone use: `coupling(x0)`). Concretes supply `_generate`, the map evaluation. This is the reflow / rectified-flow shape; iterative-Markovian-fitting couplings (DSBM) reuse the same machinery.

Parameters:

| Name    | Type  | Description                                                 | Default    |
| ------- | ----- | ----------------------------------------------------------- | ---------- |
| `model` | `Any` | The map object (sampler, module, callable) \_generate uses. | *required* |

Source code in `torchebm/core/base_coupling.py`

```python
class BaseModelCoupling(BaseCoupling):
    r"""
    Family base for couplings that generate the target from a model map.

    The template `couple` runs under `torch.no_grad()` and returns
    \((x_0, \Phi(x_0))\); any incoming ``x1`` is ignored (documented family
    behavior; the argument stays optional for standalone use:
    ``coupling(x0)``). Concretes supply `_generate`, the map evaluation.
    This is the reflow / rectified-flow shape; iterative-Markovian-fitting
    couplings (DSBM) reuse the same machinery.

    Args:
        model: The map object (sampler, module, callable) `_generate` uses.
    """

    def __init__(self, model: Any):
        self.model = model

    @torch.no_grad()
    def couple(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> CouplingResult:
        # x1 (if any) is ignored: the target is generated as Phi(x0).
        return CouplingResult(x0, self._generate(x0, **kwargs))

    @abstractmethod
    def _generate(self, x0: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""
        Generate the target batch from the source batch.

        Args:
            x0: Source samples of shape (batch_size, ...).
            **kwargs: Optional conditioning forwarded from `couple`.

        Returns:
            Target samples of shape (batch_size, ...).
        """
        raise NotImplementedError
```

## `BaseRungeKuttaIntegrator`

Bases: `BaseIntegrator`

Abstract base class for explicit Runge-Kutta ODE integrators.

Subclasses define a Butcher tableau via the abstract properties `tableau_a`, `tableau_b`, and `tableau_c` and automatically inherit generic stepping and integration logic.

For an (s)-stage explicit RK method the update reads

[ k_i = f!\\bigl(x + h \\sum\_{j=0}^{i-1} a\_{ij},k_j,; t + c_i,h\\bigr), \\quad i = 0,\\ldots,s{-}1 ]

[ x\_{n+1} = x_n + h \\sum\_{i=0}^{s-1} b_i,k_i ]

**Adaptive step-size control** is available automatically for subclasses that define `error_weights` and `order`. When `adaptive=True` is passed to `integrate` (or left as `None` for auto-detection), the integrator uses an embedded error pair to control the step size.

Parameters:

| Name            | Type                                   | Description                                                                                                              | Default        |
| --------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------- |
| `device`        | `Optional[device]`                     | Device for computations.                                                                                                 | `None`         |
| `dtype`         | `Optional[dtype]`                      | Data type for computations.                                                                                              | `None`         |
| `atol`          | `float`                                | Absolute tolerance for adaptive stepping.                                                                                | `1e-06`        |
| `rtol`          | `float`                                | Relative tolerance for adaptive stepping.                                                                                | `0.001`        |
| `max_steps`     | `int`                                  | Maximum number of steps (accepted + rejected) before raising RuntimeError.                                               | `10000`        |
| `safety`        | `float`                                | Safety factor for step-size adjustment (< 1).                                                                            | `0.9`          |
| `min_factor`    | `float`                                | Minimum step-size shrink factor.                                                                                         | `0.2`          |
| `max_factor`    | `float`                                | Maximum step-size growth factor.                                                                                         | `10.0`         |
| `max_step_size` | `float`                                | Maximum absolute step size allowed during adaptive integration. Defaults to inf (no limit).                              | `float('inf')` |
| `norm`          | `Optional[Callable[[Tensor], Tensor]]` | Callable norm(tensor) -> scalar used to measure the local error. Defaults to the RMS norm (\\sqrt{\\mathrm{mean}(x^2)}). | `None`         |

Example

```python
from torchebm.core import BaseRungeKuttaIntegrator
import torch

class MidpointIntegrator(BaseRungeKuttaIntegrator):
    @property
    def tableau_a(self):
        return ((), (0.5,))

    @property
    def tableau_b(self):
        return (0.0, 1.0)

    @property
    def tableau_c(self):
        return (0.0, 0.5)

integrator = MidpointIntegrator()
state = {"x": torch.randn(100, 2)}
drift = lambda x, t: -x
result = integrator.step(state, step_size=0.01, drift=drift)
```

Source code in `torchebm/core/base_integrator.py`

````python
class BaseRungeKuttaIntegrator(BaseIntegrator):
    r"""Abstract base class for explicit Runge-Kutta ODE integrators.

    Subclasses define a Butcher tableau via the abstract properties
    ``tableau_a``, ``tableau_b``, and ``tableau_c`` and automatically
    inherit generic stepping and integration logic.

    For an \(s\)-stage explicit RK method the update reads

    \[
    k_i = f\!\bigl(x + h \sum_{j=0}^{i-1} a_{ij}\,k_j,\;
                    t + c_i\,h\bigr),
    \quad i = 0,\ldots,s{-}1
    \]

    \[
    x_{n+1} = x_n + h \sum_{i=0}^{s-1} b_i\,k_i
    \]

    **Adaptive step-size control** is available automatically for subclasses
    that define ``error_weights`` and ``order``.  When ``adaptive=True`` is
    passed to ``integrate`` (or left as ``None`` for auto-detection), the
    integrator uses an embedded error pair to control the step size.

    Args:
        device: Device for computations.
        dtype: Data type for computations.
        atol: Absolute tolerance for adaptive stepping.
        rtol: Relative tolerance for adaptive stepping.
        max_steps: Maximum number of steps (accepted + rejected) before
            raising ``RuntimeError``.
        safety: Safety factor for step-size adjustment (< 1).
        min_factor: Minimum step-size shrink factor.
        max_factor: Maximum step-size growth factor.
        max_step_size: Maximum absolute step size allowed during adaptive
            integration.  Defaults to ``inf`` (no limit).
        norm: Callable ``norm(tensor) -> scalar`` used to measure the
            local error.  Defaults to the RMS norm
            \(\sqrt{\mathrm{mean}(x^2)}\).

    Example:
        ```python
        from torchebm.core import BaseRungeKuttaIntegrator
        import torch

        class MidpointIntegrator(BaseRungeKuttaIntegrator):
            @property
            def tableau_a(self):
                return ((), (0.5,))

            @property
            def tableau_b(self):
                return (0.0, 1.0)

            @property
            def tableau_c(self):
                return (0.0, 0.5)

        integrator = MidpointIntegrator()
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x
        result = integrator.step(state, step_size=0.01, drift=drift)
        ```
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        max_steps: int = 10_000,
        safety: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 10.0,
        max_step_size: float = float("inf"),
        norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        solver_max_iter: int = 8,
        solver_tol: float = 1e-6,
        solver_check_every: int = 0,
    ):
        super().__init__(device=device, dtype=dtype)
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.max_step_size = max_step_size
        self._norm = norm
        self.solver_max_iter = solver_max_iter
        self.solver_tol = solver_tol
        self.solver_check_every = solver_check_every
        self._register_tableau_buffers()

    def _register_tableau_buffers(self):
        r"""Pre-compute Butcher tableau as registered buffers for efficient GPU computation."""
        s = len(self.tableau_c)
        _device = self.device
        _dtype = self.dtype
        a = torch.zeros(s, s, device=_device, dtype=_dtype)
        for i, row in enumerate(self.tableau_a):
            for j, val in enumerate(row):
                a[i, j] = val
        self.register_buffer("_buf_a", a)
        self.register_buffer("_buf_b", torch.tensor(self.tableau_b, device=_device, dtype=_dtype))
        self.register_buffer("_buf_c", torch.tensor(self.tableau_c, device=_device, dtype=_dtype))
        e = self.error_weights
        self.register_buffer(
            "_buf_e",
            torch.tensor(e, device=_device, dtype=_dtype) if e is not None else None,
        )
        # Per-stage implicit flag, derived from the diagonal of `a`.
        # Stage `i` is implicit iff a[i, i] != 0 (DIRK convention).
        # Stored as a Python tuple so the per-stage dispatch in _evaluate_stages
        # is a free bool check with no GPU op or sync per step.
        self._stage_implicit: Tuple[bool, ...] = tuple(
            bool(a[i, i] != 0) for i in range(s)
        )

    # butcher tableau, must be defined by subclasses

    @property
    @abstractmethod
    def tableau_a(self) -> Tuple[Tuple[float, ...], ...]:
        r"""Lower-triangular RK matrix \(a_{ij}\).

        ``tableau_a[i]`` contains coefficients \(a_{i0}, \ldots, a_{i,i-1}\)
        for explicit methods; for diagonally-implicit (DIRK) methods the row
        extends to \(a_{ii}\) (length ``i+1``). A stage with non-zero diagonal
        entry ``a[i, i]`` is solved by Picard iteration in ``_evaluate_stages``.
        The first row is the empty tuple ``()`` for explicit methods.
        """

    @property
    @abstractmethod
    def tableau_b(self) -> Tuple[float, ...]:
        r"""Weights \(b_i\) used to combine stages into the final update."""

    @property
    @abstractmethod
    def tableau_c(self) -> Tuple[float, ...]:
        r"""Nodes \(c_i\) — time-fraction offsets for each stage evaluation."""

    @property
    def n_stages(self) -> int:
        """Number of stages *s* in the method."""
        return len(self.tableau_c)

    # adaptive properties, override in embedded-pair subclasses

    @property
    def error_weights(self) -> Optional[Tuple[float, ...]]:
        r"""Error estimation weights \(e_i = b_i - \hat{b}_i\).

        Return ``None`` (the default) for methods without an embedded pair.
        For FSAL methods the tuple has ``n_stages + 1`` entries; for
        non-FSAL methods it has ``n_stages`` entries.
        """
        return None

    @property
    def order(self) -> Optional[int]:
        r"""Order *p* of the higher-order solution.

        Used in the step-size control exponent \(-1/p\).  Return ``None``
        (the default) for methods without adaptive support.
        """
        return None

    @property
    def fsal(self) -> bool:
        r"""Whether the method has the First Same As Last property.

        When ``True`` the integrator evaluates one extra stage at the
        accepted solution and reuses it as the first stage of the next
        step, saving one drift evaluation per accepted step.
        """
        return False

    # helpers

    @staticmethod
    def _rms_norm(x: torch.Tensor) -> torch.Tensor:
        r"""Root-mean-square norm: \(\sqrt{\mathrm{mean}(x^2)}\)."""
        return torch.sqrt(torch.mean(x ** 2))

    def _tableau_on(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""Return tableau tensors (a, b, c, e) cached on (device, dtype)."""
        key = (device, dtype)
        if getattr(self, "_tableau_cache_key", None) != key:
            self._cached_a = self._buf_a.to(device=device, dtype=dtype)
            self._cached_b = self._buf_b.to(device=device, dtype=dtype)
            self._cached_c = self._buf_c.to(device=device, dtype=dtype)
            self._cached_e = (
                self._buf_e.to(device=device, dtype=dtype)
                if self._buf_e is not None
                else None
            )
            self._tableau_cache_key = key
        return self._cached_a, self._cached_b, self._cached_c, self._cached_e

    def _evaluate_stages(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        step_size: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        k0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Evaluate all RK stages and return a tensor of shape ``(s, *x.shape)``.

        Per-stage dispatch on ``self._stage_implicit[i]``: stages with
        ``a[i, i] == 0`` (explicit) use a single drift evaluation; stages
        with ``a[i, i] != 0`` (DIRK) solve the implicit equation
        \(k_i = f(x_{\text{base}} + h\,a_{ii}\,k_i,\,t_i)\) by Picard
        iteration via ``_solve_implicit_stage``.

        Args:
            x: Current position tensor.
            t: Current time tensor (batch,).
            step_size: Step size \(h\).
            drift_fn: Drift callable ``f(x, t)``.
            k0: Optional pre-computed first stage.  When provided the first
                drift evaluation is skipped (used by FSAL methods to reuse
                the last stage of the previous step).
        """
        a, _, c, _ = self._tableau_on(x.device, x.dtype)
        s = a.size(0)
        k = x.new_empty((s,) + x.shape)
        for i in range(s):
            if i == 0 and k0 is not None:
                k[0] = k0
                continue
            if i == 0:
                x_stage = x
            else:
                # Fused mul+reduce via einsum (1 kernel) instead of
                # broadcast-mul + sum (2 kernels per stage).
                x_stage = x + step_size * torch.einsum(
                    "i,i...->...", a[i, :i], k[:i]
                )
            t_stage = t + c[i] * step_size
            if self._stage_implicit[i]:
                k[i] = self._solve_implicit_stage(
                    x_stage, t_stage, step_size, a[i, i], drift_fn,
                )
            else:
                k[i] = drift_fn(x_stage, t_stage)
        return k

    def _solve_implicit_stage(
        self,
        base: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        a_ii: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        r"""Solve \(k = f(\text{base} + h\,a_{ii}\,k,\,t)\) by Picard iteration.

        Default path (``solver_check_every <= 0``) runs ``solver_max_iter``
        iterations unconditionally with no CPU-GPU sync. The opt-in path
        (``solver_check_every > 0``) checks the RMS residual every
        ``solver_check_every`` iterations and exits early when below
        ``solver_tol`` (one ``.item()`` sync per check).

        The mul-add \(\text{base} + h\,a_{ii}\,k\) uses ``torch.addcmul``
        for single-kernel fusion (cf. einsum-fusion at line 332).
        """
        coef = h * a_ii                              # scalar 0-d tensor, once
        k = drift_fn(base, t)                        # warm start
        if self.solver_check_every <= 0:
            for _ in range(self.solver_max_iter - 1):
                k = drift_fn(torch.addcmul(base, coef, k), t)
            return k
        for it in range(1, self.solver_max_iter):
            k_next = drift_fn(torch.addcmul(base, coef, k), t)
            if it % self.solver_check_every == 0:
                resid = (k_next - k).square().mean().sqrt()
                k = k_next
                # Unavoidable host sync: the convergence branch is data-dependent,
                # bounded to one per `solver_check_every` iterations.
                if resid.item() < self.solver_tol:
                    break
            else:
                k = k_next
        return k

    def _combine_stages(
        self,
        x: torch.Tensor,
        step_size: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        r"""Combine RK stages into the deterministic update \(x + h \sum b_i k_i\)."""
        _, b, _, _ = self._tableau_on(x.device, x.dtype)
        s = k.size(0)
        # Fused mul+reduce via einsum (1 kernel) instead of broadcast + sum.
        return x + step_size * torch.einsum("i,i...->...", b[:s], k)

    def _deterministic_step(
        self,
        x: torch.Tensor,
        step_size: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the deterministic RK update \(x + h \sum b_i k_i\)."""
        k = self._evaluate_stages(x, t, step_size, drift_fn)
        return self._combine_stages(x, step_size, k)

    @staticmethod
    def _build_time_grid(
        x: torch.Tensor,
        step_size: torch.Tensor,
        n_steps: int,
        t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r"""Build or validate the 1-D time grid for fixed-step integration."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if t is None:
            if not torch.is_tensor(step_size):
                step_size = torch.tensor(
                    step_size, device=x.device, dtype=x.dtype
                )
            t = (
                torch.arange(n_steps + 1, device=x.device, dtype=x.dtype)
                * step_size
            )
        if t.ndim != 1 or t.numel() < 2:
            raise ValueError("t must be a 1D tensor with length >= 2")
        return t

    def _adaptive_integrate(
        self,
        x: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t_start: float,
        t_end: float,
        h: float,
    ) -> torch.Tensor:
        r"""Core adaptive integration loop from *t_start* to *t_end*."""
        t_current = t_start
        p = self.order
        is_fsal = self.fsal
        norm_fn = self._norm if self._norm is not None else self._rms_norm

        k1_cached: Optional[torch.Tensor] = None
        if is_fsal:
            t_batch = torch.full(
                (x.size(0),), t_current, device=x.device, dtype=x.dtype
            )
            k1_cached = drift_fn(x, t_batch)

        for _ in range(self.max_steps):
            if t_current >= t_end - 1e-12 * max(abs(t_end), 1.0):
                break

            h = min(h, t_end - t_current, self.max_step_size)
            h_t = torch.tensor(h, device=x.device, dtype=x.dtype)
            t_batch = torch.full(
                (x.size(0),), t_current, device=x.device, dtype=x.dtype
            )

            k = self._evaluate_stages(x, t_batch, h_t, drift_fn, k0=k1_cached)
            y_new = self._combine_stages(x, h_t, k)

            # Error estimation
            if is_fsal:
                k_fsal = drift_fn(y_new, t_batch + h_t)
                k_err = torch.cat([k, k_fsal.unsqueeze(0)], dim=0)
            else:
                k_err = k

            _, _, _, e_buf = self._tableau_on(x.device, x.dtype)
            # Fused mul+reduce via einsum (1 kernel) instead of broadcast + sum.
            err_vec = h_t * torch.einsum(
                "i,i...->...", e_buf[: k_err.size(0)], k_err
            )

            scale = self.atol + self.rtol * torch.max(x.abs(), y_new.abs())
            # Unavoidable host sync: step accept/reject and the resize of `h` are
            # data-dependent on this ratio. One per attempted step.
            err_ratio = norm_fn(err_vec / scale).item()

            if err_ratio <= 1.0:
                x = y_new
                t_current += h
                k1_cached = k_fsal if is_fsal else None

            if err_ratio == 0.0:
                factor = self.max_factor
            else:
                factor = min(
                    self.max_factor,
                    max(
                        self.min_factor,
                        self.safety * err_ratio ** (-1.0 / p),
                    ),
                )
            h = min(h * factor, self.max_step_size)
        else:
            raise RuntimeError(
                f"{type(self).__name__}: maximum number of steps "
                f"({self.max_steps}) exceeded."
            )

        return x

    def step(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor,
        *,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Advance the state by one deterministic RK step.

        Args:
            state: Mapping containing ``"x"`` position tensor.
            step_size: Step size for the integration.
            drift: Explicit drift callable ``f(x, t)``.
            t: Current time tensor (batch,).

        Returns:
            Updated state dict ``{"x": x_new}``.
        """
        x = state["x"]
        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)
        if t is None:
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        drift_fn = self._resolve_drift(drift)
        x_new = self._deterministic_step(x, step_size, drift_fn, t)
        return {"x": x_new}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor,
        n_steps: int,
        *,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        t: Optional[torch.Tensor] = None,
        adaptive: Optional[bool] = None,
        inference_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Integrate the state over a time interval (ODE).

        Args:
            state: Mapping with key ``"x"`` holding the position tensor.
            step_size: Uniform step size (fixed mode) or initial step size
                (adaptive mode).
            n_steps: Number of integration steps (fixed mode) or, together
                with ``step_size``, defines the integration interval when
                ``t`` is ``None``.
            drift: Explicit drift callable ``f(x, t)``.
            t: 1-D time grid.  Built from ``step_size`` when ``None``.
                In adaptive mode only ``t[0]`` and ``t[-1]`` are used.
            adaptive: ``True`` for adaptive step-size control, ``False``
                for fixed-step.  When ``None`` (default) adaptive mode
                is used automatically if ``error_weights`` is defined.
            inference_mode: When ``True``, wraps computation in
                ``torch.inference_mode()`` for faster execution without
                gradient tracking.

        Returns:
            Updated state dict ``{"x": x_final}``.
        """
        if inference_mode:
            with torch.inference_mode():
                return self.integrate(
                    state, step_size, n_steps,
                    drift=drift, t=t, adaptive=adaptive,
                )

        if adaptive is None:
            adaptive = self.error_weights is not None

        # fixed-step path
        if not adaptive:
            t_grid = self._build_time_grid(state["x"], step_size, n_steps, t)
            x = state["x"]
            drift_fn = self._resolve_drift(drift)
            n = t_grid.numel() - 1
            batch_size = x.size(0)
            dts = t_grid[1:] - t_grid[:-1]
            for i in range(n):
                t_batch = t_grid[i].expand(batch_size)
                x = self._deterministic_step(x, dts[i], drift_fn, t_batch)
            return {"x": x}

        # adaptive path
        if self.error_weights is None or self.order is None:
            raise ValueError(
                f"{type(self).__name__} does not define error_weights/order "
                f"and cannot be used with adaptive=True."
            )

        x = state["x"]
        drift_fn = self._resolve_drift(drift)

        if not torch.is_tensor(step_size):
            step_size = torch.tensor(
                step_size, device=x.device, dtype=x.dtype
            )

        if t is not None:
            if t.ndim != 1 or t.numel() < 2:
                raise ValueError("t must be a 1D tensor with length >= 2")
            t_start = t[0].item()
            t_end = t[-1].item()
        else:
            t_start = 0.0
            t_end = float(n_steps) * step_size.item()

        h = min(step_size.item(), t_end - t_start, self.max_step_size)
        x = self._adaptive_integrate(x, drift_fn, t_start, t_end, h)
        return {"x": x}
````

### `error_weights`

Error estimation weights (e_i = b_i - \\hat{b}\_i).

Return `None` (the default) for methods without an embedded pair. For FSAL methods the tuple has `n_stages + 1` entries; for non-FSAL methods it has `n_stages` entries.

### `fsal`

Whether the method has the First Same As Last property.

When `True` the integrator evaluates one extra stage at the accepted solution and reuses it as the first stage of the next step, saving one drift evaluation per accepted step.

### `n_stages`

Number of stages *s* in the method.

### `order`

Order *p* of the higher-order solution.

Used in the step-size control exponent (-1/p). Return `None` (the default) for methods without adaptive support.

### `tableau_a`

Lower-triangular RK matrix (a\_{ij}).

`tableau_a[i]` contains coefficients (a\_{i0}, \\ldots, a\_{i,i-1}) for explicit methods; for diagonally-implicit (DIRK) methods the row extends to (a\_{ii}) (length `i+1`). A stage with non-zero diagonal entry `a[i, i]` is solved by Picard iteration in `_evaluate_stages`. The first row is the empty tuple `()` for explicit methods.

### `tableau_b`

Weights (b_i) used to combine stages into the final update.

### `tableau_c`

Nodes (c_i) — time-fraction offsets for each stage evaluation.

### `integrate(state, step_size, n_steps, *, drift=None, t=None, adaptive=None, inference_mode=False)`

Integrate the state over a time interval (ODE).

Parameters:

| Name             | Type                                           | Description                                                                                                                                     | Default    |
| ---------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `state`          | `Dict[str, Tensor]`                            | Mapping with key "x" holding the position tensor.                                                                                               | *required* |
| `step_size`      | `Tensor`                                       | Uniform step size (fixed mode) or initial step size (adaptive mode).                                                                            | *required* |
| `n_steps`        | `int`                                          | Number of integration steps (fixed mode) or, together with step_size, defines the integration interval when t is None.                          | *required* |
| `drift`          | `Optional[Callable[[Tensor, Tensor], Tensor]]` | Explicit drift callable f(x, t).                                                                                                                | `None`     |
| `t`              | `Optional[Tensor]`                             | 1-D time grid. Built from step_size when None. In adaptive mode only t[0] and t[-1] are used.                                                   | `None`     |
| `adaptive`       | `Optional[bool]`                               | True for adaptive step-size control, False for fixed-step. When None (default) adaptive mode is used automatically if error_weights is defined. | `None`     |
| `inference_mode` | `bool`                                         | When True, wraps computation in torch.inference_mode() for faster execution without gradient tracking.                                          | `False`    |

Returns:

| Type                | Description                        |
| ------------------- | ---------------------------------- |
| `Dict[str, Tensor]` | Updated state dict {"x": x_final}. |

Source code in `torchebm/core/base_integrator.py`

```python
def integrate(
    self,
    state: Dict[str, torch.Tensor],
    step_size: torch.Tensor,
    n_steps: int,
    *,
    drift: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    t: Optional[torch.Tensor] = None,
    adaptive: Optional[bool] = None,
    inference_mode: bool = False,
) -> Dict[str, torch.Tensor]:
    r"""Integrate the state over a time interval (ODE).

    Args:
        state: Mapping with key ``"x"`` holding the position tensor.
        step_size: Uniform step size (fixed mode) or initial step size
            (adaptive mode).
        n_steps: Number of integration steps (fixed mode) or, together
            with ``step_size``, defines the integration interval when
            ``t`` is ``None``.
        drift: Explicit drift callable ``f(x, t)``.
        t: 1-D time grid.  Built from ``step_size`` when ``None``.
            In adaptive mode only ``t[0]`` and ``t[-1]`` are used.
        adaptive: ``True`` for adaptive step-size control, ``False``
            for fixed-step.  When ``None`` (default) adaptive mode
            is used automatically if ``error_weights`` is defined.
        inference_mode: When ``True``, wraps computation in
            ``torch.inference_mode()`` for faster execution without
            gradient tracking.

    Returns:
        Updated state dict ``{"x": x_final}``.
    """
    if inference_mode:
        with torch.inference_mode():
            return self.integrate(
                state, step_size, n_steps,
                drift=drift, t=t, adaptive=adaptive,
            )

    if adaptive is None:
        adaptive = self.error_weights is not None

    # fixed-step path
    if not adaptive:
        t_grid = self._build_time_grid(state["x"], step_size, n_steps, t)
        x = state["x"]
        drift_fn = self._resolve_drift(drift)
        n = t_grid.numel() - 1
        batch_size = x.size(0)
        dts = t_grid[1:] - t_grid[:-1]
        for i in range(n):
            t_batch = t_grid[i].expand(batch_size)
            x = self._deterministic_step(x, dts[i], drift_fn, t_batch)
        return {"x": x}

    # adaptive path
    if self.error_weights is None or self.order is None:
        raise ValueError(
            f"{type(self).__name__} does not define error_weights/order "
            f"and cannot be used with adaptive=True."
        )

    x = state["x"]
    drift_fn = self._resolve_drift(drift)

    if not torch.is_tensor(step_size):
        step_size = torch.tensor(
            step_size, device=x.device, dtype=x.dtype
        )

    if t is not None:
        if t.ndim != 1 or t.numel() < 2:
            raise ValueError("t must be a 1D tensor with length >= 2")
        t_start = t[0].item()
        t_end = t[-1].item()
    else:
        t_start = 0.0
        t_end = float(n_steps) * step_size.item()

    h = min(step_size.item(), t_end - t_start, self.max_step_size)
    x = self._adaptive_integrate(x, drift_fn, t_start, t_end, h)
    return {"x": x}
```

### `step(state, step_size, *, drift=None, t=None)`

Advance the state by one deterministic RK step.

Parameters:

| Name        | Type                                           | Description                             | Default    |
| ----------- | ---------------------------------------------- | --------------------------------------- | ---------- |
| `state`     | `Dict[str, Tensor]`                            | Mapping containing "x" position tensor. | *required* |
| `step_size` | `Tensor`                                       | Step size for the integration.          | *required* |
| `drift`     | `Optional[Callable[[Tensor, Tensor], Tensor]]` | Explicit drift callable f(x, t).        | `None`     |
| `t`         | `Optional[Tensor]`                             | Current time tensor (batch,).           | `None`     |

Returns:

| Type                | Description                      |
| ------------------- | -------------------------------- |
| `Dict[str, Tensor]` | Updated state dict {"x": x_new}. |

Source code in `torchebm/core/base_integrator.py`

```python
def step(
    self,
    state: Dict[str, torch.Tensor],
    step_size: torch.Tensor,
    *,
    drift: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    t: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    r"""Advance the state by one deterministic RK step.

    Args:
        state: Mapping containing ``"x"`` position tensor.
        step_size: Step size for the integration.
        drift: Explicit drift callable ``f(x, t)``.
        t: Current time tensor (batch,).

    Returns:
        Updated state dict ``{"x": x_new}``.
    """
    x = state["x"]
    if not torch.is_tensor(step_size):
        step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)
    if t is None:
        t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    drift_fn = self._resolve_drift(drift)
    x_new = self._deterministic_step(x, step_size, drift_fn, t)
    return {"x": x_new}
```

## `BaseSDERungeKuttaIntegrator`

Bases: `BaseRungeKuttaIntegrator`

Runge-Kutta integrator with additive SDE noise.

Extends `BaseRungeKuttaIntegrator` to solve Ito SDEs of the form

[ \\mathrm{d}x = f(x,t),\\mathrm{d}t + \\sqrt{2D(x,t)},\\mathrm{d}W_t ]

The stochastic term is applied as an Euler-order additive correction after the deterministic RK update:

\[ x\_{n+1} = \\underbrace{x_n + h \\sum\_{i} b_i,k_i}\_{\\text{RK update}}

- \\sqrt{2,D(x_n, t_n)},\\Delta W_n \]

Because the noise is added independently of the RK stages, the strong convergence order is (0.5) (Euler--Maruyama level) regardless of the underlying RK scheme order. The higher-order RK tableau improves only the deterministic component.

When `diffusion` is omitted the integrator reduces to its parent ODE behaviour.

Source code in `torchebm/core/base_integrator.py`

```python
class BaseSDERungeKuttaIntegrator(BaseRungeKuttaIntegrator):
    r"""Runge-Kutta integrator with additive SDE noise.

    Extends ``BaseRungeKuttaIntegrator`` to solve Ito SDEs of the form

    \[
    \mathrm{d}x = f(x,t)\,\mathrm{d}t + \sqrt{2D(x,t)}\,\mathrm{d}W_t
    \]

    The stochastic term is applied as an Euler-order additive correction
    after the deterministic RK update:

    \[
    x_{n+1} = \underbrace{x_n + h \sum_{i} b_i\,k_i}_{\text{RK update}}
              + \sqrt{2\,D(x_n, t_n)}\,\Delta W_n
    \]

    Because the noise is added independently of the RK stages, the strong
    convergence order is \(0.5\) (Euler--Maruyama level) regardless of the
    underlying RK scheme order.  The higher-order RK tableau improves only
    the deterministic component.

    When ``diffusion`` is omitted the integrator reduces to its parent
    ODE behaviour.
    """

    @staticmethod
    def _resolve_diffusion(
        diffusion: Optional[torch.Tensor],
        noise_scale: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        r"""Return the diffusion coefficient from an explicit value or ``noise_scale``.

        A scalar ``noise_scale`` is kept as a Python float (``noise_scale ** 2``)
        rather than wrapped in a device tensor, which would force a per-step
        host->device sync; the caller's arithmetic broadcasts it against the
        state tensor without one.
        """
        if diffusion is not None:
            return diffusion
        if noise_scale is not None:
            return noise_scale ** 2
        return None

    def step(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor,
        *,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        diffusion: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Advance the state by one RK step with optional SDE noise.

        The deterministic update uses the Butcher tableau defined by the
        subclass.  When a diffusion coefficient is provided, additive
        Wiener noise is appended at Euler--Maruyama order (strong order
        \(0.5\)).

        Args:
            state: Mapping containing ``"x"`` position tensor.
            step_size: Step size for the integration.
            drift: Explicit drift callable ``f(x, t)``.
            diffusion: Diffusion coefficient \(D(x, t)\) tensor.
            noise: Pre-sampled noise tensor.  When ``None``, standard
                normal noise is generated internally.
            noise_scale: Scalar whose square is used as \(D\) when
                ``diffusion`` is not given.
            t: Current time tensor (batch,).

        Returns:
            Updated state dict ``{"x": x_new}``.
        """
        x = state["x"]
        if t is None:
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        drift_fn = self._resolve_drift(drift)
        diffusion_val = self._resolve_diffusion(
            diffusion, noise_scale, x.device, x.dtype
        )

        x_new = self._deterministic_step(x, step_size, drift_fn, t)

        if diffusion_val is not None:
            if noise is None:
                noise = torch.randn_like(x, device=self.device, dtype=self.dtype)
            # step_size/diffusion_val stay Python floats to avoid a per-step host
            # sync; ** 0.5 accepts floats and tensors, torch.sqrt would not.
            dw = noise * (step_size**0.5)
            x_new = x_new + (2.0 * diffusion_val) ** 0.5 * dw

        return {"x": x_new}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor,
        n_steps: int,
        *,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        diffusion: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        noise_scale: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        adaptive: Optional[bool] = None,
        inference_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Integrate the state over a time interval (ODE or SDE).

        When ``diffusion`` or ``noise_scale`` is provided the integration
        uses fixed-step SDE mode.  Adaptive step-size control is available
        only for the ODE case (no diffusion).

        Args:
            state: Mapping with key ``"x"``.
            step_size: Step size (fixed) or initial step size (adaptive).
            n_steps: Number of integration steps.
            drift: Explicit drift callable ``f(x, t)``.
            diffusion: Time-dependent diffusion callable ``D(x, t)``.
            noise_scale: Scalar whose square is used as \(D\) when
                ``diffusion`` is not given.
            t: 1-D time grid.
            adaptive: ``True`` for adaptive (ODE only), ``False`` for fixed.
            inference_mode: When ``True``, wraps computation in
                ``torch.inference_mode()`` for faster execution without
                gradient tracking.

        Returns:
            Updated state dict ``{"x": x_final}``.
        """
        if inference_mode:
            with torch.inference_mode():
                return self.integrate(
                    state, step_size, n_steps,
                    drift=drift, diffusion=diffusion,
                    noise_scale=noise_scale, t=t, adaptive=adaptive,
                )

        if adaptive is None:
            adaptive = self.error_weights is not None

        if adaptive:
            if diffusion is not None or noise_scale is not None:
                raise ValueError(
                    "Adaptive stepping is only supported for ODEs. "
                    "Pass adaptive=False for SDE integration."
                )
            return super().integrate(
                state, step_size, n_steps,
                drift=drift, t=t, adaptive=True,
            )

        # fixed-step SDE/ODE path
        t_grid = self._build_time_grid(state["x"], step_size, n_steps, t)
        x = state["x"]
        drift_fn = self._resolve_drift(drift)
        has_diffusion_fn = diffusion is not None
        ns_const = self._resolve_diffusion(
            None, noise_scale, x.device, x.dtype
        ) if not has_diffusion_fn else None
        n = t_grid.numel() - 1
        batch_size = x.size(0)
        for i in range(n):
            dt = t_grid[i + 1] - t_grid[i]
            t_batch = t_grid[i].expand(batch_size)
            diff_val = diffusion(x, t_batch) if has_diffusion_fn else ns_const
            x = self._deterministic_step(x, dt, drift_fn, t_batch)
            if diff_val is not None:
                x = x + torch.sqrt(2.0 * diff_val) * torch.randn_like(x) * torch.sqrt(dt)
        return {"x": x}
```

### `integrate(state, step_size, n_steps, *, drift=None, diffusion=None, noise_scale=None, t=None, adaptive=None, inference_mode=False)`

Integrate the state over a time interval (ODE or SDE).

When `diffusion` or `noise_scale` is provided the integration uses fixed-step SDE mode. Adaptive step-size control is available only for the ODE case (no diffusion).

Parameters:

| Name             | Type                                           | Description                                                                                            | Default    |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------- |
| `state`          | `Dict[str, Tensor]`                            | Mapping with key "x".                                                                                  | *required* |
| `step_size`      | `Tensor`                                       | Step size (fixed) or initial step size (adaptive).                                                     | *required* |
| `n_steps`        | `int`                                          | Number of integration steps.                                                                           | *required* |
| `drift`          | `Optional[Callable[[Tensor, Tensor], Tensor]]` | Explicit drift callable f(x, t).                                                                       | `None`     |
| `diffusion`      | `Optional[Callable[[Tensor, Tensor], Tensor]]` | Time-dependent diffusion callable D(x, t).                                                             | `None`     |
| `noise_scale`    | `Optional[Tensor]`                             | Scalar whose square is used as (D) when diffusion is not given.                                        | `None`     |
| `t`              | `Optional[Tensor]`                             | 1-D time grid.                                                                                         | `None`     |
| `adaptive`       | `Optional[bool]`                               | True for adaptive (ODE only), False for fixed.                                                         | `None`     |
| `inference_mode` | `bool`                                         | When True, wraps computation in torch.inference_mode() for faster execution without gradient tracking. | `False`    |

Returns:

| Type                | Description                        |
| ------------------- | ---------------------------------- |
| `Dict[str, Tensor]` | Updated state dict {"x": x_final}. |

Source code in `torchebm/core/base_integrator.py`

```python
def integrate(
    self,
    state: Dict[str, torch.Tensor],
    step_size: torch.Tensor,
    n_steps: int,
    *,
    drift: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    diffusion: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    noise_scale: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    adaptive: Optional[bool] = None,
    inference_mode: bool = False,
) -> Dict[str, torch.Tensor]:
    r"""Integrate the state over a time interval (ODE or SDE).

    When ``diffusion`` or ``noise_scale`` is provided the integration
    uses fixed-step SDE mode.  Adaptive step-size control is available
    only for the ODE case (no diffusion).

    Args:
        state: Mapping with key ``"x"``.
        step_size: Step size (fixed) or initial step size (adaptive).
        n_steps: Number of integration steps.
        drift: Explicit drift callable ``f(x, t)``.
        diffusion: Time-dependent diffusion callable ``D(x, t)``.
        noise_scale: Scalar whose square is used as \(D\) when
            ``diffusion`` is not given.
        t: 1-D time grid.
        adaptive: ``True`` for adaptive (ODE only), ``False`` for fixed.
        inference_mode: When ``True``, wraps computation in
            ``torch.inference_mode()`` for faster execution without
            gradient tracking.

    Returns:
        Updated state dict ``{"x": x_final}``.
    """
    if inference_mode:
        with torch.inference_mode():
            return self.integrate(
                state, step_size, n_steps,
                drift=drift, diffusion=diffusion,
                noise_scale=noise_scale, t=t, adaptive=adaptive,
            )

    if adaptive is None:
        adaptive = self.error_weights is not None

    if adaptive:
        if diffusion is not None or noise_scale is not None:
            raise ValueError(
                "Adaptive stepping is only supported for ODEs. "
                "Pass adaptive=False for SDE integration."
            )
        return super().integrate(
            state, step_size, n_steps,
            drift=drift, t=t, adaptive=True,
        )

    # fixed-step SDE/ODE path
    t_grid = self._build_time_grid(state["x"], step_size, n_steps, t)
    x = state["x"]
    drift_fn = self._resolve_drift(drift)
    has_diffusion_fn = diffusion is not None
    ns_const = self._resolve_diffusion(
        None, noise_scale, x.device, x.dtype
    ) if not has_diffusion_fn else None
    n = t_grid.numel() - 1
    batch_size = x.size(0)
    for i in range(n):
        dt = t_grid[i + 1] - t_grid[i]
        t_batch = t_grid[i].expand(batch_size)
        diff_val = diffusion(x, t_batch) if has_diffusion_fn else ns_const
        x = self._deterministic_step(x, dt, drift_fn, t_batch)
        if diff_val is not None:
            x = x + torch.sqrt(2.0 * diff_val) * torch.randn_like(x) * torch.sqrt(dt)
    return {"x": x}
```

### `step(state, step_size, *, drift=None, diffusion=None, noise=None, noise_scale=None, t=None)`

Advance the state by one RK step with optional SDE noise.

The deterministic update uses the Butcher tableau defined by the subclass. When a diffusion coefficient is provided, additive Wiener noise is appended at Euler--Maruyama order (strong order (0.5)).

Parameters:

| Name          | Type                                           | Description                                                                         | Default    |
| ------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------- | ---------- |
| `state`       | `Dict[str, Tensor]`                            | Mapping containing "x" position tensor.                                             | *required* |
| `step_size`   | `Tensor`                                       | Step size for the integration.                                                      | *required* |
| `drift`       | `Optional[Callable[[Tensor, Tensor], Tensor]]` | Explicit drift callable f(x, t).                                                    | `None`     |
| `diffusion`   | `Optional[Tensor]`                             | Diffusion coefficient (D(x, t)) tensor.                                             | `None`     |
| `noise`       | `Optional[Tensor]`                             | Pre-sampled noise tensor. When None, standard normal noise is generated internally. | `None`     |
| `noise_scale` | `Optional[Tensor]`                             | Scalar whose square is used as (D) when diffusion is not given.                     | `None`     |
| `t`           | `Optional[Tensor]`                             | Current time tensor (batch,).                                                       | `None`     |

Returns:

| Type                | Description                      |
| ------------------- | -------------------------------- |
| `Dict[str, Tensor]` | Updated state dict {"x": x_new}. |

Source code in `torchebm/core/base_integrator.py`

```python
def step(
    self,
    state: Dict[str, torch.Tensor],
    step_size: torch.Tensor,
    *,
    drift: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    diffusion: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    noise_scale: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    r"""Advance the state by one RK step with optional SDE noise.

    The deterministic update uses the Butcher tableau defined by the
    subclass.  When a diffusion coefficient is provided, additive
    Wiener noise is appended at Euler--Maruyama order (strong order
    \(0.5\)).

    Args:
        state: Mapping containing ``"x"`` position tensor.
        step_size: Step size for the integration.
        drift: Explicit drift callable ``f(x, t)``.
        diffusion: Diffusion coefficient \(D(x, t)\) tensor.
        noise: Pre-sampled noise tensor.  When ``None``, standard
            normal noise is generated internally.
        noise_scale: Scalar whose square is used as \(D\) when
            ``diffusion`` is not given.
        t: Current time tensor (batch,).

    Returns:
        Updated state dict ``{"x": x_new}``.
    """
    x = state["x"]
    if t is None:
        t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    drift_fn = self._resolve_drift(drift)
    diffusion_val = self._resolve_diffusion(
        diffusion, noise_scale, x.device, x.dtype
    )

    x_new = self._deterministic_step(x, step_size, drift_fn, t)

    if diffusion_val is not None:
        if noise is None:
            noise = torch.randn_like(x, device=self.device, dtype=self.dtype)
        # step_size/diffusion_val stay Python floats to avoid a per-step host
        # sync; ** 0.5 accepts floats and tensors, torch.sqrt would not.
        dw = noise * (step_size**0.5)
        x_new = x_new + (2.0 * diffusion_val) ** 0.5 * dw

    return {"x": x_new}
```

## `BaseSampler`

Bases: `Schedulable`, `TorchEBMModule`, `ABC`

Abstract base class for samplers.

Parameters:

| Name     | Type                           | Description                                                                                                                               | Default    |
| -------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `model`  | `Module`                       | The model to sample from. For MCMC samplers, this is typically a BaseModel energy function; for learned samplers it may be any nn.Module. | *required* |
| `dtype`  | `dtype`                        | The data type for computations.                                                                                                           | `float32`  |
| `device` | `Optional[Union[str, device]]` | The device for computations.                                                                                                              | `None`     |

Sampling output contract

`sample(return_diagnostics=False)` → `Tensor`. `sample(return_diagnostics=True)` → `(Tensor, Dict[str, Tensor])` where the dict's keys are sampler-specific metric names and values have shape `[n_kept, ...]`. Standard keys (when produced):

- `"mean"` (`[n_kept, *data_shape]`): batch-mean of `x` at each kept step.
- `"var"` (`[n_kept, *data_shape]`): batch-variance of `x`.
- `"energy"` (`[n_kept]`): batch-mean energy at each kept step.
- `"acceptance_rate"` (`[n_kept]`): MH-acceptance fraction (HMC).

When `thin > 1`, `n_kept = n_steps // thin`. Otherwise `n_kept = n_steps`.

Each sampler's `sample()` docstring lists the keys it produces.

Source code in `torchebm/core/base_sampler.py`

```python
class BaseSampler(Schedulable, TorchEBMModule, ABC):
    r"""
    Abstract base class for samplers.

    Args:
        model (nn.Module): The model to sample from. For MCMC samplers, this is
            typically a `BaseModel` energy function; for learned samplers it may be
            any `nn.Module`.
        dtype (torch.dtype): The data type for computations.
        device (Optional[Union[str, torch.device]]): The device for computations.

    Sampling output contract:
        `sample(return_diagnostics=False)` → `Tensor`.
        `sample(return_diagnostics=True)` → `(Tensor, Dict[str, Tensor])` where
        the dict's keys are sampler-specific metric names and values have shape
        `[n_kept, ...]`. Standard keys (when produced):

        - ``"mean"`` (`[n_kept, *data_shape]`): batch-mean of `x` at each kept step.
        - ``"var"`` (`[n_kept, *data_shape]`): batch-variance of `x`.
        - ``"energy"`` (`[n_kept]`): batch-mean energy at each kept step.
        - ``"acceptance_rate"`` (`[n_kept]`): MH-acceptance fraction (HMC).

        When ``thin > 1``, ``n_kept = n_steps // thin``. Otherwise ``n_kept = n_steps``.

        Each sampler's `sample()` docstring lists the keys it produces.
    """

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.model = model

    def _init_state(
        self,
        x: Optional[torch.Tensor],
        dim: Optional[Union[int, Tuple[int, ...]]],
        n_samples: int,
    ) -> torch.Tensor:
        r"""Coerce `x` to the sampler's device/dtype, or draw from `N(0, I)`.

        Args:
            x: Initial state, or `None` to synthesize one.
            dim: State dimension (int) or shape (tuple), used when `x is None`.
            n_samples: Number of parallel chains, used when `x is None`.

        Returns:
            State tensor of shape `[n_samples, *shape]`.

        Raises:
            ValueError: If both `x` and `dim` are `None`.
        """
        if x is not None:
            return x.to(device=self.device, dtype=self.dtype)
        if dim is None:
            raise ValueError("dim must be provided when x is None")
        shape = (dim,) if isinstance(dim, int) else tuple(dim)
        return torch.randn(n_samples, *shape, dtype=self.dtype, device=self.device)

    def _model_gradient(
        self, x: torch.Tensor, model_kwargs: Dict[str, object]
    ) -> torch.Tensor:
        r"""Route a gradient call through the conditioning convention.

        This is the single back-compat chokepoint: when `model_kwargs` is empty
        the model is called exactly as before (``model.gradient(x)``), so
        analytic `gradient(self, x)` overrides with no `model_kwargs` parameter
        keep working. Pass an already-normalized dict (see
        `_prepare_model_kwargs`); it is reused per step with no re-normalization.
        """
        if model_kwargs:
            return self.model.gradient(x, model_kwargs=model_kwargs)
        return self.model.gradient(x)

    def _model_energy(
        self, x: torch.Tensor, model_kwargs: Dict[str, object]
    ) -> torch.Tensor:
        r"""Route an energy call through the conditioning convention.

        Mirrors `_model_gradient`: unconditional call when `model_kwargs` is
        empty, so unconditional scalar models (``forward(self, x)``) are
        untouched.
        """
        if model_kwargs:
            return self.model(x, **model_kwargs)
        return self.model(x)

    @abstractmethod
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        reset_schedulers: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""
        Run the sampling process.

        Args:
            x: Initial state. If `None`, samples from `N(0, I)`.
            dim: Dimension (int) or shape (tuple) of the state space, used when
                `x is None`. Samplers that can infer it from the model (the HMC
                family, via `model.mean`) accept `None`; otherwise a
                `ValueError` is raised.
            n_steps: Number of MCMC steps.
            n_samples: Number of parallel chains/samples.
            thin: Keep every `thin`-th sample. Final stored length is
                `n_steps // thin`. Must be `>= 1`.
            return_trajectory: If True, return the full kept trajectory of shape
                `[n_samples, n_steps // thin, *data_shape]` instead of the final
                state.
            return_diagnostics: If True, also return a `Dict[str, torch.Tensor]`
                of per-step metrics. See class docstring for the key contract.
            reset_schedulers: If True (default), reset registered schedulers
                before sampling so each call starts from step 0. Pass False for
                lifetime schedules driven by an outer training loop.

        Returns:
            Either a tensor of samples (or trajectory) of shape
            `[n_samples, *data_shape]` or `[n_samples, n_steps // thin, *data_shape]`,
            optionally paired with a diagnostics dict if `return_diagnostics=True`.

        Raises:
            ValueError: If `thin < 1`, or if `x` is `None` and `dim` is `None`
                for samplers that cannot infer the state shape.
        """
        raise NotImplementedError
```

### `sample(x=None, dim=None, n_steps=100, n_samples=1, thin=1, return_trajectory=False, return_diagnostics=False, reset_schedulers=True)`

Run the sampling process.

Parameters:

| Name                 | Type                                    | Description                                                                                                                                                                                         | Default |
| -------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `x`                  | `Optional[Tensor]`                      | Initial state. If None, samples from N(0, I).                                                                                                                                                       | `None`  |
| `dim`                | `Optional[Union[int, Tuple[int, ...]]]` | Dimension (int) or shape (tuple) of the state space, used when x is None. Samplers that can infer it from the model (the HMC family, via model.mean) accept None; otherwise a ValueError is raised. | `None`  |
| `n_steps`            | `int`                                   | Number of MCMC steps.                                                                                                                                                                               | `100`   |
| `n_samples`          | `int`                                   | Number of parallel chains/samples.                                                                                                                                                                  | `1`     |
| `thin`               | `int`                                   | Keep every thin-th sample. Final stored length is n_steps // thin. Must be >= 1.                                                                                                                    | `1`     |
| `return_trajectory`  | `bool`                                  | If True, return the full kept trajectory of shape [n_samples, n_steps // thin, \*data_shape] instead of the final state.                                                                            | `False` |
| `return_diagnostics` | `bool`                                  | If True, also return a Dict[str, torch.Tensor] of per-step metrics. See class docstring for the key contract.                                                                                       | `False` |
| `reset_schedulers`   | `bool`                                  | If True (default), reset registered schedulers before sampling so each call starts from step 0. Pass False for lifetime schedules driven by an outer training loop.                                 | `True`  |

Returns:

| Type                                              | Description                                                              |
| ------------------------------------------------- | ------------------------------------------------------------------------ |
| `Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]` | Either a tensor of samples (or trajectory) of shape                      |
| `Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]` | [n_samples, \*data_shape] or [n_samples, n_steps // thin, \*data_shape], |
| `Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]` | optionally paired with a diagnostics dict if return_diagnostics=True.    |

Raises:

| Type         | Description                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------- |
| `ValueError` | If thin < 1, or if x is None and dim is None for samplers that cannot infer the state shape. |

Source code in `torchebm/core/base_sampler.py`

```python
@abstractmethod
def sample(
    self,
    x: Optional[torch.Tensor] = None,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    n_steps: int = 100,
    n_samples: int = 1,
    thin: int = 1,
    return_trajectory: bool = False,
    return_diagnostics: bool = False,
    reset_schedulers: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    r"""
    Run the sampling process.

    Args:
        x: Initial state. If `None`, samples from `N(0, I)`.
        dim: Dimension (int) or shape (tuple) of the state space, used when
            `x is None`. Samplers that can infer it from the model (the HMC
            family, via `model.mean`) accept `None`; otherwise a
            `ValueError` is raised.
        n_steps: Number of MCMC steps.
        n_samples: Number of parallel chains/samples.
        thin: Keep every `thin`-th sample. Final stored length is
            `n_steps // thin`. Must be `>= 1`.
        return_trajectory: If True, return the full kept trajectory of shape
            `[n_samples, n_steps // thin, *data_shape]` instead of the final
            state.
        return_diagnostics: If True, also return a `Dict[str, torch.Tensor]`
            of per-step metrics. See class docstring for the key contract.
        reset_schedulers: If True (default), reset registered schedulers
            before sampling so each call starts from step 0. Pass False for
            lifetime schedules driven by an outer training loop.

    Returns:
        Either a tensor of samples (or trajectory) of shape
        `[n_samples, *data_shape]` or `[n_samples, n_steps // thin, *data_shape]`,
        optionally paired with a diagnostics dict if `return_diagnostics=True`.

    Raises:
        ValueError: If `thin < 1`, or if `x` is `None` and `dim` is `None`
            for samplers that cannot infer the state shape.
    """
    raise NotImplementedError
```

## `BaseScheduler`

Bases: `ABC`

Abstract base class for parameter schedulers.

This class provides the foundation for all parameter scheduling strategies in TorchEBM. Schedulers are used to dynamically adjust parameters such as step sizes, noise scales, learning rates, and other hyperparameters during training or sampling processes.

The scheduler maintains an internal step counter and computes parameter values based on the current step. Subclasses must implement the `_compute_value` method to define the specific scheduling strategy.

Mathematical Foundation

A scheduler defines a function (f: \\mathbb{N} \\to \\mathbb{R}) that maps step numbers to parameter values:

[v(t) = f(t)]

where (t) is the current step count and (v(t)) is the parameter value at step (t).

Parameters:

| Name          | Type    | Description                        | Default    |
| ------------- | ------- | ---------------------------------- | ---------- |
| `start_value` | `float` | Initial parameter value at step 0. | *required* |

Attributes:

| Name            | Type    | Description                                               |
| --------------- | ------- | --------------------------------------------------------- |
| `start_value`   | `float` | The initial parameter value.                              |
| `current_value` | `float` | The current parameter value.                              |
| `step_count`    | `int`   | Number of steps taken since initialization or last reset. |

Creating a Custom Scheduler

```python
class CustomScheduler(BaseScheduler):
    def __init__(self, start_value: float, factor: float):
        super().__init__(start_value)
        self.factor = factor

    def _compute_value(self) -> float:
        return self.start_value * (self.factor ** self.step_count)

scheduler = CustomScheduler(start_value=1.0, factor=0.9)
for i in range(5):
    value = scheduler.step()
    print(f"Step {i+1}: {value:.4f}")
```

State Management

```python
scheduler = ExponentialDecayScheduler(start_value=0.1, decay_rate=0.95)
# Take some steps
for _ in range(10):
    scheduler.step()

# Save state
state = scheduler.state_dict()

# Reset and restore
scheduler.reset()
scheduler.load_state_dict(state)
```

Source code in `torchebm/core/base_scheduler.py`

````python
class BaseScheduler(ABC):
    r"""
    Abstract base class for parameter schedulers.

    This class provides the foundation for all parameter scheduling strategies in TorchEBM.
    Schedulers are used to dynamically adjust parameters such as step sizes, noise scales,
    learning rates, and other hyperparameters during training or sampling processes.

    The scheduler maintains an internal step counter and computes parameter values based
    on the current step. Subclasses must implement the `_compute_value` method to define
    the specific scheduling strategy.

    !!! info "Mathematical Foundation"
        A scheduler defines a function \(f: \mathbb{N} \to \mathbb{R}\) that maps step numbers to parameter values:

        $$v(t) = f(t)$$

        where \(t\) is the current step count and \(v(t)\) is the parameter value at step \(t\).

    Args:
        start_value (float): Initial parameter value at step 0.

    Attributes:
        start_value (float): The initial parameter value.
        current_value (float): The current parameter value.
        step_count (int): Number of steps taken since initialization or last reset.

    !!! example "Creating a Custom Scheduler"
        ```python
        class CustomScheduler(BaseScheduler):
            def __init__(self, start_value: float, factor: float):
                super().__init__(start_value)
                self.factor = factor

            def _compute_value(self) -> float:
                return self.start_value * (self.factor ** self.step_count)

        scheduler = CustomScheduler(start_value=1.0, factor=0.9)
        for i in range(5):
            value = scheduler.step()
            print(f"Step {i+1}: {value:.4f}")
        ```

    !!! tip "State Management"
        ```python
        scheduler = ExponentialDecayScheduler(start_value=0.1, decay_rate=0.95)
        # Take some steps
        for _ in range(10):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()

        # Reset and restore
        scheduler.reset()
        scheduler.load_state_dict(state)
        ```
    """

    def __init__(self, start_value: float):
        r"""
        Initialize the base scheduler.

        Args:
            start_value (float): Initial parameter value. Must be a finite number.

        Raises:
            TypeError: If start_value is not a float or int.
        """
        if not isinstance(start_value, (float, int)):
            raise TypeError(
                f"{type(self).__name__} received an invalid start_value of type "
                f"{type(start_value).__name__}. Expected float or int."
            )

        self.start_value = float(start_value)
        self.current_value = self.start_value
        self.step_count = 0

    @abstractmethod
    def _compute_value(self) -> float:
        r"""
        Compute the parameter value for the current step count.

        This method must be implemented by subclasses to define the specific
        scheduling strategy. It should return the parameter value based on
        the current `self.step_count`.

        Returns:
            float: The computed parameter value for the current step.

        !!! warning "Implementation Note"
            This method is called internally by `step()` after incrementing
            the step counter. Subclasses should not call this method directly.
        """
        pass

    def step(self) -> float:
        r"""
        Advance the scheduler by one step and return the new parameter value.

        This method increments the internal step counter and computes the new
        parameter value using the scheduler's strategy. The computed value
        becomes the new current value.

        Returns:
            float: The new parameter value after stepping.

        !!! example "Basic Usage"
            ```python
            scheduler = ExponentialDecayScheduler(start_value=1.0, decay_rate=0.9)
            print(f"Initial: {scheduler.get_value()}")  # 1.0
            print(f"Step 1: {scheduler.step()}")        # 0.9
            print(f"Step 2: {scheduler.step()}")        # 0.81
            ```
        """
        self.step_count += 1
        self.current_value = self._compute_value()
        return self.current_value

    def reset(self) -> None:
        r"""
        Reset the scheduler to its initial state.

        This method resets both the step counter and current value to their
        initial states, effectively restarting the scheduling process.

        !!! example "Reset Example"
            ```python
            scheduler = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=10)
            for _ in range(5):
                scheduler.step()
            print(f"Before reset: step={scheduler.step_count}, value={scheduler.current_value}")
            scheduler.reset()
            print(f"After reset: step={scheduler.step_count}, value={scheduler.current_value}")
            ```
        """
        self.current_value = self.start_value
        self.step_count = 0

    def get_value(self) -> float:
        r"""
        Get the current parameter value without advancing the scheduler.

        This method returns the current parameter value without modifying
        the scheduler's internal state. Use this when you need to query
        the current value without stepping.

        Returns:
            float: The current parameter value.

        !!! example "Query Current Value"
            ```python
            scheduler = ConstantScheduler(start_value=0.5)
            print(scheduler.get_value())  # 0.5
            scheduler.step()
            print(scheduler.get_value())  # 0.5 (still constant)
            ```
        """
        return self.current_value

    def state_dict(self) -> Dict[str, Any]:
        r"""
        Return the state of the scheduler as a dictionary.

        This method returns a dictionary containing all the scheduler's internal
        state, which can be used to save and restore the scheduler's state.

        Returns:
            Dict[str, Any]: Dictionary containing the scheduler's state.

        !!! example "State Management"
            ```python
            scheduler = CosineScheduler(start_value=1.0, end_value=0.0, n_steps=100)
            for _ in range(50):
                scheduler.step()
            state = scheduler.state_dict()
            print(state['step_count'])  # 50
            ```
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Load the scheduler's state from a dictionary.

        This method restores the scheduler's internal state from a dictionary
        previously created by `state_dict()`. This is useful for resuming
        training or sampling from a checkpoint.

        Args:
            state_dict (Dict[str, Any]): Dictionary containing the scheduler state.
                Should be an object returned from a call to `state_dict()`.

        !!! example "State Restoration"
            ```python
            scheduler1 = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=100)
            for _ in range(25):
                scheduler1.step()
            state = scheduler1.state_dict()

            scheduler2 = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=100)
            scheduler2.load_state_dict(state)
            print(scheduler2.step_count)  # 25
            ```
        """
        self.__dict__.update(state_dict)
````

### `__init__(start_value)`

Initialize the base scheduler.

Parameters:

| Name          | Type    | Description                                       | Default    |
| ------------- | ------- | ------------------------------------------------- | ---------- |
| `start_value` | `float` | Initial parameter value. Must be a finite number. | *required* |

Raises:

| Type        | Description                           |
| ----------- | ------------------------------------- |
| `TypeError` | If start_value is not a float or int. |

Source code in `torchebm/core/base_scheduler.py`

```python
def __init__(self, start_value: float):
    r"""
    Initialize the base scheduler.

    Args:
        start_value (float): Initial parameter value. Must be a finite number.

    Raises:
        TypeError: If start_value is not a float or int.
    """
    if not isinstance(start_value, (float, int)):
        raise TypeError(
            f"{type(self).__name__} received an invalid start_value of type "
            f"{type(start_value).__name__}. Expected float or int."
        )

    self.start_value = float(start_value)
    self.current_value = self.start_value
    self.step_count = 0
```

### `get_value()`

Get the current parameter value without advancing the scheduler.

This method returns the current parameter value without modifying the scheduler's internal state. Use this when you need to query the current value without stepping.

Returns:

| Name    | Type    | Description                  |
| ------- | ------- | ---------------------------- |
| `float` | `float` | The current parameter value. |

Query Current Value

```python
scheduler = ConstantScheduler(start_value=0.5)
print(scheduler.get_value())  # 0.5
scheduler.step()
print(scheduler.get_value())  # 0.5 (still constant)
```

Source code in `torchebm/core/base_scheduler.py`

````python
def get_value(self) -> float:
    r"""
    Get the current parameter value without advancing the scheduler.

    This method returns the current parameter value without modifying
    the scheduler's internal state. Use this when you need to query
    the current value without stepping.

    Returns:
        float: The current parameter value.

    !!! example "Query Current Value"
        ```python
        scheduler = ConstantScheduler(start_value=0.5)
        print(scheduler.get_value())  # 0.5
        scheduler.step()
        print(scheduler.get_value())  # 0.5 (still constant)
        ```
    """
    return self.current_value
````

### `load_state_dict(state_dict)`

Load the scheduler's state from a dictionary.

This method restores the scheduler's internal state from a dictionary previously created by `state_dict()`. This is useful for resuming training or sampling from a checkpoint.

Parameters:

| Name         | Type             | Description                                                                                          | Default    |
| ------------ | ---------------- | ---------------------------------------------------------------------------------------------------- | ---------- |
| `state_dict` | `Dict[str, Any]` | Dictionary containing the scheduler state. Should be an object returned from a call to state_dict(). | *required* |

State Restoration

```python
scheduler1 = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=100)
for _ in range(25):
    scheduler1.step()
state = scheduler1.state_dict()

scheduler2 = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=100)
scheduler2.load_state_dict(state)
print(scheduler2.step_count)  # 25
```

Source code in `torchebm/core/base_scheduler.py`

````python
def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    r"""
    Load the scheduler's state from a dictionary.

    This method restores the scheduler's internal state from a dictionary
    previously created by `state_dict()`. This is useful for resuming
    training or sampling from a checkpoint.

    Args:
        state_dict (Dict[str, Any]): Dictionary containing the scheduler state.
            Should be an object returned from a call to `state_dict()`.

    !!! example "State Restoration"
        ```python
        scheduler1 = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=100)
        for _ in range(25):
            scheduler1.step()
        state = scheduler1.state_dict()

        scheduler2 = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=100)
        scheduler2.load_state_dict(state)
        print(scheduler2.step_count)  # 25
        ```
    """
    self.__dict__.update(state_dict)
````

### `reset()`

Reset the scheduler to its initial state.

This method resets both the step counter and current value to their initial states, effectively restarting the scheduling process.

Reset Example

```python
scheduler = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=10)
for _ in range(5):
    scheduler.step()
print(f"Before reset: step={scheduler.step_count}, value={scheduler.current_value}")
scheduler.reset()
print(f"After reset: step={scheduler.step_count}, value={scheduler.current_value}")
```

Source code in `torchebm/core/base_scheduler.py`

````python
def reset(self) -> None:
    r"""
    Reset the scheduler to its initial state.

    This method resets both the step counter and current value to their
    initial states, effectively restarting the scheduling process.

    !!! example "Reset Example"
        ```python
        scheduler = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=10)
        for _ in range(5):
            scheduler.step()
        print(f"Before reset: step={scheduler.step_count}, value={scheduler.current_value}")
        scheduler.reset()
        print(f"After reset: step={scheduler.step_count}, value={scheduler.current_value}")
        ```
    """
    self.current_value = self.start_value
    self.step_count = 0
````

### `state_dict()`

Return the state of the scheduler as a dictionary.

This method returns a dictionary containing all the scheduler's internal state, which can be used to save and restore the scheduler's state.

Returns:

| Type             | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| `Dict[str, Any]` | Dict\[str, Any\]: Dictionary containing the scheduler's state. |

State Management

```python
scheduler = CosineScheduler(start_value=1.0, end_value=0.0, n_steps=100)
for _ in range(50):
    scheduler.step()
state = scheduler.state_dict()
print(state['step_count'])  # 50
```

Source code in `torchebm/core/base_scheduler.py`

````python
def state_dict(self) -> Dict[str, Any]:
    r"""
    Return the state of the scheduler as a dictionary.

    This method returns a dictionary containing all the scheduler's internal
    state, which can be used to save and restore the scheduler's state.

    Returns:
        Dict[str, Any]: Dictionary containing the scheduler's state.

    !!! example "State Management"
        ```python
        scheduler = CosineScheduler(start_value=1.0, end_value=0.0, n_steps=100)
        for _ in range(50):
            scheduler.step()
        state = scheduler.state_dict()
        print(state['step_count'])  # 50
        ```
    """
    return {key: value for key, value in self.__dict__.items()}
````

### `step()`

Advance the scheduler by one step and return the new parameter value.

This method increments the internal step counter and computes the new parameter value using the scheduler's strategy. The computed value becomes the new current value.

Returns:

| Name    | Type    | Description                             |
| ------- | ------- | --------------------------------------- |
| `float` | `float` | The new parameter value after stepping. |

Basic Usage

```python
scheduler = ExponentialDecayScheduler(start_value=1.0, decay_rate=0.9)
print(f"Initial: {scheduler.get_value()}")  # 1.0
print(f"Step 1: {scheduler.step()}")        # 0.9
print(f"Step 2: {scheduler.step()}")        # 0.81
```

Source code in `torchebm/core/base_scheduler.py`

````python
def step(self) -> float:
    r"""
    Advance the scheduler by one step and return the new parameter value.

    This method increments the internal step counter and computes the new
    parameter value using the scheduler's strategy. The computed value
    becomes the new current value.

    Returns:
        float: The new parameter value after stepping.

    !!! example "Basic Usage"
        ```python
        scheduler = ExponentialDecayScheduler(start_value=1.0, decay_rate=0.9)
        print(f"Initial: {scheduler.get_value()}")  # 1.0
        print(f"Step 1: {scheduler.step()}")        # 0.9
        print(f"Step 2: {scheduler.step()}")        # 0.81
        ```
    """
    self.step_count += 1
    self.current_value = self._compute_value()
    return self.current_value
````

## `BaseScoreMatching`

Bases: `BaseLoss`

Abstract base class for Score Matching based loss functions.

Parameters:

| Name                      | Type                 | Description                                                | Default    |
| ------------------------- | -------------------- | ---------------------------------------------------------- | ---------- |
| `model`                   | `BaseModel`          | The energy-based model to be trained.                      | *required* |
| `noise_scale`             | `float`              | The scale of noise for perturbation in denoising variants. | `0.01`     |
| `regularization_strength` | `float`              | The coefficient for regularization terms.                  | `0.0`      |
| `use_autograd`            | `bool`               | Whether to use torch.autograd for computing derivatives.   | `True`     |
| `hutchinson_samples`      | `int`                | The number of random samples for Hutchinson's trick.       | `1`        |
| `custom_regularization`   | `Optional[Callable]` | An optional function for custom regularization.            | `None`     |

Source code in `torchebm/core/base_loss.py`

```python
class BaseScoreMatching(BaseLoss):
    """
    Abstract base class for Score Matching based loss functions.

    Args:
        model (BaseModel): The energy-based model to be trained.
        noise_scale (float): The scale of noise for perturbation in denoising variants.
        regularization_strength (float): The coefficient for regularization terms.
        use_autograd (bool): Whether to use `torch.autograd` for computing derivatives.
        hutchinson_samples (int): The number of random samples for Hutchinson's trick.
        custom_regularization (Optional[Callable]): An optional function for custom regularization.
    """

    def __init__(
        self,
        model: BaseModel,
        noise_scale: Union[float, BaseScheduler] = 0.01,
        regularization_strength: Union[float, BaseScheduler] = 0.0,
        use_autograd: bool = True,
        hutchinson_samples: int = 1,
        custom_regularization: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self._register_param("noise_scale", noise_scale)
        self._register_param("regularization_strength", regularization_strength)
        self.use_autograd = use_autograd
        self.hutchinson_samples = hutchinson_samples
        self.custom_regularization = custom_regularization

    @property
    def noise_scale(self) -> float:
        return self.get_scheduled_value("noise_scale")

    @noise_scale.setter
    def noise_scale(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("noise_scale", value)

    @property
    def regularization_strength(self) -> float:
        return self.get_scheduled_value("regularization_strength")

    @regularization_strength.setter
    def regularization_strength(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("regularization_strength", value)

    def compute_score(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        r"""
        Computes the score function, \(\nabla_x E(x)\).

        Args:
            x (torch.Tensor): The input data tensor.
            noise (Optional[torch.Tensor]): Optional noise tensor for perturbed variants.
            model_kwargs (Optional[dict]): Conditioning arguments forwarded to the
                model (e.g. class labels). This is the single funnel every
                score-matching variant routes its model call through, so passing
                it here conditions all variants.

        Returns:
            torch.Tensor: The score function evaluated at `x` or `x + noise`.
        """

        x = x.to(device=self.device, dtype=self.dtype)

        if noise is not None:
            noise = noise.to(device=self.device, dtype=self.dtype)
            x_perturbed = x + noise
        else:
            x_perturbed = x

        if not x_perturbed.requires_grad:
            x_perturbed.requires_grad_(True)

        with self.autocast_context():
            energy = self.model(x_perturbed, **(model_kwargs or {}))

        if self.use_autograd:
            score = torch.autograd.grad(energy.sum(), x_perturbed, create_graph=True)[0]
        else:
            raise NotImplementedError(
                "Custom gradient computation must be implemented in subclasses"
            )

        return score

    def perturb_data(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # todo: add more noise types
        """
        Perturbs the input data with Gaussian noise for denoising variants.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the perturbed data
                and the noise that was added.
        """

        x = x.to(device=self.device, dtype=self.dtype)
        noise = (
            torch.randn_like(x, device=self.device, dtype=self.dtype) * self.noise_scale
        )
        x_perturbed = x + noise
        return x_perturbed, noise

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the score matching loss given input data.

        Args:
            x (torch.Tensor): Input data tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed score matching loss.
        """
        pass

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the specific score matching loss variant.

        Args:
            x (torch.Tensor): Input data tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The specific score matching loss.
        """
        pass

    def add_regularization(
        self,
        loss: torch.Tensor,
        x: torch.Tensor,
        custom_reg_fn: Optional[Callable] = None,
        reg_strength: Optional[float] = None,
        model_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Adds regularization terms to the loss.

        Args:
            loss (torch.Tensor): The current loss value.
            x (torch.Tensor): The input tensor.
            custom_reg_fn (Optional[Callable]): An optional custom regularization function.
            reg_strength (Optional[float]): An optional regularization strength.

        Returns:
            torch.Tensor: The loss with the regularization term added.
        """
        strength = (
            reg_strength if reg_strength is not None else self.regularization_strength
        )

        if strength <= 0:
            return loss

        if custom_reg_fn is not None:
            reg_term = custom_reg_fn(x, self.model)

        elif self.custom_regularization is not None:
            reg_term = self.custom_regularization(x, self.model)
        # default: L2 norm of score
        else:
            score = self.compute_score(x, model_kwargs=model_kwargs)
            reg_term = score.square().sum(dim=list(range(1, len(x.shape)))).mean()

        return loss + strength * reg_term

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}(model={self.model})"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()
```

### `__repr__()`

Return a string representation of the loss function.

Source code in `torchebm/core/base_loss.py`

```python
def __repr__(self):
    """Return a string representation of the loss function."""
    return f"{self.__class__.__name__}(model={self.model})"
```

### `__str__()`

Return a string representation of the loss function.

Source code in `torchebm/core/base_loss.py`

```python
def __str__(self):
    """Return a string representation of the loss function."""
    return self.__repr__()
```

### `add_regularization(loss, x, custom_reg_fn=None, reg_strength=None, model_kwargs=None)`

Adds regularization terms to the loss.

Parameters:

| Name            | Type                 | Description                                 | Default    |
| --------------- | -------------------- | ------------------------------------------- | ---------- |
| `loss`          | `Tensor`             | The current loss value.                     | *required* |
| `x`             | `Tensor`             | The input tensor.                           | *required* |
| `custom_reg_fn` | `Optional[Callable]` | An optional custom regularization function. | `None`     |
| `reg_strength`  | `Optional[float]`    | An optional regularization strength.        | `None`     |

Returns:

| Type     | Description                                                |
| -------- | ---------------------------------------------------------- |
| `Tensor` | torch.Tensor: The loss with the regularization term added. |

Source code in `torchebm/core/base_loss.py`

```python
def add_regularization(
    self,
    loss: torch.Tensor,
    x: torch.Tensor,
    custom_reg_fn: Optional[Callable] = None,
    reg_strength: Optional[float] = None,
    model_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Adds regularization terms to the loss.

    Args:
        loss (torch.Tensor): The current loss value.
        x (torch.Tensor): The input tensor.
        custom_reg_fn (Optional[Callable]): An optional custom regularization function.
        reg_strength (Optional[float]): An optional regularization strength.

    Returns:
        torch.Tensor: The loss with the regularization term added.
    """
    strength = (
        reg_strength if reg_strength is not None else self.regularization_strength
    )

    if strength <= 0:
        return loss

    if custom_reg_fn is not None:
        reg_term = custom_reg_fn(x, self.model)

    elif self.custom_regularization is not None:
        reg_term = self.custom_regularization(x, self.model)
    # default: L2 norm of score
    else:
        score = self.compute_score(x, model_kwargs=model_kwargs)
        reg_term = score.square().sum(dim=list(range(1, len(x.shape)))).mean()

    return loss + strength * reg_term
```

### `compute_loss(x, *args, **kwargs)`

Computes the specific score matching loss variant.

Parameters:

| Name       | Type     | Description                      | Default    |
| ---------- | -------- | -------------------------------- | ---------- |
| `x`        | `Tensor` | Input data tensor.               | *required* |
| `*args`    |          | Additional positional arguments. | `()`       |
| `**kwargs` |          | Additional keyword arguments.    | `{}`       |

Returns:

| Type     | Description                                     |
| -------- | ----------------------------------------------- |
| `Tensor` | torch.Tensor: The specific score matching loss. |

Source code in `torchebm/core/base_loss.py`

```python
@abstractmethod
def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Computes the specific score matching loss variant.

    Args:
        x (torch.Tensor): Input data tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The specific score matching loss.
    """
    pass
```

### `compute_score(x, noise=None, model_kwargs=None)`

Computes the score function, (\\nabla_x E(x)).

Parameters:

| Name           | Type               | Description                                                                                                                                                                                          | Default    |
| -------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `x`            | `Tensor`           | The input data tensor.                                                                                                                                                                               | *required* |
| `noise`        | `Optional[Tensor]` | Optional noise tensor for perturbed variants.                                                                                                                                                        | `None`     |
| `model_kwargs` | `Optional[dict]`   | Conditioning arguments forwarded to the model (e.g. class labels). This is the single funnel every score-matching variant routes its model call through, so passing it here conditions all variants. | `None`     |

Returns:

| Type     | Description                                                   |
| -------- | ------------------------------------------------------------- |
| `Tensor` | torch.Tensor: The score function evaluated at x or x + noise. |

Source code in `torchebm/core/base_loss.py`

```python
def compute_score(
    self,
    x: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    model_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    r"""
    Computes the score function, \(\nabla_x E(x)\).

    Args:
        x (torch.Tensor): The input data tensor.
        noise (Optional[torch.Tensor]): Optional noise tensor for perturbed variants.
        model_kwargs (Optional[dict]): Conditioning arguments forwarded to the
            model (e.g. class labels). This is the single funnel every
            score-matching variant routes its model call through, so passing
            it here conditions all variants.

    Returns:
        torch.Tensor: The score function evaluated at `x` or `x + noise`.
    """

    x = x.to(device=self.device, dtype=self.dtype)

    if noise is not None:
        noise = noise.to(device=self.device, dtype=self.dtype)
        x_perturbed = x + noise
    else:
        x_perturbed = x

    if not x_perturbed.requires_grad:
        x_perturbed.requires_grad_(True)

    with self.autocast_context():
        energy = self.model(x_perturbed, **(model_kwargs or {}))

    if self.use_autograd:
        score = torch.autograd.grad(energy.sum(), x_perturbed, create_graph=True)[0]
    else:
        raise NotImplementedError(
            "Custom gradient computation must be implemented in subclasses"
        )

    return score
```

### `forward(x, *args, **kwargs)`

Computes the score matching loss given input data.

Parameters:

| Name       | Type     | Description                      | Default    |
| ---------- | -------- | -------------------------------- | ---------- |
| `x`        | `Tensor` | Input data tensor.               | *required* |
| `*args`    |          | Additional positional arguments. | `()`       |
| `**kwargs` |          | Additional keyword arguments.    | `{}`       |

Returns:

| Type     | Description                                     |
| -------- | ----------------------------------------------- |
| `Tensor` | torch.Tensor: The computed score matching loss. |

Source code in `torchebm/core/base_loss.py`

```python
@abstractmethod
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Computes the score matching loss given input data.

    Args:
        x (torch.Tensor): Input data tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The computed score matching loss.
    """
    pass
```

### `perturb_data(x)`

Perturbs the input data with Gaussian noise for denoising variants.

Parameters:

| Name | Type     | Description        | Default    |
| ---- | -------- | ------------------ | ---------- |
| `x`  | `Tensor` | Input data tensor. | *required* |

Returns:

| Type                    | Description                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| `Tuple[Tensor, Tensor]` | Tuple\[torch.Tensor, torch.Tensor\]: A tuple containing the perturbed data and the noise that was added. |

Source code in `torchebm/core/base_loss.py`

```python
def perturb_data(
    self, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:  # todo: add more noise types
    """
    Perturbs the input data with Gaussian noise for denoising variants.

    Args:
        x (torch.Tensor): Input data tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the perturbed data
            and the noise that was added.
    """

    x = x.to(device=self.device, dtype=self.dtype)
    noise = (
        torch.randn_like(x, device=self.device, dtype=self.dtype) * self.noise_scale
    )
    x_perturbed = x + noise
    return x_perturbed, noise
```

## `BaseSymplecticIntegrator`

Bases: `BaseIntegrator`

Base class for symplectic integrators of Hamiltonian dynamics.

Symplectic integrators advance a phase-space state dict with keys `"x"` (position) and `"p"` (momentum) while preserving the symplectic form, which makes them time-reversible and suitable for Metropolis-Hastings proposals (HMC family). Two callback contracts exist, distinguished by the `separable` class attribute:

- `separable = True`: separable Hamiltonians ( H(x, p) = U(x) + K(p) ). Subclasses take a `drift(x, t)` callable (( = -\\nabla_x U )) and an optional `mass` term (`LeapfrogIntegrator`).
- `separable = False`: general non-separable Hamiltonians. Subclasses take `force(x, p, t)` and `velocity(x, p, t)` callables (`GeneralisedLeapfrogIntegrator`).

Subclass `integrate` implementations must honor `inference_mode=True` by re-entering themselves under `torch.inference_mode()`.

Attributes:

| Name          | Type    | Description                                             |
| ------------- | ------- | ------------------------------------------------------- |
| `separable`   | `bool`  | Whether the integrator assumes a separable Hamiltonian. |
| `_SAFE_CLAMP` | `float` | Magnitude bound applied to forces/state in safe mode.   |

Source code in `torchebm/core/base_integrator.py`

```python
class BaseSymplecticIntegrator(BaseIntegrator):
    r"""Base class for symplectic integrators of Hamiltonian dynamics.

    Symplectic integrators advance a phase-space state dict with keys
    ``"x"`` (position) and ``"p"`` (momentum) while preserving the
    symplectic form, which makes them time-reversible and suitable for
    Metropolis-Hastings proposals (HMC family). Two callback contracts
    exist, distinguished by the ``separable`` class attribute:

    - ``separable = True``: separable Hamiltonians
      \( H(x, p) = U(x) + K(p) \). Subclasses take a ``drift(x, t)``
      callable (\( = -\nabla_x U \)) and an optional ``mass`` term
      (`LeapfrogIntegrator`).
    - ``separable = False``: general non-separable Hamiltonians.
      Subclasses take ``force(x, p, t)`` and ``velocity(x, p, t)``
      callables (`GeneralisedLeapfrogIntegrator`).

    Subclass ``integrate`` implementations must honor
    ``inference_mode=True`` by re-entering themselves under
    ``torch.inference_mode()``.

    Attributes:
        separable: Whether the integrator assumes a separable Hamiltonian.
        _SAFE_CLAMP: Magnitude bound applied to forces/state in safe mode.
    """

    separable: bool = True
    _SAFE_CLAMP: float = 1e6

    @staticmethod
    def _validate_n_steps(n_steps: int) -> None:
        r"""Raise if ``n_steps`` is not a positive integer count."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

    @staticmethod
    def _unpack_state(
        state: Dict[str, torch.Tensor],
        step_size,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Extract ``(x, p)``, tensorize ``step_size``, build batch time.

        Returns:
            Tuple ``(x, p, step_size, t)`` with ``step_size`` on
            ``x.device``/``x.dtype`` and ``t`` a zero tensor of shape
            ``(batch,)`` (autonomous Hamiltonians; ``t`` exists for
            callback signature uniformity).
        """
        x = state["x"]
        p = state["p"]
        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)
        t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        return x, p, step_size, t

    def _safe_clamp_(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""In-place magnitude clamp used by safe mode."""
        return tensor.clamp_(min=-self._SAFE_CLAMP, max=self._SAFE_CLAMP)

    @staticmethod
    def _sanitize_state_(x: torch.Tensor, p: torch.Tensor) -> None:
        r"""Replace NaNs by zeros in-place on freshly owned tensors.

        Unconditional ``nan_to_num_`` is idempotent on clean tensors and
        avoids the per-step CPU sync that
        ``isnan(x).any() or isnan(p).any()`` would force (Python ``or``
        materialises both 0-d bools to host).
        """
        x.nan_to_num_(nan=0.0)
        p.nan_to_num_(nan=0.0)
```

## `ConstantScheduler`

Bases: `BaseScheduler`

Scheduler that maintains a constant parameter value.

This scheduler returns the same value at every step, effectively providing no scheduling. It's useful as a baseline or when you want to disable scheduling for certain parameters while keeping the scheduler interface.

Mathematical Formula

[v(t) = v_0 \\text{ for all } t \\geq 0]

where (v_0) is the start_value.

Parameters:

| Name          | Type    | Description                     | Default    |
| ------------- | ------- | ------------------------------- | ---------- |
| `start_value` | `float` | The constant value to maintain. | *required* |

Basic Usage

```python
scheduler = ConstantScheduler(start_value=0.01)
for i in range(5):
    value = scheduler.step()
    print(f"Step {i+1}: {value}")  # Always prints 0.01
```

Using with Samplers

```python
from torchebm.samplers import LangevinDynamics
constant_step = ConstantScheduler(start_value=0.05)
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=constant_step,
    noise_scale=0.1
)
```

Source code in `torchebm/core/base_scheduler.py`

````python
class ConstantScheduler(BaseScheduler):
    r"""
    Scheduler that maintains a constant parameter value.

    This scheduler returns the same value at every step, effectively providing
    no scheduling. It's useful as a baseline or when you want to disable
    scheduling for certain parameters while keeping the scheduler interface.

    !!! info "Mathematical Formula"
        $$v(t) = v_0 \text{ for all } t \geq 0$$

        where \(v_0\) is the start_value.

    Args:
        start_value (float): The constant value to maintain.

    !!! example "Basic Usage"
        ```python
        scheduler = ConstantScheduler(start_value=0.01)
        for i in range(5):
            value = scheduler.step()
            print(f"Step {i+1}: {value}")  # Always prints 0.01
        ```

    !!! tip "Using with Samplers"
        ```python
        from torchebm.samplers import LangevinDynamics
        constant_step = ConstantScheduler(start_value=0.05)
        sampler = LangevinDynamics(
            energy_function=energy_fn,
            step_size=constant_step,
            noise_scale=0.1
        )
        ```
    """

    def _compute_value(self) -> float:
        r"""
        Return the constant value.

        Returns:
            float: The constant start_value.
        """
        return self.start_value
````

## `CosineScheduler`

Bases: `BaseScheduler`

Scheduler with cosine annealing.

This scheduler implements cosine annealing, which provides a smooth transition from the start value to the end value following a cosine curve. Cosine annealing is popular in deep learning as it provides fast initial decay followed by slower decay, which can help with convergence.

Mathematical Formula

[v(t) = \\begin{cases} v\_{end} + (v_0 - v\_{end}) \\times \\frac{1 + \\cos(\\pi t/T)}{2}, & \\text{if } t < T \\ v\_{end}, & \\text{if } t \\geq T \\end{cases}]

where:

- (v_0) is the start_value
- (v\_{end}) is the end_value
- (T) is n_steps
- (t) is the current step count

Cosine Curve Properties

The cosine function creates a smooth S-shaped curve that starts with rapid decay and gradually slows down as it approaches the end value.

Parameters:

| Name          | Type    | Description                               | Default    |
| ------------- | ------- | ----------------------------------------- | ---------- |
| `start_value` | `float` | Starting parameter value.                 | *required* |
| `end_value`   | `float` | Target parameter value.                   | *required* |
| `n_steps`     | `int`   | Number of steps to reach the final value. | *required* |

Raises:

| Type         | Description                 |
| ------------ | --------------------------- |
| `ValueError` | If n_steps is not positive. |

Step Size Annealing

```python
scheduler = CosineScheduler(start_value=0.1, end_value=0.001, n_steps=100)
values = []
for i in range(10):
    value = scheduler.step()
    values.append(value)
    if i < 3:  # Show first few values
        print(f"Step {i+1}: {value:.6f}")
# Shows smooth decay: 0.099951, 0.099606, 0.098866, ...
```

Learning Rate Scheduling

```python
lr_scheduler = CosineScheduler(
    start_value=0.01, end_value=0.0001, n_steps=1000
)
# In training loop
for epoch in range(1000):
    lr = lr_scheduler.step()
    # Update optimizer learning rate
```

Noise Scale Annealing

```python
noise_scheduler = CosineScheduler(
    start_value=1.0, end_value=0.01, n_steps=500
)
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01,
    noise_scale=noise_scheduler
)
```

Source code in `torchebm/core/base_scheduler.py`

````python
class CosineScheduler(BaseScheduler):
    r"""
    Scheduler with cosine annealing.

    This scheduler implements cosine annealing, which provides a smooth transition
    from the start value to the end value following a cosine curve. Cosine annealing
    is popular in deep learning as it provides fast initial decay followed by
    slower decay, which can help with convergence.

    !!! info "Mathematical Formula"
        $$v(t) = \begin{cases}
        v_{end} + (v_0 - v_{end}) \times \frac{1 + \cos(\pi t/T)}{2}, & \text{if } t < T \\
        v_{end}, & \text{if } t \geq T
        \end{cases}$$

        where:

        - \(v_0\) is the start_value
        - \(v_{end}\) is the end_value  
        - \(T\) is n_steps
        - \(t\) is the current step count

    !!! note "Cosine Curve Properties"
        The cosine function creates a smooth S-shaped curve that starts with rapid
        decay and gradually slows down as it approaches the end value.

    Args:
        start_value (float): Starting parameter value.
        end_value (float): Target parameter value.
        n_steps (int): Number of steps to reach the final value.

    Raises:
        ValueError: If n_steps is not positive.

    !!! example "Step Size Annealing"
        ```python
        scheduler = CosineScheduler(start_value=0.1, end_value=0.001, n_steps=100)
        values = []
        for i in range(10):
            value = scheduler.step()
            values.append(value)
            if i < 3:  # Show first few values
                print(f"Step {i+1}: {value:.6f}")
        # Shows smooth decay: 0.099951, 0.099606, 0.098866, ...
        ```

    !!! tip "Learning Rate Scheduling"
        ```python
        lr_scheduler = CosineScheduler(
            start_value=0.01, end_value=0.0001, n_steps=1000
        )
        # In training loop
        for epoch in range(1000):
            lr = lr_scheduler.step()
            # Update optimizer learning rate
        ```

    !!! example "Noise Scale Annealing"
        ```python
        noise_scheduler = CosineScheduler(
            start_value=1.0, end_value=0.01, n_steps=500
        )
        sampler = LangevinDynamics(
            energy_function=energy_fn,
            step_size=0.01,
            noise_scale=noise_scheduler
        )
        ```
    """

    def __init__(self, start_value: float, end_value: float, n_steps: int):
        r"""
        Initialize the cosine scheduler.

        Args:
            start_value (float): Starting parameter value.
            end_value (float): Target parameter value.
            n_steps (int): Number of steps to reach the final value.

        Raises:
            ValueError: If n_steps is not positive.
        """
        super().__init__(start_value)
        if n_steps <= 0:
            raise ValueError(f"n_steps must be a positive integer, got {n_steps}")

        self.end_value = end_value
        self.n_steps = n_steps

    def _compute_value(self) -> float:
        r"""
        Compute the cosine annealed value.

        Returns:
            float: The annealed value following cosine schedule.
        """
        if self.step_count >= self.n_steps:
            return self.end_value
        else:
            # Cosine schedule from start_value to end_value
            progress = self.step_count / self.n_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return self.end_value + (self.start_value - self.end_value) * cosine_factor
````

### `__init__(start_value, end_value, n_steps)`

Initialize the cosine scheduler.

Parameters:

| Name          | Type    | Description                               | Default    |
| ------------- | ------- | ----------------------------------------- | ---------- |
| `start_value` | `float` | Starting parameter value.                 | *required* |
| `end_value`   | `float` | Target parameter value.                   | *required* |
| `n_steps`     | `int`   | Number of steps to reach the final value. | *required* |

Raises:

| Type         | Description                 |
| ------------ | --------------------------- |
| `ValueError` | If n_steps is not positive. |

Source code in `torchebm/core/base_scheduler.py`

```python
def __init__(self, start_value: float, end_value: float, n_steps: int):
    r"""
    Initialize the cosine scheduler.

    Args:
        start_value (float): Starting parameter value.
        end_value (float): Target parameter value.
        n_steps (int): Number of steps to reach the final value.

    Raises:
        ValueError: If n_steps is not positive.
    """
    super().__init__(start_value)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps}")

    self.end_value = end_value
    self.n_steps = n_steps
```

## `CouplingResult`

Result of a coupling.

Iterates as the pair, so the idiomatic unpacking always works:

```python
x0, x1 = coupling(x0, x1)
```

Extra information rides as attributes and is never part of the iteration, so appending optional fields can never break unpacking. Consumers that need extras access them by name:

```python
res = coupling(x0, x1)
x0, x1 = res
if res.weights is not None:
    ...  # per-pair importance weights (unbalanced / reweighted OT)
```

Note

`tuple(result)` and `list(result)` yield the pair only; extras are attribute-access. Future extras (e.g. transport plans, alignment transforms) are appended as `Optional[...] = None` fields.

Attributes:

| Name      | Type               | Description                                                                                                                                                               |
| --------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x0`      | `Tensor`           | Source samples of shape (batch_size, ...).                                                                                                                                |
| `x1`      | `Tensor`           | Target samples of shape (batch_size, ...).                                                                                                                                |
| `weights` | `Optional[Tensor]` | Optional per-pair weights of shape (batch_size,). None means uniform. Produced by unbalanced/reweighted couplings; weight-aware consumers use them as importance weights. |

Source code in `torchebm/core/base_coupling.py`

````python
@dataclass(frozen=True)
class CouplingResult:
    r"""
    Result of a coupling.

    Iterates as the pair, so the idiomatic unpacking always works:

    ```python
    x0, x1 = coupling(x0, x1)
    ```

    Extra information rides as attributes and is never part of the iteration,
    so appending optional fields can never break unpacking. Consumers that
    need extras access them by name:

    ```python
    res = coupling(x0, x1)
    x0, x1 = res
    if res.weights is not None:
        ...  # per-pair importance weights (unbalanced / reweighted OT)
    ```

    Note:
        `tuple(result)` and `list(result)` yield the pair only; extras are
        attribute-access. Future extras (e.g. transport plans, alignment
        transforms) are appended as `Optional[...] = None` fields.

    Attributes:
        x0: Source samples of shape (batch_size, ...).
        x1: Target samples of shape (batch_size, ...).
        weights: Optional per-pair weights of shape (batch_size,). ``None``
            means uniform. Produced by unbalanced/reweighted couplings;
            weight-aware consumers use them as importance weights.
    """

    x0: torch.Tensor
    x1: torch.Tensor
    weights: Optional[torch.Tensor] = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter((self.x0, self.x1))
````

## `DoubleWellModel`

Bases: `BaseModel`

Energy-based model for a double-well potential.

Parameters:

| Name             | Type    | Description                                                       | Default |
| ---------------- | ------- | ----------------------------------------------------------------- | ------- |
| `barrier_height` | `float` | The height of the energy barrier between the wells.               | `2.0`   |
| `b`              | `float` | The position of the wells (default is 1.0, creating wells at ±1). | `1.0`   |

Source code in `torchebm/core/base_model.py`

```python
class DoubleWellModel(BaseModel):
    r"""
    Energy-based model for a double-well potential.

    Args:
        barrier_height (float): The height of the energy barrier between the wells.
        b (float): The position of the wells (default is 1.0, creating wells at ±1).
    """
    def __init__(self, barrier_height: float = 2.0, b: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.barrier_height = barrier_height
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the double well energy: \(h \sum_{i=1}^{n} (x_i^2 - b^2)^2\)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        return self.barrier_height * (x.pow(2) - self.b**2).pow(2).sum(dim=-1)
```

### `forward(x)`

Computes the double well energy: (h \\sum\_{i=1}^{n} (x_i^2 - b^2)^2).

Source code in `torchebm/core/base_model.py`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    r"""Computes the double well energy: \(h \sum_{i=1}^{n} (x_i^2 - b^2)^2\)."""
    if x.ndim == 1:
        x = x.unsqueeze(0)

    return self.barrier_height * (x.pow(2) - self.b**2).pow(2).sum(dim=-1)
```

## `ExponentialDecayScheduler`

Bases: `BaseScheduler`

Scheduler with exponential decay.

This scheduler implements exponential decay of the parameter value according to: (v(t) = \\max(v\_{min}, v_0 \\times \\gamma^t))

Exponential decay is commonly used for step sizes in optimization and sampling algorithms, as it provides rapid initial decay that slows down over time, allowing for both exploration and convergence.

Mathematical Formula

[v(t) = \\max(v\_{min}, v_0 \\times \\gamma^t)]

where:

- (v_0) is the start_value
- (\\gamma) is the decay_rate ((0 < \\gamma \\leq 1))
- (t) is the step count
- (v\_{min}) is the min_value (lower bound)

Parameters:

| Name          | Type    | Description                                            | Default    |
| ------------- | ------- | ------------------------------------------------------ | ---------- |
| `start_value` | `float` | Initial parameter value.                               | *required* |
| `decay_rate`  | `float` | Decay factor applied at each step. Must be in (0, 1\]. | *required* |
| `min_value`   | `float` | Minimum value to clamp the result. Defaults to 0.0.    | `0.0`      |

Raises:

| Type         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `ValueError` | If decay_rate is not in (0, 1\] or min_value is negative. |

Basic Exponential Decay

```python
scheduler = ExponentialDecayScheduler(
    start_value=1.0, decay_rate=0.9, min_value=0.01
)
for i in range(5):
    value = scheduler.step()
    print(f"Step {i+1}: {value:.4f}")
# Output: 0.9000, 0.8100, 0.7290, 0.6561, 0.5905
```

Training Loop Integration

```python
step_scheduler = ExponentialDecayScheduler(
    start_value=0.1, decay_rate=0.995, min_value=0.001
)
# In training loop
for epoch in range(1000):
    current_step_size = step_scheduler.step()
    # Use current_step_size in your algorithm
```

Decay Rate Selection

- **Aggressive decay**: Use smaller decay_rate (e.g., 0.5)
- **Gentle decay**: Use larger decay_rate (e.g., 0.99)

```python
# Aggressive decay
aggressive = ExponentialDecayScheduler(start_value=1.0, decay_rate=0.5)
# Gentle decay
gentle = ExponentialDecayScheduler(start_value=1.0, decay_rate=0.99)
```

Source code in `torchebm/core/base_scheduler.py`

````python
class ExponentialDecayScheduler(BaseScheduler):
    r"""
    Scheduler with exponential decay.

    This scheduler implements exponential decay of the parameter value according to:
    \(v(t) = \max(v_{min}, v_0 \times \gamma^t)\)

    Exponential decay is commonly used for step sizes in optimization and sampling
    algorithms, as it provides rapid initial decay that slows down over time,
    allowing for both exploration and convergence.

    !!! info "Mathematical Formula"
        $$v(t) = \max(v_{min}, v_0 \times \gamma^t)$$

        where:

        - \(v_0\) is the start_value
        - \(\gamma\) is the decay_rate \((0 < \gamma \leq 1)\)
        - \(t\) is the step count
        - \(v_{min}\) is the min_value (lower bound)

    Args:
        start_value (float): Initial parameter value.
        decay_rate (float): Decay factor applied at each step. Must be in (0, 1].
        min_value (float, optional): Minimum value to clamp the result. Defaults to 0.0.

    Raises:
        ValueError: If decay_rate is not in (0, 1] or min_value is negative.

    !!! example "Basic Exponential Decay"
        ```python
        scheduler = ExponentialDecayScheduler(
            start_value=1.0, decay_rate=0.9, min_value=0.01
        )
        for i in range(5):
            value = scheduler.step()
            print(f"Step {i+1}: {value:.4f}")
        # Output: 0.9000, 0.8100, 0.7290, 0.6561, 0.5905
        ```

    !!! tip "Training Loop Integration"
        ```python
        step_scheduler = ExponentialDecayScheduler(
            start_value=0.1, decay_rate=0.995, min_value=0.001
        )
        # In training loop
        for epoch in range(1000):
            current_step_size = step_scheduler.step()
            # Use current_step_size in your algorithm
        ```

    !!! note "Decay Rate Selection"
        - **Aggressive decay**: Use smaller decay_rate (e.g., 0.5)
        - **Gentle decay**: Use larger decay_rate (e.g., 0.99)

        ```python
        # Aggressive decay
        aggressive = ExponentialDecayScheduler(start_value=1.0, decay_rate=0.5)
        # Gentle decay
        gentle = ExponentialDecayScheduler(start_value=1.0, decay_rate=0.99)
        ```
    """

    def __init__(
        self,
        start_value: float,
        decay_rate: float,
        min_value: float = 0.0,
    ):
        r"""
        Initialize the exponential decay scheduler.

        Args:
            start_value (float): Initial parameter value.
            decay_rate (float): Decay factor applied at each step. Must be in (0, 1].
            min_value (float, optional): Minimum value to clamp the result. Defaults to 0.0.

        Raises:
            ValueError: If decay_rate is not in (0, 1] or min_value is negative.
        """
        super().__init__(start_value)
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}")
        if min_value < 0:
            raise ValueError(f"min_value must be non-negative, got {min_value}")
        self.decay_rate: float = decay_rate
        self.min_value: float = min_value

    def _compute_value(self) -> float:
        r"""
        Compute the exponentially decayed value.

        Returns:
            float: The decayed value, clamped to min_value.
        """
        val = self.start_value * (self.decay_rate**self.step_count)
        return max(self.min_value, val)
````

### `__init__(start_value, decay_rate, min_value=0.0)`

Initialize the exponential decay scheduler.

Parameters:

| Name          | Type    | Description                                            | Default    |
| ------------- | ------- | ------------------------------------------------------ | ---------- |
| `start_value` | `float` | Initial parameter value.                               | *required* |
| `decay_rate`  | `float` | Decay factor applied at each step. Must be in (0, 1\]. | *required* |
| `min_value`   | `float` | Minimum value to clamp the result. Defaults to 0.0.    | `0.0`      |

Raises:

| Type         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `ValueError` | If decay_rate is not in (0, 1\] or min_value is negative. |

Source code in `torchebm/core/base_scheduler.py`

```python
def __init__(
    self,
    start_value: float,
    decay_rate: float,
    min_value: float = 0.0,
):
    r"""
    Initialize the exponential decay scheduler.

    Args:
        start_value (float): Initial parameter value.
        decay_rate (float): Decay factor applied at each step. Must be in (0, 1].
        min_value (float, optional): Minimum value to clamp the result. Defaults to 0.0.

    Raises:
        ValueError: If decay_rate is not in (0, 1] or min_value is negative.
    """
    super().__init__(start_value)
    if not 0.0 < decay_rate <= 1.0:
        raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}")
    if min_value < 0:
        raise ValueError(f"min_value must be non-negative, got {min_value}")
    self.decay_rate: float = decay_rate
    self.min_value: float = min_value
```

## `GaussianModel`

Bases: `BaseModel`

Energy-based model for a Gaussian distribution.

Parameters:

| Name   | Type     | Description                                             | Default    |
| ------ | -------- | ------------------------------------------------------- | ---------- |
| `mean` | `Tensor` | The mean vector (μ) of the Gaussian distribution.       | *required* |
| `cov`  | `Tensor` | The covariance matrix (Σ) of the Gaussian distribution. | *required* |

Source code in `torchebm/core/base_model.py`

```python
class GaussianModel(BaseModel):
    r"""
    Energy-based model for a Gaussian distribution.

    Args:
        mean (torch.Tensor): The mean vector (μ) of the Gaussian distribution.
        cov (torch.Tensor): The covariance matrix (Σ) of the Gaussian distribution.
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mean.ndim != 1:
            raise ValueError("Mean must be a 1D tensor.")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance must be a 2D square matrix.")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError(
                "Mean vector dimension must match covariance matrix dimension."
            )

        self.register_buffer("mean", mean.to(dtype=self.dtype, device=self.device))
        try:
            cov_inv = torch.inverse(cov)
            self.register_buffer(
                "cov_inv", cov_inv.to(dtype=self.dtype, device=self.device)
            )
        except RuntimeError as e:
            raise ValueError(
                f"Failed to invert covariance matrix: {e}. Ensure it is invertible."
            ) from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Gaussian energy: \(E(x) = \frac{1}{2} (x - \mu)^{\top} \Sigma^{-1} (x - \mu)\)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"Input x expected batch_shape (batch_size, {self.mean.shape[0]}), but got {x.shape}"
            )

        x = x.to(dtype=self.dtype, device=self.device)
        # mean = self.mean.to(device=x.device)
        cov_inv = self.cov_inv.to(dtype=self.dtype, device=x.device)

        delta = (
            x - self.mean
        )  # avoid detaching or converting x to maintain grad tracking
        # energy = 0.5 * torch.einsum("bi,ij,bj->b", delta, cov_inv, delta)

        if delta.shape[0] > 1:
            delta_expanded = delta.unsqueeze(-1)  # (batch_size, dim, 1)
            cov_inv_expanded = cov_inv.unsqueeze(0).expand(
                delta.shape[0], -1, -1
            )  # (batch_size, dim, dim)

            temp = torch.bmm(cov_inv_expanded, delta_expanded)  # (batch_size, dim, 1)
            energy = 0.5 * torch.bmm(delta.unsqueeze(1), temp).squeeze(-1).squeeze(-1)
        else:
            energy = 0.5 * torch.sum(delta * torch.matmul(delta, cov_inv), dim=-1)

        return energy
```

### `forward(x)`

Computes the Gaussian energy: (E(x) = \\frac{1}{2} (x - \\mu)^{\\top} \\Sigma^{-1} (x - \\mu)).

Source code in `torchebm/core/base_model.py`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    r"""Computes the Gaussian energy: \(E(x) = \frac{1}{2} (x - \mu)^{\top} \Sigma^{-1} (x - \mu)\)."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
        raise ValueError(
            f"Input x expected batch_shape (batch_size, {self.mean.shape[0]}), but got {x.shape}"
        )

    x = x.to(dtype=self.dtype, device=self.device)
    # mean = self.mean.to(device=x.device)
    cov_inv = self.cov_inv.to(dtype=self.dtype, device=x.device)

    delta = (
        x - self.mean
    )  # avoid detaching or converting x to maintain grad tracking
    # energy = 0.5 * torch.einsum("bi,ij,bj->b", delta, cov_inv, delta)

    if delta.shape[0] > 1:
        delta_expanded = delta.unsqueeze(-1)  # (batch_size, dim, 1)
        cov_inv_expanded = cov_inv.unsqueeze(0).expand(
            delta.shape[0], -1, -1
        )  # (batch_size, dim, dim)

        temp = torch.bmm(cov_inv_expanded, delta_expanded)  # (batch_size, dim, 1)
        energy = 0.5 * torch.bmm(delta.unsqueeze(1), temp).squeeze(-1).squeeze(-1)
    else:
        energy = 0.5 * torch.sum(delta * torch.matmul(delta, cov_inv), dim=-1)

    return energy
```

## `HarmonicModel`

Bases: `BaseModel`

Energy-based model for a harmonic oscillator.

Parameters:

| Name | Type    | Description          | Default |
| ---- | ------- | -------------------- | ------- |
| `k`  | `float` | The spring constant. | `1.0`   |

Source code in `torchebm/core/base_model.py`

```python
class HarmonicModel(BaseModel):
    r"""
    Energy-based model for a harmonic oscillator.

    Args:
        k (float): The spring constant.
    """
    def __init__(self, k: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the harmonic oscillator energy: \(\frac{1}{2} k \sum_{i=1}^{n} x_i^{2}\)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        return 0.5 * self.k * x.pow(2).sum(dim=-1)
```

### `forward(x)`

Computes the harmonic oscillator energy: (\\frac{1}{2} k \\sum\_{i=1}^{n} x_i^{2}).

Source code in `torchebm/core/base_model.py`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    r"""Computes the harmonic oscillator energy: \(\frac{1}{2} k \sum_{i=1}^{n} x_i^{2}\)."""
    if x.ndim == 1:
        x = x.unsqueeze(0)

    return 0.5 * self.k * x.pow(2).sum(dim=-1)
```

## `LinearScheduler`

Bases: `BaseScheduler`

Scheduler with linear interpolation between start and end values.

This scheduler linearly interpolates between a start value and an end value over a specified number of steps. After reaching the end value, it remains constant. Linear scheduling is useful when you want predictable, uniform changes in parameter values.

Mathematical Formula

[v(t) = \\begin{cases} v_0 + (v\_{end} - v_0) \\times \\frac{t}{T}, & \\text{if } t < T \\ v\_{end}, & \\text{if } t \\geq T \\end{cases}]

where:

- (v_0) is the start_value
- (v\_{end}) is the end_value
- (T) is n_steps
- (t) is the current step count

Parameters:

| Name          | Type    | Description                               | Default    |
| ------------- | ------- | ----------------------------------------- | ---------- |
| `start_value` | `float` | Starting parameter value.                 | *required* |
| `end_value`   | `float` | Target parameter value.                   | *required* |
| `n_steps`     | `int`   | Number of steps to reach the final value. | *required* |

Raises:

| Type         | Description                 |
| ------------ | --------------------------- |
| `ValueError` | If n_steps is not positive. |

Linear Decay

```python
scheduler = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=5)
for i in range(7):  # Go beyond n_steps to see clamping
    value = scheduler.step()
    print(f"Step {i+1}: {value:.2f}")
# Output: 0.80, 0.60, 0.40, 0.20, 0.00, 0.00, 0.00
```

Warmup Strategy

```python
warmup_scheduler = LinearScheduler(
    start_value=0.0, end_value=0.1, n_steps=100
)
# Use for learning rate warmup
for epoch in range(100):
    lr = warmup_scheduler.step()
    # Set learning rate in optimizer
```

MCMC Integration

```python
step_scheduler = LinearScheduler(
    start_value=0.1, end_value=0.001, n_steps=1000
)
# Use in MCMC sampler
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=step_scheduler
)
```

Source code in `torchebm/core/base_scheduler.py`

````python
class LinearScheduler(BaseScheduler):
    r"""
    Scheduler with linear interpolation between start and end values.

    This scheduler linearly interpolates between a start value and an end value
    over a specified number of steps. After reaching the end value, it remains
    constant. Linear scheduling is useful when you want predictable, uniform
    changes in parameter values.

    !!! info "Mathematical Formula"
        $$v(t) = \begin{cases}
        v_0 + (v_{end} - v_0) \times \frac{t}{T}, & \text{if } t < T \\
        v_{end}, & \text{if } t \geq T
        \end{cases}$$

        where:

        - \(v_0\) is the start_value
        - \(v_{end}\) is the end_value
        - \(T\) is n_steps
        - \(t\) is the current step count

    Args:
        start_value (float): Starting parameter value.
        end_value (float): Target parameter value.
        n_steps (int): Number of steps to reach the final value.

    Raises:
        ValueError: If n_steps is not positive.

    !!! example "Linear Decay"
        ```python
        scheduler = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=5)
        for i in range(7):  # Go beyond n_steps to see clamping
            value = scheduler.step()
            print(f"Step {i+1}: {value:.2f}")
        # Output: 0.80, 0.60, 0.40, 0.20, 0.00, 0.00, 0.00
        ```

    !!! tip "Warmup Strategy"
        ```python
        warmup_scheduler = LinearScheduler(
            start_value=0.0, end_value=0.1, n_steps=100
        )
        # Use for learning rate warmup
        for epoch in range(100):
            lr = warmup_scheduler.step()
            # Set learning rate in optimizer
        ```

    !!! example "MCMC Integration"
        ```python
        step_scheduler = LinearScheduler(
            start_value=0.1, end_value=0.001, n_steps=1000
        )
        # Use in MCMC sampler
        sampler = LangevinDynamics(
            energy_function=energy_fn,
            step_size=step_scheduler
        )
        ```
    """

    def __init__(self, start_value: float, end_value: float, n_steps: int):
        r"""
        Initialize the linear scheduler.

        Args:
            start_value (float): Starting parameter value.
            end_value (float): Target parameter value.
            n_steps (int): Number of steps to reach the final value.

        Raises:
            ValueError: If n_steps is not positive.
        """
        super().__init__(start_value)
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        self.end_value = end_value
        self.n_steps = n_steps
        self.step_size: float = (end_value - start_value) / n_steps

    def _compute_value(self) -> float:
        r"""
        Compute the linearly interpolated value.

        Returns:
            float: The interpolated value, clamped to end_value after n_steps.
        """
        if self.step_count >= self.n_steps:
            return self.end_value
        else:
            return self.start_value + self.step_size * self.step_count
````

### `__init__(start_value, end_value, n_steps)`

Initialize the linear scheduler.

Parameters:

| Name          | Type    | Description                               | Default    |
| ------------- | ------- | ----------------------------------------- | ---------- |
| `start_value` | `float` | Starting parameter value.                 | *required* |
| `end_value`   | `float` | Target parameter value.                   | *required* |
| `n_steps`     | `int`   | Number of steps to reach the final value. | *required* |

Raises:

| Type         | Description                 |
| ------------ | --------------------------- |
| `ValueError` | If n_steps is not positive. |

Source code in `torchebm/core/base_scheduler.py`

```python
def __init__(self, start_value: float, end_value: float, n_steps: int):
    r"""
    Initialize the linear scheduler.

    Args:
        start_value (float): Starting parameter value.
        end_value (float): Target parameter value.
        n_steps (int): Number of steps to reach the final value.

    Raises:
        ValueError: If n_steps is not positive.
    """
    super().__init__(start_value)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    self.end_value = end_value
    self.n_steps = n_steps
    self.step_size: float = (end_value - start_value) / n_steps
```

## `MultiStepScheduler`

Bases: `BaseScheduler`

Scheduler that reduces the parameter value at specific milestone steps.

This scheduler maintains the current value until reaching predefined milestone steps, at which point it multiplies the value by a decay factor (gamma). This creates a step-wise decay pattern commonly used in learning rate scheduling.

Mathematical Formula

[v(t) = v_0 \\times \\gamma^k]

where:

- (v_0) is the start_value
- (\\gamma) is the gamma decay factor
- (k) is the number of milestones that have been reached by step (t)

Parameters:

| Name          | Type        | Description                                                                             | Default    |
| ------------- | ----------- | --------------------------------------------------------------------------------------- | ---------- |
| `start_value` | `float`     | Initial parameter value.                                                                | *required* |
| `milestones`  | `List[int]` | List of step numbers at which to apply decay. Must be positive and strictly increasing. | *required* |
| `gamma`       | `float`     | Multiplicative factor for decay. Defaults to 0.1.                                       | `0.1`      |

Raises:

| Type         | Description                                                |
| ------------ | ---------------------------------------------------------- |
| `ValueError` | If milestones are not positive or not strictly increasing. |

Step-wise Learning Rate Decay

```python
scheduler = MultiStepScheduler(
    start_value=0.1,
    milestones=[30, 60, 90],
    gamma=0.1
)
# Simulate training steps
for step in [0, 29, 30, 31, 59, 60, 61, 89, 90, 91]:
    if step > 0:
        scheduler.step_count = step
        value = scheduler._compute_value()
    else:
        value = scheduler.get_value()
    print(f"Step {step}: {value:.4f}")
# Output shows: 0.1 until step 30, then 0.01, then 0.001 at step 60, etc.
```

Different Decay Strategies

```python
# Gentle decay
gentle_scheduler = MultiStepScheduler(
    start_value=1.0, milestones=[100, 200], gamma=0.5
)

# Aggressive decay
aggressive_scheduler = MultiStepScheduler(
    start_value=1.0, milestones=[50, 100], gamma=0.01
)
```

Training Loop Integration

```python
step_scheduler = MultiStepScheduler(
    start_value=0.01,
    milestones=[500, 1000, 1500],
    gamma=0.2
)
# In training loop
for epoch in range(2000):
    current_step_size = step_scheduler.step()
    # Use current_step_size in your algorithm
```

Source code in `torchebm/core/base_scheduler.py`

````python
class MultiStepScheduler(BaseScheduler):
    r"""
    Scheduler that reduces the parameter value at specific milestone steps.

    This scheduler maintains the current value until reaching predefined milestone
    steps, at which point it multiplies the value by a decay factor (gamma).
    This creates a step-wise decay pattern commonly used in learning rate scheduling.

    !!! info "Mathematical Formula"
        $$v(t) = v_0 \times \gamma^k$$

        where:

        - \(v_0\) is the start_value
        - \(\gamma\) is the gamma decay factor
        - \(k\) is the number of milestones that have been reached by step \(t\)

    Args:
        start_value (float): Initial parameter value.
        milestones (List[int]): List of step numbers at which to apply decay.
            Must be positive and strictly increasing.
        gamma (float, optional): Multiplicative factor for decay. Defaults to 0.1.

    Raises:
        ValueError: If milestones are not positive or not strictly increasing.

    !!! example "Step-wise Learning Rate Decay"
        ```python
        scheduler = MultiStepScheduler(
            start_value=0.1,
            milestones=[30, 60, 90],
            gamma=0.1
        )
        # Simulate training steps
        for step in [0, 29, 30, 31, 59, 60, 61, 89, 90, 91]:
            if step > 0:
                scheduler.step_count = step
                value = scheduler._compute_value()
            else:
                value = scheduler.get_value()
            print(f"Step {step}: {value:.4f}")
        # Output shows: 0.1 until step 30, then 0.01, then 0.001 at step 60, etc.
        ```

    !!! tip "Different Decay Strategies"
        ```python
        # Gentle decay
        gentle_scheduler = MultiStepScheduler(
            start_value=1.0, milestones=[100, 200], gamma=0.5
        )

        # Aggressive decay
        aggressive_scheduler = MultiStepScheduler(
            start_value=1.0, milestones=[50, 100], gamma=0.01
        )
        ```

    !!! example "Training Loop Integration"
        ```python
        step_scheduler = MultiStepScheduler(
            start_value=0.01,
            milestones=[500, 1000, 1500],
            gamma=0.2
        )
        # In training loop
        for epoch in range(2000):
            current_step_size = step_scheduler.step()
            # Use current_step_size in your algorithm
        ```
    """

    def __init__(self, start_value: float, milestones: List[int], gamma: float = 0.1):
        r"""
        Initialize the multi-step scheduler.

        Args:
            start_value (float): Initial parameter value.
            milestones (List[int]): List of step numbers at which to apply decay.
                Must be positive and strictly increasing.
            gamma (float, optional): Multiplicative factor for decay. Defaults to 0.1.

        Raises:
            ValueError: If milestones are not positive or not strictly increasing.
        """
        super().__init__(start_value)
        if not all(m > 0 for m in milestones):
            raise ValueError("Milestone steps must be positive integers.")
        if not all(
            milestones[i] < milestones[i + 1] for i in range(len(milestones) - 1)
        ):
            raise ValueError("Milestones must be strictly increasing.")
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def _compute_value(self) -> float:
        r"""
        Compute the value based on reached milestones.

        Returns:
            float: The parameter value after applying decay for reached milestones.
        """
        power = sum(1 for m in self.milestones if self.step_count >= m)
        return self.start_value * (self.gamma**power)
````

### `__init__(start_value, milestones, gamma=0.1)`

Initialize the multi-step scheduler.

Parameters:

| Name          | Type        | Description                                                                             | Default    |
| ------------- | ----------- | --------------------------------------------------------------------------------------- | ---------- |
| `start_value` | `float`     | Initial parameter value.                                                                | *required* |
| `milestones`  | `List[int]` | List of step numbers at which to apply decay. Must be positive and strictly increasing. | *required* |
| `gamma`       | `float`     | Multiplicative factor for decay. Defaults to 0.1.                                       | `0.1`      |

Raises:

| Type         | Description                                                |
| ------------ | ---------------------------------------------------------- |
| `ValueError` | If milestones are not positive or not strictly increasing. |

Source code in `torchebm/core/base_scheduler.py`

```python
def __init__(self, start_value: float, milestones: List[int], gamma: float = 0.1):
    r"""
    Initialize the multi-step scheduler.

    Args:
        start_value (float): Initial parameter value.
        milestones (List[int]): List of step numbers at which to apply decay.
            Must be positive and strictly increasing.
        gamma (float, optional): Multiplicative factor for decay. Defaults to 0.1.

    Raises:
        ValueError: If milestones are not positive or not strictly increasing.
    """
    super().__init__(start_value)
    if not all(m > 0 for m in milestones):
        raise ValueError("Milestone steps must be positive integers.")
    if not all(
        milestones[i] < milestones[i + 1] for i in range(len(milestones) - 1)
    ):
        raise ValueError("Milestones must be strictly increasing.")
    self.milestones = sorted(milestones)
    self.gamma = gamma
```

## `RastriginModel`

Bases: `BaseModel`

Energy-based model for the Rastrigin function.

Parameters:

| Name | Type    | Description                                | Default |
| ---- | ------- | ------------------------------------------ | ------- |
| `a`  | `float` | The a parameter of the Rastrigin function. | `10.0`  |

Source code in `torchebm/core/base_model.py`

```python
class RastriginModel(BaseModel):
    r"""
    Energy-based model for the Rastrigin function.

    Args:
        a (float): The `a` parameter of the Rastrigin function.
    """
    def __init__(self, a: float = 10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Rastrigin energy."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        n = x.shape[-1]
        return self.a * n + torch.sum(
            x**2 - self.a * torch.cos(2 * math.pi * x), dim=-1
        )
```

### `forward(x)`

Computes the Rastrigin energy.

Source code in `torchebm/core/base_model.py`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    r"""Computes the Rastrigin energy."""
    if x.ndim == 1:
        x = x.unsqueeze(0)

    n = x.shape[-1]
    return self.a * n + torch.sum(
        x**2 - self.a * torch.cos(2 * math.pi * x), dim=-1
    )
```

## `RosenbrockModel`

Bases: `BaseModel`

Energy-based model for the Rosenbrock function.

Parameters:

| Name | Type    | Description                                 | Default |
| ---- | ------- | ------------------------------------------- | ------- |
| `a`  | `float` | The a parameter of the Rosenbrock function. | `1.0`   |
| `b`  | `float` | The b parameter of the Rosenbrock function. | `100.0` |

Source code in `torchebm/core/base_model.py`

```python
class RosenbrockModel(BaseModel):
    r"""
    Energy-based model for the Rosenbrock function.

    Args:
        a (float): The `a` parameter of the Rosenbrock function.
        b (float): The `b` parameter of the Rosenbrock function.
    """
    def __init__(self, a: float = 1.0, b: float = 100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Rosenbrock energy: \(\sum_{i=1}^{n-1} \left[ b(x_{i+1} - x_i^2)^2 + (a - x_i)^2 \right]\)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] < 2:
            raise ValueError(
                f"Rosenbrock energy function requires at least 2 dimensions, got {x.shape[-1]}"
            )

        # return (self.a - x[..., 0]) ** 2 + self.b * (x[..., 1] - x[..., 0] ** 2) ** 2
        # return sum(
        #     self.b * (x[..., i + 1] - x[..., i] ** 2) ** 2 + (self.a - x[i]) ** 2
        #     for i in range(len(x) - 1)
        # )

        x_i = x[:, :-1]
        x_ip1 = x[:, 1:]
        term1 = (self.a - x_i).pow(2)
        term2 = self.b * (x_ip1 - x_i.pow(2)).pow(2)
        return (term1 + term2).sum(dim=-1)
```

### `forward(x)`

Computes the Rosenbrock energy: (\\sum\_{i=1}^{n-1} \\left[ b(x\_{i+1} - x_i^2)^2 + (a - x_i)^2 \\right]).

Source code in `torchebm/core/base_model.py`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    r"""Computes the Rosenbrock energy: \(\sum_{i=1}^{n-1} \left[ b(x_{i+1} - x_i^2)^2 + (a - x_i)^2 \right]\)."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[-1] < 2:
        raise ValueError(
            f"Rosenbrock energy function requires at least 2 dimensions, got {x.shape[-1]}"
        )

    # return (self.a - x[..., 0]) ** 2 + self.b * (x[..., 1] - x[..., 0] ** 2) ** 2
    # return sum(
    #     self.b * (x[..., i + 1] - x[..., i] ** 2) ** 2 + (self.a - x[i]) ** 2
    #     for i in range(len(x) - 1)
    # )

    x_i = x[:, :-1]
    x_ip1 = x[:, 1:]
    term1 = (self.a - x_i).pow(2)
    term2 = self.b * (x_ip1 - x_i.pow(2)).pow(2)
    return (term1 + term2).sum(dim=-1)
```

## `Schedulable`

Mixin: add scheduler registration + recursive stepping.

Use alongside `TorchEBMModule` (or any `nn.Module` subclass). The host's MRO must invoke `super().__init__()` so this mixin can initialize the scheduler dict on construction.

Source code in `torchebm/core/schedulable.py`

```python
class Schedulable:
    r"""Mixin: add scheduler registration + recursive stepping.

    Use alongside `TorchEBMModule` (or any `nn.Module` subclass). The host's
    MRO must invoke `super().__init__()` so this mixin can initialize the
    scheduler dict on construction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedulers: Dict[str, BaseScheduler] = {}

    def register_scheduler(self, name: str, scheduler: BaseScheduler) -> None:
        r"""Register a scheduler under `name`. Replaces any existing entry."""
        self.schedulers[name] = scheduler

    def _register_param(
        self,
        name: str,
        value: Union[float, BaseScheduler],
        *,
        positive: bool = False,
    ) -> None:
        r"""Register a numeric param as either a `BaseScheduler` or a constant.

        For a float input, optionally enforce ``value > 0`` and wrap in
        `ConstantScheduler`. For a `BaseScheduler` input, store it directly.
        """
        if isinstance(value, BaseScheduler):
            self.schedulers[name] = value
            return
        if positive and value <= 0:
            raise ValueError(f"{name} must be positive")
        self.schedulers[name] = ConstantScheduler(float(value))

    def get_schedulers(self) -> Dict[str, BaseScheduler]:
        return self.schedulers

    def get_scheduled_value(self, name: str) -> float:
        if name not in self.schedulers:
            raise KeyError(f"No scheduler registered for parameter '{name}'")
        return self.schedulers[name].get_value()

    def step_schedulers(self) -> None:
        r"""Advance every scheduler in this module subtree by one step."""
        def _step(m):
            if isinstance(m, Schedulable):
                for s in m.schedulers.values():
                    s.step()
        # `apply` is an nn.Module method; the host class must inherit from nn.Module.
        self.apply(_step)

    def reset_schedulers(self) -> None:
        r"""Reset every scheduler in this module subtree to step 0."""
        def _reset(m):
            if isinstance(m, Schedulable):
                for s in m.schedulers.values():
                    s.reset()
        self.apply(_reset)
```

### `register_scheduler(name, scheduler)`

Register a scheduler under `name`. Replaces any existing entry.

Source code in `torchebm/core/schedulable.py`

```python
def register_scheduler(self, name: str, scheduler: BaseScheduler) -> None:
    r"""Register a scheduler under `name`. Replaces any existing entry."""
    self.schedulers[name] = scheduler
```

### `reset_schedulers()`

Reset every scheduler in this module subtree to step 0.

Source code in `torchebm/core/schedulable.py`

```python
def reset_schedulers(self) -> None:
    r"""Reset every scheduler in this module subtree to step 0."""
    def _reset(m):
        if isinstance(m, Schedulable):
            for s in m.schedulers.values():
                s.reset()
    self.apply(_reset)
```

### `step_schedulers()`

Advance every scheduler in this module subtree by one step.

Source code in `torchebm/core/schedulable.py`

```python
def step_schedulers(self) -> None:
    r"""Advance every scheduler in this module subtree by one step."""
    def _step(m):
        if isinstance(m, Schedulable):
            for s in m.schedulers.values():
                s.step()
    # `apply` is an nn.Module method; the host class must inherit from nn.Module.
    self.apply(_step)
```

## `TemperatureScheduler`

Bases: `BaseScheduler`

Piecewise-linear temperature schedule (\\epsilon(t)) for two-regime sampling.

Implements the Energy Matching temperature profile (Balcerak et al., 2025, arXiv:2504.10612): zero temperature during the transport regime, a linear ramp near the data, and a constant Boltzmann temperature afterwards.

Mathematical Formula

[\\epsilon(t) = \\begin{cases} 0, & \\text{if } t < \\tau^\* \\ \\epsilon\_{max} \\frac{t - \\tau^\*}{1 - \\tau^\*}, & \\text{if } \\tau^\* \\leq t < 1 \\ \\epsilon\_{max}, & \\text{if } t \\geq 1 \\end{cases}]

The scheduler maps its step count to virtual time (t = t\_{start} + (t\_{end} - t\_{start}) \\min(n, N)/N) where (N) is `n_steps`, then returns (\\sqrt{\\epsilon(t)}) (default) or (\\epsilon(t)).

With `sqrt=True` the returned value is a drop-in `noise_scale` for `LangevinDynamics`: the integrator noise is `noise_scale * sqrt(2 * step_size)`, so the update noise becomes the paper's (\\sqrt{2 \\epsilon(t) \\Delta t}, \\eta). Use `sqrt=False` for the raw temperature (e.g. as an interaction strength or for logging).

Parameters:

| Name          | Type    | Description                                                                                            | Default    |
| ------------- | ------- | ------------------------------------------------------------------------------------------------------ | ---------- |
| `epsilon_max` | `float` | Plateau temperature (\\epsilon\_{max}). Must be >= 0.                                                  | *required* |
| `tau_star`    | `float` | Transport/diffusion switch time (\\tau^\*) in \[0, 1).                                                 | `0.8`      |
| `n_steps`     | `int`   | Steps to sweep from t_start to t_end; holds the end value afterwards.                                  | `200`      |
| `t_start`     | `float` | Virtual time at step 0.                                                                                | `0.0`      |
| `t_end`       | `float` | Virtual time at step n_steps. May exceed 1 for equilibration (e.g. inference with total time (T > 1)). | `1.0`      |
| `sqrt`        | `bool`  | Return (\\sqrt{\\epsilon(t)}) if True (noise_scale semantics), else (\\epsilon(t)).                    | `True`     |

Raises:

| Type         | Description                                                                     |
| ------------ | ------------------------------------------------------------------------------- |
| `ValueError` | If epsilon_max < 0, tau_star not in \[0, 1), n_steps \<= 0, or t_end < t_start. |

Energy Matching generation (one SDE pass)

```python
from torchebm.core import TemperatureScheduler
from torchebm.samplers import LangevinDynamics

# T = 3.25, dt = 0.01 -> 325 steps; noise-free below tau*, Langevin above
temp = TemperatureScheduler(
    epsilon_max=0.15, tau_star=0.8, n_steps=325, t_end=3.25
)
sampler = LangevinDynamics(model=potential, step_size=0.01, noise_scale=temp)
samples = sampler.sample(x=torch.randn(1000, 2), n_steps=325)
```

Source code in `torchebm/core/base_scheduler.py`

````python
class TemperatureScheduler(BaseScheduler):
    r"""
    Piecewise-linear temperature schedule \(\epsilon(t)\) for two-regime sampling.

    Implements the Energy Matching temperature profile (Balcerak et al., 2025,
    arXiv:2504.10612): zero temperature during the transport regime, a linear
    ramp near the data, and a constant Boltzmann temperature afterwards.

    !!! info "Mathematical Formula"
        $$\epsilon(t) = \begin{cases}
        0, & \text{if } t < \tau^* \\
        \epsilon_{max} \frac{t - \tau^*}{1 - \tau^*}, & \text{if } \tau^* \leq t < 1 \\
        \epsilon_{max}, & \text{if } t \geq 1
        \end{cases}$$

        The scheduler maps its step count to virtual time
        \(t = t_{start} + (t_{end} - t_{start}) \min(n, N)/N\) where \(N\) is
        `n_steps`, then returns \(\sqrt{\epsilon(t)}\) (default) or \(\epsilon(t)\).

    With `sqrt=True` the returned value is a drop-in `noise_scale` for
    `LangevinDynamics`: the integrator noise is
    `noise_scale * sqrt(2 * step_size)`, so the update noise becomes the
    paper's \(\sqrt{2 \epsilon(t) \Delta t}\, \eta\). Use `sqrt=False` for the
    raw temperature (e.g. as an interaction strength or for logging).

    Args:
        epsilon_max (float): Plateau temperature \(\epsilon_{max}\). Must be >= 0.
        tau_star (float): Transport/diffusion switch time \(\tau^*\) in [0, 1).
        n_steps (int): Steps to sweep from `t_start` to `t_end`; holds the
            end value afterwards.
        t_start (float): Virtual time at step 0.
        t_end (float): Virtual time at step `n_steps`. May exceed 1 for
            equilibration (e.g. inference with total time \(T > 1\)).
        sqrt (bool): Return \(\sqrt{\epsilon(t)}\) if True (noise_scale
            semantics), else \(\epsilon(t)\).

    Raises:
        ValueError: If `epsilon_max < 0`, `tau_star` not in [0, 1),
            `n_steps <= 0`, or `t_end < t_start`.

    !!! example "Energy Matching generation (one SDE pass)"
        ```python
        from torchebm.core import TemperatureScheduler
        from torchebm.samplers import LangevinDynamics

        # T = 3.25, dt = 0.01 -> 325 steps; noise-free below tau*, Langevin above
        temp = TemperatureScheduler(
            epsilon_max=0.15, tau_star=0.8, n_steps=325, t_end=3.25
        )
        sampler = LangevinDynamics(model=potential, step_size=0.01, noise_scale=temp)
        samples = sampler.sample(x=torch.randn(1000, 2), n_steps=325)
        ```
    """

    def __init__(
        self,
        epsilon_max: float,
        tau_star: float = 0.8,
        n_steps: int = 200,
        t_start: float = 0.0,
        t_end: float = 1.0,
        sqrt: bool = True,
    ):
        r"""
        Initialize the temperature scheduler.

        Args:
            epsilon_max (float): Plateau temperature. Must be >= 0.
            tau_star (float): Switch time in [0, 1).
            n_steps (int): Steps to sweep from `t_start` to `t_end`.
            t_start (float): Virtual time at step 0.
            t_end (float): Virtual time at step `n_steps`.
            sqrt (bool): Return sqrt(epsilon) if True, else epsilon.

        Raises:
            ValueError: On invalid arguments (see class docstring).
        """
        if epsilon_max < 0:
            raise ValueError(f"epsilon_max must be >= 0, got {epsilon_max}")
        if not 0.0 <= tau_star < 1.0:
            raise ValueError(f"tau_star must be in [0, 1), got {tau_star}")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if t_end < t_start:
            raise ValueError(
                f"t_end ({t_end}) must be >= t_start ({t_start})"
            )

        self.epsilon_max = float(epsilon_max)
        self.tau_star = float(tau_star)
        self.n_steps = int(n_steps)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.sqrt = bool(sqrt)
        super().__init__(self._value_at_time(self.t_start))

    def epsilon_at(self, t: float) -> float:
        r"""Evaluate the raw temperature \(\epsilon(t)\) at virtual time `t`."""
        if t < self.tau_star:
            return 0.0
        if t < 1.0:
            return self.epsilon_max * (t - self.tau_star) / (1.0 - self.tau_star)
        return self.epsilon_max

    def _value_at_time(self, t: float) -> float:
        eps = self.epsilon_at(t)
        return math.sqrt(eps) if self.sqrt else eps

    def _compute_value(self) -> float:
        r"""Map step count to virtual time and evaluate the profile."""
        frac = min(self.step_count, self.n_steps) / self.n_steps
        t = self.t_start + (self.t_end - self.t_start) * frac
        return self._value_at_time(t)
````

### `__init__(epsilon_max, tau_star=0.8, n_steps=200, t_start=0.0, t_end=1.0, sqrt=True)`

Initialize the temperature scheduler.

Parameters:

| Name          | Type    | Description                                 | Default    |
| ------------- | ------- | ------------------------------------------- | ---------- |
| `epsilon_max` | `float` | Plateau temperature. Must be >= 0.          | *required* |
| `tau_star`    | `float` | Switch time in \[0, 1).                     | `0.8`      |
| `n_steps`     | `int`   | Steps to sweep from t_start to t_end.       | `200`      |
| `t_start`     | `float` | Virtual time at step 0.                     | `0.0`      |
| `t_end`       | `float` | Virtual time at step n_steps.               | `1.0`      |
| `sqrt`        | `bool`  | Return sqrt(epsilon) if True, else epsilon. | `True`     |

Raises:

| Type         | Description                                 |
| ------------ | ------------------------------------------- |
| `ValueError` | On invalid arguments (see class docstring). |

Source code in `torchebm/core/base_scheduler.py`

```python
def __init__(
    self,
    epsilon_max: float,
    tau_star: float = 0.8,
    n_steps: int = 200,
    t_start: float = 0.0,
    t_end: float = 1.0,
    sqrt: bool = True,
):
    r"""
    Initialize the temperature scheduler.

    Args:
        epsilon_max (float): Plateau temperature. Must be >= 0.
        tau_star (float): Switch time in [0, 1).
        n_steps (int): Steps to sweep from `t_start` to `t_end`.
        t_start (float): Virtual time at step 0.
        t_end (float): Virtual time at step `n_steps`.
        sqrt (bool): Return sqrt(epsilon) if True, else epsilon.

    Raises:
        ValueError: On invalid arguments (see class docstring).
    """
    if epsilon_max < 0:
        raise ValueError(f"epsilon_max must be >= 0, got {epsilon_max}")
    if not 0.0 <= tau_star < 1.0:
        raise ValueError(f"tau_star must be in [0, 1), got {tau_star}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if t_end < t_start:
        raise ValueError(
            f"t_end ({t_end}) must be >= t_start ({t_start})"
        )

    self.epsilon_max = float(epsilon_max)
    self.tau_star = float(tau_star)
    self.n_steps = int(n_steps)
    self.t_start = float(t_start)
    self.t_end = float(t_end)
    self.sqrt = bool(sqrt)
    super().__init__(self._value_at_time(self.t_start))
```

### `epsilon_at(t)`

Evaluate the raw temperature (\\epsilon(t)) at virtual time `t`.

Source code in `torchebm/core/base_scheduler.py`

```python
def epsilon_at(self, t: float) -> float:
    r"""Evaluate the raw temperature \(\epsilon(t)\) at virtual time `t`."""
    if t < self.tau_star:
        return 0.0
    if t < 1.0:
        return self.epsilon_max * (t - self.tau_star) / (1.0 - self.tau_star)
    return self.epsilon_max
```

## `TorchEBMModule`

Bases: `Module`

Base `nn.Module` with cached, parameter-derived device/dtype access.

Source code in `torchebm/core/base_module.py`

```python
class TorchEBMModule(nn.Module):
    r"""Base `nn.Module` with cached, parameter-derived device/dtype access."""

    def __init__(
        self,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        probe_dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.register_buffer(
            "_torchebm_probe",
            torch.empty(0, dtype=probe_dtype, device=device),
            persistent=False,
        )
        self.use_mixed_precision: bool = False
        self.autocast_available: bool = False
        self._amp_dtype: torch.dtype = torch.float16
        self._cached_device: Optional[torch.device] = None
        self._cached_dtype: Optional[torch.dtype] = None

    def _resolve_device_dtype(self) -> None:
        try:
            p = next(self.parameters())
            self._cached_device = _normalize(p.device)
            self._cached_dtype = p.dtype
            return
        except StopIteration:
            pass
        probe = self._torchebm_probe
        self._cached_device = _normalize(probe.device)
        self._cached_dtype = probe.dtype

    @property
    def device(self) -> torch.device:
        if self._cached_device is None:
            self._resolve_device_dtype()
        return self._cached_device

    @property
    def dtype(self) -> torch.dtype:
        if self._cached_dtype is None:
            self._resolve_device_dtype()
        return self._cached_dtype

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._cached_device = None
        self._cached_dtype = None
        return result

    def _prepare_model_kwargs(
        self, model_kwargs: Optional[dict]
    ) -> dict:
        r"""Normalize conditioning `model_kwargs` once at a call boundary.

        The single entry point for the library-wide conditioning convention:
        every ``sample()``/``forward()`` that forwards conditioning to the model
        calls this once, then reuses the returned dict (e.g. captured by a
        per-step drift closure) without re-normalizing.

        GPU-first: tensor values are moved to `self.device` a single time with
        ``non_blocking=True`` and are **not** dtype-cast (integer class labels /
        token ids must stay integral for embedding lookups). Non-tensor values
        pass through untouched. A fresh dict is always returned, so callers may
        mutate it without aliasing the caller's mapping.

        Args:
            model_kwargs: Conditioning mapping forwarded to the model, or None.

        Returns:
            A new dict with tensor values on `self.device`; ``{}`` when
            `model_kwargs` is None or empty.

        Raises:
            TypeError: If `model_kwargs` is neither None nor a mapping.
        """
        if not model_kwargs:
            return {}
        if not isinstance(model_kwargs, dict):
            raise TypeError(
                f"model_kwargs must be a dict, got {type(model_kwargs).__name__}"
            )
        device = self.device
        return {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in model_kwargs.items()
        }

    def setup_mixed_precision(
        self,
        use_mixed_precision: bool,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        r"""Configure autocast for `autocast_context()`.

        Args:
            use_mixed_precision: Enable autocast.
            amp_dtype: Autocast dtype (`torch.float16` default, `torch.bfloat16` supported).
        """
        self.use_mixed_precision = bool(use_mixed_precision)
        self._amp_dtype = amp_dtype
        if not self.use_mixed_precision:
            self.autocast_available = False
            return
        if self.device.type != "cuda":
            warnings.warn(
                f"Mixed precision requested but device is {self.device}. "
                f"Requires CUDA. Falling back to full precision.",
                UserWarning,
            )
            self.use_mixed_precision = False
            self.autocast_available = False
            return
        self.autocast_available = True

    def autocast_context(self):
        r"""Return an autocast context if mixed precision is enabled, else `nullcontext`."""
        if self.use_mixed_precision and self.autocast_available:
            return torch.amp.autocast(
                device_type=self.device.type, dtype=self._amp_dtype
            )
        return nullcontext()
```

### `autocast_context()`

Return an autocast context if mixed precision is enabled, else `nullcontext`.

Source code in `torchebm/core/base_module.py`

```python
def autocast_context(self):
    r"""Return an autocast context if mixed precision is enabled, else `nullcontext`."""
    if self.use_mixed_precision and self.autocast_available:
        return torch.amp.autocast(
            device_type=self.device.type, dtype=self._amp_dtype
        )
    return nullcontext()
```

### `setup_mixed_precision(use_mixed_precision, amp_dtype=torch.float16)`

Configure autocast for `autocast_context()`.

Parameters:

| Name                  | Type    | Description                                                       | Default    |
| --------------------- | ------- | ----------------------------------------------------------------- | ---------- |
| `use_mixed_precision` | `bool`  | Enable autocast.                                                  | *required* |
| `amp_dtype`           | `dtype` | Autocast dtype (torch.float16 default, torch.bfloat16 supported). | `float16`  |

Source code in `torchebm/core/base_module.py`

```python
def setup_mixed_precision(
    self,
    use_mixed_precision: bool,
    amp_dtype: torch.dtype = torch.float16,
) -> None:
    r"""Configure autocast for `autocast_context()`.

    Args:
        use_mixed_precision: Enable autocast.
        amp_dtype: Autocast dtype (`torch.float16` default, `torch.bfloat16` supported).
    """
    self.use_mixed_precision = bool(use_mixed_precision)
    self._amp_dtype = amp_dtype
    if not self.use_mixed_precision:
        self.autocast_available = False
        return
    if self.device.type != "cuda":
        warnings.warn(
            f"Mixed precision requested but device is {self.device}. "
            f"Requires CUDA. Falling back to full precision.",
            UserWarning,
        )
        self.use_mixed_precision = False
        self.autocast_available = False
        return
    self.autocast_available = True
```

## `WarmupScheduler`

Bases: `BaseScheduler`

Scheduler that combines linear warmup with another scheduler.

This scheduler implements a two-phase approach: first, it linearly increases the parameter value from a small initial value to the target value over a warmup period, then it follows the schedule defined by the main scheduler. Warmup is commonly used in deep learning to stabilize training in the initial phases.

Mathematical Formula

[v(t) = \\begin{cases} v\_{init} + (v\_{target} - v\_{init}) \\times \\frac{t}{T\_{warmup}}, & \\text{if } t < T\_{warmup} \\ \\text{main_scheduler}(t - T\_{warmup}), & \\text{if } t \\geq T\_{warmup} \\end{cases}]

where:

- (v\_{init} = v\_{target} \\times \\text{warmup_init_factor})
- (v\_{target}) is the main scheduler's start_value
- (T\_{warmup}) is warmup_steps
- (t) is the current step count

Parameters:

| Name                 | Type            | Description                                                 | Default    |
| -------------------- | --------------- | ----------------------------------------------------------- | ---------- |
| `main_scheduler`     | `BaseScheduler` | The scheduler to use after warmup.                          | *required* |
| `warmup_steps`       | `int`           | Number of warmup steps.                                     | *required* |
| `warmup_init_factor` | `float`         | Factor to determine initial warmup value. Defaults to 0.01. | `0.01`     |

Learning Rate Warmup + Cosine Annealing

```python
main_scheduler = CosineScheduler(
    start_value=0.1, end_value=0.001, n_steps=1000
)
warmup_scheduler = WarmupScheduler(
    main_scheduler=main_scheduler,
    warmup_steps=100,
    warmup_init_factor=0.01
)

# First 100 steps: linear warmup from 0.001 to 0.1
# Next 1000 steps: cosine annealing from 0.1 to 0.001
for i in range(10):
    value = warmup_scheduler.step()
    print(f"Warmup step {i+1}: {value:.6f}")
```

MCMC Sampling with Warmup

```python
decay_scheduler = ExponentialDecayScheduler(
    start_value=0.05, decay_rate=0.999, min_value=0.001
)
step_scheduler = WarmupScheduler(
    main_scheduler=decay_scheduler,
    warmup_steps=50,
    warmup_init_factor=0.1
)

sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=step_scheduler
)
```

Noise Scale Warmup

```python
linear_scheduler = LinearScheduler(
    start_value=1.0, end_value=0.01, n_steps=500
)
noise_scheduler = WarmupScheduler(
    main_scheduler=linear_scheduler,
    warmup_steps=25,
    warmup_init_factor=0.05
)
```

Source code in `torchebm/core/base_scheduler.py`

````python
class WarmupScheduler(BaseScheduler):
    r"""
    Scheduler that combines linear warmup with another scheduler.

    This scheduler implements a two-phase approach: first, it linearly increases
    the parameter value from a small initial value to the target value over a
    warmup period, then it follows the schedule defined by the main scheduler.
    Warmup is commonly used in deep learning to stabilize training in the
    initial phases.

    !!! info "Mathematical Formula"
        $$v(t) = \begin{cases}
        v_{init} + (v_{target} - v_{init}) \times \frac{t}{T_{warmup}}, & \text{if } t < T_{warmup} \\
        \text{main\_scheduler}(t - T_{warmup}), & \text{if } t \geq T_{warmup}
        \end{cases}$$

        where:

        - \(v_{init} = v_{target} \times \text{warmup\_init\_factor}\)
        - \(v_{target}\) is the main scheduler's start_value
        - \(T_{warmup}\) is warmup_steps
        - \(t\) is the current step count

    Args:
        main_scheduler (BaseScheduler): The scheduler to use after warmup.
        warmup_steps (int): Number of warmup steps.
        warmup_init_factor (float, optional): Factor to determine initial warmup value.
            Defaults to 0.01.

    !!! example "Learning Rate Warmup + Cosine Annealing"
        ```python
        main_scheduler = CosineScheduler(
            start_value=0.1, end_value=0.001, n_steps=1000
        )
        warmup_scheduler = WarmupScheduler(
            main_scheduler=main_scheduler,
            warmup_steps=100,
            warmup_init_factor=0.01
        )

        # First 100 steps: linear warmup from 0.001 to 0.1
        # Next 1000 steps: cosine annealing from 0.1 to 0.001
        for i in range(10):
            value = warmup_scheduler.step()
            print(f"Warmup step {i+1}: {value:.6f}")
        ```

    !!! tip "MCMC Sampling with Warmup"
        ```python
        decay_scheduler = ExponentialDecayScheduler(
            start_value=0.05, decay_rate=0.999, min_value=0.001
        )
        step_scheduler = WarmupScheduler(
            main_scheduler=decay_scheduler,
            warmup_steps=50,
            warmup_init_factor=0.1
        )

        sampler = LangevinDynamics(
            energy_function=energy_fn,
            step_size=step_scheduler
        )
        ```

    !!! example "Noise Scale Warmup"
        ```python
        linear_scheduler = LinearScheduler(
            start_value=1.0, end_value=0.01, n_steps=500
        )
        noise_scheduler = WarmupScheduler(
            main_scheduler=linear_scheduler,
            warmup_steps=25,
            warmup_init_factor=0.05
        )
        ```
    """

    def __init__(
        self,
        main_scheduler: BaseScheduler,
        warmup_steps: int,
        warmup_init_factor: float = 0.01,
    ):
        r"""
        Initialize the warmup scheduler.

        Args:
            main_scheduler (BaseScheduler): The scheduler to use after warmup.
            warmup_steps (int): Number of warmup steps.
            warmup_init_factor (float, optional): Factor to determine initial warmup value.
                The initial value will be main_scheduler.start_value * warmup_init_factor.
                Defaults to 0.01.
        """
        super().__init__(main_scheduler.start_value * warmup_init_factor)
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_init_factor = warmup_init_factor
        self.target_value = main_scheduler.start_value
        self.main_scheduler.reset()

    def _compute_value(self) -> float:
        r"""Pure read: linear-interpolate during warmup, otherwise mirror main scheduler.

        After warmup, `step()` drives the main scheduler in lockstep, so its
        `current_value` is always at the right step.
        """
        if self.step_count <= self.warmup_steps:
            progress = self.step_count / self.warmup_steps
            return self.start_value + progress * (self.target_value - self.start_value)
        return self.main_scheduler.current_value

    def step(self) -> float:
        r"""Advance one step, driving the main scheduler in lockstep post-warmup."""
        self.step_count += 1
        if self.step_count > self.warmup_steps:
            self.main_scheduler.step()
        self.current_value = self._compute_value()
        return self.current_value

    def reset(self) -> None:
        super().reset()
        self.main_scheduler.reset()
````

### `__init__(main_scheduler, warmup_steps, warmup_init_factor=0.01)`

Initialize the warmup scheduler.

Parameters:

| Name                 | Type            | Description                                                                                                                            | Default    |
| -------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `main_scheduler`     | `BaseScheduler` | The scheduler to use after warmup.                                                                                                     | *required* |
| `warmup_steps`       | `int`           | Number of warmup steps.                                                                                                                | *required* |
| `warmup_init_factor` | `float`         | Factor to determine initial warmup value. The initial value will be main_scheduler.start_value * warmup_init_factor. Defaults to 0.01. | `0.01`     |

Source code in `torchebm/core/base_scheduler.py`

```python
def __init__(
    self,
    main_scheduler: BaseScheduler,
    warmup_steps: int,
    warmup_init_factor: float = 0.01,
):
    r"""
    Initialize the warmup scheduler.

    Args:
        main_scheduler (BaseScheduler): The scheduler to use after warmup.
        warmup_steps (int): Number of warmup steps.
        warmup_init_factor (float, optional): Factor to determine initial warmup value.
            The initial value will be main_scheduler.start_value * warmup_init_factor.
            Defaults to 0.01.
    """
    super().__init__(main_scheduler.start_value * warmup_init_factor)
    self.main_scheduler = main_scheduler
    self.warmup_steps = warmup_steps
    self.warmup_init_factor = warmup_init_factor
    self.target_value = main_scheduler.start_value
    self.main_scheduler.reset()
```

### `step()`

Advance one step, driving the main scheduler in lockstep post-warmup.

Source code in `torchebm/core/base_scheduler.py`

```python
def step(self) -> float:
    r"""Advance one step, driving the main scheduler in lockstep post-warmup."""
    self.step_count += 1
    if self.step_count > self.warmup_steps:
        self.main_scheduler.step()
    self.current_value = self._compute_value()
    return self.current_value
```

## `expand_t_like_x(t, x)`

Expand time tensor to match spatial dimensions of x.

Parameters:

| Name | Type     | Description                                  | Default    |
| ---- | -------- | -------------------------------------------- | ---------- |
| `t`  | `Tensor` | Time tensor of shape (batch_size,).          | *required* |
| `x`  | `Tensor` | Reference tensor of shape (batch_size, ...). | *required* |

Returns:

| Type     | Description                                            |
| -------- | ------------------------------------------------------ |
| `Tensor` | Time tensor expanded to shape (batch_size, 1, 1, ...). |

Source code in `torchebm/core/base_interpolant.py`

```python
def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Expand time tensor to match spatial dimensions of x.

    Args:
        t: Time tensor of shape (batch_size,).
        x: Reference tensor of shape (batch_size, ...).

    Returns:
        Time tensor expanded to shape (batch_size, 1, 1, ...).
    """
    dims = [1] * (x.ndim - 1)
    return t.view(t.size(0), *dims)
```

## `warn_once(key, message, category=DeprecationWarning, stacklevel=3)`

Emit a warning at most once per process, keyed by `key`.

Deprecation paths on hot code (per-step sampler loops, per-batch losses) must not call `warnings.warn` every iteration: even when the filter shows a warning only once, the per-call filter processing adds avoidable overhead. This guard skips the call entirely after the first hit.

Source code in `torchebm/core/base_module.py`

```python
def warn_once(
    key: str,
    message: str,
    category: type = DeprecationWarning,
    stacklevel: int = 3,
) -> None:
    r"""Emit a warning at most once per process, keyed by `key`.

    Deprecation paths on hot code (per-step sampler loops, per-batch losses)
    must not call `warnings.warn` every iteration: even when the filter shows a
    warning only once, the per-call filter processing adds avoidable overhead.
    This guard skips the call entirely after the first hit.
    """
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    warnings.warn(message, category, stacklevel=stacklevel)
```
