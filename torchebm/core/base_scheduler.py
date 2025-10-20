r"""
Parameter schedulers for MCMC samplers and optimization algorithms.

This module provides a comprehensive set of parameter schedulers that can be used to
dynamically adjust parameters during training or sampling processes. Schedulers are
particularly useful for controlling step sizes, noise scales, learning rates, and
other hyperparameters that benefit from adaptive adjustment over time.

The schedulers implement various mathematical schedules including exponential decay,
linear annealing, cosine annealing, multi-step schedules, and warmup strategies.
All schedulers inherit from the `BaseScheduler` abstract base class, ensuring a
consistent interface across different scheduling strategies.

!!! info "Mathematical Foundation"
    Parameter scheduling involves updating a parameter value \(v(t)\) at each time step \(t\)
    according to a predefined schedule. Common patterns include:
    
    - **Exponential decay**: \(v(t) = v_0 \times \gamma^t\)
    - **Linear annealing**: \(v(t) = v_0 + (v_{end} - v_0) \times t/T\)
    - **Cosine annealing**: \(v(t) = v_{end} + (v_0 - v_{end}) \times (1 + \cos(\pi t/T))/2\)
    
    where \(v_0\) is the initial value, \(v_{end}\) is the final value, \(T\) is the total number
    of steps, and \(\gamma\) is the decay rate.

!!! example "Basic Usage"
    Basic usage with different scheduler types:
    
    ```python
    import torch
    from torchebm.core import ExponentialDecayScheduler, CosineScheduler
    
    # Exponential decay for step size
    step_scheduler = ExponentialDecayScheduler(
        start_value=0.1, decay_rate=0.99, min_value=0.001
    )
    
    # Cosine annealing for noise scale
    noise_scheduler = CosineScheduler(
        start_value=1.0, end_value=0.01, n_steps=1000
    )
    
    # Use in training loop
    for epoch in range(100):
        current_step_size = step_scheduler.step()
        current_noise = noise_scheduler.step()
        # Use current_step_size and current_noise in your algorithm
    ```

!!! tip "Integration with Samplers"
    ```python
    from torchebm.samplers import LangevinDynamics
    from torchebm.core import LinearScheduler
    
    # Create scheduler for adaptive step size
    step_scheduler = LinearScheduler(
        start_value=0.1, end_value=0.001, n_steps=500
    )
    
    # Use with Langevin sampler
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=step_scheduler,
        noise_scale=0.1
    )
    ```
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union


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
        # Initialize based on the main scheduler's initial value
        super().__init__(main_scheduler.start_value * warmup_init_factor)
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_init_factor = warmup_init_factor
        self.target_value = main_scheduler.start_value  # Store the target after warmup

        # Reset main scheduler as warmup controls the initial phase
        self.main_scheduler.reset()

    def _compute_value(self) -> float:
        r"""
        Compute the value based on warmup phase or main scheduler.

        Returns:
            float: The parameter value from warmup or main scheduler.
        """
        if self.step_count < self.warmup_steps:
            # Linear warmup phase
            progress = self.step_count / self.warmup_steps
            return self.start_value + progress * (self.target_value - self.start_value)
        else:
            # Main scheduler phase
            # We need its value based on steps *after* warmup
            main_scheduler_step = self.step_count - self.warmup_steps
            # Temporarily set main scheduler state, get value, restore state
            original_step = self.main_scheduler.step_count
            original_value = self.main_scheduler.current_value
            self.main_scheduler.step_count = main_scheduler_step
            computed_main_value = self.main_scheduler._compute_value()
            # Restore state
            self.main_scheduler.step_count = original_step
            self.main_scheduler.current_value = original_value
            return computed_main_value
