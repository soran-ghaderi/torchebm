"""
Scheduler classes for MCMC samplers and related algorithms.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseScheduler(ABC):
    """Base class for parameter schedulers.

    Args:
    start_value: Initial parameter value
    """

    def __init__(self, start_value: float):
        if not isinstance(start_value, (float, int)):
            raise TypeError(f"{type(self).__name__} received an invalid start_value")

        self.start_value = start_value
        self.current_value = start_value
        self.step_count = 0

    @abstractmethod
    def _compute_value(self) -> float:
        """Compute the value for the current step count. To be implemented by subclasses."""
        pass

    def step(self) -> float:
        """Advance the scheduler by one step and return the new value."""
        self.step_count += 1
        self.current_value = self._compute_value()
        return self.current_value

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_value = self.start_value
        self.step_count = 0

    def get_value(self) -> float:
        """Get current value without updating."""
        return self.current_value

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the scheduler as a :class:`dict`."""
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)


class ConstantScheduler(BaseScheduler):
    """Scheduler that maintains a constant value."""

    def _compute_value(self) -> float:
        """Return constant value."""
        return self.current_value


class ExponentialDecayScheduler(BaseScheduler):
    """Scheduler with exponential decay: value = initial * (decay_rate ** step_count)."""

    def __init__(
        self,
        start_value: float,
        decay_rate: float,
        min_value: float = 0.0,  # Default min_value to 0
    ):
        super().__init__(start_value)
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError("decay_rate must be in (0, 1]")
        if min_value < 0:
            raise ValueError("min_value must be non-negative")
        self.decay_rate: float = decay_rate
        self.min_value: float = min_value

    def _compute_value(self) -> float:
        """Compute value with exponential decay."""
        # Use self.step_count which was incremented in step()
        val = self.start_value * (self.decay_rate**self.step_count)
        return max(self.min_value, val)


class LinearScheduler(BaseScheduler):
    """Scheduler with linear annealing.

    Args:
        start_value: Starting parameter value
        end_value: Target parameter value
        n_steps: Number of steps to reach final value
    """

    def __init__(self, start_value: float, end_value: float, n_steps: int):
        super().__init__(start_value)
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.end_value = end_value
        self.n_steps = n_steps
        if n_steps > 0:
            self.step_size: float = (end_value - start_value) / n_steps
        else:
            self.step_size = 0.0  # Or handle n_steps=0 case appropriately

    def _compute_value(self) -> float:
        """Update value with linear change."""
        if self.step_count >= self.n_steps:
            self.current_value = self.end_value
        else:
            self.current_value = self.start_value + self.step_size * self.step_count
        return self.current_value


class CosineScheduler(BaseScheduler):
    """Scheduler with cosine annealing.

    Args:
        start_value: Starting parameter value
        end_value: Target parameter value
        n_steps: Number of steps to reach final value
    """

    def __init__(self, start_value: float, end_value: float, n_steps: int):
        super().__init__(start_value)
        if n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")

        self.end_value = end_value
        self.n_steps = n_steps

    def _compute_value(self) -> float:
        """Update value with cosine annealing."""
        if self.step_count >= self.n_steps:
            self.current_value = self.end_value
        else:
            # Cosine schedule from start_value to end_value
            progress = self.step_count / self.n_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            self.current_value = (
                self.end_value + (self.start_value - self.end_value) * cosine_factor
            )
        return self.current_value


class MultiStepScheduler(BaseScheduler):
    def __init__(self, start_value: float, milestones: List[int], gamma: float = 0.1):
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
        power = sum(1 for m in self.milestones if self.step_count >= m)
        return self.start_value * (self.gamma**power)


class WarmupScheduler(BaseScheduler):
    def __init__(
        self,
        main_scheduler: BaseScheduler,
        warmup_steps: int,
        warmup_init_factor: float = 0.01,
    ):
        # Initialize based on the main scheduler's initial value
        super().__init__(main_scheduler.start_value * warmup_init_factor)
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_init_factor = warmup_init_factor
        self.target_value = main_scheduler.start_value  # Store the target after warmup

        # Reset main scheduler as warmup controls the initial phase
        self.main_scheduler.reset()

    def _compute_value(self) -> float:
        if self.step_count < self.warmup_steps:
            # Linear warmup
            progress = self.step_count / self.warmup_steps
            return self.start_value + progress * (self.target_value - self.start_value)
        else:
            # After warmup, step the main scheduler
            # We need its value based on steps *after* warmup
            main_scheduler_step = self.step_count - self.warmup_steps
            # Temporarily set main scheduler state, get value, restore state (a bit hacky)
            original_step = self.main_scheduler.step_count
            original_value = self.main_scheduler.current_value
            self.main_scheduler.step_count = main_scheduler_step
            computed_main_value = self.main_scheduler._compute_value()
            # Restore state
            self.main_scheduler.step_count = original_step
            self.main_scheduler.current_value = original_value
            return computed_main_value
