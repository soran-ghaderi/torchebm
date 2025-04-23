"""
Scheduler classes for MCMC samplers and related algorithms.
"""

import math
from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    """Base class for parameter schedulers."""

    def __init__(self, initial_value: float):
        """
        Initialize scheduler with initial value.

        Args:
            initial_value: Initial parameter value
        """
        self.initial_value = initial_value
        self.current_value = initial_value
        self.step_count = 0

    @abstractmethod
    def step(self) -> float:
        """Update internal state and return current value."""
        pass

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_value = self.initial_value
        self.step_count = 0

    def get_value(self) -> float:
        """Get current value without updating."""
        return self.current_value


class ConstantScheduler(BaseScheduler):
    """Scheduler that maintains a constant value."""

    def step(self) -> float:
        """Return constant value."""
        self.step_count += 1
        return self.current_value


class ExponentialDecayScheduler(BaseScheduler):
    """Scheduler with exponential decay."""

    def __init__(
        self, initial_value: float, decay_rate: float, min_value: float = 1e-10
    ):
        """
        Initialize scheduler with exponential decay.

        Args:
            initial_value: Initial parameter value
            decay_rate: Rate of exponential decay (0 to 1)
            min_value: Minimum value allowed
        """
        super().__init__(initial_value)
        self.decay_rate = decay_rate
        self.min_value = min_value

    def step(self) -> float:
        """Update value by applying exponential decay."""
        self.step_count += 1
        self.current_value = max(
            self.min_value, self.initial_value * (self.decay_rate**self.step_count)
        )
        return self.current_value


class LinearScheduler(BaseScheduler):
    """Scheduler with linear annealing."""

    def __init__(self, initial_value: float, final_value: float, total_steps: int):
        """
        Initialize scheduler with linear annealing.

        Args:
            initial_value: Starting parameter value
            final_value: Target parameter value
            total_steps: Number of steps to reach final value
        """
        super().__init__(initial_value)
        self.final_value = final_value
        self.total_steps = total_steps
        self.step_size = (final_value - initial_value) / total_steps

    def step(self) -> float:
        """Update value with linear change."""
        self.step_count += 1
        if self.step_count >= self.total_steps:
            self.current_value = self.final_value
        else:
            self.current_value = self.initial_value + self.step_size * self.step_count
        return self.current_value


class CosineScheduler(BaseScheduler):
    """Scheduler with cosine annealing."""

    def __init__(self, initial_value: float, final_value: float, total_steps: int):
        """
        Initialize scheduler with cosine annealing.

        Args:
            initial_value: Starting parameter value
            final_value: Target parameter value
            total_steps: Number of steps to reach final value
        """
        super().__init__(initial_value)
        self.final_value = final_value
        self.total_steps = total_steps

    def step(self) -> float:
        """Update value with cosine annealing."""
        self.step_count += 1
        if self.step_count >= self.total_steps:
            self.current_value = self.final_value
        else:
            # Cosine schedule from initial_value to final_value
            progress = self.step_count / self.total_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            self.current_value = (
                self.final_value
                + (self.initial_value - self.final_value) * cosine_factor
            )
        return self.current_value
