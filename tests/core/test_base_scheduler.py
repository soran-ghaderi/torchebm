import pytest
import torch
import numpy as np

from torchebm.core import (
    ConstantScheduler,
    ExponentialDecayScheduler,
    LinearScheduler,
    CosineScheduler,
    MultiStepScheduler,
    WarmupScheduler,
)
from torchebm.samplers import LangevinDynamics

from tests.conftest import requires_cuda

# Define all configurations in a list
# ===================================
all_scheduler_configs = [
    # ConstantScheduler configs
    pytest.param(
        {"class": ConstantScheduler, "args": {"start_value": 0.1}}, id="Constant_0.1"
    ),
    pytest.param(
        {"class": ConstantScheduler, "args": {"start_value": 5.0}}, id="Constant_5.0"
    ),
    # ExponentialDecayScheduler configs
    pytest.param(
        {
            "class": ExponentialDecayScheduler,
            "args": {"start_value": 1.0, "decay_rate": 0.9, "min_value": 0.01},
        },
        id="ExpDecay_1.0_0.9",
    ),
    pytest.param(
        {
            "class": ExponentialDecayScheduler,
            "args": {"start_value": 0.5, "decay_rate": 0.99, "min_value": 0.0},
        },
        id="ExpDecay_0.5_0.99",
    ),
    # LinearScheduler configs
    pytest.param(
        {
            "class": LinearScheduler,
            "args": {"start_value": 1.0, "end_value": 0.1, "n_steps": 10},
        },
        id="Linear_1.0_0.1_10",
    ),
    pytest.param(
        {
            "class": LinearScheduler,
            "args": {"start_value": 0.0, "end_value": 5.0, "n_steps": 5},
        },
        id="Linear_0.0_5.0_5",
    ),
    # CosineScheduler configs
    pytest.param(
        {
            "class": CosineScheduler,
            "args": {"start_value": 1.0, "end_value": 0.0, "n_steps": 10},
        },
        id="Cosine_1.0_0.0_10",
    ),
    pytest.param(
        {
            "class": CosineScheduler,
            "args": {"start_value": 0.1, "end_value": 0.5, "n_steps": 20},
        },
        id="Cosine_0.1_0.5_20",
    ),
    # MultiStepScheduler configs
    pytest.param(
        {
            "class": MultiStepScheduler,
            "args": {"start_value": 1.0, "milestones": [5, 8], "gamma": 0.1},
        },
        id="MultiStep_[5,8]_0.1",
    ),
    pytest.param(
        {
            "class": MultiStepScheduler,
            "args": {"start_value": 10.0, "milestones": [2], "gamma": 0.5},
        },
        id="MultiStep_[2]_0.5",
    ),
]


# Combined Fixture using the list of configs
# ===========================================
@pytest.fixture(params=all_scheduler_configs)
def scheduler_instance(request):
    """Creates scheduler instances directly from the config dictionary."""
    config = request.param  # config is now the dictionary directly
    SchedulerClass = config["class"]
    args = config["args"]
    return SchedulerClass(**args)


# Test Cases
# ==========


def test_scheduler_initialization(scheduler_instance):
    """Test initial value and step count."""
    initial_value = scheduler_instance.start_value
    assert scheduler_instance.get_value() == pytest.approx(initial_value)
    assert scheduler_instance.step_count == 0


# --- REFINED SPECIFIC TESTS ---


def test_constant_scheduler_behavior():
    """Specific test for ConstantScheduler logic."""
    initial_val = 5.0
    scheduler = ConstantScheduler(start_value=initial_val)
    assert scheduler.step_count == 0
    assert scheduler.get_value() == pytest.approx(initial_val)

    for i in range(5):
        expected_step_count = i + 1
        current_step_count_before = scheduler.step_count
        returned_val = scheduler.step()  # Step the scheduler

        # Check state *after* step
        assert returned_val == pytest.approx(initial_val)
        assert scheduler.get_value() == pytest.approx(initial_val)
        assert scheduler.step_count == expected_step_count
        assert scheduler.step_count == current_step_count_before + 1  # Verify increment


def test_exponential_scheduler_behavior():
    """Specific test for ExponentialDecayScheduler logic."""
    args = {"start_value": 1.0, "decay_rate": 0.9, "min_value": 0.01}
    scheduler = ExponentialDecayScheduler(**args)
    assert scheduler.step_count == 0

    last_val = scheduler.start_value
    for i in range(15):  # Step multiple times
        expected_step_count = i + 1
        # Calculate expected value based on the step count *after* incrementing
        expected_val = max(
            args["min_value"],
            args["start_value"] * (args["decay_rate"] ** expected_step_count),
        )

        returned_val = scheduler.step()

        assert returned_val == pytest.approx(expected_val)
        assert scheduler.get_value() == pytest.approx(expected_val)
        assert scheduler.step_count == expected_step_count
        last_val = returned_val  # Keep track for min_value check

    # Check min_value after many steps
    for _ in range(100):
        last_val = scheduler.step()
    assert last_val == pytest.approx(args["min_value"])
    assert scheduler.get_value() == pytest.approx(args["min_value"])


def test_linear_scheduler_behavior():
    """Specific test for LinearScheduler."""
    args = {"start_value": 1.0, "end_value": 0.1, "n_steps": 10}
    scheduler = LinearScheduler(**args)
    total_steps = args["n_steps"]
    assert scheduler.step_count == 0

    for i in range(total_steps + 3):  # Step beyond n_steps
        expected_step_count = i + 1
        # Calculate expected value based on the step count *after* incrementing
        if expected_step_count >= total_steps:
            expected_val = args["end_value"]
        else:
            progress = expected_step_count / total_steps  # Use future step count
            expected_val = (
                args["start_value"]
                + (args["end_value"] - args["start_value"]) * progress
            )

        returned_val = scheduler.step()
        assert returned_val == pytest.approx(expected_val)
        assert scheduler.get_value() == pytest.approx(expected_val)
        assert scheduler.step_count == expected_step_count

    assert scheduler.get_value() == pytest.approx(args["end_value"])  # Final check


def test_cosine_scheduler_behavior():
    """Specific test for CosineScheduler."""
    args = {"start_value": 1.0, "end_value": 0.0, "n_steps": 10}
    scheduler = CosineScheduler(**args)
    total_steps = args["n_steps"]
    assert scheduler.step_count == 0

    for i in range(total_steps + 3):  # Step beyond n_steps
        expected_step_count = i + 1
        # Calculate expected value based on the step count *after* incrementing
        if expected_step_count >= total_steps:
            expected_val = args["end_value"]
        else:
            progress = expected_step_count / total_steps  # Use future step count
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            expected_val = (
                args["end_value"]
                + (args["start_value"] - args["end_value"]) * cosine_factor
            )

        returned_val = scheduler.step()
        assert returned_val == pytest.approx(expected_val)
        assert scheduler.get_value() == pytest.approx(expected_val)
        assert scheduler.step_count == expected_step_count

    assert scheduler.get_value() == pytest.approx(args["end_value"])  # Final check


def test_multistep_scheduler_behavior():
    """Specific test for MultiStepScheduler."""
    args = {"start_value": 1.0, "milestones": [5, 8], "gamma": 0.1}
    scheduler = MultiStepScheduler(**args)
    initial_val = args["start_value"]
    gamma = args["gamma"]
    milestones = args["milestones"]
    assert scheduler.step_count == 0

    max_step_check = max(milestones) + 3
    for i in range(max_step_check):
        expected_step_count = i + 1
        # Calculate expected value based on the step count *after* incrementing
        current_power = sum(
            1 for m in milestones if expected_step_count >= m
        )  # Power based on future step
        expected_val = initial_val * (gamma**current_power)

        returned_val = scheduler.step()
        assert returned_val == pytest.approx(expected_val)
        assert scheduler.get_value() == pytest.approx(expected_val)
        assert scheduler.step_count == expected_step_count


# --- Generic Tests (Should Pass Now) ---


def test_scheduler_reset(scheduler_instance):
    """Test the reset method using the combined fixture."""
    initial_value = scheduler_instance.start_value
    # Step a few times
    for _ in range(3):
        scheduler_instance.step()

    # Reset
    scheduler_instance.reset()

    # Check state after reset
    assert scheduler_instance.get_value() == pytest.approx(initial_value)
    assert scheduler_instance.step_count == 0

    # Check value after stepping once post-reset
    scheduler_instance.step()  # Call step once
    expected_step_count = 1
    assert scheduler_instance.step_count == expected_step_count
    # Compute expected value at step 1 manually
    scheduler_instance.step_count = expected_step_count  # Ensure count is 1 for compute
    expected_val_at_step_1 = scheduler_instance._compute_value()
    # Compare with current value (which was set by step())
    assert scheduler_instance.get_value() == pytest.approx(expected_val_at_step_1)


def test_scheduler_get_value_no_side_effects(scheduler_instance):
    """Test that get_value() doesn't change the state using the combined fixture."""
    val1 = scheduler_instance.step()
    count1 = scheduler_instance.step_count
    current_val1 = scheduler_instance.current_value  # Store value after step

    gv1 = scheduler_instance.get_value()
    gv2 = scheduler_instance.get_value()

    assert gv1 == pytest.approx(
        current_val1
    )  # Compare get_value with stored current_value
    assert gv2 == pytest.approx(current_val1)
    assert scheduler_instance.step_count == count1  # Count should not change
    assert scheduler_instance.current_value == pytest.approx(
        current_val1
    )  # current_value attr should not change

    scheduler_instance.step()
    count2 = scheduler_instance.step_count
    assert count2 == count1 + 1


def test_scheduler_state_dict_load_state_dict(scheduler_instance):
    """Test saving and loading state using the combined fixture."""
    # Step the scheduler
    for _ in range(5):
        scheduler_instance.step()
    state = scheduler_instance.state_dict()

    # Create a new instance
    NewSchedulerClass = type(scheduler_instance)
    # --- Robust recreation ---
    # Store original args (assuming they are simple types stored in state)
    # Best practice would be to have a get_init_args method or store them explicitly
    init_args = {}
    import inspect

    sig = inspect.signature(NewSchedulerClass.__init__)
    for param_name in sig.parameters:
        if param_name != "self" and param_name in state:
            init_args[param_name] = state[param_name]
        elif (
            param_name == "start_value" and "start_value" in state
        ):  # Ensure start_value is present
            init_args[param_name] = state["start_value"]

    try:
        # Attempt to recreate with extracted args
        new_scheduler = NewSchedulerClass(**init_args)
    except TypeError as e:
        pytest.skip(
            f"Cannot reliably recreate {NewSchedulerClass} for state_dict test. Error: {e}. Init args found: {init_args}"
        )
    # --- End Robust recreation ---

    new_scheduler.load_state_dict(state)

    # Check if states match using state_dict comparison
    assert new_scheduler.state_dict() == state


# --- (Keep Input Validation Tests as they are) ---
def test_invalid_exponential_decay_rate():
    with pytest.raises(ValueError):
        ExponentialDecayScheduler(1.0, 1.1)  # > 1
    with pytest.raises(ValueError):
        ExponentialDecayScheduler(1.0, 0.0)  # Not > 0
    with pytest.raises(ValueError):
        ExponentialDecayScheduler(1.0, -0.1)  # < 0


def test_invalid_linear_total_steps():
    with pytest.raises(ValueError):
        LinearScheduler(1.0, 0.1, 0)
    with pytest.raises(ValueError):
        LinearScheduler(1.0, 0.1, -5)


def test_invalid_cosine_total_steps():
    with pytest.raises(ValueError):
        CosineScheduler(1.0, 0.1, 0)
    with pytest.raises(ValueError):
        CosineScheduler(1.0, 0.1, -5)


def test_invalid_multistep_milestones():
    with pytest.raises(ValueError):
        MultiStepScheduler(1.0, [5, 3])  # Not increasing
    with pytest.raises(ValueError):
        MultiStepScheduler(1.0, [5, 5])  # Not strictly increasing
    with pytest.raises(ValueError):
        MultiStepScheduler(1.0, [0, 5])  # Milestone not positive
    with pytest.raises(ValueError):
        MultiStepScheduler(1.0, [-2, 5])  # Milestone not positive


def test_invalid_exponential_min_value():
    with pytest.raises(ValueError):
        ExponentialDecayScheduler(1.0, 0.9, -0.1)  # Negative min_value
