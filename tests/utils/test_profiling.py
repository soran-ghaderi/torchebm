"""Tests for the public profiling helpers."""

from torch.profiler import ProfilerActivity

from torchebm.utils.profiling import profile_context, record_function


def test_profile_context_prints_recorded_timing(capsys):
    with profile_context(
        activities=[ProfilerActivity.CPU],
        profile_memory=False,
    ) as profiler:
        with record_function("profiled_block"):
            sum(range(10))

    output = capsys.readouterr().out
    assert "profiled_block" in output
    assert "CPU time" in output
    assert "profiled_block" in {event.key for event in profiler.key_averages()}


def test_record_function_decorator_preserves_return_value():
    @record_function("profiled_function")
    def add(left, right):
        return left + right

    with profile_context(
        activities=[ProfilerActivity.CPU],
        profile_memory=False,
        print_table=False,
    ) as profiler:
        result = add(20, 22)

    assert result == 42
    assert "profiled_function" in {event.key for event in profiler.key_averages()}
