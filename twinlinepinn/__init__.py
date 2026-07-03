"""High-level public API for the TwinLinePINN scientific package."""

from .api import (
    benchmark_line_params,
    solve_reference,
    make_benchmark_problem,
    run_smoke_training,
)

__all__ = [
    "benchmark_line_params",
    "solve_reference",
    "make_benchmark_problem",
    "run_smoke_training",
]
