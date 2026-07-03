"""Convenience functions for users who want the core workflow directly.

The original research scripts remain available at the repository root. This
module provides a small stable API for examples, notebooks, and external users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fdtd_reference import LineParams, fdtd_solve, interp_I_to_V_grid
from pinn_core import Problem
from run_forward import make_sets, train, eval_grid


@dataclass(frozen=True)
class ReferenceSolution:
    """FDTD reference data on the voltage-node grid."""

    x: np.ndarray
    t: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    info: dict[str, Any]


def benchmark_line_params() -> LineParams:
    """Return the manuscript benchmark line parameters."""

    return LineParams(
        Rp=1.0e-1,
        Lp=2.5e-7,
        Gp=1.0e-5,
        Cp=1.0e-10,
        length=1.0,
        Zs=50.0,
        ZL=50.0,
    )


def make_benchmark_problem(T: float = 12e-9) -> tuple[LineParams, Problem]:
    """Return matching FDTD and PINN problem objects for the paper benchmark."""

    params = benchmark_line_params()
    problem = Problem(
        Rp=params.Rp,
        Lp=params.Lp,
        Gp=params.Gp,
        Cp=params.Cp,
        length=params.length,
        T=T,
        V_star=1.0,
    )
    return params, problem


def solve_reference(T: float = 12e-9, Nx: int = 201, record_every: int = 2) -> ReferenceSolution:
    """Solve the manuscript FDTD reference case."""

    params = benchmark_line_params()
    x, t, voltage, current_half, info = fdtd_solve(
        params,
        T=T,
        Nx=Nx,
        cfl=0.95,
        record_every=record_every,
    )
    current = interp_I_to_V_grid(x, current_half, info["xI"])
    return ReferenceSolution(x=x, t=t, voltage=voltage, current=current, info=info)


def run_smoke_training(n_epochs: int = 20, seed: int = 42) -> dict[str, Any]:
    """Run a small PINN training job to verify installation.

    This is intentionally much smaller than the manuscript run. Use
    ``python main.py`` for the full reproduction pipeline.
    """

    params, problem = make_benchmark_problem()
    ref = solve_reference(Nx=61, record_every=2)
    sets = make_sets(
        problem,
        params,
        ref.x,
        ref.t,
        ref.voltage,
        ref.current,
        Np=96,
        Ni=24,
        Nb=48,
        Nd=12,
        noise_pct=0.03,
        seed=seed,
    )
    net, history, weights = train(
        problem,
        sets,
        layers=(2, 16, 16, 2),
        n_epochs=n_epochs,
        lr=2e-3,
        w0=(1e-2, 10.0, 10.0, 5.0),
        adapt_every=max(10, n_epochs + 1),
        log_every=max(1, n_epochs // 5),
        seed=seed,
        verbose_every=max(1, n_epochs + 1),
    )
    voltage_pred, current_pred, err_voltage, err_current = eval_grid(
        net,
        problem,
        ref.x,
        ref.t,
        ref.voltage,
        ref.current,
    )
    return {
        "net": net,
        "history": history,
        "weights": weights,
        "reference": ref,
        "voltage_pred": voltage_pred,
        "current_pred": current_pred,
        "relative_l2_voltage": err_voltage,
        "relative_l2_current": err_current,
    }
