"""
fdtd_reference.py
=================
Yee-staggered, leap-frog FDTD solver for the lossy telegrapher equations:

    ∂V/∂x + L' ∂I/∂t + R' I = 0
    ∂I/∂x + C' ∂V/∂t + G' V = 0

Voltages V live on integer x-nodes (i=0..Nx-1) and integer time steps.
Currents I live on half-integer x-nodes (cell centers, Nx-1 of them) and
half-integer time steps. The discretization is second-order accurate.

Used as ground truth for the PINN benchmark.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class LineParams:
    Rp: float                # Ω/m
    Lp: float                # H/m
    Gp: float                # S/m
    Cp: float                # F/m
    length: float = 1.0      # m
    Zs: float = 50.0         # source impedance (Ω)
    ZL: float = 50.0         # load impedance (Ω)

    @property
    def c(self):  return 1.0 / np.sqrt(self.Lp * self.Cp)

    @property
    def Z0(self): return np.sqrt(self.Lp / self.Cp)

    @property
    def alpha(self):
        """Low-loss attenuation constant (Np/m)."""
        return 0.5 * (self.Rp / self.Z0 + self.Gp * self.Z0)


def gaussian_pulse(t, t0=2e-9, sigma=0.5e-9, amp=1.0):
    """Smooth band-limited source, well-suited for benchmarking."""
    return amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def fdtd_solve(params: LineParams, T: float,
               Nx: int = 201, cfl: float = 0.95,
               source=gaussian_pulse, record_every: int = 1):
    """
    Solve on [0, length] x [0, T]. Returns x_grid (Nx,), t_hist (Nt_rec,),
    V_hist (Nt_rec, Nx), I_hist (Nt_rec, Nx-1), info dict.
    """
    L = params.length
    dx = L / (Nx - 1)
    dt = cfl * dx / params.c
    Nt = int(np.ceil(T / dt)) + 1
    dt = T / (Nt - 1)

    x_grid = np.linspace(0.0, L, Nx)
    xI_grid = 0.5 * (x_grid[:-1] + x_grid[1:])

    V = np.zeros(Nx)
    I = np.zeros(Nx - 1)

    Rp, Lp, Gp, Cp = params.Rp, params.Lp, params.Gp, params.Cp
    aV = (Cp / dt - Gp / 2.0); bV = (Cp / dt + Gp / 2.0)
    aI = (Lp / dt - Rp / 2.0); bI = (Lp / dt + Rp / 2.0)

    n_rec = Nt // record_every + 2
    V_hist = np.zeros((n_rec, Nx))
    I_hist = np.zeros((n_rec, Nx - 1))
    t_hist = np.zeros(n_rec)
    V_hist[0] = V; I_hist[0] = I; t_hist[0] = 0.0
    rec = 1

    Zs, ZL = params.Zs, params.ZL
    for n in range(1, Nt):
        t_now = n * dt
        V_new = V.copy()
        V_new[1:-1] = (aV * V[1:-1] - (I[1:] - I[:-1]) / dx) / bV
        # Source (Thevenin): V(0) = Vs - Zs * I(0+)
        V_new[0] = source(t_now) - Zs * I[0]
        # Load: V(L) = ZL * I(L-)
        V_new[-1] = ZL * I[-1]
        I_new = (aI * I - (V_new[1:] - V_new[:-1]) / dx) / bI
        V, I = V_new, I_new

        if n % record_every == 0 and rec < n_rec:
            V_hist[rec] = V; I_hist[rec] = I; t_hist[rec] = t_now
            rec += 1

    info = dict(dx=dx, dt=dt, Nt=Nt, c=params.c, Z0=params.Z0,
                alpha=params.alpha, xI=xI_grid, cfl=cfl)
    return x_grid, t_hist[:rec], V_hist[:rec], I_hist[:rec], info


def interp_I_to_V_grid(x_grid, I_hist, xI):
    """Resample currents onto the voltage grid (linear)."""
    out = np.zeros((I_hist.shape[0], x_grid.size))
    for k in range(I_hist.shape[0]):
        out[k] = np.interp(x_grid, xI, I_hist[k])
    return out


if __name__ == "__main__":
    p = LineParams(Rp=0.1, Lp=2.5e-7, Gp=1e-5, Cp=1.0e-10)
    print(f"c={p.c:.3e}  Z0={p.Z0:.2f}  α={p.alpha:.3e}")
    x, t, V, I, info = fdtd_solve(p, T=12e-9, Nx=201, cfl=0.95)
    print(f"grid {x.size}x{t.size}  dt={info['dt']:.3e}  max|V|={np.max(np.abs(V)):.3f}")
