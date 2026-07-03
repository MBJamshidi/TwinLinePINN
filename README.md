# TwinLinePINN

> **Citation notice:** If you use TwinLinePINN in research, publications, or
> derivative software, please cite: Mohammad (Behdad) Jamshidi, **"A
> Physics-Informed Neural Network Framework for Lossy Telegrapher Equations
> with a Formulated Multi-Physics Environmental Extension"**, *Computation*,
> July 2026.

Open-access reference software for physics-informed neural network modelling of
lossy transmission lines.

This package accompanies the paper:

> Mohammad (Behdad) Jamshidi, **A Physics-Informed Neural Network Framework for
> Lossy Telegrapher Equations with a Formulated Multi-Physics Environmental
> Extension**, *Computation*, July 2026.

TwinLinePINN is designed as a readable scientific implementation. It keeps the
core numerical method transparent: pure NumPy neural-network training,
hand-coded input derivatives, a finite-difference time-domain reference solver,
inverse RLGC parameter identification, and figure/table generation for the
reported baseline experiments.

## What This Software Implements

The validated core is a four-term electromagnetic PINN for the lossy
telegrapher equations:

```text
(x, t) -> (V, I)
```

with residuals:

```text
rV = dV/dx + L' dI/dt + R' I
rI = dI/dx + C' dV/dt + G' V
```

and loss:

```text
L4 = wp L_pde + wi L_ic + wb L_bc + wd L_data
```

The benchmark is a matched 50 ohm line over a 1 m, 12 ns electromagnetic
transient window. The package validates the PINN against a Yee-staggered FDTD
reference solver and includes inverse identification of `(R', L', G', C')`.

## Formulated Extension Scope

The paper also formulates a multi-physics environmental extension:

```text
(x, t, e) -> (V, I, T_line)
```

where `e` is an ambient vector including temperature, wind, solar irradiance,
humidity, and related operating conditions. TwinLinePINN includes utility
functions for temperature-adjusted resistance and steady IEEE-style thermal
balance calculations, but the reported numerical validation intentionally keeps
thermal, dynamic-line-rating, sag-tension, and online-update terms inactive.

This separation is important: nanosecond electromagnetic wave propagation and
minute-to-hour conductor heating are physically different time scales.

## Installation

From a local clone:

```bash
pip install -e .
```

For the optional dashboard dependencies:

```bash
pip install -e ".[dashboard]"
```

The minimal scientific stack is NumPy and Matplotlib.

## Quick Start

Run a small smoke example:

```bash
python examples/quickstart.py
```

Or use the package API:

```python
from twinlinepinn import solve_reference, run_smoke_training

ref = solve_reference()
print(ref.voltage.shape, ref.current.shape)

result = run_smoke_training(n_epochs=20)
print(result["relative_l2_voltage"], result["relative_l2_current"])
```

The smoke example is intentionally small. It verifies the installation but does
not reproduce the full manuscript numbers.

## Reproduce the Paper Baseline

Run the complete baseline pipeline:

```bash
python main.py
```

The stages are:

```text
python gradcheck.py      # analytic-gradient verification
python run_forward.py    # FDTD, clean PINN, noisy-sensor PINN, data-only ANN
python run_inverse.py    # inverse RLGC identification and FIM diagnostics
python make_figures.py   # manuscript figures and CSV tables
```

Outputs:

```text
results/forward_runs.npz
results/inverse_runs.npz
figures/fig*.png
figures/table_*.csv
```

Runtime depends on CPU and BLAS configuration. On CPU-only machines, the full
forward + inverse + figure pipeline can take more than 10 minutes.

## Repository Layout

| Path | Purpose |
|---|---|
| `twinlinepinn/api.py` | High-level user API |
| `fdtd_reference.py` | Yee-staggered FDTD reference solver |
| `pinn_core.py` | Pure NumPy MLP, input Jacobians, residuals, and gradients |
| `run_forward.py` | Forward PINN and data-only baseline experiments |
| `run_inverse.py` | Inverse RLGC identification and Fisher Information Matrix |
| `environmental_coupling.py` | Thermal and weather utility functions |
| `make_figures.py` | Manuscript figure and table generation |
| `digital_twin_dashboard.py` | Offline visualization prototype |
| `docs/methodology.md` | Paper-core methodology summary |
| `docs/implementation_guide.md` | Practical usage and implementation guide |
| `SOFTWARE_SCOPE.md` | Equation-to-file correspondence and active scope |

## Citation

If this software supports your research, please cite:

```text
Mohammad (Behdad) Jamshidi,
"A Physics-Informed Neural Network Framework for Lossy Telegrapher Equations
with a Formulated Multi-Physics Environmental Extension",
Computation, July 2026.
```

A machine-readable citation file is provided in `CITATION.cff`.

## License

MIT License. See `LICENSE`.
