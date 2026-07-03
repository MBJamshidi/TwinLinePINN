# Implementation Guide

This guide explains how to use TwinLinePINN as an open scientific package.

## Installation

From a local clone:

```bash
pip install -e .
```

For dashboard extras:

```bash
pip install -e ".[dashboard]"
```

## Quick Smoke Test

Run a short installation check:

```bash
python examples/quickstart.py
```

This uses a much smaller network and fewer collocation points than the paper.
It is intended to verify that imports, FDTD generation, PINN training, and
evaluation all work.

## Full Reproduction Pipeline

To regenerate the baseline artifacts:

```bash
python main.py
```

The stages are:

```text
gradcheck.py     verifies hand-coded gradients
run_forward.py   trains clean, noisy-sensor, and data-only forward runs
run_inverse.py   performs inverse RLGC identification and FIM diagnostics
make_figures.py  writes manuscript figures and CSV tables
```

The full pipeline can take more than 10 minutes on a CPU-only machine. Run
stages individually if only one artifact is required.

## Programmatic API

```python
from twinlinepinn import solve_reference, run_smoke_training

ref = solve_reference()
print(ref.voltage.shape, ref.current.shape)

result = run_smoke_training(n_epochs=20)
print(result["relative_l2_voltage"], result["relative_l2_current"])
```

## Main Files

| File | Purpose |
|---|---|
| `fdtd_reference.py` | Yee-staggered FDTD reference solver |
| `pinn_core.py` | NumPy MLP, input Jacobians, residuals, and gradients |
| `run_forward.py` | Forward PINN and data-only baseline experiments |
| `run_inverse.py` | Inverse parameter identification and FIM diagnostic |
| `environmental_coupling.py` | Thermal and weather utility functions |
| `make_figures.py` | Manuscript figure and table generation |
| `digital_twin_dashboard.py` | Offline visualization prototype |
| `twinlinepinn/api.py` | High-level package API |

## Scientific Scope

Implemented and validated:

- lossy telegrapher FDTD reference,
- four-term electromagnetic PINN,
- sparse noisy sensor reconstruction,
- unconstrained ANN/data-only baseline,
- inverse identification of `R'`, `L'`, `G'`, and `C'`,
- Fisher Information Matrix identifiability diagnostic.

Formulated but not active in the baseline training loop:

- augmented environmental PINN outputting line temperature,
- IEEE-style thermal residual,
- dynamic line rating penalty,
- sag-tension residual,
- online streaming update.

## Citation

Please cite:

Mohammad (Behdad) Jamshidi, "A Physics-Informed Neural Network Framework for
Lossy Telegrapher Equations with a Formulated Multi-Physics Environmental
Extension", *Computation*, July 2026.
