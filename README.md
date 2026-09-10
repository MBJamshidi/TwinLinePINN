# TwinLinePINN

> **Citation notice:** If you use TwinLinePINN in research, publications, or
> derivative software, please cite: Jamshidi M. A Physics-Informed Neural
> Network Framework for Lossy Telegrapher Equations with a Formulated
> Multi-Physics Environmental Extension. *Computation*. 2026; 14(9):208.
> https://doi.org/10.3390/computation14090208

Open-access reference software for physics-informed neural network modelling of
lossy transmission lines.

This package accompanies the paper:

> Jamshidi M. A Physics-Informed Neural Network Framework for Lossy Telegrapher
> Equations with a Formulated Multi-Physics Environmental Extension.
> *Computation*. 2026; 14(9):208.
> https://doi.org/10.3390/computation14090208

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

> Jamshidi M. A Physics-Informed Neural Network Framework for Lossy Telegrapher
> Equations with a Formulated Multi-Physics Environmental Extension.
> *Computation*. 2026; 14(9):208.
> [https://doi.org/10.3390/computation14090208](https://doi.org/10.3390/computation14090208)

```bibtex
@Article{computation14090208,
AUTHOR = {Jamshidi, Mohammad (Behdad)},
TITLE = {A Physics-Informed Neural Network Framework for Lossy Telegrapher Equations with a Formulated Multi-Physics Environmental Extension},
JOURNAL = {Computation},
VOLUME = {14},
YEAR = {2026},
NUMBER = {9},
ARTICLE-NUMBER = {208},
URL = {https://www.mdpi.com/2079-3197/14/9/208},
ISSN = {2079-3197},
ABSTRACT = {This paper develops a physics-informed neural network (PINN) framework for the lossy telegrapher equations and presents a coupled IEEE 738 thermal balance formulation intended as a structural blueprint for environmentally aware transmission-line digital twins. The baseline electromagnetic PINN maps (x,t)↦(V̂,Î) and is empirically validated against a finite-difference time-domain (FDTD) reference solver. An augmented parametric framework Nθ:(x,t,e)↦(V̂,Î,T̂line) is mathematically derived, wherein the ambient vector e modulates a temperature-dependent resistance R′(Tline) and couples to the telegrapher residuals via a non-linear thermal balance residual rT. Two further constraints, a sag-tension consistency residual rS and a dynamic line rating (DLR) one-sided penalty rDLR, are formulated for completeness but are explicitly designated as architectural extension hooks running at zero weight (ωth=ωsag=ωdlr=0) within the reported microscale numerical benchmarks. Consequently, the empirical validation presented herein strictly concerns the baseline electromagnetic telegrapher PINN. The numerical results demonstrate robust L2 field convergence against FDTD reference data, highly structured error accumulation along physical characteristic curves, and reliable recovery of strongly observable parameters (L′,C′) from sparse, noisy terminal measurements. Conversely, the recovery of loss parameters (R′,G′) exhibits a severe structural weak identifiability that precisely matches the analytical predictions of a comprehensive Fisher Information Matrix analysis. The core contributions of this work are primarily methodological: (i) a dimensionally consistent, corrected residual formulation for the lossy telegrapher equations; (ii) an explicit positioning of the proposed multi-physics framework within the parametric PINN literature; (iii) a Fisher information identifiability diagnostic illustrating the near-degeneracy of baseline parameter estimation; and (iv) a clean algorithmic separation of forward training, inverse parameter identification, and prospective online updates.},
DOI = {10.3390/computation14090208}
}
```

A machine-readable citation file is provided in `CITATION.cff`.

## License

MIT License. See `LICENSE`.
