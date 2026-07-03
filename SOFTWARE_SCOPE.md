# Software Scope and Manuscript Alignment

This file records the package version of the methodology and formulation used
by the manuscript:

> A Physics-Informed Neural Network Framework for Lossy Telegrapher Equations
> with a Formulated Multi-Physics Environmental Extension

## Validated Baseline

The implemented and validated model is the electromagnetic telegrapher PINN

```text
(x, t) -> (V, I)
```

with the four-term loss

```text
L4 = wp L_pde + wi L_ic + wb L_bc + wd L_data
```

and the corrected strong-form residuals

```text
rV = dV/dx + L' dI/dt + R' I
rI = dI/dx + C' dV/dt + G' V
```

The implementation uses normalized inputs internally and rescales derivatives
back to physical units before forming the PDE residuals.

## Equation-to-File Correspondence

| Manuscript object | Software location | Status |
|---|---|---|
| Telegrapher residuals `rV`, `rI` | `pinn_core.total_loss_and_grad` | Implemented and active |
| Baseline ansatz `(x,t)->(V,I)` | `pinn_core.MLP` | Implemented and active |
| Forward training algorithm | `run_forward.py` | Implemented and active |
| Inverse RLGC identification | `run_inverse.py` | Implemented and active |
| Positivity via log-parameters | `run_inverse.grad_phi` and main loop | Implemented and active |
| FDTD reference benchmark | `fdtd_reference.py` | Implemented and active |
| Fisher Information Matrix diagnostic | `run_inverse.fim_via_fdtd` | Implemented and active |
| Data-only ANN baseline | `run_forward.py`, Run C | Implemented and active |
| Figure and CSV generation | `make_figures.py` | Implemented |
| Temperature-adjusted resistance utility | `environmental_coupling.temperature_adjusted_rp` | Utility only |
| Steady IEEE-style heat-balance utility | `environmental_coupling.solve_line_temperature_c` | Utility only |
| Ambient/weather utility | `environmental_coupling.current_environment_for_model` | Utility/dashboard only |
| Augmented ansatz `(x,t,e)->(V,I,T_line)` | Not in active training code | Future work |
| Thermal residual `r_T` in the PINN loss | Not in active training code | Inactive extension formulation |
| DLR one-sided penalty `r_DLR` | Not in active training code | Inactive extension formulation |
| Sag-tension residual `r_S` | Not in active training code | Inactive extension formulation |
| Online warm-start update | Not in active training code | Future work |

## Reference Configuration

The default scripts match the manuscript reproducibility table:

```text
Line length:              1.0 m
Simulation horizon:       12 ns
R', L', G', C':           0.1, 2.5e-7, 1.0e-5, 1.0e-10
Network:                  [2, 30, 30, 30, 2], tanh activations
Forward collocation:      1500
Forward IC / BC points:   100 / 200
Noisy sensor case:        64 sensors, 3 percent noise
Forward optimizer:        Adam, 4000 epochs, lr=2e-3
Inverse collocation:      1000
Inverse IC / BC points:   80 / 160
Inverse sensors:          60
Inverse optimizer:        Adam, 1500 epochs, theta lr=2e-3, log-phi lr=5e-2
L-BFGS:                   not used
Thermal/DLR/sag weights:  zero / inactive
```

## Verification Performed

`python gradcheck.py` passed on Python 3.13.7 with maximum sampled relative
gradient error `2.060e-10`, verifying the hand-coded NumPy gradient path used
by the PINN loss.

`python main.py` was also attempted in this workspace. It successfully
regenerated `results/forward_runs.npz` but did not complete the inverse and
figure stages inside a 10-minute command window on this machine.

The full numerical pipeline is available through:

```bash
python main.py
```

This regenerates `results/forward_runs.npz`, `results/inverse_runs.npz`, and
the manuscript figures/tables when allowed to run to completion.
