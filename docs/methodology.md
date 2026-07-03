# Methodology Summary

TwinLinePINN implements the validated core of the paper:

**A Physics-Informed Neural Network Framework for Lossy Telegrapher Equations
with a Formulated Multi-Physics Environmental Extension**, Mohammad (Behdad)
Jamshidi, *Computation*, July 2026.

## Governing Equations

The implemented baseline models a uniform lossy transmission line with voltage
`V(x,t)` and current `I(x,t)`:

```text
dV/dx = -L' dI/dt - R' I
dI/dx = -C' dV/dt - G' V
```

The PINN residuals are therefore:

```text
rV = dV/dx + L' dI/dt + R' I
rI = dI/dx + C' dV/dt + G' V
```

These are the corrected strong-form residuals. The spatial derivatives are
essential; replacing `dV/dx` or `dI/dx` by the field values would no longer be
the telegrapher PDE.

## Validated Neural Map

The active validated network is:

```text
(x, t) -> (V, I)
```

The manuscript also formulates an environmental extension:

```text
(x, t, e) -> (V, I, T_line)
```

where `e` contains ambient temperature, wind, solar irradiance, humidity, and
ice-loading quantities. In this repository, the thermal/weather functions are
available as utilities, but the active training loop remains the electromagnetic
baseline.

## Loss Terms

The validated loss is:

```text
L4 = wp L_pde + wi L_ic + wb L_bc + wd L_data
```

The thermal residual `r_T`, dynamic line rating penalty `r_DLR`, and sag-tension
residual `r_S` are not active in the reported numerical experiments. This is a
physical design choice: the electromagnetic benchmark is 1 m over 12 ns, while
conductor thermal dynamics occur on minute-to-hour horizons.

## Reference Benchmark

The default case is a matched 50 ohm low-loss line:

```text
R' = 1.0e-1 ohm/m
L' = 2.5e-7 H/m
G' = 1.0e-5 S/m
C' = 1.0e-10 F/m
Length = 1 m
T = 12 ns
Zs = ZL = 50 ohm
```

The finite-difference time-domain solver in `fdtd_reference.py` provides the
reference solution used by the PINN experiments.

## Inverse Identification

`run_inverse.py` optimizes `(R', L', G', C')` in log space so the identified
parameters remain positive. The paper and software both show that `L'` and `C'`
are strongly observable in the short matched-line benchmark, while `R'` and
`G'` are weakly identifiable without temperature observations or stronger
loss/attenuation signatures.
