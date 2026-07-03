"""Short TwinLinePINN smoke example.

This is not the full manuscript reproduction. It is a fast check that the
package imports, generates an FDTD reference, trains a small PINN, and evaluates
field errors.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twinlinepinn import benchmark_line_params, solve_reference, run_smoke_training


def main() -> None:
    params = benchmark_line_params()
    ref = solve_reference(Nx=61, record_every=2)
    print("Benchmark line")
    print(f"  c  = {params.c:.3e} m/s")
    print(f"  Z0 = {params.Z0:.2f} ohm")
    print(f"  reference grid = {ref.voltage.shape[1]} x {ref.voltage.shape[0]}")

    result = run_smoke_training(n_epochs=20)
    print("Smoke-training errors")
    print(f"  relative L2 voltage = {result['relative_l2_voltage']:.3e}")
    print(f"  relative L2 current = {result['relative_l2_current']:.3e}")


if __name__ == "__main__":
    main()
