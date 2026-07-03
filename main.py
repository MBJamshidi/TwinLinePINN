"""
main.py
=======
End-to-end driver. Run this to reproduce all results and figures.

Stages:
  1) gradcheck      - verify analytic backprop matches finite differences
  2) run_forward    - train PINN against FDTD (3 scenarios)
  3) run_inverse    - identify (R',L',G',C') under noise
  4) make_figures   - produce all publication-quality figures and tables

Outputs:
  results/forward_runs.npz       (raw arrays + training histories)
  results/inverse_runs.npz       (parameter recovery + FIM)
  figures/fig*.png               (PNG, 400 dpi)
  figures/table_*.csv            (CSV summary tables)

Runtime depends strongly on CPU and BLAS configuration. On a CPU-only
workstation the full forward + inverse + figure pipeline can take more than
10 minutes; run individual stages when only part of the artifact set is needed.
"""
import sys, time, subprocess


def run_stage(label, module):
    print(f"\n{'='*60}\n>>> Stage: {label}\n{'='*60}", flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable, "-u", module])
    dt = time.time() - t0
    if rc != 0:
        print(f"!!! {label} failed (rc={rc})")
        sys.exit(rc)
    print(f"--- {label} done in {dt:.1f}s ---", flush=True)


if __name__ == "__main__":
    print("Running manuscript baseline: fixed RLGC electromagnetic validation.", flush=True)
    print("Thermal, DLR, sag, and online live-weather components remain inactive extension hooks.", flush=True)

    run_stage("Gradient verification", "gradcheck.py")
    run_stage("Forward problem (FDTD vs PINN)", "run_forward.py")
    run_stage("Inverse problem (parameter ID)", "run_inverse.py")
    run_stage("Figures and tables", "make_figures.py")
    print("\n[ALL STAGES COMPLETE]")
    print("Open ./figures/  for all PNGs and CSV tables.")
