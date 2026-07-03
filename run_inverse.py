"""
run_inverse.py
==============
Inverse problem: identify (R', L', G', C') from sparse noisy
measurements of V and I. Optimize log-parameters for passivity.

We sweep noise level x random seed to estimate the variance of
identification, and compute the Fisher Information Matrix (FIM)
numerically using FDTD as the forward model; this gives us a
Cramer-Rao bound to compare PINN identification variance against.
"""
from __future__ import annotations
import os, time, json
import numpy as np

from fdtd_reference import LineParams, fdtd_solve, interp_I_to_V_grid
from pinn_core import (MLP, Adam, Problem, total_loss_and_grad,
                       forward, forward_with_jac)
from run_forward import make_problem, make_sets, eval_grid, update_weights


# ============================================================
def grad_phi(net, prob, sets, log_phi, weights):
    """Compute gradient wrt log-parameters via the linear residual."""
    Rp = float(np.exp(log_phi[0])); Lp = float(np.exp(log_phi[1]))
    Gp = float(np.exp(log_phi[2])); Cp = float(np.exp(log_phi[3]))

    L, g_theta, comps, _ = total_loss_and_grad(
        net, prob, sets, weights, phi_override=(Rp, Lp, Gp, Cp))

    # Compute dL_pde / d(phi) directly from the residual structure
    coll = sets['coll']
    u_n, J_n, _ = forward_with_jac(net, coll)
    Vs_, Is_ = prob.V_star, prob.I_star
    Lx, Tt = prob.length, prob.T
    V = Vs_ * u_n[:, 0]; I_ = Is_ * u_n[:, 1]
    Vx = (Vs_ / Lx) * J_n[:, 0, 0]; Vt = (Vs_ / Tt) * J_n[:, 1, 0]
    Ix = (Is_ / Lx) * J_n[:, 0, 1]; It = (Is_ / Tt) * J_n[:, 1, 1]
    rV = Vx + Lp * It + Rp * I_
    rI = Ix + Cp * Vt + Gp * V
    Np = coll.shape[0]; wp = weights[0]
    dL_dRp = wp * (2.0 / Np) * np.sum(rV * I_)
    dL_dLp = wp * (2.0 / Np) * np.sum(rV * It)
    dL_dGp = wp * (2.0 / Np) * np.sum(rI * V)
    dL_dCp = wp * (2.0 / Np) * np.sum(rI * Vt)
    g_phi = np.array([dL_dRp * Rp, dL_dLp * Lp,
                      dL_dGp * Gp, dL_dCp * Cp])
    return L, g_theta, g_phi, comps, (Rp, Lp, Gp, Cp)


# ============================================================
def fim_via_fdtd(base_params, sensor_xt, sigma_V, sigma_I, T_sim,
                 rel_eps=2e-2, Nx=201):
    """
    Numerically compute the Fisher Information Matrix at base parameters,
    in log-parameter space, by central differences of FDTD solutions.

    F_kj = (1/sigma^2) Σ_i [ (∂u/∂log_phi_k)^T Σ^{-1} (∂u/∂log_phi_j) ]_i
    """
    base_phi = np.array([base_params.Rp, base_params.Lp,
                         base_params.Gp, base_params.Cp])
    log_base = np.log(base_phi)

    Nd = sensor_xt.shape[0]
    S = np.zeros((Nd, 2, 4))            # ∂u/∂log_phi at sensors
    for k in range(4):
        log_p = log_base.copy(); log_p[k] += rel_eps
        phi_p = np.exp(log_p)
        params_p = LineParams(*phi_p,
                              length=base_params.length,
                              Zs=base_params.Zs, ZL=base_params.ZL)
        fxp, ftp, Vfp, Ifp_h, infop = fdtd_solve(params_p, T=T_sim,
                                                 Nx=Nx, cfl=0.95)
        Ifp = interp_I_to_V_grid(fxp, Ifp_h, infop['xI'])

        log_m = log_base.copy(); log_m[k] -= rel_eps
        phi_m = np.exp(log_m)
        params_m = LineParams(*phi_m,
                              length=base_params.length,
                              Zs=base_params.Zs, ZL=base_params.ZL)
        fxm, ftm, Vfm, Ifm_h, infom = fdtd_solve(params_m, T=T_sim,
                                                 Nx=Nx, cfl=0.95)
        Ifm = interp_I_to_V_grid(fxm, Ifm_h, infom['xI'])

        from run_forward import bilinear
        Vp_at = bilinear(Vfp, fxp, ftp, sensor_xt[:, 0], sensor_xt[:, 1])
        Ip_at = bilinear(Ifp, fxp, ftp, sensor_xt[:, 0], sensor_xt[:, 1])
        Vm_at = bilinear(Vfm, fxm, ftm, sensor_xt[:, 0], sensor_xt[:, 1])
        Im_at = bilinear(Ifm, fxm, ftm, sensor_xt[:, 0], sensor_xt[:, 1])

        S[:, 0, k] = (Vp_at - Vm_at) / (2 * rel_eps)
        S[:, 1, k] = (Ip_at - Im_at) / (2 * rel_eps)

    Sigma_inv = np.diag([1.0 / sigma_V ** 2, 1.0 / sigma_I ** 2])
    F = np.zeros((4, 4))
    for i in range(Nd):
        F += S[i].T @ Sigma_inv @ S[i]
    return F, S


# ============================================================
def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(12345)
    params, T = make_problem()
    print(">>> Inverse problem", flush=True)
    print(f"    truth: R={params.Rp:.3e}  L={params.Lp:.3e}  "
          f"G={params.Gp:.3e}  C={params.Cp:.3e}", flush=True)

    print(">>> FDTD reference for measurements", flush=True)
    fdtd_x, fdtd_t, V_fdtd, I_fdtd_h, info = fdtd_solve(
        params, T=T, Nx=201, cfl=0.95, record_every=2)
    I_fdtd = interp_I_to_V_grid(fdtd_x, I_fdtd_h, info['xI'])

    prob = Problem(Rp=params.Rp, Lp=params.Lp, Gp=params.Gp, Cp=params.Cp,
                   length=params.length, T=T, V_star=1.0)

    noise_levels = [0.0, 0.03, 0.10]
    n_seeds = 2
    summary = []
    histories = {}

    for noise in noise_levels:
        for seed in range(n_seeds):
            tag = f"n{int(noise*100):02d}_s{seed}"
            t0 = time.time()
            print(f"\n--- {tag}: noise={noise:.2f} seed={seed} ---", flush=True)

            sets = make_sets(prob, params, fdtd_x, fdtd_t, V_fdtd, I_fdtd,
                             Np=1000, Ni=80, Nb=160, Nd=60,
                             noise_pct=noise, seed=10 + seed)

            init_jitter = rng.normal(0, 0.3, 4)
            log_phi = np.log(np.array([params.Rp, params.Lp,
                                       params.Gp, params.Cp])) + init_jitter

            net = MLP(layers=[2, 30, 30, 30, 2], seed=100 + seed)
            opt_th = Adam(lr=2e-3); opt_ph = Adam(lr=5e-2)
            p_th = net.get_params()
            w = (1e-2, 10.0, 10.0, 5.0)

            n_epochs = 1500
            log_w, phi_track, loss_track = [], [], []
            for ep in range(n_epochs):
                L, g_th, g_ph, comps, phi_now = grad_phi(net, prob, sets, log_phi, w)
                p_th = opt_th.step(p_th, g_th); net.set_params(p_th)
                log_phi = opt_ph.step(log_phi, g_ph)
                if ep % 15 == 0:
                    log_w.append(ep)
                    phi_track.append(np.exp(log_phi).copy())
                    loss_track.append(comps['total'])
                if ep % 500 == 0:
                    Rp, Lp, Gp, Cp = phi_now
                    print(f"   ep={ep:4d}  L={comps['total']:.2e}  "
                          f"R={Rp:.2e} L={Lp:.2e} G={Gp:.2e} C={Cp:.2e}",
                          flush=True)

            phi_final = np.exp(log_phi)
            phi_true = np.array([params.Rp, params.Lp, params.Gp, params.Cp])
            rel = (phi_final - phi_true) / phi_true * 100
            Vp, Ip, eV, eI = eval_grid(net, prob, fdtd_x, fdtd_t, V_fdtd, I_fdtd)
            print(f"   FINAL: rel-err(%) R={rel[0]:+.2f} L={rel[1]:+.2f} "
                  f"G={rel[2]:+.2f} C={rel[3]:+.2f}  "
                  f"|sol| eV={eV:.2e} eI={eI:.2e}  ({time.time()-t0:.1f}s)",
                  flush=True)
            summary.append(dict(noise=float(noise), seed=int(seed),
                                Rp=float(phi_final[0]), Lp=float(phi_final[1]),
                                Gp=float(phi_final[2]), Cp=float(phi_final[3]),
                                rel_err=rel.tolist(),
                                eV=float(eV), eI=float(eI),
                                time_s=float(time.time() - t0)))
            histories[tag] = dict(epoch=log_w,
                                  phi_hist=np.array(phi_track),
                                  loss=loss_track)

    # ---------- Fisher Information at the true parameters ----------
    print("\n>>> Fisher Information Matrix (FDTD-based numerical)", flush=True)
    rng_fim = np.random.default_rng(999)
    sensor_xt = np.column_stack([
        rng_fim.uniform(0, params.length, 60),
        rng_fim.uniform(0, T, 60)
    ])
    sigma_V = 0.05 * np.max(np.abs(V_fdtd))
    sigma_I = 0.05 * np.max(np.abs(I_fdtd))
    F, S = fim_via_fdtd(params, sensor_xt, sigma_V, sigma_I, T_sim=T,
                        rel_eps=2e-2, Nx=201)
    eigs = np.linalg.eigvalsh(F)
    cond = float(eigs[-1] / max(eigs[0], 1e-30))
    F_inv = np.linalg.inv(F + 1e-12 * np.eye(4))
    cr = np.sqrt(np.diag(F_inv))                      # std-bound on log-phi
    print(f"   FIM eigenvalues: {eigs}", flush=True)
    print(f"   condition: {cond:.3e}", flush=True)
    print(f"   CR std(log phi): R={cr[0]:.3f} L={cr[1]:.3f} G={cr[2]:.3f} C={cr[3]:.3f}",
          flush=True)

    np.savez_compressed("results/inverse_runs.npz",
                        true_phi=np.array([params.Rp, params.Lp,
                                           params.Gp, params.Cp]),
                        summary=json.dumps(summary),
                        histories=json.dumps({
                            k: dict(epoch=v['epoch'],
                                    phi_hist=v['phi_hist'].tolist(),
                                    loss=v['loss'])
                            for k, v in histories.items()
                        }),
                        F=F, S=S, eigs=eigs, cond=cond, cr=cr,
                        sensor_xt=sensor_xt,
                        sigma_V=sigma_V, sigma_I=sigma_I,
                        noise_levels=np.array(noise_levels))
    print("\nSaved results/inverse_runs.npz", flush=True)


if __name__ == "__main__":
    main()
