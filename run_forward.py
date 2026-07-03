"""
run_forward.py
==============
Forward problem: train a PINN to reproduce the FDTD reference solution,
both with and without sparse noisy measurements.
"""
from __future__ import annotations
import os, time, json
import numpy as np

from fdtd_reference import LineParams, fdtd_solve, interp_I_to_V_grid
from pinn_core import MLP, Adam, Problem, total_loss_and_grad, forward


# ---------------- Problem ----------------
def make_problem():
    p = LineParams(Rp=0.1, Lp=2.5e-7, Gp=1e-5, Cp=1.0e-10,
                   length=1.0, Zs=50.0, ZL=50.0)
    p.base_Rp = p.Rp
    T = 12e-9
    return p, T


# -------------- LHS sampler --------------
def lhs(n, d, rng):
    out = np.zeros((n, d))
    for j in range(d):
        cuts = (np.arange(n) + rng.uniform(0, 1, n)) / n
        rng.shuffle(cuts)
        out[:, j] = cuts
    return out


def bilinear(F, fdtd_x, fdtd_t, x, t):
    ix = np.clip(np.searchsorted(fdtd_x, x) - 1, 0, fdtd_x.size - 2)
    it = np.clip(np.searchsorted(fdtd_t, t) - 1, 0, fdtd_t.size - 2)
    ax = (x - fdtd_x[ix]) / (fdtd_x[ix + 1] - fdtd_x[ix])
    at = (t - fdtd_t[it]) / (fdtd_t[it + 1] - fdtd_t[it])
    return ((1 - ax) * (1 - at) * F[it,     ix] +
            ax * (1 - at)       * F[it,     ix + 1] +
            (1 - ax) * at       * F[it + 1, ix] +
            ax * at             * F[it + 1, ix + 1])


def make_sets(prob, params, fdtd_x, fdtd_t, V_fdtd, I_fdtd,
              Np=1500, Ni=100, Nb=200, Nd=0, noise_pct=0.0, seed=1):
    """Build collocation/IC/BC/data sets in NORMALIZED coordinates."""
    rng = np.random.default_rng(seed)
    coll = lhs(Np, 2, rng)

    # IC: zero everywhere (line at rest)
    x_ic = rng.uniform(0, 1, Ni)
    ic_pts = np.column_stack([x_ic, np.zeros(Ni)])
    ic_vals = np.zeros((Ni, 2))

    # BC: sample at x=0 and x=L; target from FDTD trace
    Nb_each = Nb // 2
    t_bc = rng.uniform(0, 1, Nb_each)
    bc_pts = np.vstack([np.column_stack([np.zeros(Nb_each), t_bc]),
                        np.column_stack([np.ones(Nb_each),  t_bc])])
    Vbc = bilinear(V_fdtd, fdtd_x, fdtd_t,
                   bc_pts[:, 0] * params.length, bc_pts[:, 1] * prob.T)
    Ibc = bilinear(I_fdtd, fdtd_x, fdtd_t,
                   bc_pts[:, 0] * params.length, bc_pts[:, 1] * prob.T)
    bc_vals = np.column_stack([Vbc / prob.V_star, Ibc / prob.I_star])

    # Sparse noisy measurements
    if Nd > 0:
        data_xt = lhs(Nd, 2, rng)
        Vc = bilinear(V_fdtd, fdtd_x, fdtd_t,
                      data_xt[:, 0] * params.length, data_xt[:, 1] * prob.T)
        Ic = bilinear(I_fdtd, fdtd_x, fdtd_t,
                      data_xt[:, 0] * params.length, data_xt[:, 1] * prob.T)
        sV = noise_pct * np.max(np.abs(V_fdtd))
        sI = noise_pct * np.max(np.abs(I_fdtd))
        Vd = Vc + rng.normal(0, sV, Nd)
        Id = Ic + rng.normal(0, sI, Nd)
        data_pts  = data_xt
        data_vals = np.column_stack([Vd / prob.V_star, Id / prob.I_star])
        data_clean = np.column_stack([Vc / prob.V_star, Ic / prob.I_star])
    else:
        data_pts  = np.zeros((0, 2)); data_vals = np.zeros((0, 2))
        data_clean = np.zeros((0, 2))

    return dict(coll=coll,
                ic_pts=ic_pts, ic_vals=ic_vals,
                bc_pts=bc_pts, bc_vals=bc_vals,
                data_pts=data_pts, data_vals=data_vals,
                data_clean=data_clean)


# ----------- adaptive loss balancing -----------
def update_weights(net, prob, sets, w, eta=0.1):
    """Equalize per-task gradient norms (Wang–Teng–Perdikaris)."""
    norms = []
    for k in range(4):
        ww = [0.0, 0.0, 0.0, 0.0]; ww[k] = 1.0
        if k == 3 and len(sets['data_pts']) == 0:
            norms.append(None); continue
        _, g, _, _ = total_loss_and_grad(net, prob, sets, tuple(ww))
        norms.append(float(np.linalg.norm(g) + 1e-12))
    target = max(n for n in norms if n is not None)
    new = list(w)
    for k, n in enumerate(norms):
        if n is None: continue
        new[k] = (1 - eta) * w[k] + eta * (target / n)
    return tuple(new)


# ----------- training -----------
def train(prob, sets, layers=(2, 30, 30, 30, 2), n_epochs=4000,
          lr=2e-3, w0=(1e-2, 10.0, 10.0, 5.0), adapt_every=400,
          log_every=50, seed=42, verbose_every=400):
    net = MLP(layers=list(layers), seed=seed)
    opt = Adam(lr=lr); p = net.get_params(); w = w0
    hist = dict(epoch=[], total=[], pde=[], ic=[], bc=[], data=[],
                wp=[], wi=[], wb=[], wd=[], wallclock=[])
    t0 = time.time()
    for ep in range(n_epochs):
        L, g, c, _ = total_loss_and_grad(net, prob, sets, w)
        p = opt.step(p, g); net.set_params(p)
        if (ep + 1) % adapt_every == 0:
            w = update_weights(net, prob, sets, w, eta=0.1)
        if ep % log_every == 0 or ep == n_epochs - 1:
            hist['epoch'].append(ep); hist['total'].append(c['total'])
            hist['pde'].append(c['pde']); hist['ic'].append(c['ic'])
            hist['bc'].append(c['bc']);  hist['data'].append(c['data'])
            hist['wp'].append(w[0]); hist['wi'].append(w[1])
            hist['wb'].append(w[2]); hist['wd'].append(w[3])
            hist['wallclock'].append(time.time() - t0)
        if ep % verbose_every == 0:
            print(f"   ep={ep:5d}  L={c['total']:.3e}  "
                  f"pde={c['pde']:.2e} ic={c['ic']:.2e} "
                  f"bc={c['bc']:.2e} data={c['data']:.2e}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return net, hist, w


# ----------- evaluation -----------
def eval_grid(net, prob, fdtd_x, fdtd_t, V_fdtd, I_fdtd):
    Xs = fdtd_x / prob.length; Ts = fdtd_t / prob.T
    XX, TT = np.meshgrid(Xs, Ts)
    pts = np.column_stack([XX.ravel(), TT.ravel()])
    u, _ = forward(net, pts)
    Vp = (prob.V_star * u[:, 0]).reshape(XX.shape)
    Ip = (prob.I_star * u[:, 1]).reshape(XX.shape)
    eV = np.linalg.norm(Vp - V_fdtd) / (np.linalg.norm(V_fdtd) + 1e-12)
    eI = np.linalg.norm(Ip - I_fdtd) / (np.linalg.norm(I_fdtd) + 1e-12)
    return Vp, Ip, float(eV), float(eI)


# ----------- main -----------
def main():
    os.makedirs("results", exist_ok=True)
    params, T = make_problem()
    print(">>> Baseline electromagnetic benchmark", flush=True)
    print("   constant RLGC parameters; live thermal/weather coupling inactive", flush=True)
    print(">>> FDTD ground truth", flush=True)
    fdtd_x, fdtd_t, V_fdtd, I_fdtd_half, info = fdtd_solve(
        params, T=T, Nx=201, cfl=0.95, record_every=2)
    I_fdtd = interp_I_to_V_grid(fdtd_x, I_fdtd_half, info['xI'])
    print(f"   grid {fdtd_x.size} x {fdtd_t.size}, dt={info['dt']:.2e}", flush=True)
    print(f"   c={params.c:.3e}, Z0={params.Z0:.2f}, alpha={params.alpha:.3e}", flush=True)

    prob = Problem(Rp=params.Rp, Lp=params.Lp, Gp=params.Gp, Cp=params.Cp,
                   length=params.length, T=T, V_star=1.0)

    runs = {}

    # ---- Run A: PDE + IC + BC, no measurements ----
    print("\n>>> Run A: PDE-only (clean)", flush=True)
    sets_A = make_sets(prob, params, fdtd_x, fdtd_t, V_fdtd, I_fdtd,
                       Np=1500, Ni=100, Nb=200, Nd=0, seed=1)
    netA, histA, _ = train(prob, sets_A, n_epochs=4000, seed=42,
                           w0=(1e-2, 10.0, 10.0, 0.0))
    Vp, Ip, eV, eI = eval_grid(netA, prob, fdtd_x, fdtd_t, V_fdtd, I_fdtd)
    print(f"   FINAL: rel-L2 V = {eV:.3e}, rel-L2 I = {eI:.3e}", flush=True)
    runs['clean'] = dict(net_params=netA.get_params(), layers=netA.layers,
                         hist=histA, V_pred=Vp, I_pred=Ip, eV=eV, eI=eI,
                         data_pts=sets_A['data_pts'].copy(),
                         data_vals=sets_A['data_vals'].copy(),
                         data_clean=sets_A['data_clean'].copy())

    # ---- Run B: with noisy measurements (3% noise, 64 sensors) ----
    print("\n>>> Run B: with sparse noisy measurements (3%, 64 sensors)", flush=True)
    sets_B = make_sets(prob, params, fdtd_x, fdtd_t, V_fdtd, I_fdtd,
                       Np=1500, Ni=100, Nb=200, Nd=64, noise_pct=0.03, seed=2)
    netB, histB, _ = train(prob, sets_B, n_epochs=4000, seed=42,
                           w0=(1e-2, 10.0, 10.0, 5.0))
    Vp, Ip, eV, eI = eval_grid(netB, prob, fdtd_x, fdtd_t, V_fdtd, I_fdtd)
    print(f"   FINAL: rel-L2 V = {eV:.3e}, rel-L2 I = {eI:.3e}", flush=True)
    runs['noisy'] = dict(net_params=netB.get_params(), layers=netB.layers,
                         hist=histB, V_pred=Vp, I_pred=Ip, eV=eV, eI=eI,
                         data_pts=sets_B['data_pts'].copy(),
                         data_vals=sets_B['data_vals'].copy(),
                         data_clean=sets_B['data_clean'].copy())

    # ---- Run C: data-only (no physics), to highlight the value of physics ----
    print("\n>>> Run C: data-only (no PDE; same noisy sensors)", flush=True)
    netC, histC, _ = train(prob, sets_B, n_epochs=4000, seed=42,
                           w0=(0.0, 0.0, 0.0, 5.0))
    Vp, Ip, eV, eI = eval_grid(netC, prob, fdtd_x, fdtd_t, V_fdtd, I_fdtd)
    print(f"   FINAL: rel-L2 V = {eV:.3e}, rel-L2 I = {eI:.3e}", flush=True)
    runs['dataonly'] = dict(net_params=netC.get_params(), layers=netC.layers,
                            hist=histC, V_pred=Vp, I_pred=Ip, eV=eV, eI=eI)

    # ---- Save ----
    payload = dict(
        fdtd_x=fdtd_x, fdtd_t=fdtd_t, V_fdtd=V_fdtd, I_fdtd=I_fdtd,
        Rp=params.Rp, Lp=params.Lp, Gp=params.Gp, Cp=params.Cp,
        base_Rp=getattr(params, "base_Rp", params.Rp),
        validation_scope=np.array("baseline electromagnetic; thermal, DLR, and sag weights inactive"),
        noisy_sensor_count=np.array(64),
        noisy_noise_pct=np.array(0.03),
        collocation_count=np.array(1500),
        length=params.length, T=T, Zs=params.Zs, ZL=params.ZL,
        Z0=params.Z0, c=params.c, alpha=params.alpha,
        dt_fdtd=info['dt'], dx_fdtd=info['dx'],
    )
    for name, r in runs.items():
        for k, v in r.items():
            if k == 'hist':
                payload[f"{name}_hist"] = json.dumps(
                    {kk: list(map(float, vv)) for kk, vv in v.items()})
            elif k == 'layers':
                payload[f"{name}_layers"] = np.array(v)
            elif k == 'net_params':
                payload[f"{name}_params"] = v
            else:
                payload[f"{name}_{k}"] = v
    np.savez_compressed("results/forward_runs.npz", **payload)
    print("\nSaved results/forward_runs.npz", flush=True)


if __name__ == "__main__":
    main()
