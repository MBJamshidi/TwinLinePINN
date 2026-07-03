"""Gradient verification - critical sanity check."""
import numpy as np
from pinn_core import MLP, Problem, total_loss_and_grad

def main():
    rng = np.random.default_rng(7)
    net = MLP(layers=[2, 12, 12, 2], seed=3)
    prob = Problem(Rp=0.1, Lp=2.5e-7, Gp=1e-5, Cp=1.0e-10,
                   length=1.0, T=12e-9)

    sets = dict(
        coll=rng.uniform(0, 1, (32, 2)),
        ic_pts=np.column_stack([rng.uniform(0, 1, 8), np.zeros(8)]),
        ic_vals=rng.normal(0, 0.1, (8, 2)),
        bc_pts=np.column_stack([rng.choice([0.0, 1.0], 8), rng.uniform(0, 1, 8)]),
        bc_vals=rng.normal(0, 0.1, (8, 2)),
        data_pts=rng.uniform(0, 1, (8, 2)),
        data_vals=rng.normal(0, 0.1, (8, 2)),
    )
    w = (1e-3, 1.0, 1.0, 1.0)
    L0, g, _, _ = total_loss_and_grad(net, prob, sets, w)

    p0 = net.get_params()
    eps = 1e-5
    idx = rng.choice(p0.size, size=25, replace=False)
    max_rel = 0.0
    print(f"{'idx':>5} {'analytic':>14} {'numeric':>14} {'rel_err':>10}")
    for i in idx:
        pp = p0.copy(); pp[i] += eps; net.set_params(pp)
        Lp, _, _, _ = total_loss_and_grad(net, prob, sets, w)
        pm = p0.copy(); pm[i] -= eps; net.set_params(pm)
        Lm, _, _, _ = total_loss_and_grad(net, prob, sets, w)
        num = (Lp - Lm) / (2 * eps)
        rel = abs(num - g[i]) / (abs(num) + abs(g[i]) + 1e-12)
        max_rel = max(max_rel, rel)
        print(f"{i:5d} {g[i]:14.6e} {num:14.6e} {rel:10.2e}")
    print(f"\nmax relative error: {max_rel:.3e}")
    assert max_rel < 1e-4, "Gradient mismatch!"
    print("PASS")
    net.set_params(p0)


if __name__ == "__main__":
    main()
