"""
make_figures.py
===============
Produce a comprehensive set of publication-quality figures and tables
from the saved results.  Everything is regenerated from the .npz files
produced by run_forward.py and run_inverse.py.
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import csv

from fdtd_reference import LineParams, fdtd_solve

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------- global style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.labelweight": "normal",
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.2",
    "legend.fancybox": False,
    "figure.dpi": 130,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.35,
    "image.cmap": "viridis",
})

AXIS_LABEL_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 9
COLORBAR_LABEL_FONTSIZE = 10
COLORBAR_TICK_FONTSIZE = 8.5

def _save_polished(fig, filename):
    """Save with no title and high journal-print quality."""
    for ax in fig.get_axes():
        if hasattr(ax, "set_title"):
            ax.set_title("")
        if hasattr(ax, "grid"):
            ax.grid(False)
        if hasattr(ax, "tick_params"):
            ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
        for spine in getattr(ax, "spines", {}).values():
            spine.set_linewidth(0.8)
    if hasattr(fig, "suptitle"):
        fig.suptitle("")
    fig.savefig(f"{OUT}/{filename}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")

# ---------- load forward run ----------
fwd = np.load("results/forward_runs.npz", allow_pickle=True)
fdtd_x = fwd["fdtd_x"]; fdtd_t = fwd["fdtd_t"]
V_fdtd = fwd["V_fdtd"]; I_fdtd = fwd["I_fdtd"]
T = float(fwd["T"]); Lx = float(fwd["length"])
Rp_t = float(fwd["Rp"]); Lp_t = float(fwd["Lp"])
Gp_t = float(fwd["Gp"]); Cp_t = float(fwd["Cp"])
c_speed = float(fwd["c"]); Z0 = float(fwd["Z0"]); alpha = float(fwd["alpha"])

clean_V = fwd["clean_V_pred"]; clean_I = fwd["clean_I_pred"]
noisy_V = fwd["noisy_V_pred"]; noisy_I = fwd["noisy_I_pred"]
do_V = fwd["dataonly_V_pred"]; do_I = fwd["dataonly_I_pred"]
clean_eV = float(fwd["clean_eV"]); clean_eI = float(fwd["clean_eI"])
noisy_eV = float(fwd["noisy_eV"]); noisy_eI = float(fwd["noisy_eI"])
do_eV = float(fwd["dataonly_eV"]); do_eI = float(fwd["dataonly_eI"])

clean_hist = json.loads(str(fwd["clean_hist"]))
noisy_hist = json.loads(str(fwd["noisy_hist"]))
do_hist    = json.loads(str(fwd["dataonly_hist"]))
noisy_data_pts   = fwd["noisy_data_pts"]
noisy_data_vals  = fwd["noisy_data_vals"]
noisy_data_clean = fwd["noisy_data_clean"]

def _fwd_scalar(name, default=np.nan):
    return float(fwd[name]) if name in fwd.files else float(default)

env_values = {
    "ambient_c": _fwd_scalar("env_ambient_c"),
    "line_temp_c": _fwd_scalar("env_line_temp_c"),
    "wind_kmh": _fwd_scalar("env_wind_kmh"),
    "solar_w_m2": _fwd_scalar("env_solar_w_m2"),
    "q_joule_w_m": _fwd_scalar("env_q_joule_w_m"),
    "q_solar_w_m": _fwd_scalar("env_q_solar_w_m"),
    "q_convection_w_m": _fwd_scalar("env_q_convection_w_m"),
    "q_radiation_w_m": _fwd_scalar("env_q_radiation_w_m"),
    "base_Rp": _fwd_scalar("base_Rp", Rp_t),
    "effective_Rp": Rp_t,
    "temp_coeff": _fwd_scalar("env_temp_coeff", 0.0039),
}
env_source = str(fwd["env_source"]) if "env_source" in fwd.files else "inactive in reported electromagnetic experiments"
NOISY_ND = int(fwd["noisy_sensor_count"]) if "noisy_sensor_count" in fwd.files else int(noisy_data_pts.shape[0])
NOISY_PCT = 100.0 * (float(fwd["noisy_noise_pct"]) if "noisy_noise_pct" in fwd.files else 0.03)
NP_COLLOCATION = int(fwd["collocation_count"]) if "collocation_count" in fwd.files else 1500

# ---------- load inverse run ----------
inv = np.load("results/inverse_runs.npz", allow_pickle=True)
true_phi = inv["true_phi"]
summary = json.loads(str(inv["summary"]))
histories = json.loads(str(inv["histories"]))
F = inv["F"]; eigs = inv["eigs"]; cond = float(inv["cond"]); cr = inv["cr"]
sensor_xt = inv["sensor_xt"]
sigma_V = float(inv["sigma_V"]); sigma_I = float(inv["sigma_I"])
ts_ns = fdtd_t * 1e9

# =========================================================
def fig01_overview():
    fig, ax = plt.subplots(figsize=(6, 5))
    for t0 in np.linspace(0, T, 9):
        t_end_f = min(T, t0 + Lx / c_speed)
        ax.plot([t0 * 1e9, t_end_f * 1e9], [0, c_speed * (t_end_f - t0)], color="C0", alpha=0.5, lw=1.2)
        t_end_b = min(T, t0 + Lx / c_speed)
        ax.plot([t0 * 1e9, t_end_b * 1e9], [Lx, Lx - c_speed * (t_end_b - t0)], color="C3", alpha=0.5, lw=1.2)
    ax.axvline(2.0, color="k", linestyle="--", lw=1.0, alpha=0.8)
    ax.text(2.05, 0.05, "source\n pulse", fontsize=10, fontweight='bold')
    ax.set_xlim(0, T * 1e9); ax.set_ylim(0, Lx)
    ax.set_xlabel("t (ns)", fontweight='bold'); ax.set_ylabel("x (m)", fontweight='bold')
    _save_polished(fig, "fig01a_characteristics.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ts_ns, V_fdtd[:, 0], color="C0", label="$V(0,t)$ (source)")
    ax.plot(ts_ns, V_fdtd[:, -1], color="C3", label="$V(L,t)$ (load)")
    ax.set_xlabel("t (ns)", fontweight='bold'); ax.set_ylabel("V (V)", fontweight='bold')
    ax.legend()
    _save_polished(fig, "fig01b_terminal_waveforms.png")

    fig, ax = plt.subplots(figsize=(6, 5)); ax.axis("off")
    rows = [[r"R'", f"{Rp_t:.3e}", "Ω/m"], [r"L'", f"{Lp_t:.3e}", "H/m"], [r"G'", f"{Gp_t:.3e}", "S/m"], [r"C'", f"{Cp_t:.3e}", "F/m"], [r"c = 1/$\sqrt{L'C'}$", f"{c_speed:.3e}", "m/s"], [r"$Z_0 = \sqrt{L'/C'}$", f"{Z0:.2f}", "Ω"], [r"$\alpha$", f"{alpha:.3e}", "Np/m"], [r"length", f"{Lx:.2f}", "m"], [r"$T_{sim}$", f"{T*1e9:.1f}", "ns"], [r"$Z_s, Z_L$", "50, 50", "Ω"]]
    tbl = ax.table(cellText=rows, colLabels=["Quantity", "Value", "Units"], cellLoc="center", colLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.5)
    _save_polished(fig, "fig01c_parameters.png")

def _heatmap_polished(F, vmin, vmax, cmap="magma", label="", filename=""):
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    im = ax.imshow(
        F, origin="lower", aspect="auto",
        extent=[fdtd_x[0], fdtd_x[-1], ts_ns[0], ts_ns[-1]],
        vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest",
        rasterized=True
    )
    ax.set_xlabel("$x$ (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("$t$ (ns)", fontsize=AXIS_LABEL_FONTSIZE)
    cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.025)
    cb.set_label(label, fontsize=COLORBAR_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    _save_polished(fig, filename)

def fig02_fields_clean():
    Vmax = max(np.abs(V_fdtd).max(), np.abs(clean_V).max()); Imax = max(np.abs(I_fdtd).max(), np.abs(clean_I).max())
    _heatmap_polished(V_fdtd, -Vmax, Vmax, "RdBu_r", "V (V)", "fig02a_V_fdtd.png")
    _heatmap_polished(clean_V, -Vmax, Vmax, "RdBu_r", "V (V)", "fig02b_V_pinn.png")
    _heatmap_polished(np.abs(clean_V - V_fdtd), 0, None, "viridis", r"$|V_{\rm PINN}-V_{\rm FDTD}|$", "fig02c_V_error.png")
    _heatmap_polished(I_fdtd, -Imax, Imax, "RdBu_r", "I (A)", "fig02d_I_fdtd.png")
    _heatmap_polished(clean_I, -Imax, Imax, "RdBu_r", "I (A)", "fig02e_I_pinn.png")
    _heatmap_polished(np.abs(clean_I - I_fdtd), 0, None, "viridis", r"$|I_{\rm PINN}-I_{\rm FDTD}|$", "fig02f_I_error.png")

def fig03_fields_noisy():
    Vmax = max(np.abs(V_fdtd).max(), np.abs(noisy_V).max(), np.abs(do_V).max())
    Imax = max(np.abs(I_fdtd).max(), np.abs(noisy_I).max(), np.abs(do_I).max())
    _heatmap_polished(V_fdtd, -Vmax, Vmax, "RdBu_r", "V (V)", "fig03a_V_truth.png")
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    im = ax.imshow(noisy_V, origin="lower", aspect="auto", extent=[fdtd_x[0], fdtd_x[-1], ts_ns[0], ts_ns[-1]], vmin=-Vmax, vmax=Vmax, cmap="RdBu_r")
    ax.set_xlabel("$x$ (m)", fontsize=AXIS_LABEL_FONTSIZE); ax.set_ylabel("$t$ (ns)", fontsize=AXIS_LABEL_FONTSIZE)
    sx = noisy_data_pts[:, 0] * Lx; st = noisy_data_pts[:, 1] * T * 1e9
    ax.scatter(sx, st, marker="o", s=7, facecolor="none", edgecolor="0.15", linewidths=0.45, alpha=0.8, zorder=5)
    cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.025); cb.set_label("V (V)", fontsize=COLORBAR_LABEL_FONTSIZE); cb.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    _save_polished(fig, "fig03b_V_pinn_noisy.png")
    _heatmap_polished(np.abs(noisy_V - V_fdtd), 0, None, "viridis", r"$|V_{\rm PINN}-V_{\rm FDTD}|$", "fig03c_V_noisy_error.png")
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    im = ax.imshow(do_V, origin="lower", aspect="auto", extent=[fdtd_x[0], fdtd_x[-1], ts_ns[0], ts_ns[-1]], vmin=-Vmax, vmax=Vmax, cmap="RdBu_r")
    ax.set_xlabel("$x$ (m)", fontsize=AXIS_LABEL_FONTSIZE); ax.set_ylabel("$t$ (ns)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.scatter(sx, st, marker="o", s=7, facecolor="none", edgecolor="0.15", linewidths=0.45, alpha=0.8, zorder=5)
    cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.025); cb.set_label("V (V)", fontsize=COLORBAR_LABEL_FONTSIZE); cb.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    _save_polished(fig, "fig03d_V_dataonly.png")
    _heatmap_polished(np.abs(do_V - V_fdtd), 0, None, "viridis", r"$|V_{\rm ANN}-V_{\rm FDTD}|$", "fig03e_V_do_error.png")
    _heatmap_polished(I_fdtd, -Imax, Imax, "RdBu_r", "I (A)", "fig03f_I_truth.png")
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    im = ax.imshow(noisy_I, origin="lower", aspect="auto", extent=[fdtd_x[0], fdtd_x[-1], ts_ns[0], ts_ns[-1]], vmin=-Imax, vmax=Imax, cmap="RdBu_r")
    ax.set_xlabel("$x$ (m)", fontsize=AXIS_LABEL_FONTSIZE); ax.set_ylabel("$t$ (ns)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.scatter(sx, st, marker="o", s=7, facecolor="none", edgecolor="0.15", linewidths=0.45, alpha=0.8, zorder=5)
    cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.025); cb.set_label("I (A)", fontsize=COLORBAR_LABEL_FONTSIZE); cb.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    _save_polished(fig, "fig03g_I_pinn_noisy.png")
    _heatmap_polished(np.abs(noisy_I - I_fdtd), 0, None, "viridis", r"$|I_{\rm PINN}-I_{\rm FDTD}|$", "fig03h_I_noisy_error.png")
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    im = ax.imshow(do_I, origin="lower", aspect="auto", extent=[fdtd_x[0], fdtd_x[-1], ts_ns[0], ts_ns[-1]], vmin=-Imax, vmax=Imax, cmap="RdBu_r")
    ax.set_xlabel("$x$ (m)", fontsize=AXIS_LABEL_FONTSIZE); ax.set_ylabel("$t$ (ns)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.scatter(sx, st, marker="o", s=7, facecolor="none", edgecolor="0.15", linewidths=0.45, alpha=0.8, zorder=5)
    cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.025); cb.set_label("I (A)", fontsize=COLORBAR_LABEL_FONTSIZE); cb.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    _save_polished(fig, "fig03i_I_dataonly.png")
    _heatmap_polished(np.abs(do_I - I_fdtd), 0, None, "viridis", r"$|I_{\rm ANN}-I_{\rm FDTD}|$", "fig03j_I_do_error.png")

def fig04_snapshots():
    snap_t_ns = [3.0, 5.0, 8.0, 11.0]
    for t_ns in snap_t_ns:
        idx = np.argmin(np.abs(ts_ns - t_ns))
        for field, f_name, f_fdtd, f_clean, f_noisy, label in [("V", "Voltage", V_fdtd, clean_V, noisy_V, "V (V)"), ("I", "Current", I_fdtd, clean_I, noisy_I, "I (A)")]:
            fig, ax = plt.subplots(figsize=(3.55, 2.65))
            ax.plot(fdtd_x, f_fdtd[idx], "k-", lw=2.2, label="FDTD (Ref)")
            ax.plot(fdtd_x, f_clean[idx], "C0--", lw=2.0, label="PINN (Clean)")
            ax.plot(fdtd_x, f_noisy[idx], "C3:", lw=2.0, label="PINN (Noisy)")
            ax.set_xlabel("Position $x$ (m)")
            ax.set_ylabel(label)
            ax.legend(loc="upper right")
            _save_polished(fig, f"fig04{field.lower()}_snapshot_t{t_ns:.0f}ns.png")

def fig05_terminal_traces():
    pos_targets = [0.0, 0.5 * Lx, Lx]; pos_names = ["0", "L2", "L"]
    for j, xt in enumerate(pos_targets):
        ix = np.argmin(np.abs(fdtd_x - xt))
        for field, f_name, f_fdtd, f_clean, f_noisy, label in [("V", "Voltage", V_fdtd, clean_V, noisy_V, "V (V)"), ("I", "Current", I_fdtd, clean_I, noisy_I, "I (A)")]:
            fig, ax = plt.subplots(figsize=(3.55, 2.65))
            ax.plot(ts_ns, f_fdtd[:, ix], "k-", lw=2.0, label="FDTD")
            ax.plot(ts_ns, f_clean[:, ix], "C0--", lw=1.8, label="PINN clean")
            ax.plot(ts_ns, f_noisy[:, ix], "C3:", lw=1.8, label="PINN+data")
            if field == "V":
                nearby = np.abs(noisy_data_pts[:, 0] * Lx - xt) < 0.05
                if nearby.any():
                    ts_sensor = noisy_data_pts[nearby, 1] * T * 1e9
                    ax.scatter(ts_sensor, noisy_data_vals[nearby, 0], s=40, marker="x", c="C3", alpha=0.8, label="sensors", zorder=4)
            ax.set_xlabel("$t$ (ns)"); ax.set_ylabel(label); ax.legend()
            _save_polished(fig, f"fig05{field.lower()}_trace_x{pos_names[j]}.png")

def fig06_3d_surface():
    """Publication-style space-time maps replacing perspective 3-D surfaces."""
    def _space_time_map(F, label, filename, vmax, contour_color="0.15"):
        fig, ax = plt.subplots(figsize=(3.55, 2.65))
        norm = Normalize(vmin=-vmax, vmax=vmax)
        im = ax.imshow(
            F, origin="lower", aspect="auto",
            extent=[fdtd_x[0], fdtd_x[-1], ts_ns[0], ts_ns[-1]],
            cmap="RdBu_r", norm=norm, interpolation="nearest"
        )
        levels = np.linspace(-vmax, vmax, 13)
        levels = levels[np.abs(levels) > 0.08 * vmax]
        ax.contour(fdtd_x, ts_ns, F, levels=levels, colors=contour_color,
                   linewidths=0.55, alpha=0.65)
        ax.axvline(0.0, color="k", lw=1.0, alpha=0.45)
        ax.axvline(Lx, color="k", lw=1.0, alpha=0.45)
        ax.set_xlabel("Position $x$ (m)")
        ax.set_ylabel("Time $t$ (ns)")
        cb = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.025)
        cb.set_label(label)
        _save_polished(fig, filename)

    vlim = max(np.abs(V_fdtd).max(), np.abs(clean_V).max())
    elim = max(np.abs(V_fdtd - clean_V).max(), 1e-12)
    _space_time_map(V_fdtd, "Reference voltage $V$ (V)", "fig06a_3d_V_truth.png", vlim)
    _space_time_map(clean_V, "PINN voltage $V$ (V)", "fig06b_3d_V_pinn.png", vlim)
    _space_time_map(V_fdtd - clean_V, r"Signed error $\Delta V$ (V)", "fig06c_3d_V_error.png", elim)

def fig07_loss_curves():
    fig, ax = plt.subplots(figsize=(6, 5))
    for h, name, color in [(clean_hist, "PINN clean", "C0"), (noisy_hist, "PINN+data", "C3"), (do_hist, "data-only", "C2")]:
        ax.semilogy(h["epoch"], np.array(h["total"]) + 1e-15, color=color, label=name)
    ax.set_xlabel("epoch", fontweight='bold'); ax.set_ylabel("total loss", fontweight='bold'); ax.legend(fontsize=13)
    _save_polished(fig, "fig07a_loss_total.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, color in zip(["pde", "ic", "bc", "data"], ["C0", "C2", "C3", "C4"]):
        ax.semilogy(noisy_hist["epoch"], np.array(noisy_hist[k]) + 1e-15, color=color, label=k)
    ax.set_xlabel("epoch", fontweight='bold'); ax.set_ylabel("loss component", fontweight='bold'); ax.legend(fontsize=13)
    _save_polished(fig, "fig07b_loss_components.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, color in zip(["wp", "wi", "wb", "wd"], ["C0", "C2", "C3", "C4"]):
        ax.semilogy(noisy_hist["epoch"], np.array(noisy_hist[k]) + 1e-15, color=color, label=k)
    ax.set_xlabel("epoch", fontweight='bold'); ax.set_ylabel("adaptive weight", fontweight='bold'); ax.legend(fontsize=13)
    _save_polished(fig, "fig07c_adaptive_weights.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(clean_hist["wallclock"], np.array(clean_hist["total"]) + 1e-15, "C0-", label="PINN clean")
    ax.plot(noisy_hist["wallclock"], np.array(noisy_hist["total"]) + 1e-15, "C3-", label="PINN+data")
    ax.plot(do_hist["wallclock"], np.array(do_hist["total"]) + 1e-15, "C2-", label="data-only")
    ax.set_yscale("log"); ax.set_xlabel("wallclock (s)", fontweight='bold'); ax.set_ylabel("total loss", fontweight='bold'); ax.legend(fontsize=13)
    _save_polished(fig, "fig07d_wallclock_convergence.png")

def fig08_residual_field():
    def res(V, I, x, t):
        dx, dt = x[1]-x[0], t[1]-t[0]
        Vx, Vt = np.gradient(V, dx, axis=1), np.gradient(V, dt, axis=0)
        Ix, It = np.gradient(I, dx, axis=1), np.gradient(I, dt, axis=0)
        return Vx + Lp_t*It + Rp_t*I, Ix + Cp_t*Vt + Gp_t*V
    rVc, rIc = res(clean_V, clean_I, fdtd_x, fdtd_t); rVn, rIn = res(noisy_V, noisy_I, fdtd_x, fdtd_t)
    vmax, imax = max(np.abs(rVc).max(), np.abs(rVn).max()), max(np.abs(rIc).max(), np.abs(rIn).max())
    _heatmap_polished(rVc, -vmax, vmax, "RdBu_r", "$r_V$", "fig08a_resid_V_clean.png")
    _heatmap_polished(rVn, -vmax, vmax, "RdBu_r", "$r_V$", "fig08b_resid_V_noisy.png")
    _heatmap_polished(rIc, -imax, imax, "RdBu_r", "$r_I$", "fig08c_resid_I_clean.png")
    _heatmap_polished(rIn, -imax, imax, "RdBu_r", "$r_I$", "fig08d_resid_I_noisy.png")

def fig09_inverse_traj():
    names = ["R'", "L'", "G'", "C'"]; truths = [Rp_t, Lp_t, Gp_t, Cp_t]; colors = {0:"C0", 3:"C2", 10:"C3"}
    for k in range(4):
        fig, ax = plt.subplots(figsize=(6, 5))
        for tag, h in histories.items():
            n, s = int(tag[1:3]), int(tag[5:])
            ax.semilogy(h["epoch"], np.array(h["phi_hist"])[:, k], color=colors.get(n, "C0"), alpha=0.6, label=f"{n}% noise" if s==0 else None)
        ax.axhline(truths[k], color="k", linestyle="--", lw=1.5, label="truth"); ax.set_xlabel("epoch", fontweight='bold'); ax.set_ylabel(names[k], fontweight='bold'); ax.legend(fontsize=11, ncol=2)
        _save_polished(fig, f"fig09_traj_{names[k][0]}.png")

def fig10_param_recovery():
    fig, ax = plt.subplots(figsize=(8, 6)); bn = {}
    for s in summary: bn.setdefault(s["noise"], []).append(s["rel_err"])
    noises = sorted(bn.keys()); width, xc = 0.18, np.arange(len(noises)); colors, names, hatches = ["C0", "C2", "C3", "C4"], [r"$R'$", r"$L'$", r"$G'$", r"$C'$"], ["/", "\\", "|", "-"]
    for k in range(4):
        vals = [np.array([abs(rr[k]) for rr in bn[n]]) for n in noises]; pos = xc + (k-1.5)*width
        bp = ax.boxplot(vals, positions=pos, widths=width*0.9, showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=1.8))
        for p in bp['boxes']: p.set_facecolor(colors[k]); p.set_alpha(0.7); p.set_hatch(hatches[k]); p.set_edgecolor("black"); p.set_linewidth(1.2)
        for ix, v in enumerate(vals): ax.scatter(np.full(v.size, pos[ix]) + np.random.uniform(-0.02, 0.02, v.size), v, c="black", s=25, alpha=0.6, zorder=4)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[k], alpha=0.7, hatch=hatches[k], ec="k") for k in range(4)]
    ax.legend(handles, names, ncol=4, loc="upper left", fontsize=13); ax.set_xticks(xc); ax.set_xticklabels([f"{int(n*100)}%" for n in noises]); ax.set_xlabel("Measurement Noise (%)", fontweight='bold'); ax.set_ylabel(r"Relative Error Magnitude $| \epsilon |$ (%)", fontweight='bold'); ax.set_yscale("log")
    _save_polished(fig, "fig10_recovery_error.png")

def fig11_fim_eigvecs():
    ev, ec = np.linalg.eigh(F)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(np.arange(4), ev, color="skyblue", edgecolor="navy", alpha=0.8, hatch="//", linewidth=1.5)
    ax.set_yscale("log"); ax.set_xticks(np.arange(4)); ax.set_xticklabels([fr"$\lambda_{i+1}$" for i in range(4)]); ax.set_ylabel(r"FIM Eigenvalue $\lambda$", fontweight='bold')
    for i, v in enumerate(ev): ax.text(i, v*1.3, f"{v:.1e}", ha="center", fontsize=11, fontweight='bold', fontfamily='serif')
    _save_polished(fig, "fig11a_fim_spectrum.png")
    fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(np.abs(ec.T), cmap="PuBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([r"$\log R'$", r"$\log L'$", r"$\log G'$", r"$\log C'$"]); ax.set_yticks(range(4)); ax.set_yticklabels([f"Mode $v_{i+1}$" for i in range(4)])
    for i in range(4):
        for j in range(4):
            val = abs(ec[j, i]); color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label="Component Magnitude"); _save_polished(fig, "fig11b_fim_modes.png")

def fig12_sensitivity():
    fig, ax = plt.subplots(figsize=(8, 6)); names = ["R'", "L'", "G'", "C'"]; truths = [Rp_t, Lp_t, Gp_t, Cp_t]; bn = {}
    for s in summary: bn.setdefault(s["noise"], []).append([s[k] for k in ["Rp", "Lp", "Gp", "Cp"]])
    noises = sorted(bn.keys()); width, xc = 0.2, np.arange(4); colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(noises)))
    for i, n in enumerate(noises):
        arr = np.array(bn[n]); log_arr, log_truth = np.log(np.maximum(arr, 1e-30)), np.log(truths)
        std = np.std(log_arr - log_truth, axis=0, ddof=1) if arr.shape[0] > 1 else np.abs(log_arr - log_truth).mean(axis=0)
        ax.bar(xc + (i - len(noises)/2 + 0.5)*width, std, width=width*0.9, color=colors[i], edgecolor="k", alpha=0.8, label=f"noise {int(n*100)}%")
    ax.plot(xc, cr, "kD", ms=8, label="CR bound (5%)", zorder=5); ax.set_yscale("log"); ax.set_xticks(xc); ax.set_xticklabels(names); ax.set_ylabel(r"$\sigma(\log \phi)$", fontweight='bold'); ax.legend(ncol=2, fontsize=13)
    _save_polished(fig, "fig12_uncertainty.png")

def fig13_sensor_layout():
    fig, ax = plt.subplots(figsize=(7, 5)); im = ax.imshow(np.abs(V_fdtd), origin="lower", aspect="auto", extent=[fdtd_x[0], fdtd_x[-1], ts_ns[0], ts_ns[-1]], cmap="Greys", alpha=0.6)
    ax.scatter(noisy_data_pts[:, 0]*Lx, noisy_data_pts[:, 1]*T*1e9, s=40, marker="o", facecolor="C3", edgecolor="k", lw=0.5, zorder=3, label="sensors")
    ax.set_xlabel("x (m)", fontweight='bold'); ax.set_ylabel("t (ns)", fontweight='bold'); ax.legend(fontsize=13); fig.colorbar(im, ax=ax, shrink=0.8, label="|V| (V)"); _save_polished(fig, "fig13a_sensor_layout.png")
    fig, ax = plt.subplots(figsize=(7, 5)); order = np.argsort(noisy_data_clean[:, 0])
    ax.plot(np.arange(noisy_data_clean.shape[0]), noisy_data_clean[order, 0], "k-", lw=1.5, label="true V")
    ax.plot(np.arange(noisy_data_clean.shape[0]), noisy_data_vals[order, 0], "C3o", ms=5, alpha=0.7, label="noisy measured V")
    ax.set_xlabel("sensor index", fontweight='bold'); ax.set_ylabel("V (normalized)", fontweight='bold'); ax.legend(fontsize=13); _save_polished(fig, "fig13b_noise_samples.png")

def fig14_long_term_behavior():
    params = LineParams(Rp=Rp_t, Lp=Lp_t, Gp=Gp_t, Cp=Cp_t,
                        length=Lx, Zs=float(fwd["Zs"]), ZL=float(fwd["ZL"]))
    T_long = 60e-9
    x_long, t_long, V_long, I_half_long, info = fdtd_solve(
        params, T=T_long, Nx=401, cfl=0.95, record_every=2)
    t_long_ns = t_long * 1e9
    xI_long = info["xI"]
    I_long = np.zeros((I_half_long.shape[0], x_long.size))
    for k in range(I_half_long.shape[0]):
        I_long[k] = np.interp(x_long, xI_long, I_half_long[k])

    vmax = max(np.abs(V_long).max(), 1e-12)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    im = ax.imshow(V_long, origin="lower", aspect="auto",
                   extent=[x_long[0], x_long[-1], t_long_ns[0], t_long_ns[-1]],
                   cmap="RdBu_r", norm=Normalize(vmin=-vmax, vmax=vmax),
                   interpolation="nearest")
    levels = np.linspace(-vmax, vmax, 15)
    levels = levels[np.abs(levels) > 0.08 * vmax]
    ax.contour(x_long, t_long_ns, V_long, levels=levels,
               colors="0.15", linewidths=0.45, alpha=0.55)
    ax.set_xlabel("Position $x$ (m)", fontweight='bold')
    ax.set_ylabel("Time $t$ (ns)", fontweight='bold')
    cb = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.025)
    cb.set_label("Voltage $V$ (V)", fontweight='bold')
    _save_polished(fig, "fig14a_longterm_voltage_map.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(t_long_ns, V_long[:, 0], color="C0", lw=2.1, label=r"$V(0,t)$")
    ax.plot(t_long_ns, V_long[:, x_long.size // 2], color="C2", lw=2.1, label=r"$V(L/2,t)$")
    ax.plot(t_long_ns, V_long[:, -1], color="C3", lw=2.1, label=r"$V(L,t)$")
    ax.axvline(Lx / params.c * 1e9, color="0.25", lw=1.0, ls="--", alpha=0.8)
    ax.axvline(2 * Lx / params.c * 1e9, color="0.25", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlabel("Time $t$ (ns)", fontweight='bold')
    ax.set_ylabel("Voltage $V$ (V)", fontweight='bold')
    ax.legend(loc="upper right", fontsize=12)
    _save_polished(fig, "fig14b_longterm_terminal_traces.png")

    energy_density = 0.5 * params.Cp * V_long**2 + 0.5 * params.Lp * I_long**2
    energy = np.trapezoid(energy_density, x_long, axis=1)
    peak_energy = max(np.max(energy), 1e-30)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.semilogy(t_long_ns, energy / peak_energy, color="k", lw=2.2)
    ax.set_ylim(1e-8, 2)
    ax.set_xlabel("Time $t$ (ns)", fontweight='bold')
    ax.set_ylabel(r"Normalized line energy $E(t)/E_{\max}$", fontweight='bold')
    _save_polished(fig, "fig14c_longterm_energy_decay.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for t_ns, color in zip([3, 6, 10, 20, 40, 60], ["C0", "C1", "C2", "C3", "C4", "0.25"]):
        idx = np.argmin(np.abs(t_long_ns - t_ns))
        ax.plot(x_long, V_long[idx], color=color, lw=1.9, label=f"{t_long_ns[idx]:.0f} ns")
    ax.set_xlabel("Position $x$ (m)", fontweight='bold')
    ax.set_ylabel("Voltage $V$ (V)", fontweight='bold')
    ax.legend(loc="upper right", fontsize=11, ncol=2)
    _save_polished(fig, "fig14d_longterm_spatial_snapshots.png")

def fig15_weather_conditions():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    labels = ["Ambient", "Line"]
    temps = [env_values["ambient_c"], env_values["line_temp_c"]]
    bars = ax.bar(labels, temps, color=["#38bdf8", "#f97316"], edgecolor="k", linewidth=1.2)
    ax.set_ylabel("Temperature (deg C)", fontweight="bold")
    ax.text(0.03, 0.94, env_source, transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.5"))
    for bar, val in zip(bars, temps):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    _save_polished(fig, "fig15a_weather_temperatures.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    labels = ["Wind\n(km/h)", "Solar\n(W/m2)", "R' shift\n(%)"]
    rp_shift = (env_values["effective_Rp"] / max(env_values["base_Rp"], 1e-30) - 1.0) * 100
    vals = [env_values["wind_kmh"], env_values["solar_w_m2"], rp_shift]
    colors = ["#0f766e", "#facc15", "#7c3aed"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="k", linewidth=1.2)
    ax.set_ylabel("Recorded weather/coupling value", fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3g}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    _save_polished(fig, "fig15b_weather_coupling_summary.png")

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    heat_in = [env_values["q_joule_w_m"], env_values["q_solar_w_m"]]
    heat_out = [-env_values["q_convection_w_m"], -env_values["q_radiation_w_m"]]
    labels = ["Joule", "Solar", "Convection", "Radiation"]
    vals = heat_in + heat_out
    colors = ["#ef4444", "#f59e0b", "#2563eb", "#0f766e"]
    ax.axhline(0.0, color="k", lw=1.0)
    bars = ax.bar(labels, vals, color=colors, edgecolor="k", linewidth=1.2)
    ax.set_ylabel("Heat flow per metre (W/m)", fontweight="bold")
    for bar, val in zip(bars, vals):
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2e}", ha="center", va=va, fontsize=11, fontweight="bold")
    _save_polished(fig, "fig15c_heat_balance.png")

def write_tables():
    with open(f"{OUT}/table_forward_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "rel_L2_V", "rel_L2_I", "max_abs_err_V", "max_abs_err_I", "n_collocation", "n_data_sensors", "noise_pct"])
        w.writerow(["PINN clean", f"{clean_eV:.4e}", f"{clean_eI:.4e}", f"{np.max(np.abs(clean_V - V_fdtd)):.4e}", f"{np.max(np.abs(clean_I - I_fdtd)):.4e}", NP_COLLOCATION, 0, 0.0])
        w.writerow([f"PINN + {NOISY_ND} noisy sensors", f"{noisy_eV:.4e}", f"{noisy_eI:.4e}", f"{np.max(np.abs(noisy_V - V_fdtd)):.4e}", f"{np.max(np.abs(noisy_I - I_fdtd)):.4e}", NP_COLLOCATION, NOISY_ND, f"{NOISY_PCT:.1f}"])
        w.writerow(["data-only baseline", f"{do_eV:.4e}", f"{do_eI:.4e}", f"{np.max(np.abs(do_V - V_fdtd)):.4e}", f"{np.max(np.abs(do_I - I_fdtd)):.4e}", 0, NOISY_ND, f"{NOISY_PCT:.1f}"])
    with open(f"{OUT}/table_inverse_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["noise_pct", "seed", "Rp_recovered", "Lp_recovered", "Gp_recovered", "Cp_recovered", "rel_err_R(%)", "rel_err_L(%)", "rel_err_G(%)", "rel_err_C(%)", "rel_L2_V", "rel_L2_I", "wallclock_s"])
        for s in summary: w.writerow([f"{s['noise']*100:.0f}", s['seed'], f"{s['Rp']:.4e}", f"{s['Lp']:.4e}", f"{s['Gp']:.4e}", f"{s['Cp']:.4e}", f"{s['rel_err'][0]:+.2f}", f"{s['rel_err'][1]:+.2f}", f"{s['rel_err'][2]:+.2f}", f"{s['rel_err'][3]:+.2f}", f"{s['eV']:.4e}", f"{s['eI']:.4e}", f"{s['time_s']:.1f}"])
    with open(f"{OUT}/table_fim_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["FIM eigenvalue", "value"])
        for i, v in enumerate(eigs): w.writerow([f"lambda_{i+1}", f"{v:.4e}"])
        w.writerow([]); w.writerow(["FIM condition number", f"{cond:.4e}"]); w.writerow([]); w.writerow(["parameter", "Cramer-Rao std(log phi) at 5% noise"])
        for n, v in zip(["R'", "L'", "G'", "C'"], cr): w.writerow([n, f"{v:.4e}"])
    with open(f"{OUT}/table_validation_scope.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "reported_experiment_status"])
        w.writerow(["telegrapher PDE residuals r_V, r_I", "active"])
        w.writerow(["IC/BC/data losses", "active"])
        w.writerow(["thermal residual r_T", "inactive; extension hook"])
        w.writerow(["DLR penalty r_DLR", "inactive; extension hook"])
        w.writerow(["sag-tension residual r_S", "inactive; extension hook"])
        w.writerow(["live weather as training input", "inactive in reported validation"])

def main():
    print("Generating figures...")
    fig01_overview(); fig02_fields_clean(); fig03_fields_noisy(); fig04_snapshots(); fig05_terminal_traces(); fig06_3d_surface(); fig07_loss_curves(); fig08_residual_field(); fig09_inverse_traj(); fig10_param_recovery(); fig11_fim_eigvecs(); fig12_sensitivity(); fig13_sensor_layout(); fig14_long_term_behavior(); write_tables(); print("All figures and tables saved to ./figures/")

if __name__ == "__main__":
    main()
