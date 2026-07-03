"""
MetaMaxTwin3 Streamlit dashboard for the lossy transmission-line PINN.

Run:
    python -m streamlit run digital_twin_dashboard.py
"""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fdtd_reference import LineParams, fdtd_solve, interp_I_to_V_grid
from environmental_coupling import (
    DEFAULT_NOMINAL_CURRENT_A,
    DEFAULT_TEMP_COEFF,
    fetch_sydney_weather,
    solve_line_temperature_c,
    temperature_adjusted_rp,
)
from pinn_core import MLP, forward
from run_forward import make_problem


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "forward_runs.npz"
SYDNEY_TZ = ZoneInfo("Australia/Sydney")
SYDNEY_LAT = -33.8688
SYDNEY_LON = 151.2093
PLOT_FONT = "Times New Roman"

SCENARIOS = {
    "FDTD reference": ("fdtd", "Numerical reference"),
    "PINN clean": ("clean", "Physics + IC/BC"),
    "PINN noisy sensors": ("noisy", "Physics + sparse noisy data"),
    "Data-only baseline": ("dataonly", "Sparse data without PDE"),
}


st.set_page_config(
    page_title="MetaMaxTwin3",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1.2rem;}
    [data-testid="stSidebar"] {background: #111827;}
    [data-testid="stSidebar"] * {color: #f9fafb;}
    [data-testid="stMetricValue"] {font-size: 1.35rem;}
    .dt-title {
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: .1rem;
    }
    .dt-subtitle {
        color: #64748b;
        font-size: .95rem;
        margin-bottom: .65rem;
    }
    .dt-panel-title {
        color: #334155;
        font-size: .92rem;
        font-weight: 700;
        margin: .2rem 0 .35rem;
    }
    .dt-section {
        border-top: 1px solid #e5e7eb;
        padding-top: .75rem;
        margin-top: .85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_forward_results(path: str) -> dict:
    p = Path(path)
    if p.exists():
        raw = np.load(p, allow_pickle=True)
        return {k: raw[k] for k in raw.files}
    params, T = make_problem()
    x, t, V, I_half, info = fdtd_solve(params, T=T, Nx=181, cfl=0.95, record_every=2)
    I = interp_I_to_V_grid(x, I_half, info["xI"])
    return {
        "fdtd_x": x,
        "fdtd_t": t,
        "V_fdtd": V,
        "I_fdtd": I,
        "Rp": np.array(params.Rp),
        "base_Rp": np.array(getattr(params, "base_Rp", params.Rp)),
        "Lp": np.array(params.Lp),
        "Gp": np.array(params.Gp),
        "Cp": np.array(params.Cp),
        "length": np.array(params.length),
        "T": np.array(T),
        "Zs": np.array(params.Zs),
        "ZL": np.array(params.ZL),
        "Z0": np.array(params.Z0),
        "c": np.array(params.c),
        "alpha": np.array(params.alpha),
        "dt_fdtd": np.array(info["dt"]),
        "dx_fdtd": np.array(info["dx"]),
    }


@st.cache_data(ttl=600, show_spinner=False)
def load_sydney_weather() -> dict:
    weather = fetch_sydney_weather()
    return {
        "ok": bool(weather["ok"]),
        "temperature_c": float(weather["temperature_c"]),
        "humidity_pct": float(weather["humidity_pct"]),
        "wind_kmh": float(weather["wind_kmh"]),
        "solar_w_m2": float(weather.get("solar_w_m2", 0.0)),
        "is_day": bool(weather.get("is_day", False)),
        "weather_code": int(weather.get("weather_code", -1)),
        "observed_at": str(weather.get("observed_at", "")),
        "source": str(weather["source"]),
    }


def scalar(data: dict, key: str) -> float:
    return float(np.asarray(data[key]).item())


def available_scenarios(data: dict) -> list[str]:
    names = ["FDTD reference"]
    for label, (prefix, _) in SCENARIOS.items():
        if prefix != "fdtd" and f"{prefix}_V_pred" in data:
            names.append(label)
    return names


def field_arrays(data: dict, scenario_label: str) -> tuple[np.ndarray, np.ndarray]:
    prefix, _ = SCENARIOS[scenario_label]
    if prefix == "fdtd":
        return data["V_fdtd"], data["I_fdtd"]
    return data[f"{prefix}_V_pred"], data[f"{prefix}_I_pred"]


def pinn_predict(data: dict, scenario_label: str, x_phys: np.ndarray, t_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    prefix, _ = SCENARIOS[scenario_label]
    if prefix == "fdtd" or f"{prefix}_params" not in data or f"{prefix}_layers" not in data:
        return None
    x_norm = np.asarray(x_phys, dtype=float) / scalar(data, "length")
    t_norm = np.asarray(t_phys, dtype=float) / scalar(data, "T")
    pts = np.column_stack([x_norm.ravel(), t_norm.ravel()])
    net = MLP(layers=list(np.asarray(data[f"{prefix}_layers"], dtype=int)), seed=0)
    net.set_params(np.asarray(data[f"{prefix}_params"], dtype=float))
    u, _ = forward(net, pts)
    v = u[:, 0].reshape(np.shape(x_norm))
    i = (u[:, 1] / scalar(data, "Z0")).reshape(np.shape(x_norm))
    return v, i


def scenario_errors(data: dict, scenario_label: str) -> tuple[float | None, float | None]:
    prefix, _ = SCENARIOS[scenario_label]
    if prefix == "fdtd":
        return None, None
    return scalar(data, f"{prefix}_eV"), scalar(data, f"{prefix}_eI")


def parse_history(data: dict, scenario_label: str) -> dict | None:
    prefix, _ = SCENARIOS[scenario_label]
    key = f"{prefix}_hist"
    if key not in data:
        return None
    return json.loads(str(np.asarray(data[key]).item()))


def format_si(value: float, unit: str) -> str:
    if value == 0:
        return f"0 {unit}"
    exponent = int(np.floor(np.log10(abs(value)) / 3) * 3)
    exponent = max(min(exponent, 12), -12)
    prefixes = {-12: "p", -9: "n", -6: "u", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T"}
    scaled = value / (10 ** exponent)
    return f"{scaled:.3g} {prefixes[exponent]}{unit}"


def lighting_from_sydney_time(now: datetime, override_theme: str | None = None) -> dict:
    hour = now.hour + now.minute / 60.0
    daylight = 0.5 + 0.5 * math.sin((hour - 6.0) / 12.0 * math.pi)
    daylight = float(np.clip(daylight, 0.08, 1.0))
    
    if override_theme == "Light":
        daylight = 1.0
    elif override_theme == "Dark":
        daylight = 0.08

    if daylight > 0.72:
        sky = "#ffffff"
        ground = "#6b8f71"
        conductor = "#b6c2cc"
        text_color = "#0f172a"  # High-contrast black for daylight
        name = "Daylight"
    elif daylight > 0.28:
        sky = "#fde68a"
        ground = "#60715f"
        conductor = "#d6b36a"
        text_color = "#1e293b"
        name = "Twilight"
    else:
        sky = "#0f172a"
        ground = "#1f2937"
        conductor = "#7dd3fc"
        text_color = "#f8fafc"  # White for night
        name = "Night"
    
    return {
        "daylight": daylight,
        "sky": sky,
        "ground": ground,
        "conductor": conductor,
        "text_color": text_color,
        "name": name,
        "light_x": 0.25 + 1.5 * daylight,
        "light_y": -1.0,
        "light_z": 0.25 + 1.8 * daylight,
    }


def temperature_adjusted_line(
    base_rp: float,
    ambient_c: float,
    wind_kmh: float,
    solar_w_m2: float,
    current: np.ndarray,
    time_idx: int,
    temp_coeff: float,
    nominal_current_a: float,
) -> tuple[float, float, float, dict]:
    rms_i = float(np.sqrt(np.mean(current[time_idx] ** 2)))
    max_rms = float(np.sqrt(np.max(np.mean(current ** 2, axis=1)))) + 1e-12
    effective_current = max(nominal_current_a, DEFAULT_NOMINAL_CURRENT_A * (rms_i / max_rms))
    line_temp, thermal = solve_line_temperature_c(
        ambient_c=ambient_c,
        wind_kmh=wind_kmh,
        solar_w_m2=solar_w_m2,
        base_rp=base_rp,
        nominal_current_a=effective_current,
    )
    rp_temp = temperature_adjusted_rp(base_rp, line_temp, temp_coeff)
    attenuation_ratio = rp_temp / max(base_rp, 1e-12)
    thermal["effective_current_a"] = effective_current
    return line_temp, rp_temp, attenuation_ratio, thermal


@st.cache_data(show_spinner=False)
def solve_temperature_adjusted_reference(
    rp: float,
    lp: float,
    gp: float,
    cp: float,
    length: float,
    zs: float,
    zl: float,
    total_time: float,
    nx: int,
    record_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = LineParams(Rp=rp, Lp=lp, Gp=gp, Cp=cp, length=length, Zs=zs, ZL=zl)
    x_ref, t_ref, v_ref, i_half, info = fdtd_solve(params, T=total_time, Nx=nx, cfl=0.95, record_every=record_every)
    i_ref = interp_I_to_V_grid(x_ref, i_half, info["xI"])
    return x_ref, t_ref, v_ref, i_ref


def resample_like_reference(
    x_target: np.ndarray,
    t_target: np.ndarray,
    x_source: np.ndarray,
    t_source: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    if field.shape == (t_target.size, x_target.size) and np.allclose(x_source, x_target) and np.allclose(t_source, t_target):
        return field
    time_resampled = np.vstack([np.interp(x_target, x_source, row) for row in field])
    return np.vstack([np.interp(t_target, t_source, time_resampled[:, col]) for col in range(x_target.size)]).T


def thermally_coupled_fields(
    data: dict,
    scenario_label: str,
    base_v: np.ndarray,
    base_i: np.ndarray,
    rp_temp: float,
    apply_temperature: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    base_ref_v = data["V_fdtd"]
    base_ref_i = data["I_fdtd"]
    if not apply_temperature:
        return base_v, base_i, base_ref_v, base_ref_i, "temperature monitoring only"

    x_ref, t_ref, thermal_v_raw, thermal_i_raw = solve_temperature_adjusted_reference(
        rp=float(rp_temp),
        lp=scalar(data, "Lp"),
        gp=scalar(data, "Gp"),
        cp=scalar(data, "Cp"),
        length=scalar(data, "length"),
        zs=scalar(data, "Zs"),
        zl=scalar(data, "ZL"),
        total_time=scalar(data, "T"),
        nx=int(data["fdtd_x"].size),
        record_every=2,
    )
    thermal_v = resample_like_reference(data["fdtd_x"], data["fdtd_t"], x_ref, t_ref, thermal_v_raw)
    thermal_i = resample_like_reference(data["fdtd_x"], data["fdtd_t"], x_ref, t_ref, thermal_i_raw)

    prefix, _ = SCENARIOS[scenario_label]
    if prefix == "fdtd":
        return thermal_v, thermal_i, thermal_v, thermal_i, "temperature-adjusted FDTD solve"

    corrected_v = base_v + (thermal_v - base_ref_v)
    corrected_i = base_i + (thermal_i - base_ref_i)
    return corrected_v, corrected_i, thermal_v, thermal_i, "PINN plus temperature physics correction"


def cylinder_x(x: np.ndarray, y0: float, z0: float, radius: float, n_theta: int = 20):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    xx, tt = np.meshgrid(x, theta)
    yy = y0 + radius * np.cos(tt)
    zz = z0 + radius * np.sin(tt)
    return xx, yy, zz


def cylinder_z(x0: float, y0: float, z0: float, z1: float, radius: float, n_theta: int = 14):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.array([z0, z1])
    zz, tt = np.meshgrid(z, theta)
    xx = x0 + radius * np.cos(tt)
    yy = y0 + radius * np.sin(tt)
    return xx, yy, zz


def cone_mesh(
    x0: float,
    y0: float,
    z0: float,
    radius: float,
    height: float,
    n_theta: int = 14,
    y_scale: float = 0.72,
    twist: float = 0.0,
):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False) + twist
    x_ring = x0 + radius * np.cos(theta)
    y_ring = y0 + radius * y_scale * np.sin(theta)
    z_ring = z0 + 0.012 * height * np.sin(3 * theta)
    x_mesh = np.concatenate([[x0], x_ring, [x0]])
    y_mesh = np.concatenate([[y0], y_ring, [y0]])
    z_mesh = np.concatenate([[z0 + height], z_ring, [z0]])

    apex = np.zeros(n_theta, dtype=int)
    ring = np.arange(1, n_theta + 1)
    ring_next = np.roll(ring, -1)
    base_center = np.full(n_theta, n_theta + 1, dtype=int)
    return x_mesh, y_mesh, z_mesh, apex, ring, ring_next, base_center


def terrain_height_at(gx: np.ndarray, gy: np.ndarray, gzz: np.ndarray, tx: float, ty: float) -> float:
    row = int(np.clip(np.searchsorted(gy, ty), 1, gy.size - 1))
    y0, y1 = gy[row - 1], gy[row]
    z0 = np.interp(tx, gx, gzz[row - 1, :])
    z1 = np.interp(tx, gx, gzz[row, :])
    weight = 0.0 if y1 == y0 else (ty - y0) / (y1 - y0)
    return float((1.0 - weight) * z0 + weight * z1)


def dashed_3d_line(x_values: np.ndarray, y_values: np.ndarray, z_values: np.ndarray, dash_points: int = 4, gap_points: int = 2):
    x_dash: list[float | None] = []
    y_dash: list[float | None] = []
    z_dash: list[float | None] = []
    step = dash_points + gap_points
    for start in range(0, len(x_values) - 1, step):
        end = min(start + dash_points, len(x_values))
        x_dash.extend(x_values[start:end])
        y_dash.extend(y_values[start:end])
        z_dash.extend(z_values[start:end])
        x_dash.append(None)
        y_dash.append(None)
        z_dash.append(None)
    return x_dash, y_dash, z_dash


def make_physical_twin(
    x: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    time_idx: int,
    time_ns: float,
    field_label: str,
    sensors: np.ndarray | None,
    length: float,
    lighting: dict,
    ambient_c: float,
    line_temp_c: float,
    waveform_gain: float = 1.0,
) -> go.Figure:
    v = voltage[time_idx]
    i = current[time_idx]
    v_norm = v / max(np.max(np.abs(voltage)), 1e-12)
    i_norm = i / max(np.max(np.abs(current)), 1e-12)
    color_values = v if field_label == "Voltage" else i
    color_title = "V" if field_label == "Voltage" else "A"
    text_color = lighting["text_color"]

    fig = go.Figure()
    
    # Dynamic Water Physics & Landscape
    gx = np.linspace(0.0, length, 120)
    gy = np.linspace(-0.60, 0.60, 80)
    gxx, gyy = np.meshgrid(gx, gy)
    
    # Subtle wave movement based on time_ns
    wave_phase = time_ns * 0.2
    gzz = -0.2 + 0.015 * np.sin(2 * np.pi * gxx / (max(length, 1e-12) * 0.5) + wave_phase) * np.exp(-1.5 * np.abs(gyy))
    shoreline_wave = 0.018 * np.sin(8 * np.pi * gxx / length + 16 * gyy + wave_phase)
    fine_ripple = 0.007 * np.sin(18 * np.pi * gxx / length - 10 * gyy + 0.6 * wave_phase)
    beach_fade = np.clip((-gyy + 0.08) / 0.30, 0.0, 1.0)
    gzz += np.where(gyy < 0.08, (shoreline_wave + fine_ripple) * beach_fade, 0)
    
    # Sea texture with light reflection simulation (using lighting parameters)
    landscape_color = np.where(gyy < -0.18, 0.0, np.where(gyy > 0.22, 1.0, 0.5))
    fig.add_trace(
        go.Surface(
            x=gxx,
            y=gyy,
            z=gzz,
            surfacecolor=landscape_color,
            colorscale=[[0, "#0e7490"], [0.49, "#38bdf8"], [0.5, "#e7d8a0"], [0.74, "#d6c18b"], [0.75, "#14532d"], [1, "#166534"]],
            opacity=0.88,
            showscale=False,
            hoverinfo="skip",
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.5, roughness=0.1, fresnel=0.2),
            name="Sydney coast landscape",
        )
    )

    mesh_x = np.linspace(-0.1 * length, 1.1 * length, 15)
    mesh_y = np.linspace(-0.8, 0.8, 12)
    mesh_z = np.linspace(-0.25, 0.75, 8)
    cage_color = "rgba(0, 0, 0, 0.28)" if lighting["name"] == "Daylight" else "rgba(100, 116, 139, 0.15)"
    for mz in [mesh_z[0], mesh_z[-1]]:
        for mx in mesh_x:
            fig.add_trace(go.Scatter3d(x=[mx, mx], y=[mesh_y[0], mesh_y[-1]], z=[mz, mz], mode="lines", line=dict(color=cage_color, width=1), showlegend=False, hoverinfo="skip"))
        for my in mesh_y:
            fig.add_trace(go.Scatter3d(x=[mesh_x[0], mesh_x[-1]], y=[my, my], z=[mz, mz], mode="lines", line=dict(color=cage_color, width=1), showlegend=False, hoverinfo="skip"))

    # Layered 3D vegetation on the green corridor.
    rng = np.random.default_rng(42)
    is_night = lighting["name"] == "Night"
    trunk_color = "#5b3a29" if not is_night else "#3b241a"
    leaf_palette = ["#0f5132", "#166534", "#1f7a3a", "#2f8f46"] if not is_night else ["#082f1e", "#0f3d29", "#14532d"]
    branch_color = "rgba(91,58,41,0.82)" if not is_night else "rgba(59,36,26,0.75)"

    n_trees = 54
    tree_x = rng.uniform(0.06 * length, 0.94 * length, n_trees)
    green_band_mask = rng.random(n_trees) < 0.34
    tree_y = np.where(
        green_band_mask,
        rng.uniform(0.10, 0.24, n_trees),
        rng.uniform(0.27, 0.57, n_trees),
    )

    for tx, ty in zip(tree_x, tree_y):
        tz = terrain_height_at(gx, gy, gzz, float(tx), float(ty))
        depth_scale = 0.85 + 0.9 * (ty - 0.10) / 0.47
        tree_h = 0.24 * rng.uniform(0.08, 0.18) * depth_scale
        trunk_h = tree_h * rng.uniform(0.42, 0.58)
        trunk_r = max(0.004, tree_h * 0.045)
        crown_r = tree_h * rng.uniform(0.19, 0.28)
        crown_z = tz + trunk_h * 0.72
        leaf_color = leaf_palette[int(rng.integers(0, len(leaf_palette)))]

        txx, tyy, tzz = cylinder_z(float(tx), float(ty), tz, tz + trunk_h, trunk_r, n_theta=10)
        fig.add_trace(
            go.Surface(
                x=txx,
                y=tyy,
                z=tzz,
                colorscale=[[0, trunk_color], [1, trunk_color]],
                showscale=False,
                opacity=0.95,
                lighting=dict(ambient=0.35, diffuse=0.75, specular=0.08, roughness=0.95),
                lightposition=dict(x=lighting["light_x"], y=lighting["light_y"], z=lighting["light_z"]),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        branch_x = [tx, tx - 0.35 * crown_r, None, tx, tx + 0.38 * crown_r, None]
        branch_y = [ty, ty + 0.18 * crown_r, None, ty, ty - 0.20 * crown_r, None]
        branch_z = [tz + trunk_h * 0.68, crown_z + 0.10 * tree_h, None, tz + trunk_h * 0.74, crown_z + 0.16 * tree_h, None]
        fig.add_trace(go.Scatter3d(x=branch_x, y=branch_y, z=branch_z, mode="lines", line=dict(color=branch_color, width=max(2, int(18 * trunk_r))), hoverinfo="skip", showlegend=False))

        for layer, scale in enumerate([1.0, 0.72]):
            cone_z = crown_z + layer * tree_h * 0.21
            mx, my, mz, mi, mj, mk, base = cone_mesh(
                float(tx) + rng.uniform(-0.12, 0.12) * crown_r,
                float(ty) + rng.uniform(-0.10, 0.10) * crown_r,
                cone_z,
                crown_r * scale,
                tree_h * (0.42 if layer == 0 else 0.34),
                n_theta=14,
                y_scale=rng.uniform(0.58, 0.82),
                twist=rng.uniform(0, np.pi),
            )
            fig.add_trace(
                go.Mesh3d(
                    x=mx,
                    y=my,
                    z=mz,
                    i=np.concatenate([mi, base]),
                    j=np.concatenate([mj, mk]),
                    k=np.concatenate([mk, mj]),
                    color=leaf_color,
                    opacity=0.92,
                    flatshading=False,
                    lighting=dict(ambient=0.38, diffuse=0.82, specular=0.12, roughness=0.85),
                    lightposition=dict(x=lighting["light_x"], y=lighting["light_y"], z=lighting["light_z"]),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    n_scrub = 90
    scrub_x = rng.uniform(0.04 * length, 0.96 * length, n_scrub)
    scrub_y = rng.uniform(0.07, 0.58, n_scrub)
    scrub_z = np.array([terrain_height_at(gx, gy, gzz, float(sx), float(sy)) + rng.uniform(0.005, 0.018) for sx, sy in zip(scrub_x, scrub_y)])
    scrub_colors = rng.choice(leaf_palette, n_scrub)
    scrub_sizes = 2.5 + 7.5 * (scrub_y - scrub_y.min()) / max(float(np.ptp(scrub_y)), 1e-12)
    fig.add_trace(
        go.Scatter3d(
            x=scrub_x,
            y=scrub_y,
            z=scrub_z,
            mode="markers",
            marker=dict(size=scrub_sizes, color=scrub_colors, opacity=0.72, symbol="circle"),
            name="Green corridor scrub",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Volumetric Clouds & Skybox
    if lighting["name"] == "Night":
        fig.add_trace(
            go.Scatter3d(
                x=[0.12 * length], y=[-0.36], z=[0.42],
                mode="markers",
                marker=dict(size=22, color="#f8fafc", symbol="circle", opacity=0.9),
                hoverinfo="skip",
                name="Moon",
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=[0.1 * length], y=[-0.5], z=[0.6],
                mode="markers",
                marker=dict(size=28, color="#fde68a", symbol="circle", opacity=0.9),
                hoverinfo="skip",
                name="Sun",
            )
        )
        # Volumetric clouds using stacked surfaces
        def add_cloud(cx, cy, cz, scale):
            for i in range(3):
                theta = np.linspace(0, 2 * np.pi, 12)
                phi = np.linspace(0, np.pi, 6)
                tt, pp = np.meshgrid(theta, phi)
                fig.add_trace(
                    go.Surface(
                        x=cx + scale * (1 + 0.2*i) * np.sin(pp) * np.cos(tt),
                        y=cy + scale * 0.5 * np.sin(pp) * np.sin(tt),
                        z=cz + 0.02*i + scale * 0.3 * np.cos(pp),
                        colorscale=[[0, "#ffffff"], [1, "#f1f5f9"]],
                        showscale=False,
                        opacity=0.4 - 0.1*i,
                        hoverinfo="skip",
                    )
                )
        add_cloud(0.22 * length, -0.4, 0.38, 0.06)
        add_cloud(0.65 * length, -0.38, 0.42, 0.08)

    # Conductors with enhanced rendering
    for y0 in (-0.055, 0.055):
        xx, yy, zz = cylinder_x(x, y0, np.zeros_like(x), 0.012)
        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=zz,
                surfacecolor=np.tile(color_values, (yy.shape[0], 1)),
                colorscale="Turbo",
                colorbar=dict(title=dict(text=color_title, font=dict(color=text_color, family=PLOT_FONT)), thickness=14) if y0 > 0 else None,
                showscale=y0 > 0,
                lighting=dict(
                    ambient=0.3 + 0.3 * lighting["daylight"],
                    diffuse=0.8,
                    specular=0.8,
                    roughness=0.2,
                    fresnel=0.3,
                ),
                lightposition=dict(x=lighting["light_x"], y=lighting["light_y"], z=lighting["light_z"]),
                hovertemplate=f"Line temperature={line_temp_c:.1f} C<extra></extra>",
                name="Conductors",
            )
        )

    # Telemetry Overlay: dashed voltage and current waveforms.
    voltage_wave_color = "#0369a1" if lighting["name"] == "Daylight" else "#22d3ee"
    current_wave_color = "#166534" if lighting["name"] == "Daylight" else "#4ade80"
    wave_z = 0.28 + 0.12 * waveform_gain * v_norm
    voltage_dash_x, voltage_dash_y, voltage_dash_z = dashed_3d_line(x, np.full_like(x, -0.05), wave_z)
    fig.add_trace(
        go.Scatter3d(
            x=voltage_dash_x,
            y=voltage_dash_y,
            z=voltage_dash_z,
            mode="lines",
            line=dict(color=voltage_wave_color, width=5),
            hovertemplate="V-Wave: %{z:.4f}<extra></extra>",
            name="Voltage waveform",
        )
    )
    wave_y = 0.05 + 0.15 * waveform_gain * i_norm
    current_dash_x, current_dash_y, current_dash_z = dashed_3d_line(x, wave_y, np.full_like(x, 0.32))
    fig.add_trace(
        go.Scatter3d(
            x=current_dash_x,
            y=current_dash_y,
            z=current_dash_z,
            mode="lines",
            line=dict(color=current_wave_color, width=5),
            hovertemplate="I-Wave: %{y:.4f}<extra></extra>",
            name="Current waveform",
        )
    )

    # Sensors with connection lines (telemetry overlay)
    if sensors is not None and sensors.size:
        sx = sensors[:, 0] * length
        sy = np.full(sensors.shape[0], -0.22)
        sz = np.full(sensors.shape[0], 0.035)
        fig.add_trace(
            go.Scatter3d(
                x=sx, y=sy, z=sz,
                mode="markers",
                marker=dict(size=8, color="#f59e0b", symbol="diamond", line=dict(color="#111827", width=1)),
                hovertemplate="sensor x=%{x:.3f} m<extra></extra>",
                name="Sensors",
            )
        )
        # Sensor connection lines
        for sxi in sx:
            fig.add_trace(
                go.Scatter3d(
                    x=[sxi, sxi], y=[-0.22, -0.055], z=[0.035, 0.012],
                    mode="lines",
                    line=dict(color="rgba(245, 158, 11, 0.4)", width=2, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    tower_detail_text = {
        "Lattice Steel Framework": "Interconnected steel angles provide high strength, durability, and resistance to wind, ice loads, and vibration.",
        "Crossarms": "Horizontal steel arms hold conductors far apart so wires do not touch each other or the tower.",
        "Tower Top & Peak": "The peak carries the overhead ground or shield wire, protecting lower power lines from lightning.",
        "Base/Legs": "Four square base points anchor the tower into concrete foundations.",
    }
    tower_positions = np.linspace(0.08 * length, 0.92 * length, 5)
    for xp in tower_positions:
        tower_lines = [
            ("Lattice Steel Framework", [xp, xp], [-0.13, -0.055], [-0.16, 0.04], "#334155", 5),
            ("Lattice Steel Framework", [xp, xp], [0.13, 0.055], [-0.16, 0.04], "#334155", 5),
            ("Base/Legs", [xp, xp], [-0.16, 0.16], [-0.16, -0.16], "#1e293b", 7),
            ("Crossarms", [xp, xp], [-0.12, 0.12], [-0.04, -0.04], "#475569", 6),
            ("Crossarms", [xp, xp], [-0.09, 0.09], [0.04, 0.04], "#475569", 6),
            ("Tower Top & Peak", [xp, xp], [-0.06, 0.06], [0.09, 0.09], "#0f172a", 6),
            ("Tower Top & Peak", [xp, xp], [0.0, 0.0], [0.09, 0.15], "#0f172a", 5),
        ]
        for component, lx_, ly_, lz_, line_color, line_width in tower_lines:
            fig.add_trace(
                go.Scatter3d(
                    x=lx_, y=ly_, z=lz_,
                    mode="lines",
                    line=dict(color=line_color, width=line_width),
                    hovertemplate=f"<b>{component}</b><br>{tower_detail_text[component]}<extra></extra>",
                    showlegend=False,
                )
            )
        if np.isclose(xp, tower_positions[len(tower_positions) // 2]):
            fig.add_trace(
                go.Scatter3d(
                    x=[xp, xp, xp, xp],
                    y=[0.18, 0.16, 0.13, 0.0],
                    z=[0.10, -0.04, -0.16, 0.16],
                    mode="text",
                    text=["Crossarms", "Lattice steel", "Base/legs", "Shield wire peak"],
                    textfont=dict(color=text_color, size=12, family=PLOT_FONT),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # 3D source/load icons: wind turbine source and residential load.
    source_x = 0.0
    load_x = length
    icon_y = 0.0
    ground_z = -0.16
    turbine_hub_z = 0.20
    turbine_r = 0.055
    fig.add_trace(
        go.Scatter3d(
            x=[source_x, source_x],
            y=[icon_y, icon_y],
            z=[ground_z, turbine_hub_z],
            mode="lines",
            line=dict(color="#e5e7eb" if lighting["name"] == "Night" else "#64748b", width=8),
            hovertemplate="<b>SOURCE</b><br>3D wind turbine source<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[source_x],
            y=[icon_y],
            z=[turbine_hub_z],
            mode="markers",
            marker=dict(size=8, color="#16a34a", symbol="circle", line=dict(color="#052e16", width=2)),
            hovertemplate="<b>SOURCE</b><br>Wind turbine generator<extra></extra>",
            showlegend=False,
        )
    )
    blade_lines = []
    for blade_angle in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3):
        blade_lines.extend([source_x, icon_y, turbine_hub_z])
        blade_lines.extend([source_x, icon_y + turbine_r * np.cos(blade_angle), turbine_hub_z + turbine_r * np.sin(blade_angle)])
        blade_lines.extend([None, None, None])
    fig.add_trace(
        go.Scatter3d(
            x=blade_lines[0::3],
            y=blade_lines[1::3],
            z=blade_lines[2::3],
            mode="lines",
            line=dict(color="#f8fafc" if lighting["name"] == "Night" else "#0f172a", width=5),
            hovertemplate="<b>SOURCE</b><br>Wind turbine blades<extra></extra>",
            showlegend=False,
        )
    )

    house_w = 0.045 * length
    house_d = 0.075
    house_h = 0.07
    roof_h = 0.045
    hx0, hx1 = load_x - house_w, load_x
    hy0, hy1 = icon_y - house_d / 2, icon_y + house_d / 2
    hz0, hz1 = ground_z, ground_z + house_h
    house_x = [hx0, hx1, hx1, hx0, hx0, hx1, hx1, hx0]
    house_y = [hy0, hy0, hy1, hy1, hy0, hy0, hy1, hy1]
    house_z = [hz0, hz0, hz0, hz0, hz1, hz1, hz1, hz1]
    fig.add_trace(
        go.Mesh3d(
            x=house_x,
            y=house_y,
            z=house_z,
            i=[0, 0, 0, 4, 4, 4, 0, 1, 2, 3, 0, 1],
            j=[1, 2, 3, 5, 6, 7, 1, 2, 3, 0, 4, 5],
            k=[2, 3, 0, 6, 7, 4, 5, 6, 7, 4, 5, 6],
            color="#f8fafc" if lighting["name"] == "Night" else "#fef3c7",
            opacity=0.96,
            hovertemplate="<b>LOAD</b><br>3D residential load<extra></extra>",
            showlegend=False,
        )
    )
    roof_x = [hx0, hx1, hx1, hx0, (hx0 + hx1) / 2, (hx0 + hx1) / 2]
    roof_y = [hy0, hy0, hy1, hy1, hy0, hy1]
    roof_z = [hz1, hz1, hz1, hz1, hz1 + roof_h, hz1 + roof_h]
    fig.add_trace(
        go.Mesh3d(
            x=roof_x,
            y=roof_y,
            z=roof_z,
            i=[0, 1, 2, 3, 0, 1],
            j=[1, 2, 3, 0, 4, 5],
            k=[4, 5, 5, 4, 5, 4],
            color="#dc2626",
            opacity=0.98,
            hovertemplate="<b>LOAD</b><br>House roof load marker<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[source_x, load_x - house_w / 2],
            y=[icon_y, icon_y],
            z=[turbine_hub_z + 0.045, hz1 + roof_h + 0.03],
            mode="text",
            text=["<b>SOURCE</b>", "<b>LOAD</b>"],
            textfont=dict(color=text_color, size=17, family=PLOT_FONT),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"<b>Sydney {lighting['name']}</b><br>Ambient: {ambient_c:.1f} °C<br>Conductor: {line_temp_c:.1f} °C",
        showarrow=False,
        bgcolor="rgba(15,23,42,.65)" if lighting["daylight"] > 0.5 else "rgba(255,255,255,.15)",
        bordercolor="rgba(255,255,255,.2)",
        font=dict(color="#f8fafc" if lighting["daylight"] > 0.5 else text_color, size=14, family=PLOT_FONT),
        align="left",
    )

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=lighting["sky"],
        font=dict(color=text_color, family=PLOT_FONT),
        scene=dict(
            bgcolor=lighting["sky"],
            xaxis=dict(title=dict(text="Position x (m)", font=dict(color=text_color, family=PLOT_FONT)), tickfont=dict(color=text_color, family=PLOT_FONT), backgroundcolor="rgba(255,255,255,.05)", gridcolor="rgba(100,116,139,.2)"),
            yaxis=dict(title=dict(text="Lateral offset", font=dict(color=text_color, family=PLOT_FONT)), tickfont=dict(color=text_color, family=PLOT_FONT), backgroundcolor="rgba(255,255,255,.05)", gridcolor="rgba(100,116,139,.2)"),
            zaxis=dict(title=dict(text="Field height", font=dict(color=text_color, family=PLOT_FONT)), tickfont=dict(color=text_color, family=PLOT_FONT), backgroundcolor="rgba(255,255,255,.05)", gridcolor="rgba(100,116,139,.2)"),
            aspectmode="manual",
            aspectratio=dict(x=3.2, y=1.0, z=0.9),
            camera=dict(eye=dict(x=1.6, y=-1.8, z=0.95), center=dict(x=0.0, y=0.0, z=-0.05)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=text_color, family=PLOT_FONT)),
    )
    return fig


def downsample_surface(z: np.ndarray, x: np.ndarray, t: np.ndarray, max_x: int = 120, max_t: int = 120):
    sx = max(1, int(np.ceil(x.size / max_x)))
    stp = max(1, int(np.ceil(t.size / max_t)))
    return x[::sx], t[::stp], z[::stp, ::sx]


def make_surface(x: np.ndarray, t: np.ndarray, z: np.ndarray, title: str, unit: str) -> go.Figure:
    xs, ts, zs = downsample_surface(z, x, t)
    fig = go.Figure(
        data=[
            go.Surface(
                x=xs,
                y=ts * 1e9,
                z=zs,
                colorscale="Turbo",
                colorbar=dict(title=unit, thickness=14),
                contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "#111827"}},
            )
        ]
    )
    fig.update_layout(
        height=470,
        margin=dict(l=0, r=0, t=28, b=0),
        title=dict(text=title, x=0.02, y=0.98, font=dict(size=15)),
        scene=dict(
            xaxis_title="Position x (m)",
            yaxis_title="Time (ns)",
            zaxis_title=unit,
            camera=dict(eye=dict(x=1.65, y=-1.7, z=0.85)),
        ),
    )
    return fig


def make_heatmap(x: np.ndarray, t: np.ndarray, z: np.ndarray, title: str, unit: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=x,
                y=t * 1e9,
                z=z,
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title=unit, thickness=14),
            )
        ]
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=34, b=0),
        title=dict(text=title, x=0.01, font=dict(size=15)),
        xaxis_title="Position x (m)",
        yaxis_title="Time (ns)",
    )
    return fig


def make_trace_plot(x: np.ndarray, field: np.ndarray, time_idx: int, label: str, unit: str, lighting: dict) -> go.Figure:
    text = lighting["text_color"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=field[time_idx],
            mode="lines",
            line=dict(color="#2563eb", width=3),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, .15)",
            name=label,
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=18, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=dict(text="Position x (m)", font=dict(color=text)), tickfont=dict(color=text), gridcolor="rgba(100,116,139,.1)"),
        yaxis=dict(title=dict(text=unit, font=dict(color=text)), tickfont=dict(color=text), gridcolor="rgba(100,116,139,.1)"),
        showlegend=False,
    )
    return fig


def make_time_trace(t: np.ndarray, field: np.ndarray, x_idx: int, label: str, unit: str, lighting: dict) -> go.Figure:
    text = lighting["text_color"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t * 1e9,
            y=field[:, x_idx],
            mode="lines",
            line=dict(color="#0f766e", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(15, 118, 110, .1)",
            name=label,
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=18, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=dict(text="Time (ns)", font=dict(color=text)), tickfont=dict(color=text), gridcolor="rgba(100,116,139,.1)"),
        yaxis=dict(title=dict(text=unit, font=dict(color=text)), tickfont=dict(color=text), gridcolor="rgba(100,116,139,.1)"),
        showlegend=False,
    )
    return fig


def make_top_view_line(
    x: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    time_idx: int,
    field_label: str,
    sensors: np.ndarray | None,
    length: float,
    lighting: dict,
    probe_x: float,
) -> go.Figure:
    v = voltage[time_idx]
    i = current[time_idx]
    color_values = v if field_label == "Voltage" else i
    color_title = "V" if field_label == "Voltage" else "A"
    is_night = lighting["name"] == "Night"
    bg = lighting["sky"]
    text = lighting["text_color"]
    grid = "rgba(148,163,184,.2)" if is_night else "rgba(15,23,42,.1)"
    
    water = "#0e7490" if is_night else "#38bdf8"
    sand = "#a78b5f" if is_night else "#e7d8a0"
    bush = "#14532d" if is_night else "#15803d"
    conductor = lighting["conductor"]

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=length, y0=-0.36, y1=-0.16, fillcolor=water, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=length, y0=-0.16, y1=0.02, fillcolor=sand, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=length, y0=0.02, y1=0.36, fillcolor=bush, line_width=0, layer="below")

    for offset in (-0.055, 0.055):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.full_like(x, offset),
                mode="lines",
                line=dict(color=conductor, width=13),
                hoverinfo="skip",
                name="conductor",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.full_like(x, offset),
                mode="markers",
                marker=dict(
                    size=8,
                    color=color_values,
                    colorscale="Turbo",
                    colorbar=dict(title=dict(text=color_title, font=dict(color=text, family=PLOT_FONT)), thickness=12) if offset > 0 else None,
                    showscale=offset > 0,
                ),
                customdata=np.column_stack([v, i]),
                hovertemplate="x=%{x:.3f} m<br>V=%{customdata[0]:.4f} V<br>I=%{customdata[1]:.5f} A<extra></extra>",
                name="live field",
                showlegend=offset > 0,
            )
        )

    for xp in np.linspace(0.08 * length, 0.92 * length, 5):
        fig.add_shape(type="line", x0=xp, x1=xp, y0=-0.12, y1=0.12, line=dict(color="#64748b", width=4))

    fig.add_trace(
        go.Scatter(
            x=[0, length],
            y=[0, 0],
            mode="markers+text",
            marker=dict(
                size=[22, 24],
                color=["#16a34a", "#dc2626"],
                symbol=["diamond", "square"],
                line=dict(color=["#052e16", "#450a0a"], width=3),
            ),
            text=["<b>SOURCE</b>", "<b>LOAD</b>"],
            textfont=dict(color=text, size=17, family=PLOT_FONT),
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
            name="Terminals",
        )
    )
    fig.add_shape(type="line", x0=probe_x, x1=probe_x, y0=-0.34, y1=0.34, line=dict(color="#f97316", width=2, dash="dot"))
    fig.add_annotation(x=probe_x, y=0.34, text="probe", showarrow=False, font=dict(color=text, size=12, family=PLOT_FONT), yshift=10)

    if sensors is not None and sensors.size:
        fig.add_trace(
            go.Scatter(
                x=sensors[:, 0] * length,
                y=np.full(sensors.shape[0], -0.24),
                mode="markers",
                marker=dict(size=10, color="#f59e0b", symbol="diamond", line=dict(color="#111827", width=1)),
                hovertemplate="sensor x=%{x:.3f} m<extra></extra>",
                name="Sensors",
            )
        )

    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        xaxis=dict(title=dict(text="Position x (m)", font=dict(color=text, family=PLOT_FONT)), tickfont=dict(color=text, family=PLOT_FONT), range=[-0.02 * length, 1.02 * length], gridcolor=grid, color=text),
        yaxis=dict(title=dict(text="Top-view corridor", font=dict(color=text, family=PLOT_FONT)), tickfont=dict(color=text, family=PLOT_FONT), range=[-0.38, 0.38], gridcolor=grid, color=text, zeroline=False),
        legend=dict(orientation="h", y=1.1, font=dict(color=text, family=PLOT_FONT)),
        font=dict(color=text, family=PLOT_FONT),
    )
    return fig


def make_live_probe_prediction(
    data: dict,
    scenario_label: str,
    t: np.ndarray,
    probe_x: float,
    field_label: str,
    fallback_field: np.ndarray,
    fallback_x_idx: int,
    lighting: dict,
) -> go.Figure:
    pred = pinn_predict(data, scenario_label, np.full_like(t, probe_x), t)
    unit = "V" if field_label == "Voltage" else "A"
    text = lighting["text_color"]
    if pred is None:
        y = fallback_field[:, fallback_x_idx]
        source = "FDTD reference"
    else:
        v_pred, i_pred = pred
        y = v_pred if field_label == "Voltage" else i_pred
        source = "PINN inference"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t * 1e9,
            y=y,
            mode="lines",
            line=dict(color="#2563eb", width=3.5),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,.15)",
            name=source,
        )
    )
    peak_t = t[np.argmax(np.abs(y))] * 1e9
    fig.add_vline(x=peak_t, line_color="#ef4444", line_dash="dot")
    fig.add_annotation(x=peak_t, y=0.9, yref="paper", text="peak", showarrow=False, font=dict(color=text))
    
    fig.update_layout(
        height=270,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"PINN Forecast at x={probe_x:.2f} m", x=0.01, font=dict(size=14, color=text)),
        xaxis=dict(title=dict(text="Horizon (ns)", font=dict(color=text)), tickfont=dict(color=text), gridcolor="rgba(100,116,139,.1)"),
        yaxis=dict(title=dict(text=unit, font=dict(color=text)), tickfont=dict(color=text), gridcolor="rgba(100,116,139,.1)"),
        showlegend=False,
    )
    return fig


def make_technical_phase_plot(t: np.ndarray, voltage: np.ndarray, current: np.ndarray, x_idx: int, lighting: dict) -> go.Figure:
    text = lighting["text_color"]
    z_inst = np.divide(voltage[:, x_idx], current[:, x_idx], out=np.full_like(voltage[:, x_idx], np.nan), where=np.abs(current[:, x_idx]) > 1e-7)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t * 1e9, y=voltage[:, x_idx], mode="lines", name="V (V)", line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(x=t * 1e9, y=50.0 * current[:, x_idx], mode="lines", name="50xI (A)", line=dict(color="#0f766e", width=2)))
    fig.add_trace(go.Scatter(x=t * 1e9, y=np.clip(z_inst, -200, 200), mode="lines", name="V/I (Ω)", yaxis="y2", line=dict(color="#f97316", width=1.5, dash="dash")))
    fig.update_layout(
        height=285,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Electrical Phase & Impedance", x=0.01, font=dict(size=14, color=text)),
        xaxis=dict(title=dict(text="Time (ns)", font=dict(color=text)), tickfont=dict(color=text), gridcolor="rgba(100,116,139,.1)"),
        yaxis=dict(title=dict(text="V / 50I", font=dict(color=text)), tickfont=dict(color=text), gridcolor="rgba(100,116,139,.1)"),
        yaxis2=dict(title=dict(text="Z (Ω)", font=dict(color=text, size=11)), tickfont=dict(color=text, size=10), overlaying="y", side="right", range=[-220, 220]),
        legend=dict(orientation="h", y=1.12, font=dict(color=text, size=10)),
    )
    return fig


def make_heat_balance_chart(thermal_terms: dict, lighting: dict) -> go.Figure:
    text = lighting["text_color"]
    labels = ["Joule", "Solar", "Convection", "Radiation"]
    values = [
        thermal_terms["q_joule_w_m"],
        thermal_terms["q_solar_w_m"],
        -thermal_terms["q_convection_w_m"],
        -thermal_terms["q_radiation_w_m"],
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=["#ef4444", "#f59e0b", "#2563eb", "#0f766e"], opacity=0.85))
    fig.add_hline(y=0, line_color=text, line_width=1)
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=28, b=0),
        title=dict(text="Thermal Flux Balance", x=0.01, font=dict(size=14, color=text)),
        xaxis=dict(tickfont=dict(color=text)),
        yaxis=dict(title=dict(text="W/m", font=dict(color=text)), tickfont=dict(color=text), gridcolor="rgba(100,116,139,.1)"),
        showlegend=False,
    )
    return fig


data = load_forward_results(str(RESULTS_PATH))
weather = load_sydney_weather()

# Configuration Sidebar
with st.sidebar:
    st.markdown("### MetaMaxTwin3 Controls")
    theme_mode = st.radio("Display Theme", ["Auto", "Light", "Dark"], horizontal=True)
    
    scenario_label = st.selectbox("PINN / reference source", available_scenarios(data), index=1 if "PINN clean" in available_scenarios(data) else 0)
    field_label = st.radio("Field coloring", ["Voltage", "Current"], horizontal=True)
    
    t = data["fdtd_t"]
    time_ns = st.slider(
        "Simulation time",
        float(t.min() * 1e9),
        float(t.max() * 1e9),
        float(t[np.argmax(np.max(np.abs(data["V_fdtd"]), axis=1))] * 1e9),
        step=0.05,
    )
    waveform_gain = st.slider(
        "3D signal amplitude gain",
        0.5,
        8.0,
        1.0,
        step=0.1,
        help="Amplifies only the 3D dashed voltage/current waveform height and lateral swing for observability.",
    )
    
    length = scalar(data, "length")
    probe_x = st.slider("Probe position", 0.0, float(length), float(length * 0.5), step=0.01)
    temp_coeff = st.slider("Conductor temp coefficient", 0.0010, 0.0060, DEFAULT_TEMP_COEFF, step=0.0001, format="%.4f / C")
    nominal_current_a = st.slider("Thermal RMS current", 0.0, 0.25, DEFAULT_NOMINAL_CURRENT_A, step=0.005, format="%.3f A")
    apply_temperature_to_model = st.toggle("Apply temperature to fields", value=True)
    compare_reference = st.toggle("PINN error map", value=scenario_label != "FDTD reference")

    st.markdown('<div class="dt-section"></div>', unsafe_allow_html=True)
    st.caption("Line parameters")
    st.write(f"R' = {format_si(scalar(data, 'Rp'), 'ohm/m')}")
    st.write(f"L' = {format_si(scalar(data, 'Lp'), 'H/m')}")
    st.write(f"G' = {format_si(scalar(data, 'Gp'), 'S/m')}")
    st.write(f"C' = {format_si(scalar(data, 'Cp'), 'F/m')}")

now_sydney = datetime.now(SYDNEY_TZ)
lighting = lighting_from_sydney_time(now_sydney, override_theme=None if theme_mode == "Auto" else theme_mode)
x = data["fdtd_x"]
t = data["fdtd_t"]
length = scalar(data, "length")

base_V, base_I = field_arrays(data, scenario_label)
time_idx = int(np.argmin(np.abs(t * 1e9 - time_ns)))
probe_idx = int(np.argmin(np.abs(x - probe_x)))
prefix, scenario_note = SCENARIOS[scenario_label]
sensors = data.get(f"{prefix}_data_pts") if prefix != "fdtd" else data.get("noisy_data_pts")
ambient_c = float(weather["temperature_c"])
wind_kmh = 0.0 if np.isnan(weather["wind_kmh"]) else float(weather["wind_kmh"])
solar_w_m2 = float(weather.get("solar_w_m2", 0.0))
line_temp_c, rp_temp, attenuation_ratio, thermal_terms = temperature_adjusted_line(
    scalar(data, "base_Rp") if "base_Rp" in data else scalar(data, "Rp"),
    ambient_c,
    wind_kmh,
    solar_w_m2,
    base_I,
    time_idx,
    temp_coeff,
    nominal_current_a,
)
V, I, reference_V, reference_I, coupling_note = thermally_coupled_fields(
    data, scenario_label, base_V, base_I, rp_temp, apply_temperature_to_model
)
primary = V if field_label == "Voltage" else I
unit = "V" if field_label == "Voltage" else "A"

st.markdown('<div class="dt-title">MetaMaxTwin3: Weather-Coupled PINN Digital Twin</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="dt-subtitle">{scenario_label} · {scenario_note} · Sydney {now_sydney.strftime("%Y-%m-%d %H:%M")} · '
    f'{lighting["name"]} lighting · {coupling_note} · state at {t[time_idx] * 1e9:.2f} ns</div>',
    unsafe_allow_html=True,
)

left, center, right = st.columns([0.82, 2.2, 0.95], gap="large")

max_v = float(np.max(np.abs(V[time_idx])))
max_i = float(np.max(np.abs(I[time_idx])))
energy_proxy = float(np.trapezoid(V[time_idx] ** 2 + (scalar(data, "Z0") * I[time_idx]) ** 2, x))
e_v, e_i = scenario_errors(data, scenario_label)
if scenario_label != "FDTD reference":
    e_v = float(np.linalg.norm(V - reference_V) / (np.linalg.norm(reference_V) + 1e-12))
    e_i = float(np.linalg.norm(I - reference_I) / (np.linalg.norm(reference_I) + 1e-12))

with left:
    st.markdown('<div class="dt-panel-title">Environmental Telemetry</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Ambient", f"{ambient_c:.1f} °C")
    m2.metric("Conductor", f"{line_temp_c:.1f} °C", f"{line_temp_c - ambient_c:+.1f} Δ")
    
    m3, m4 = st.columns(2)
    m3.metric("Wind", f"{wind_kmh:.1f} km/h")
    m4.metric("Solar", f"{solar_w_m2:.0f} W/m²")
    
    st.caption(f"Source: {weather['source']}")

    st.markdown('<div class="dt-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dt-panel-title">Physics-Informed Metrics</div>', unsafe_allow_html=True)
    
    # Resistance and Attenuation
    st.metric("Resistance (R')", format_si(rp_temp, "Ω/m"), f"{(attenuation_ratio - 1.0) * 100:+.2f}% thermal shift")
    
    # Thermal Balance Breakdown
    st.markdown("**Thermal Flux Balance (W/m)**")
    t_cols = st.columns(2)
    t_cols[0].write(f"In: {thermal_terms['q_solar_w_m'] + thermal_terms['q_joule_w_m']:.2f}")
    t_cols[1].write(f"Out: {thermal_terms['q_convection_w_m'] + thermal_terms['q_radiation_w_m']:.2f}")
    
    # Progress bars for flux components
    total_in = thermal_terms['q_solar_w_m'] + thermal_terms['q_joule_w_m'] + 1e-9
    st.progress(thermal_terms['q_solar_w_m'] / total_in, text="Solar Gain")
    st.progress(thermal_terms['q_joule_w_m'] / total_in, text="Joule Heating")

    if e_v is not None and e_i is not None:
        st.markdown('<div class="dt-section"></div>', unsafe_allow_html=True)
        st.markdown('<div class="dt-panel-title">PINN Inference Integrity</div>', unsafe_allow_html=True)
        st.metric("L2 Voltage Error", f"{100 * e_v:.3f}%", delta_color="inverse")
        st.metric("L2 Current Error", f"{100 * e_i:.3f}%", delta_color="inverse")

with center:
    st.plotly_chart(
        make_physical_twin(x, V, I, time_idx, time_ns, field_label, sensors, length, lighting, ambient_c, line_temp_c, waveform_gain),
        use_container_width=True,
    )
    st.plotly_chart(
        make_top_view_line(x, V, I, time_idx, field_label, sensors, length, lighting, probe_x),
        use_container_width=True,
    )

with right:
    st.markdown('<div class="dt-panel-title">Real-time Electrical State</div>', unsafe_allow_html=True)
    
    r1, r2 = st.columns(2)
    r1.metric("Peak Voltage", f"{max_v:.2f} V")
    r2.metric("Peak Current", f"{max_i * 1e3:.1f} mA")
    
    r3, r4 = st.columns(2)
    r3.metric("Impedance", f"{scalar(data, 'Z0'):.1f} Ω")
    r4.metric("Phase Vel.", f"{scalar(data, 'c') / 1e8:.2f}c")
    
    st.plotly_chart(make_trace_plot(x, primary, time_idx, field_label, unit, lighting), use_container_width=True)
    st.plotly_chart(make_time_trace(t, primary, probe_idx, f"{field_label} at probe", unit, lighting), use_container_width=True)

live_left, live_mid, live_right = st.columns([1.15, 1.0, 0.9], gap="large")

with live_left:
    st.markdown('<div class="dt-section"></div>', unsafe_allow_html=True)
    st.plotly_chart(
        make_live_probe_prediction(data, scenario_label, t, probe_x, field_label, primary, probe_idx, lighting),
        use_container_width=True,
    )

with live_mid:
    st.markdown('<div class="dt-section"></div>', unsafe_allow_html=True)
    st.plotly_chart(make_technical_phase_plot(t, V, I, probe_idx, lighting), use_container_width=True)

with live_right:
    st.markdown('<div class="dt-section"></div>', unsafe_allow_html=True)
    st.plotly_chart(make_heat_balance_chart(thermal_terms, lighting), use_container_width=True)

if compare_reference and scenario_label != "FDTD reference":
    ref = reference_V if field_label == "Voltage" else reference_I
    err = primary - ref
    err_trace = np.linalg.norm(err, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
    err_fig = go.Figure()
    err_fig.add_trace(go.Scatter(x=t * 1e9, y=100 * err_trace, mode="lines", line=dict(color="#dc2626", width=2.5)))
    err_fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=28, b=0),
        title=dict(text="Live PINN reference error over prediction horizon", x=0.01, font=dict(size=14)),
        xaxis_title="Time (ns)",
        yaxis_title="Relative error (%)",
        showlegend=False,
    )
    st.plotly_chart(err_fig, width="stretch")

if not weather["ok"]:
    st.warning("Live Sydney weather could not be reached. The dashboard is using a 20.0 C fallback for thermal estimates.")

if not RESULTS_PATH.exists():
    st.warning("Saved PINN results were not found. The dashboard is showing a lightweight FDTD-only twin.")
