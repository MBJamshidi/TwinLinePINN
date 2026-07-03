"""Shared environmental coupling for the transmission-line model and dashboard."""
from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass

from fdtd_reference import LineParams


SYDNEY_LAT = -33.8688
SYDNEY_LON = 151.2093
DEFAULT_TEMP_COEFF = 0.0039
DEFAULT_DIAMETER_M = 4.95e-3
DEFAULT_ABSORPTIVITY = 0.72
DEFAULT_EMISSIVITY = 0.82
DEFAULT_NOMINAL_CURRENT_A = 2.0e-2
STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass(frozen=True)
class EnvironmentState:
    ambient_c: float
    line_temp_c: float
    temp_coeff: float
    base_rp: float
    effective_rp: float
    source: str
    weather_ok: bool
    humidity_pct: float = float("nan")
    wind_kmh: float = float("nan")
    solar_w_m2: float = 0.0
    is_day: bool = False
    convective_h_w_m2k: float = 0.0
    q_joule_w_m: float = 0.0
    q_solar_w_m: float = 0.0
    q_convection_w_m: float = 0.0
    q_radiation_w_m: float = 0.0
    conductor_diameter_m: float = DEFAULT_DIAMETER_M
    absorptivity: float = DEFAULT_ABSORPTIVITY
    emissivity: float = DEFAULT_EMISSIVITY
    nominal_current_a: float = DEFAULT_NOMINAL_CURRENT_A


def fetch_sydney_weather(timeout_s: float = 6.0) -> dict:
    query = urllib.parse.urlencode(
        {
            "latitude": SYDNEY_LAT,
            "longitude": SYDNEY_LON,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,is_day,weather_code",
            "timezone": "Australia/Sydney",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        current = payload["current"]
        return {
            "ok": True,
            "temperature_c": float(current["temperature_2m"]),
            "humidity_pct": float(current["relative_humidity_2m"]),
            "wind_kmh": float(current["wind_speed_10m"]),
            "solar_w_m2": float(current.get("shortwave_radiation", 0.0)),
            "is_day": bool(current.get("is_day", 0)),
            "observed_at": str(current.get("time", "")),
            "source": "Open-Meteo Sydney live weather",
        }
    except Exception as exc:
        return {
            "ok": False,
            "temperature_c": 20.0,
            "humidity_pct": float("nan"),
            "wind_kmh": float("nan"),
            "solar_w_m2": 0.0,
            "is_day": False,
            "observed_at": "",
            "source": f"20 C fallback; weather unavailable: {exc.__class__.__name__}",
        }


def temperature_adjusted_rp(base_rp: float, line_temp_c: float, temp_coeff: float = DEFAULT_TEMP_COEFF) -> float:
    return base_rp * (1.0 + temp_coeff * (line_temp_c - 20.0))


def convective_coefficient_w_m2k(wind_kmh: float) -> float:
    if math.isnan(wind_kmh):
        wind_kmh = 0.0
    wind_m_s = max(0.0, wind_kmh / 3.6)
    return 5.7 + 3.8 * wind_m_s


def solve_line_temperature_c(
    ambient_c: float,
    wind_kmh: float,
    solar_w_m2: float,
    base_rp: float,
    nominal_current_a: float = DEFAULT_NOMINAL_CURRENT_A,
    conductor_diameter_m: float = DEFAULT_DIAMETER_M,
    absorptivity: float = DEFAULT_ABSORPTIVITY,
    emissivity: float = DEFAULT_EMISSIVITY,
) -> tuple[float, dict]:
    area_per_m = math.pi * conductor_diameter_m
    projected_area_per_m = conductor_diameter_m
    h = convective_coefficient_w_m2k(wind_kmh)
    q_joule = nominal_current_a ** 2 * base_rp
    q_solar = max(0.0, solar_w_m2) * absorptivity * projected_area_per_m
    ambient_k = ambient_c + 273.15

    def residual(temp_c: float) -> float:
        temp_k = temp_c + 273.15
        q_conv = h * area_per_m * (temp_c - ambient_c)
        q_rad = emissivity * STEFAN_BOLTZMANN * area_per_m * (temp_k ** 4 - ambient_k ** 4)
        return q_joule + q_solar - q_conv - q_rad

    lo = ambient_c - 40.0
    hi = ambient_c + 120.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    line_temp_c = 0.5 * (lo + hi)
    line_k = line_temp_c + 273.15
    q_convection = h * area_per_m * (line_temp_c - ambient_c)
    q_radiation = emissivity * STEFAN_BOLTZMANN * area_per_m * (line_k ** 4 - ambient_k ** 4)
    return line_temp_c, {
        "convective_h_w_m2k": h,
        "q_joule_w_m": q_joule,
        "q_solar_w_m": q_solar,
        "q_convection_w_m": q_convection,
        "q_radiation_w_m": q_radiation,
    }


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def current_environment_for_model(base_rp: float, temp_coeff: float = DEFAULT_TEMP_COEFF) -> EnvironmentState:
    override = os.getenv("PINN_AMBIENT_TEMP_C")
    if override is not None:
        ambient_c = float(override)
        weather = {
            "ok": True,
            "temperature_c": ambient_c,
            "humidity_pct": float("nan"),
            "wind_kmh": _float_env("PINN_WIND_KMH", 0.0),
            "solar_w_m2": _float_env("PINN_SOLAR_W_M2", 0.0),
            "is_day": _float_env("PINN_SOLAR_W_M2", 0.0) > 0.0,
            "source": "PINN_AMBIENT_TEMP_C override",
        }
    else:
        weather = fetch_sydney_weather()
        ambient_c = float(weather["temperature_c"])

    wind_kmh = float(weather["wind_kmh"])
    if math.isnan(wind_kmh):
        wind_kmh = 0.0
    solar_w_m2 = max(0.0, float(weather.get("solar_w_m2", 0.0)))
    nominal_current_a = _float_env("PINN_NOMINAL_CURRENT_A", DEFAULT_NOMINAL_CURRENT_A)
    conductor_diameter_m = _float_env("PINN_CONDUCTOR_DIAMETER_M", DEFAULT_DIAMETER_M)
    absorptivity = _float_env("PINN_SOLAR_ABSORPTIVITY", DEFAULT_ABSORPTIVITY)
    emissivity = _float_env("PINN_THERMAL_EMISSIVITY", DEFAULT_EMISSIVITY)

    line_temp_c, thermal = solve_line_temperature_c(
        ambient_c=ambient_c,
        wind_kmh=wind_kmh,
        solar_w_m2=solar_w_m2,
        base_rp=base_rp,
        nominal_current_a=nominal_current_a,
        conductor_diameter_m=conductor_diameter_m,
        absorptivity=absorptivity,
        emissivity=emissivity,
    )
    effective_rp = temperature_adjusted_rp(base_rp, line_temp_c, temp_coeff)
    return EnvironmentState(
        ambient_c=ambient_c,
        line_temp_c=line_temp_c,
        temp_coeff=temp_coeff,
        base_rp=base_rp,
        effective_rp=effective_rp,
        source=str(weather["source"]),
        weather_ok=bool(weather["ok"]),
        humidity_pct=float(weather["humidity_pct"]),
        wind_kmh=wind_kmh,
        solar_w_m2=solar_w_m2,
        is_day=bool(weather.get("is_day", False)),
        convective_h_w_m2k=float(thermal["convective_h_w_m2k"]),
        q_joule_w_m=float(thermal["q_joule_w_m"]),
        q_solar_w_m=float(thermal["q_solar_w_m"]),
        q_convection_w_m=float(thermal["q_convection_w_m"]),
        q_radiation_w_m=float(thermal["q_radiation_w_m"]),
        conductor_diameter_m=conductor_diameter_m,
        absorptivity=absorptivity,
        emissivity=emissivity,
        nominal_current_a=nominal_current_a,
    )


def make_environmental_line_params(
    base_params: LineParams,
    temp_coeff: float = DEFAULT_TEMP_COEFF,
) -> tuple[LineParams, EnvironmentState]:
    env = current_environment_for_model(base_params.Rp, temp_coeff=temp_coeff)
    params = LineParams(
        Rp=env.effective_rp,
        Lp=base_params.Lp,
        Gp=base_params.Gp,
        Cp=base_params.Cp,
        length=base_params.length,
        Zs=base_params.Zs,
        ZL=base_params.ZL,
    )
    return params, env
