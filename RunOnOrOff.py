"""Single-day battery SOC simulation for an electric vehicle.

The model includes traction power, auxiliaries, battery thermal behavior and
regenerative braking (RBS). Battery aging and external/overnight charging are
intentionally outside the scope of this module.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from BatteryHeat import Heat, HeatTransfer_btms, HeatTransfer_hvac
from ModelPara import BatElecInputs, Lookup_OCVandR, VehicleInfo
from VehDynModel import BatPower, DischargCurrent, RBSCurrent
from lookuptable import TableLookup1D, TableLookup2D


DEFAULT_CAPACITY_AH = 198.5
DEFAULT_INITIAL_SOC = 1.0
MAX_REGEN_C_RATE = 0.6
RUN_TIMESTEP_S = 1.0
# Parameters used only by the original reference-power comparison path.
WHEEL_RADIUS_M = 0.30
GEAR_RATIO = 9.0
DRIVETRAIN_EFFICIENCY = 0.95


def BREVO_RunTrips(
    BTMSOnOrOff="Off",
    HVACOnOrOff="Off",
    RBSOnOrOff="On",
    initial_soc=DEFAULT_INITIAL_SOC,
    capacity_ah=DEFAULT_CAPACITY_AH,
    trip_file=None,
    output_dir=".",
):
    """Run one driving-cycle CSV and save its SOC trace and summary.

    Negative battery current is produced only by RBS. No external charging or
    battery-aging calculation is performed.
    """
    trip_path = _resolve_trip_file(trip_file)
    trip = _load_trip(trip_path)
    date_label = trip_path.stem.removeprefix("Trip_File_")
    try:
        year = int(date_label[:4])
    except ValueError:
        year = 1
    result = DailyTripSimulation(
        trip,
        BTMSOnOrOff=BTMSOnOrOff,
        HVACOnOrOff=HVACOnOrOff,
        RBSOnOrOff=RBSOnOrOff,
        initial_soc=initial_soc,
        capacity_ah=capacity_ah,
        date_label=date_label,
        year=year,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trace_file = output_path / "HiDrive_SOC_Timeseries.csv"
    summary_file = output_path / "HiDrive_SOC_Summary.csv"
    result["trace"].to_csv(trace_file, index=False, encoding="utf-8")
    pd.DataFrame([result["summary"]]).to_csv(summary_file, index=False, encoding="utf-8")

    print(f"Trip: {trip_path}")
    print(f"Final SOC: {result['summary']['final_soc']:.6f}")
    return result


def DailyTripSimulation(
    trip,
    BTMSOnOrOff="Off",
    HVACOnOrOff="Off",
    RBSOnOrOff="On",
    initial_soc=DEFAULT_INITIAL_SOC,
    capacity_ah=DEFAULT_CAPACITY_AH,
    date_label="",
    year=1,
):
    """Calculate the SOC trajectory for one day of driving data."""
    if capacity_ah <= 0:
        raise ValueError("capacity_ah must be positive")
    if not 0 <= initial_soc <= 1:
        raise ValueError("initial_soc must be between 0 and 1")

    (
        A_d,
        c_d,
        c_r,
        f_acc,
        f_pt_max,
        f_pt_min,
        f_rbs,
        m_v,
        grav,
        rho_air,
        sigma,
        _theta,
        v_wind,
        P_pt_fixed,
    ) = VehicleInfo()
    (
        C_b,
        C_c,
        f_btms,
        K_ac,
        K_ab,
        K_bc,
        K_btms,
        _P_d,
        Q_rad,
        T_low,
        T_up,
        _V_max,
        _V_min,
    ) = BatElecInputs()
    SOC_OCVx, SOC_OCVy, SOC_Tba_Rx, SOC_Tba_Ry, SOC_Tba_Rz = Lookup_OCVandR()

    time_s = trip["Time Step (s)"].to_numpy(dtype=float)
    speed = trip["Cycle Speed (m/s)"].to_numpy(dtype=float)
    grade = trip["Grade"].to_numpy(dtype=float)
    ambient = trip["Temperature (C)"].to_numpy(dtype=float)
    status = trip["Running Status"].astype(str).str.strip().str.lower().to_numpy()
    acceleration = np.gradient(speed, time_s) if len(trip) > 1 else np.zeros(1)

    soc = float(initial_soc)
    battery_temp = float(ambient[0])
    cabin_temp = float(ambient[0])
    distance_m = 0.0
    net_energy_ws = 0.0
    rbs_energy_ws = 0.0
    rows = []

    for i in range(len(trip)):
        dt = RUN_TIMESTEP_S if i == 0 else max(time_s[i] - time_s[i - 1], 0.0)
        running = status[i] in {"runon", "runons"}
        current = 0.0
        battery_power_w = 0.0
        reference_power_w = np.nan
        F_d = abs(0.5 * rho_air * A_d * c_d * (speed[i] - v_wind) ** 2)
        F_r = abs(c_r * m_v * grav * np.cos(grade[i]))
        F_g = m_v * grav * np.sin(grade[i])
        F_a = sigma * m_v * acceleration[i]
        total_force_n = F_d + F_r + F_g + F_a

        if running and dt > 0 and soc > 0:
            Q_hvac, K_ac_hvac = HeatTransfer_hvac(HVACOnOrOff, K_ac, ambient[i], cabin_temp)
            Q_btms, P_btms = HeatTransfer_btms(
                BTMSOnOrOff, f_btms, K_btms, battery_temp, T_low, T_up
            )
            ocv = TableLookup1D(SOC_OCVx, SOC_OCVy, soc)
            resistance = TableLookup2D(
                SOC_Tba_Rx, SOC_Tba_Ry, SOC_Tba_Rz, soc, battery_temp
            )
            battery_power_w = BatPower(
                acceleration[i], A_d, c_d, c_r, f_acc, f_pt_max, f_pt_min,
                f_rbs, grav, m_v, ocv, P_btms, Q_hvac, rho_air, resistance,
                RBSOnOrOff, sigma, grade[i], speed[i], v_wind, P_pt_fixed,
            ) * 1000.0

            if battery_power_w >= 0:
                current = DischargCurrent(ocv, battery_power_w, resistance)
            else:
                # RBS is retained: negative current increases SOC.
                current = RBSCurrent(ocv, battery_power_w, resistance)
                current = max(current, -MAX_REGEN_C_RATE * capacity_ah)
                battery_power_w = ocv * current - resistance * current * current
                rbs_energy_ws += -battery_power_w * dt

            reference_power_w = _reference_battery_power(
                total_force_n, speed[i], ocv, soc, RBSOnOrOff
            )

            soc = float(np.clip(soc - current * dt / (capacity_ah * 3600.0), 0.0, 1.0))
            battery_temp, cabin_temp = Heat(
                current, C_b, C_c, K_ab, K_ac_hvac, K_bc, Q_btms, Q_hvac,
                Q_rad, resistance, dt, ambient[i], battery_temp, cabin_temp,
            )
            distance_m += speed[i] * dt
            net_energy_ws += battery_power_w * dt

        rows.append({
            "Date": date_label,
            "Year": year,
            "t_idx": int(time_s[i]),
            "v_mps": float(speed[i]),
            "F_total_N": float(total_force_n),
            "SOC": float(soc),
            "P_model_W": float(battery_power_w),
            "P_ref_W": reference_power_w,
            "current_A": float(current),
            "c_rate": float(abs(current) / capacity_ah),
        })

    trace = pd.DataFrame(rows)
    summary = {
        "initial_soc": float(initial_soc),
        "final_soc": soc,
        "soc_change": soc - float(initial_soc),
        "distance_km": distance_m / 1000.0,
        "net_battery_energy_kwh": net_energy_ws / 3_600_000.0,
        "rbs_energy_kwh": rbs_energy_ws / 3_600_000.0,
        "rbs_enabled": RBSOnOrOff == "On",
    }
    return {"trace": trace, "summary": summary}


def _reference_battery_power(total_force_n, speed_mps, ocv, soc, rbs_enabled):
    """Original empirical motor-to-battery reference-power calculation."""
    powertrain_w = total_force_n * speed_mps
    if speed_mps <= 0:
        return np.nan
    if powertrain_w < 0 and (speed_mps < 1.389 or rbs_enabled != "On"):
        return np.nan

    motor_torque = (
        total_force_n * WHEEL_RADIUS_M / (GEAR_RATIO * DRIVETRAIN_EFFICIENCY)
    )
    speed_kmh = speed_mps * 3.6
    if powertrain_w >= 0:
        motor_current = (
            2.333 + 1.302 * motor_torque + 0.045 * speed_kmh
            - 7.780e-4 * motor_torque**2
            + 2.222e-4 * motor_torque * speed_kmh
            + 3.325e-4 * speed_kmh**2
        )
        battery_current = (
            0.221 - 0.018 * motor_current + 4.45e-3 * speed_kmh
            + 2.285e-4 * motor_current**2
            + 0.016 * motor_current * speed_kmh
            - 1.670e-4 * speed_kmh**2
        )
    else:
        if soc < 0.9:
            motor_current = (
                -2.278 + 1.307 * motor_torque + 0.085 * speed_kmh
                + 9.635e-4 * motor_torque**2
                + 2.929e-4 * motor_torque * speed_kmh
                - 7.392e-4 * speed_kmh**2
            )
        else:
            motor_current = (
                -1.991 + 1.302 * motor_torque + 0.064 * speed_kmh
                + 9.527e-4 * motor_torque**2
                + 1.858e-4 * motor_torque * speed_kmh
                - 4.774e-4 * speed_kmh**2
            )
        battery_current = (
            -1.323 - 0.049 * motor_current - 0.036 * speed_kmh
            - 5.106e-4 * motor_current**2
            + 0.013 * motor_current * speed_kmh
            + 4.824e-4 * speed_kmh**2
        )
    return float(battery_current * ocv)


def _resolve_trip_file(trip_file):
    if trip_file is not None:
        path = Path(trip_file)
        if not path.is_file():
            raise FileNotFoundError(f"Trip file not found: {path}")
        return path

    files = sorted(Path("DayTrips").glob("*.csv"))
    if not files:
        raise FileNotFoundError("No driving-cycle CSV found in DayTrips")
    if len(files) > 1:
        raise ValueError("Multiple trip files found; pass trip_file explicitly for a single-day run")
    return files[0]


def _load_trip(path):
    trip = pd.read_csv(path)
    required = {
        "Time Step (s)", "Cycle Speed (m/s)", "Grade",
        "Temperature (C)", "Running Status",
    }
    missing = required.difference(trip.columns)
    if missing:
        raise ValueError(f"Missing trip columns: {', '.join(sorted(missing))}")

    # The original files contain a second row with units/text labels.
    trip["Time Step (s)"] = pd.to_numeric(trip["Time Step (s)"], errors="coerce")
    trip = trip.dropna(subset=["Time Step (s)"]).copy()
    for column in ["Cycle Speed (m/s)", "Grade", "Temperature (C)"]:
        trip[column] = pd.to_numeric(trip[column], errors="raise")
    return trip.reset_index(drop=True)
