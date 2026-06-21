"""
data_generator.py
=================
Generates synthetic mountain gorilla GPS tracking data that simulates
SMART Conservation Software exports from the Virunga Massif.

Usage:
    python src/data_generator.py

Output:
    data/raw/gorilla_gps_data.csv   (~216,000 rows)

Biological parameters are based on published literature:
  - Robbins et al. (2009) Am. J. Primatol.
  - Grueter et al. (2013) Behav. Ecol. Sociobiol.
  - Doran-Sheehy et al. (2007) Am. J. Primatol.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)

# ── Study parameters ─────────────────────────────────────────────────────────
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2022, 12, 31)
GPS_INTERVAL_MIN = 10           # GPS fix every 10 minutes
FOLLOW_START_HOUR = 6           # 06:00 local time
FOLLOW_END_HOUR   = 14          # 14:00 local time
MISSING_RATE      = 0.02        # 2% random missing data

# Approximate degrees per kilometer at Virunga latitude (-1.47°S)
DEG_PER_KM_LAT = 1.0 / 110.574
DEG_PER_KM_LON = 1.0 / 111.320

# ── Group definitions ─────────────────────────────────────────────────────────
# Real Virunga mountain gorilla groups used as templates
# Home range centers based on known territories in the Virunga Massif
GROUPS = {
    "Susa": {
        "center_lat": -1.435, "center_lon": 29.522,
        "group_size": 28, "silverbacks": 2,
        "range_km": 5.5,   # home range radius (km)
        "mean_daily_km": 1.45,
        "altitude_base": 3200,
    },
    "Hirwa": {
        "center_lat": -1.461, "center_lon": 29.538,
        "group_size": 13, "silverbacks": 1,
        "range_km": 4.2,
        "mean_daily_km": 1.20,
        "altitude_base": 2800,
    },
    "Amahoro": {
        "center_lat": -1.478, "center_lon": 29.558,
        "group_size": 17, "silverbacks": 1,
        "range_km": 4.8,
        "mean_daily_km": 1.35,
        "altitude_base": 3000,
    },
    "Umubano": {
        "center_lat": -1.452, "center_lon": 29.571,
        "group_size": 12, "silverbacks": 1,
        "range_km": 4.0,
        "mean_daily_km": 1.15,
        "altitude_base": 2900,
    },
    "Pablo": {
        "center_lat": -1.490, "center_lon": 29.542,
        "group_size": 24, "silverbacks": 3,
        "range_km": 6.0,
        "mean_daily_km": 1.55,
        "altitude_base": 2600,
    },
    "Kwitonda": {
        "center_lat": -1.505, "center_lon": 29.525,
        "group_size": 18, "silverbacks": 2,
        "range_km": 5.0,
        "mean_daily_km": 1.30,
        "altitude_base": 2700,
    },
}

# Habitat types by altitude
HABITATS = {
    "bamboo_zone":     (2200, 2800),
    "montane_forest":  (2800, 3500),
    "mixed_vegetation":(3500, 3800),
    "subalpine_zone":  (3800, 4500),
}

# Rwanda seasons (approximate)
def get_season(month):
    if month in [6, 7, 8]:        return "dry_season"
    elif month in [3, 4, 5]:      return "long_rains"
    elif month in [10, 11]:       return "short_rains"
    else:                         return "dry_season"


def haversine_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in km between two WGS84 points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def correlated_random_walk(n_steps, step_mean_km, step_sd_km,
                           autocorr=0.6, center_pull=0.05,
                           center_lat=0, center_lon=0,
                           current_lat=0, current_lon=0):
    """
    Generate a correlated random walk (CRW) simulating gorilla movement.
    - autocorr:     direction autocorrelation (persistence)
    - center_pull:  attraction to home range center (site fidelity)
    Returns arrays of delta_lat, delta_lon for each step.
    """
    lats = [current_lat]
    lons = [current_lon]
    angle = np.random.uniform(0, 2 * np.pi)  # initial direction

    for _ in range(n_steps - 1):
        # Step length (gamma-distributed for realistic movement)
        step_km = np.random.gamma(
            shape=(step_mean_km / step_sd_km)**2,
            scale=step_sd_km**2 / step_mean_km
        )
        step_km = np.clip(step_km, 0, step_mean_km * 3)

        # Turn angle with persistence (wrapped Cauchy-like)
        turn = np.random.vonmises(0, autocorr * 3)
        angle = (angle + turn) % (2 * np.pi)

        # Center-pull vector (home range fidelity)
        dlat_center = center_lat - lats[-1]
        dlon_center = center_lon - lons[-1]
        pull_angle = np.arctan2(dlat_center, dlon_center)
        dist_from_center = np.sqrt(dlat_center**2 + dlon_center**2)
        pull_strength = center_pull * dist_from_center * 50

        angle = angle + pull_strength * np.sin(pull_angle - angle)

        # Convert to lat/lon degrees
        dlat = step_km * np.sin(angle) * DEG_PER_KM_LAT
        dlon = step_km * np.cos(angle) * DEG_PER_KM_LON

        lats.append(lats[-1] + dlat)
        lons.append(lons[-1] + dlon)

    return np.array(lats), np.array(lons)


def generate_daily_track(group_name, group_params, date, nest_lat, nest_lon):
    """
    Generate one day's GPS track for a gorilla group.
    Returns a DataFrame of GPS fixes for that day.
    """
    # Build timestamps for follow period
    start_dt = datetime(date.year, date.month, date.day, FOLLOW_START_HOUR, 0)
    end_dt   = datetime(date.year, date.month, date.day, FOLLOW_END_HOUR,   0)
    timestamps = []
    t = start_dt
    while t <= end_dt:
        timestamps.append(t)
        t += timedelta(minutes=GPS_INTERVAL_MIN)

    n_fixes = len(timestamps)

    # Steps per fix (10-min intervals)
    # Mean step size between consecutive 10-min fixes
    daily_km  = group_params["mean_daily_km"]
    n_active_hours = FOLLOW_END_HOUR - FOLLOW_START_HOUR
    step_mean = daily_km / (n_active_hours * 6)  # 6 fixes per hour
    step_sd   = step_mean * 0.8

    # Generate CRW
    lats, lons = correlated_random_walk(
        n_steps       = n_fixes,
        step_mean_km  = step_mean,
        step_sd_km    = step_sd,
        autocorr      = 0.65,
        center_pull   = 0.08,
        center_lat    = group_params["center_lat"],
        center_lon    = group_params["center_lon"],
        current_lat   = nest_lat,
        current_lon   = nest_lon,
    )

    # Altitude (correlated with latitude position in Virunga)
    base_alt = group_params["altitude_base"]
    altitudes = base_alt + np.random.normal(0, 80, n_fixes)
    altitudes = altitudes + (lats - group_params["center_lat"]) * 8000  # terrain effect
    altitudes = np.clip(altitudes, 2200, 4500)

    # Habitat type by altitude
    def alt_to_habitat(alt):
        if alt < 2800:   return "bamboo_zone"
        elif alt < 3500: return "montane_forest"
        elif alt < 3800: return "mixed_vegetation"
        else:            return "subalpine_zone"

    # Random observer
    observer = f"OBS_{np.random.randint(1, 8):02d}"

    # Apply random missing data
    mask = np.random.random(n_fixes) > MISSING_RATE

    records = []
    for i, (ts, lat, lon, alt) in enumerate(zip(timestamps, lats, lons, altitudes)):
        if not mask[i]:
            continue
        records.append({
            "timestamp":    ts,
            "group_id":     group_name,
            "latitude":     round(lat, 7),
            "longitude":    round(lon, 7),
            "altitude_m":   round(alt, 1),
            "observer_id":  observer,
            "group_size":   group_params["group_size"],
            "silverbacks":  group_params["silverbacks"],
            "nest_site":    (i == 0),
            "date":         date.date(),
            "day_of_year":  date.timetuple().tm_yday,
            "hour":         ts.hour,
            "habitat_type": alt_to_habitat(alt),
            "season":       get_season(date.month),
        })

    return pd.DataFrame(records)


def generate_gorilla_dataset():
    """
    Generate the full GPS dataset for all groups over the study period.
    Returns a DataFrame with all GPS fixes.
    """
    all_records = []
    date_range = pd.date_range(START_DATE, END_DATE, freq="D")

    print(f"Generating GPS data for {len(GROUPS)} groups over {len(date_range)} days...")

    for group_name, group_params in GROUPS.items():
        print(f"  Processing group: {group_name} ...", end=" ", flush=True)

        # Initialize nest site at group home range center
        nest_lat = group_params["center_lat"] + np.random.normal(0, 0.005)
        nest_lon = group_params["center_lon"] + np.random.normal(0, 0.005)

        for date in date_range:
            # Generate daily track
            daily_df = generate_daily_track(group_name, group_params, date, nest_lat, nest_lon)

            if len(daily_df) > 0:
                all_records.append(daily_df)
                # Next nest site = end of today's track (last fix)
                nest_lat = float(daily_df.iloc[-1]["latitude"])
                nest_lon = float(daily_df.iloc[-1]["longitude"])

                # Add small random overnight movement (nest building)
                nest_lat += np.random.normal(0, 0.0003)
                nest_lon += np.random.normal(0, 0.0003)

        print("done")

    df = pd.concat(all_records, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["group_id", "timestamp"]).reset_index(drop=True)
    return df


def main():
    # Determine output path (works whether called from project root or src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_path = os.path.join(project_root, "data", "raw", "gorilla_gps_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = generate_gorilla_dataset()
    df.to_csv(out_path, index=False)

    print(f"\n✅ Dataset saved to: {out_path}")
    print(f"   Total rows:   {len(df):,}")
    print(f"   Date range:   {df['date'].min()} to {df['date'].max()}")
    print(f"   Groups:       {df['group_id'].nunique()}")
    print(f"   Columns:      {list(df.columns)}")


if __name__ == "__main__":
    main()
