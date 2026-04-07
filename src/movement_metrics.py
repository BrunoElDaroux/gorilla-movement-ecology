"""
movement_metrics.py
===================
Functions for computing movement metrics from GPS tracking data:
  - Haversine distances between consecutive fixes
  - Daily path lengths
  - Step lengths and turning angles
  - Movement speed
  - Net squared displacement
  - First Passage Time (FPT)
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2, degrees


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance (km) between two WGS84 coordinate pairs.

    Parameters
    ----------
    lat1, lon1 : float  origin coordinates (decimal degrees)
    lat2, lon2 : float  destination coordinates (decimal degrees)

    Returns
    -------
    float : distance in kilometres
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlam / 2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance for numpy arrays. Returns km array."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Compute initial bearing (degrees from North) between two points.
    Returns angle in [0, 360).
    """
    phi1, phi2 = radians(lat1), radians(lat2)
    dlam = radians(lon2 - lon1)
    x = sin(dlam) * cos(phi2)
    y = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlam)
    bearing = (degrees(atan2(x, y)) + 360) % 360
    return bearing


def add_step_metrics(df, group_col="group_id", lat_col="latitude",
                     lon_col="longitude", time_col="timestamp"):
    """
    Add step-level movement metrics to GPS DataFrame.
    Computes per-row: step_length_km, bearing_deg, turning_angle_deg, speed_kmh.

    Parameters
    ----------
    df : pd.DataFrame  sorted by group and timestamp

    Returns
    -------
    pd.DataFrame with added columns
    """
    df = df.copy().sort_values([group_col, time_col])

    # Shift within group
    df["_lat_prev"] = df.groupby(group_col)[lat_col].shift(1)
    df["_lon_prev"] = df.groupby(group_col)[lon_col].shift(1)
    df["_time_prev"] = df.groupby(group_col)[time_col].shift(1)

    # Step length (km)
    mask = df["_lat_prev"].notna()
    df.loc[mask, "step_length_km"] = haversine_vectorized(
        df.loc[mask, "_lat_prev"].values,
        df.loc[mask, "_lon_prev"].values,
        df.loc[mask, lat_col].values,
        df.loc[mask, lon_col].values,
    )
    df["step_length_m"] = df["step_length_km"] * 1000

    # Time elapsed (hours)
    df["time_elapsed_hr"] = (df[time_col] - df["_time_prev"]).dt.total_seconds() / 3600

    # Speed (km/h)
    df["speed_kmh"] = df["step_length_km"] / df["time_elapsed_hr"].replace(0, np.nan)

    # Bearing
    bearing_vals = []
    for _, row in df[mask].iterrows():
        bearing_vals.append(compute_bearing(
            row["_lat_prev"], row["_lon_prev"],
            row[lat_col], row[lon_col]
        ))
    df.loc[mask, "bearing_deg"] = bearing_vals

    # Turning angle
    df["_bearing_prev"] = df.groupby(group_col)["bearing_deg"].shift(1)
    df["turning_angle_deg"] = (df["bearing_deg"] - df["_bearing_prev"] + 180) % 360 - 180

    # Drop helper columns
    df.drop(columns=["_lat_prev", "_lon_prev", "_time_prev", "_bearing_prev"],
            inplace=True, errors="ignore")
    return df


def daily_path_length(df, group_col="group_id", date_col="date",
                      step_col="step_length_km"):
    """
    Compute daily path length (km) per group per day.

    Returns
    -------
    pd.DataFrame with columns: [group_id, date, daily_path_km, n_fixes]
    """
    daily = (
        df.groupby([group_col, date_col])
        .agg(
            daily_path_km=(step_col, "sum"),
            n_fixes=(step_col, "count"),
        )
        .reset_index()
    )
    return daily


def net_squared_displacement(df, group_col="group_id", date_col="date",
                             lat_col="latitude", lon_col="longitude",
                             time_col="timestamp"):
    """
    Compute net squared displacement (NSD) from first fix of each day.
    NSD is useful for detecting migration vs residency patterns.

    Returns
    -------
    pd.DataFrame with NSD in km² per fix
    """
    df = df.copy().sort_values([group_col, time_col])

    # First fix per group per day = nest site
    first_fixes = (
        df.groupby([group_col, date_col])
        .first()
        .reset_index()[[group_col, date_col, lat_col, lon_col]]
        .rename(columns={lat_col: "nest_lat", lon_col: "nest_lon"})
    )

    df = df.merge(first_fixes, on=[group_col, date_col])
    df["nsd_km2"] = haversine_vectorized(
        df["nest_lat"].values, df["nest_lon"].values,
        df[lat_col].values, df[lon_col].values,
    ) ** 2

    df.drop(columns=["nest_lat", "nest_lon"], inplace=True)
    return df


def nest_to_nest_distance(df, group_col="group_id", date_col="date",
                          lat_col="latitude", lon_col="longitude",
                          nest_col="nest_site"):
    """
    Compute distance from today's nest site to yesterday's nest site.
    Reveals how far the group moves between sleeping sites.

    Returns
    -------
    pd.DataFrame: [group_id, date, nest_lat, nest_lon, nest_to_nest_km]
    """
    nests = (
        df[df[nest_col] == True]
        .groupby([group_col, date_col])
        .agg(nest_lat=(lat_col, "first"), nest_lon=(lon_col, "first"))
        .reset_index()
        .sort_values([group_col, date_col])
    )

    nests["prev_lat"] = nests.groupby(group_col)["nest_lat"].shift(1)
    nests["prev_lon"] = nests.groupby(group_col)["nest_lon"].shift(1)

    mask = nests["prev_lat"].notna()
    nests.loc[mask, "nest_to_nest_km"] = haversine_vectorized(
        nests.loc[mask, "prev_lat"].values,
        nests.loc[mask, "prev_lon"].values,
        nests.loc[mask, "nest_lat"].values,
        nests.loc[mask, "nest_lon"].values,
    )

    nests.drop(columns=["prev_lat", "prev_lon"], inplace=True)
    return nests


def movement_summary(daily_df, group_col="group_id", path_col="daily_path_km"):
    """
    Compute summary statistics of daily movement per group.

    Returns
    -------
    pd.DataFrame: summary statistics per group
    """
    summary = (
        daily_df.groupby(group_col)[path_col]
        .agg(
            mean_km="mean",
            median_km="median",
            std_km="std",
            min_km="min",
            max_km="max",
            cv=lambda x: x.std() / x.mean(),
            n_days="count",
        )
        .round(4)
        .reset_index()
    )
    return summary
