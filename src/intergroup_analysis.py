"""
intergroup_analysis.py — pure numpy/scipy (no geopandas)
"""
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

_LAT_CENTER = -1.47
_DEG_LAT_TO_M = 110574.0
_DEG_LON_TO_M = 111320.0 * np.cos(np.radians(_LAT_CENTER))


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))


def pairwise_distances_by_timestamp(df, group_col="group_id",
                                     lat_col="latitude", lon_col="longitude",
                                     time_col="timestamp"):
    groups = df[group_col].unique()
    records = []
    df = df.copy()
    df["time_bin"] = pd.to_datetime(df[time_col]).dt.floor("10min")
    centroids = (df.groupby([group_col, "time_bin"])
                 .agg(lat=(lat_col,"mean"), lon=(lon_col,"mean"))
                 .reset_index())
    for g1, g2 in combinations(groups, 2):
        s1 = centroids[centroids[group_col]==g1].rename(columns={"lat":"lat1","lon":"lon1"})
        s2 = centroids[centroids[group_col]==g2].rename(columns={"lat":"lat2","lon":"lon2"})
        merged = s1.merge(s2, on="time_bin")
        if len(merged) == 0: continue
        merged["distance_km"] = haversine_vectorized(
            merged["lat1"].values, merged["lon1"].values,
            merged["lat2"].values, merged["lon2"].values)
        merged["group_a"] = g1; merged["group_b"] = g2
        records.append(merged.rename(columns={"time_bin":"timestamp"})
                       [["timestamp","group_a","group_b","distance_km"]])
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def daily_min_intergroup_distance(pairwise_df):
    df = pairwise_df.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return (df.groupby(["date","group_a","group_b"])["distance_km"]
            .agg(min_distance_km="min", mean_distance_km="mean",
                 median_distance_km="median", n_obs="count")
            .reset_index())


def intergroup_distance_matrix(daily_min_df, value_col="min_distance_km"):
    groups = sorted(set(daily_min_df["group_a"]) | set(daily_min_df["group_b"]))
    n = len(groups); idx = {g:i for i,g in enumerate(groups)}
    mat = np.zeros((n,n))
    means = daily_min_df.groupby(["group_a","group_b"])[value_col].mean().reset_index()
    for _, row in means.iterrows():
        i, j = idx[row["group_a"]], idx[row["group_b"]]
        mat[i,j] = row[value_col]; mat[j,i] = row[value_col]
    return pd.DataFrame(mat, index=groups, columns=groups)


def spatial_overlap_bhattacharyya(df, group_a, group_b, group_col="group_id",
                                   lat_col="latitude", lon_col="longitude", grid_n=100):
    def to_m(lats, lons):
        ys = (np.array(lats) - _LAT_CENTER) * _DEG_LAT_TO_M
        xs = (np.array(lons) - np.mean(lons)) * _DEG_LON_TO_M
        return xs, ys
    pts_a = df[df[group_col]==group_a][[lat_col,lon_col]].dropna()
    pts_b = df[df[group_col]==group_b][[lat_col,lon_col]].dropna()
    if len(pts_a) < 10 or len(pts_b) < 10: return np.nan
    xa, ya = to_m(pts_a[lat_col].values, pts_a[lon_col].values)
    xb, yb = to_m(pts_b[lat_col].values, pts_b[lon_col].values)
    xmin = min(xa.min(),xb.min())-1000; xmax = max(xa.max(),xb.max())+1000
    ymin = min(ya.min(),yb.min())-1000; ymax = max(ya.max(),yb.max())+1000
    xi = np.linspace(xmin,xmax,grid_n); yi = np.linspace(ymin,ymax,grid_n)
    Xi, Yi = np.meshgrid(xi, yi); pts_grid = np.vstack([Xi.ravel(), Yi.ravel()])
    kde_a = gaussian_kde(np.vstack([xa,ya]))(pts_grid).reshape(Xi.shape)
    kde_b = gaussian_kde(np.vstack([xb,yb]))(pts_grid).reshape(Xi.shape)
    kde_a /= kde_a.sum(); kde_b /= kde_b.sum()
    return round(float(np.sum(np.sqrt(kde_a * kde_b))), 4)


def overlap_matrix(df, group_col="group_id", lat_col="latitude", lon_col="longitude"):
    groups = sorted(df[group_col].unique()); n = len(groups)
    mat = np.eye(n)
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if j <= i: continue
            bc = spatial_overlap_bhattacharyya(df, g1, g2, group_col, lat_col, lon_col)
            mat[i,j] = bc; mat[j,i] = bc
    return pd.DataFrame(mat, index=groups, columns=groups)


def encounter_frequency(pairwise_df, threshold_km=0.5):
    close = pairwise_df[pairwise_df["distance_km"] < threshold_km].copy()
    if "date" not in close.columns:
        close["date"] = pd.to_datetime(close["timestamp"]).dt.date
    n_days = pd.to_datetime(pairwise_df["timestamp"]).dt.date.nunique()
    counts = (close.groupby(["group_a","group_b"])
              .agg(n_encounters=("distance_km","count")).reset_index())
    counts["rate_per_day"] = (counts["n_encounters"]/n_days).round(4)
    return counts
