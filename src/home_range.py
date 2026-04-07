"""
home_range.py — pure numpy/scipy implementation (no geopandas required)
Estimates home ranges: MCP (95%, 50%) and KDE (95%, 50%) using UTM projection.
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings("ignore")

# Approximate conversion at Virunga latitude (-1.47°S)
_LAT_CENTER = -1.47
_DEG_LAT_TO_M = 110574.0
_DEG_LON_TO_M = 111320.0 * np.cos(np.radians(_LAT_CENTER))


def _to_metres(lats, lons):
    """Convert WGS84 arrays to approximate local metres (flat-earth)."""
    ys = (np.array(lats) - _LAT_CENTER) * _DEG_LAT_TO_M
    xs = (np.array(lons) - np.mean(lons)) * _DEG_LON_TO_M
    return xs, ys


def _convex_hull_area_m2(xs, ys):
    """Convex hull area in m² using scipy."""
    pts = np.column_stack([xs, ys])
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2-D, hull.volume = area
    except Exception:
        return 0.0


def _hull_vertices_wgs84(lats, lons, xs, ys):
    """Return convex hull vertices back in WGS84."""
    pts = np.column_stack([xs, ys])
    try:
        hull = ConvexHull(pts)
        vx = xs[hull.vertices]
        vy = ys[hull.vertices]
        # inverse transform
        lon_center = np.mean(lons)
        hull_lons = vx / _DEG_LON_TO_M + lon_center
        hull_lats = vy / _DEG_LAT_TO_M + _LAT_CENTER
        return hull_lats, hull_lons
    except Exception:
        return lats, lons


def mcp_home_range(lats, lons, percent=95):
    lats = np.array(lats); lons = np.array(lons)
    xs, ys = _to_metres(lats, lons)
    cx, cy = xs.mean(), ys.mean()
    if percent < 100:
        dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        thr = np.percentile(dists, percent)
        mask = dists <= thr
        xs, ys, lats, lons = xs[mask], ys[mask], lats[mask], lons[mask]
    if len(xs) < 3:
        return {"area_km2": np.nan, "hull_lats": None, "hull_lons": None, "n_points": 0}
    area_km2 = _convex_hull_area_m2(xs, ys) / 1e6
    h_lats, h_lons = _hull_vertices_wgs84(lats, lons, xs, ys)
    return {"area_km2": round(area_km2, 3), "hull_lats": h_lats, "hull_lons": h_lons, "n_points": len(xs)}


def kde_home_range(lats, lons, contour_levels=(95, 50), grid_n=150, bandwidth="scott"):
    lats = np.array(lats); lons = np.array(lons)
    xs, ys = _to_metres(lats, lons)
    if len(xs) < 10:
        return {}
    kde = gaussian_kde(np.vstack([xs, ys]), bw_method=bandwidth)
    margin = 1500
    xi = np.linspace(xs.min() - margin, xs.max() + margin, grid_n)
    yi = np.linspace(ys.min() - margin, ys.max() + margin, grid_n)
    # convert back to degrees for plotting
    lon_center = np.mean(lons)
    xi_deg = xi / _DEG_LON_TO_M + lon_center
    yi_deg = yi / _DEG_LAT_TO_M + _LAT_CENTER
    Xi, Yi = np.meshgrid(xi, yi)
    Z = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    cell_area_km2 = ((xi[1]-xi[0]) * (yi[1]-yi[0])) / 1e6
    results = {"kde_grid": Z, "xi": xi_deg, "yi": yi_deg, "cell_area_km2": cell_area_km2}
    for level in contour_levels:
        z_sorted = np.sort(Z.ravel())[::-1]
        cumsum = np.cumsum(z_sorted); cumsum /= cumsum[-1]
        idx = np.searchsorted(cumsum, level / 100.0)
        thr = z_sorted[min(idx, len(z_sorted)-1)]
        area_km2 = np.sum(Z >= thr) * cell_area_km2
        results[f"area_{level}pct_km2"] = round(area_km2, 3)
        results[f"threshold_{level}pct"] = thr
    return results


def home_range_all_groups(df, group_col="group_id", lat_col="latitude", lon_col="longitude"):
    rows = []
    for group, sub in df.groupby(group_col):
        lats = sub[lat_col].values; lons = sub[lon_col].values
        row = {"group_id": group, "n_fixes": len(sub)}
        for pct in [95, 50]:
            res = mcp_home_range(lats, lons, pct)
            row[f"mcp_{pct}_km2"] = res["area_km2"]
        res = kde_home_range(lats, lons)
        row["kde_95_km2"] = res.get("area_95pct_km2", np.nan)
        row["kde_50_km2"] = res.get("area_50pct_km2", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def seasonal_home_range(df, group_col="group_id", season_col="season",
                        lat_col="latitude", lon_col="longitude"):
    rows = []
    for (group, season), sub in df.groupby([group_col, season_col]):
        lats = sub[lat_col].values; lons = sub[lon_col].values
        if len(lats) < 10: continue
        r95 = mcp_home_range(lats, lons, 95)
        r50 = mcp_home_range(lats, lons, 50)
        rows.append({"group_id": group, "season": season,
                     "mcp_95_km2": r95["area_km2"], "mcp_50_km2": r50["area_km2"],
                     "n_fixes": len(lats)})
    return pd.DataFrame(rows)
