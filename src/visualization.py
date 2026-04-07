"""
visualization.py
================
Plotting utilities for gorilla movement ecology analysis.
All functions return matplotlib Figure objects that can be saved or displayed.

Color palette matches conservation-friendly earth tones.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ── Consistent color palette per group ───────────────────────────────────────
GROUP_COLORS = {
    "Susa":     "#2E86AB",   # ocean blue
    "Hirwa":    "#A23B72",   # deep purple
    "Amahoro":  "#F18F01",   # amber
    "Umubano":  "#C73E1D",   # brick red
    "Pablo":    "#3B1F2B",   # dark plum
    "Kwitonda": "#44BBA4",   # teal
}

SEASON_COLORS = {
    "dry_season":   "#E8A838",
    "long_rains":   "#5B9BD5",
    "short_rains":  "#70AD47",
}

FIGSIZE_DEFAULT  = (10, 6)
FIGSIZE_WIDE     = (14, 6)
FIGSIZE_SQUARE   = (8, 8)
FIGSIZE_TALL     = (8, 12)


def plot_movement_trajectories(df, groups=None, figsize=(12, 9),
                                date_range=None, alpha=0.6,
                                show_nest_sites=True):
    """
    Plot GPS movement trajectories for one or more gorilla groups.

    Parameters
    ----------
    df         : GPS DataFrame
    groups     : list of group names (None = all)
    date_range : tuple (start_date, end_date) strings or None
    """
    if groups is None:
        groups = sorted(df["group_id"].unique())

    if date_range is not None:
        df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]

    fig, ax = plt.subplots(figsize=figsize)

    for group in groups:
        sub = df[df["group_id"] == group].sort_values("timestamp")
        color = GROUP_COLORS.get(group, "#333333")

        # Plot trajectory
        ax.plot(sub["longitude"], sub["latitude"],
                color=color, alpha=alpha, linewidth=0.6, zorder=2)

        # Scatter dots for individual fixes
        ax.scatter(sub["longitude"], sub["latitude"],
                   color=color, alpha=0.2, s=1, zorder=1)

        # Nest sites (morning fixes)
        if show_nest_sites:
            nests = sub[sub["nest_site"] == True]
            ax.scatter(nests["longitude"], nests["latitude"],
                       color=color, marker="^", s=25, zorder=4,
                       edgecolors="white", linewidths=0.4, alpha=0.9)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=GROUP_COLORS.get(g, "#333"), lw=2, label=g)
        for g in groups
    ]
    if show_nest_sites:
        legend_elements.append(
            Line2D([0], [0], marker="^", color="gray", lw=0,
                   markersize=8, label="Nest sites")
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9)

    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°S)", fontsize=11)
    title_str = "Mountain Gorilla Movement Trajectories — Virunga Massif"
    if date_range:
        title_str += f"\n{date_range[0]} to {date_range[1]}"
    ax.set_title(title_str, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_daily_distances(daily_df, figsize=FIGSIZE_WIDE):
    """
    Violin + box plot of daily path lengths by group.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    groups = sorted(daily_df["group_id"].unique())
    colors = [GROUP_COLORS.get(g, "#555") for g in groups]

    # Violin plot
    ax = axes[0]
    data_by_group = [daily_df[daily_df["group_id"] == g]["daily_path_km"].dropna()
                     for g in groups]
    vp = ax.violinplot(data_by_group, positions=range(len(groups)),
                       showmedians=True, showextrema=True)
    for i, (pc, col) in enumerate(zip(vp["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Daily Path Length (km)", fontsize=11)
    ax.set_title("Distribution of Daily Movement Distances", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Time series (smoothed monthly mean)
    ax = axes[1]
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["month"] = daily_df["date"].dt.to_period("M")
    monthly = (
        daily_df.groupby(["group_id", "month"])["daily_path_km"]
        .mean().reset_index()
    )
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    for group in groups:
        sub = monthly[monthly["group_id"] == group]
        ax.plot(sub["month_dt"], sub["daily_path_km"],
                label=group, color=GROUP_COLORS.get(group, "#555"),
                linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Mean Daily Path Length (km)", fontsize=11)
    ax.set_title("Monthly Mean Daily Movement", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_home_range_comparison(hr_df, figsize=(10, 5)):
    """
    Grouped bar chart comparing MCP and KDE home range sizes.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    groups = hr_df["group_id"].tolist()
    x = np.arange(len(groups))
    colors = [GROUP_COLORS.get(g, "#555") for g in groups]

    for ax, (col95, col50, title) in zip(
        axes,
        [("mcp_95_km2", "mcp_50_km2", "MCP Home Range"),
         ("kde_95_km2", "kde_50_km2", "KDE Home Range")]
    ):
        if col95 not in hr_df.columns:
            continue
        bars95 = ax.bar(x - 0.2, hr_df[col95], 0.35, label="95%",
                        color=colors, alpha=0.85, edgecolor="white")
        bars50 = ax.bar(x + 0.2, hr_df[col50], 0.35, label="50% (core)",
                        color=colors, alpha=0.45, edgecolor=colors, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Area (km²)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Home Range Estimates — All Groups", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_intergroup_distance_heatmap(daily_min_df, figsize=(9, 7)):
    """
    Heatmap of mean daily minimum distances between all group pairs.
    """
    from src.intergroup_analysis import intergroup_distance_matrix
    mat = intergroup_distance_matrix(daily_min_df)

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.eye(len(mat), dtype=bool)
    sns.heatmap(
        mat, annot=True, fmt=".2f", mask=mask,
        cmap="YlOrRd_r", linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Mean Min. Distance (km)"},
        vmin=0, vmax=mat.values[~mask].max()
    )
    ax.set_title("Inter-Group Spatial Separation Matrix\n(Mean Daily Minimum Distance, km)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_intergroup_distance_timeseries(daily_min_df, pairs=None, figsize=(12, 8)):
    """
    Time series of daily minimum inter-group distances.
    """
    daily_min_df = daily_min_df.copy()
    daily_min_df["date"] = pd.to_datetime(daily_min_df["date"])

    all_pairs = daily_min_df[["group_a", "group_b"]].drop_duplicates()
    if pairs is None:
        pairs = [(r["group_a"], r["group_b"]) for _, r in all_pairs.iterrows()]

    n_cols = 2
    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for i, (ga, gb) in enumerate(pairs):
        sub = daily_min_df[
            (daily_min_df["group_a"] == ga) & (daily_min_df["group_b"] == gb)
        ].sort_values("date")
        ax = axes[i]
        ax.fill_between(sub["date"], sub["min_distance_km"],
                        alpha=0.3, color="#2E86AB")
        ax.plot(sub["date"], sub["min_distance_km"],
                color="#2E86AB", linewidth=0.8)
        ax.axhline(sub["min_distance_km"].mean(), color="red",
                   linestyle="--", linewidth=1.2, label=f"Mean={sub['min_distance_km'].mean():.2f}km")
        ax.set_title(f"{ga} — {gb}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Min. Distance (km)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Daily Minimum Inter-Group Distances Over Time",
                 fontsize=13, fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_hypothesis_test_results(observed, null_distribution,
                                  p_value, effect_size, figsize=(10, 5)):
    """
    Plot permutation test results: null distribution vs observed value.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Null distribution histogram with observed value
    ax = axes[0]
    ax.hist(null_distribution, bins=60, color="#5B9BD5", alpha=0.75,
            edgecolor="white", linewidth=0.4, label="Null distribution\n(random)")
    ax.axvline(observed, color="#C73E1D", linewidth=2.5,
               linestyle="--", label=f"Observed = {observed:.3f} km")
    ax.axvline(np.percentile(null_distribution, 95), color="#E8A838",
               linewidth=1.5, linestyle=":", label="95th percentile (null)")
    ax.set_xlabel("Mean Inter-Group Distance (km)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Permutation Test\np = {p_value:.4f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Effect size context
    ax = axes[1]
    categories = ["Null Mean", "Observed"]
    values = [np.mean(null_distribution), observed]
    colors_bar = ["#5B9BD5", "#C73E1D"]
    bars = ax.bar(categories, values, color=colors_bar, alpha=0.85,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{val:.3f} km", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Inter-Group Distance (km)", fontsize=11)
    ax.set_title(f"Effect Size (Cohen's d = {effect_size:.3f})",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Statistical Test: Do Gorilla Groups Maintain Spatial Separation?",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_nest_predictability(nest_df, figsize=(10, 5)):
    """
    Plot nest-to-nest movement distances (predictability of nest site location).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    groups = sorted(nest_df["group_id"].unique())
    colors = [GROUP_COLORS.get(g, "#555") for g in groups]

    # CDF of nest-to-nest distances
    ax = axes[0]
    for group, color in zip(groups, colors):
        sub = nest_df[nest_df["group_id"] == group]["nest_to_nest_km"].dropna()
        sub_sorted = np.sort(sub)
        cdf = np.arange(1, len(sub_sorted)+1) / len(sub_sorted)
        ax.plot(sub_sorted, cdf, label=group, color=color, linewidth=2)

    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
               label="500m threshold")
    ax.set_xlabel("Nest-to-Next-Day Distance (km)", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_title("CDF of Nest-to-Nest Distances", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, ax.get_xlim()[1])

    # Bar chart of mean nest predictability
    ax = axes[1]
    means = nest_df.groupby("group_id")["nest_to_nest_km"].mean().reindex(groups)
    stds  = nest_df.groupby("group_id")["nest_to_nest_km"].std().reindex(groups)
    ax.bar(groups, means.values, color=colors, alpha=0.85, edgecolor="white")
    ax.errorbar(groups, means.values, yerr=stds.values,
                fmt="none", color="black", capsize=4, linewidth=1.5)
    ax.set_ylabel("Mean Nest-to-Nest Distance (km)", fontsize=11)
    ax.set_title("Nest Site Predictability by Group", fontsize=12, fontweight="bold")
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Nest Site Predictability — Conservation Monitoring Implications",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
