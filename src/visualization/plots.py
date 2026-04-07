"""
src/visualization/plots.py
All static (matplotlib/seaborn) and interactive (folium) visualizations.
"""

import matplotlib
matplotlib.use("Agg")  

import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import FIGURE_DPI, FIGURE_SIZE_DEFAULT, MAP_ZOOM_START


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved → {path}")


# ── EDA plots ─────────────────────────────────────────────────────────────

def plot_delivery_locations(df: pd.DataFrame, save_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["Demand"], cmap="viridis", s=50)
    fig.colorbar(sc, ax=ax, label="Demand")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Delivery Locations (coloured by demand)")
    if save_path:
        _save(fig, save_path)
    else:
        plt.show()


def plot_demand_histogram(df: pd.DataFrame, save_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["Demand"], bins=20, color="skyblue", edgecolor="black")
    ax.set(title="Distribution of Delivery Demand", xlabel="Demand", ylabel="Frequency")
    if save_path:
        _save(fig, save_path)
    else:
        plt.show()


def plot_demand_boxplot(df: pd.DataFrame, save_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df["Demand"], ax=ax)
    ax.set_title("Boxplot of Delivery Demand")
    if save_path:
        _save(fig, save_path)
    else:
        plt.show()


def plot_distance_to_depot(df: pd.DataFrame, save_path: Path | None = None) -> None:
    depot_lon = df["Depot_Longitude"].iloc[0]
    depot_lat = df["Depot_Latitude"].iloc[0]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["Distance_to_Depot"], cmap="coolwarm", s=50)
    ax.scatter(depot_lon, depot_lat, color="red", marker="x", s=100, label="Depot")
    fig.colorbar(sc, ax=ax, label="Distance to Depot (m)")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Delivery Locations & Distance to Depot")
    ax.legend()
    if save_path:
        _save(fig, save_path)
    else:
        plt.show()


def plot_distance_histogram(distances: np.ndarray, save_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(distances, bins=30, color="skyblue", edgecolor="black")
    ax.set(title="Distance of Delivery Points from Depot", xlabel="Distance (m)", ylabel="Frequency")
    if save_path:
        _save(fig, save_path)
    else:
        plt.show()


# ── Clustering plots ──────────────────────────────────────────────────────

def plot_dbscan_clusters(df: pd.DataFrame, save_path: Path | None = None) -> None:
    depot_lon = df["Depot_Longitude"].iloc[0]
    depot_lat = df["Depot_Latitude"].iloc[0]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["DBSCAN_Cluster"], cmap="viridis", s=50)
    ax.scatter(depot_lon, depot_lat, color="red", marker="x", s=100, label="Depot")
    fig.colorbar(sc, ax=ax, label="DBSCAN Cluster")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="DBSCAN Clustering")
    ax.legend()
    if save_path:
        _save(fig, save_path)
    else:
        plt.show()


def plot_kmeans_clusters(
    df: pd.DataFrame,
    best_k: int,
    best_score: float,
    save_path: Path | None = None,
) -> None:
    depot_lon = df["Depot_Longitude"].iloc[0]
    depot_lat = df["Depot_Latitude"].iloc[0]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["KMeans_Cluster"], cmap="plasma", s=50)
    ax.scatter(depot_lon, depot_lat, color="red", marker="x", s=100, label="Depot")
    fig.colorbar(sc, ax=ax, label="K-Means Cluster")
    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title=f"K-Means Clustering (k={best_k}, Silhouette={best_score:.4f})",
    )
    ax.legend()
    if save_path:
        _save(fig, save_path)
    else:
        plt.show()


def plot_optimized_clusters(df: pd.DataFrame, save_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["Optimized_Cluster"], cmap="viridis", s=50)
    fig.colorbar(sc, ax=ax, label="Optimized Cluster")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="MIP-Optimized Delivery Clusters")
    if save_path:
        _save(fig, save_path)
    else:
        plt.show()


def plot_demand_by_cluster(df: pd.DataFrame, save_path: Path | None = None) -> None:
    demand_by_cluster = df.groupby("Optimized_Cluster")["Demand"].sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    demand_by_cluster.plot(kind="bar", color="skyblue", ax=ax)
    ax.set(title="Total Demand by Optimized Cluster", xlabel="Cluster", ylabel="Total Demand")
    ax.tick_params(axis="x", rotation=0)
    if save_path:
        _save(fig, save_path)
    else:
        plt.show()


# ── Interactive map ───────────────────────────────────────────────────────

def build_folium_map(df: pd.DataFrame, save_path: Path) -> folium.Map:
    """
    Build an interactive Folium map with clustered delivery markers.
    Saves to save_path as HTML.
    """
    depot_lat = df["Depot_Latitude"].iloc[0]
    depot_lon = df["Depot_Longitude"].iloc[0]

    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=MAP_ZOOM_START)

    # Depot marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

    # Delivery markers
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in df.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"ID: {row['Delivery_ID']}  Demand: {row['Demand']}  Cluster: {row.get('Optimized_Cluster', 'N/A')}",
        ).add_to(marker_cluster)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(save_path))
    print(f"[viz] Folium map saved → {save_path}")
    return m