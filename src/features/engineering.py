"""
src/features/engineering.py
Computes derived features (geodesic distances) and scales coordinates
for use in clustering models.
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from config.settings import COORDINATE_FEATURES


def compute_depot_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Distance_to_Depot column (metres) using geodesic distance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Latitude, Longitude, Depot_Latitude, Depot_Longitude.

    Returns
    -------
    pd.DataFrame
        Original dataframe with Distance_to_Depot added (or overwritten).
    """
    df = df.copy()
    df["Distance_to_Depot"] = df.apply(
        lambda row: geodesic(
            (row["Latitude"], row["Longitude"]),
            (row["Depot_Latitude"], row["Depot_Longitude"]),
        ).meters,
        axis=1,
    )
    print(
        f"[features] Distance_to_Depot: "
        f"min={df['Distance_to_Depot'].min():.1f}m  "
        f"max={df['Distance_to_Depot'].max():.1f}m  "
        f"mean={df['Distance_to_Depot'].mean():.1f}m"
    )
    return df


def compute_pairwise_distances(df: pd.DataFrame) -> np.ndarray:
    """
    Build an (n × n) geodesic distance matrix between all delivery points.
    Used by the MIP optimizer.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Latitude and Longitude.

    Returns
    -------
    np.ndarray of shape (n, n)
    """
    n = len(df)
    print(f"[features] Building {n}×{n} pairwise distance matrix …")
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic(
                (df.loc[i, "Latitude"], df.loc[i, "Longitude"]),
                (df.loc[j, "Latitude"], df.loc[j, "Longitude"]),
            ).meters
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix


def scale_coordinates(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardise Latitude/Longitude for distance-sensitive clustering.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    coords_scaled : np.ndarray of shape (n, 2)
    scaler        : fitted StandardScaler (keep for inverse_transform if needed)
    """
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(df[COORDINATE_FEATURES])
    return coords_scaled, scaler