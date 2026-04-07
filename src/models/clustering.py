"""
src/models/clustering.py
DBSCAN and K-Means clustering with silhouette-based model selection.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from config.settings import (
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    KMEANS_CLUSTER_RANGE,
    KMEANS_RANDOM_STATE,
    KMEANS_INIT,
)


# ── DBSCAN ────────────────────────────────────────────────────────────────

def run_dbscan(coords_scaled: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit DBSCAN and attach cluster labels to the dataframe.

    Points classified as noise receive label -1.

    Parameters
    ----------
    coords_scaled : np.ndarray  — standardised coordinates
    df            : pd.DataFrame — original dataframe (not mutated)

    Returns
    -------
    pd.DataFrame with a DBSCAN_Cluster column added.
    """
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = dbscan.fit_predict(coords_scaled)
    df = df.copy()
    df["DBSCAN_Cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"[DBSCAN] clusters={n_clusters}  noise points={n_noise}")

    if n_clusters > 1:
        score = silhouette_score(coords_scaled, labels)
        print(f"[DBSCAN] Silhouette Score: {score:.4f}")
    else:
        print("[DBSCAN] All points are noise — skipping silhouette score.")

    return df, dbscan


# ── K-Means ───────────────────────────────────────────────────────────────

def run_kmeans_search(
    coords_scaled: np.ndarray,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, KMeans, int, float]:
    """
    Run K-Means for each k in KMEANS_CLUSTER_RANGE, select the best k by
    silhouette score, and attach the final labels to the dataframe.

    Returns
    -------
    df          : pd.DataFrame with KMeans_Cluster column
    best_model  : fitted KMeans with best k
    best_k      : int
    best_score  : float
    """
    results = {}
    print("[KMeans] Searching over cluster counts …")

    for k in KMEANS_CLUSTER_RANGE:
        model = KMeans(
            n_clusters=k,
            init=KMEANS_INIT,
            random_state=KMEANS_RANDOM_STATE,
        )
        labels = model.fit_predict(coords_scaled)
        score = silhouette_score(coords_scaled, labels)
        results[k] = (model, labels, score)
        print(f"  k={k}  silhouette={score:.4f}")

    best_k = max(results, key=lambda k: results[k][2])
    best_model, best_labels, best_score = results[best_k]

    df = df.copy()
    df["KMeans_Cluster"] = best_labels

    print(f"[KMeans] Best k={best_k}  silhouette={best_score:.4f}")
    return df, best_model, best_k, best_score


def cluster_summary(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """
    Return a per-cluster summary table with delivery count, total demand,
    and distance statistics.
    """
    summary = (
        df.groupby(cluster_col)
        .agg(
            num_deliveries=("Delivery_ID", "count"),
            total_demand=("Demand", "sum"),
            mean_distance=("Distance_to_Depot", "mean"),
            std_distance=("Distance_to_Depot", "std"),
            min_distance=("Distance_to_Depot", "min"),
            max_distance=("Distance_to_Depot", "max"),
        )
        .reset_index()
    )
    return summary


def predict_cluster(
    latitude: float,
    longitude: float,
    model: KMeans,
    scaler,
) -> int:
    """
    Predict the K-Means cluster for a new delivery location.

    Parameters
    ----------
    latitude, longitude : float
    model               : fitted KMeans
    scaler              : fitted StandardScaler used during training

    Returns
    -------
    int  — cluster label
    """
    point = scaler.transform([[latitude, longitude]])
    return int(model.predict(point)[0])