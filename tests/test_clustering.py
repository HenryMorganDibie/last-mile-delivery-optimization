import pandas as pd
import numpy as np

from src.data.loader import load_data
from src.features.engineering import compute_depot_distances
from src.models.clustering import run_kmeans_search
from src.features.engineering import scale_coordinates
from src.optimization.optimization import run_mip_optimization
from src.features.engineering import compute_pairwise_distances


# -----------------------------
# Fixtures (sample test data)
# -----------------------------
def create_sample_data():
    # Create 10 rows to ensure we have enough points for clustering
    data = {
        "Delivery_ID": list(range(1, 11)),
        "Latitude": [37.7749, 37.7849, 37.7649, 37.7759, 37.7739, 37.7729, 37.7719, 37.7809, 37.7819, 37.7829],
        "Longitude": [-122.4194, -122.4094, -122.4294, -122.4184, -122.4174, -122.4164, -122.4154, -122.4144, -122.4134, -122.4124],
        "Demand": [10, 20, 15, 30, 10, 5, 25, 12, 18, 22],
        "Depot_Latitude": [37.7739] * 10,
        "Depot_Longitude": [-122.4313] * 10
    }
    return pd.DataFrame(data)


# -----------------------------
# Data Loading Test
# -----------------------------
def test_load_data():
    df = load_data("data/processed/OR_sample_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Latitude" in df.columns
    assert "Longitude" in df.columns
    assert "Demand" in df.columns


# -----------------------------
# Feature Engineering Test
# -----------------------------
def test_compute_distance_to_depot():
    df = create_sample_data()
    df_out = compute_depot_distances(df)

    assert "Distance_to_Depot" in df_out.columns
    assert df_out["Distance_to_Depot"].min() >= 0
    assert df_out["Distance_to_Depot"].notnull().all()


# -----------------------------
# Clustering Test
# -----------------------------
def test_kmeans_clustering():
    df = create_sample_data()
    df = compute_depot_distances(df)
    
    # K-Means needs scaled coordinates
    coords_scaled, _ = scale_coordinates(df)
    
    # Unpack 4 values: df, model, best_k, score
    df_clustered, model, best_k, score = run_kmeans_search(coords_scaled, df)
    
    assert "KMeans_Cluster" in df_clustered.columns


# -----------------------------
# Optimization Test
# -----------------------------
def test_mip_optimization():
    df = create_sample_data()
    df = compute_depot_distances(df)
    
    # 1. Generate the distance matrix required by the function
    dist_matrix = compute_pairwise_distances(df)
    
    # 2. Call with only the two expected arguments
    result_df = run_mip_optimization(df, dist_matrix)
    
    assert "Optimized_Cluster" in result_df.columns