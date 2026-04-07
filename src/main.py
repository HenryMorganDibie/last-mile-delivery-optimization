"""
main.py
Neighbourhood Clustering Optimization — pipeline entry point.

Run:
    python main.py
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    RAW_DATA_FILE,
    FIGURES_DIR,
    MAPS_DIR,
)
from src.data.loader import load_data, summarize
from src.features.engineering import (
    compute_depot_distances,
    compute_pairwise_distances,
    scale_coordinates,
)
from src.models.clustering import (
    run_dbscan,
    run_kmeans_search,
    cluster_summary,
)
from src.optimization.optimization import run_mip_optimization, mip_cluster_summary
from src.visualization.plots import (
    plot_delivery_locations,
    plot_demand_histogram,
    plot_demand_boxplot,
    plot_distance_to_depot,
    plot_dbscan_clusters,
    plot_kmeans_clusters,
    plot_optimized_clusters,
    plot_demand_by_cluster,
    build_folium_map,
)


def main() -> None:
    print("\n══ Neighbourhood Clustering Optimization ══\n")

    # ── 1. Load & validate data ───────────────────────────────────────────
    df = load_data(RAW_DATA_FILE)
    summarize(df)

    # ── 2. Feature engineering ────────────────────────────────────────────
    df = compute_depot_distances(df)
    coords_scaled, scaler = scale_coordinates(df)

    # ── 3. EDA plots ──────────────────────────────────────────────────────
    plot_delivery_locations(df, save_path=FIGURES_DIR / "01_delivery_locations.png")
    plot_demand_histogram(df, save_path=FIGURES_DIR / "02_demand_histogram.png")
    plot_demand_boxplot(df, save_path=FIGURES_DIR / "03_demand_boxplot.png")
    plot_distance_to_depot(df, save_path=FIGURES_DIR / "04_distance_to_depot.png")

    # ── 4. DBSCAN ────────────────────────────────────────────────────────
    df, dbscan_model = run_dbscan(coords_scaled, df)
    plot_dbscan_clusters(df, save_path=FIGURES_DIR / "05_dbscan_clusters.png")

    # ── 5. K-Means (with silhouette search) ──────────────────────────────
    df, kmeans_model, best_k, best_score = run_kmeans_search(coords_scaled, df)
    plot_kmeans_clusters(
        df,
        best_k=best_k,
        best_score=best_score,
        save_path=FIGURES_DIR / "06_kmeans_clusters.png",
    )

    kmeans_summary = cluster_summary(df, "KMeans_Cluster")
    print("\n[KMeans] Cluster Summary:")
    print(kmeans_summary.to_string(index=False))

    # ── 6. MIP Optimization ───────────────────────────────────────────────
    pairwise_dist = compute_pairwise_distances(df)
    df = run_mip_optimization(df, pairwise_dist)
    mip_cluster_summary(df)

    plot_optimized_clusters(df, save_path=FIGURES_DIR / "07_optimized_clusters.png")
    plot_demand_by_cluster(df, save_path=FIGURES_DIR / "08_demand_by_cluster.png")

    # ── 7. Interactive map ────────────────────────────────────────────────
    build_folium_map(df, save_path=MAPS_DIR / "cluster_map.html")

    # ── 8. Save final dataframe ───────────────────────────────────────────
    output_csv = Path("outputs") / "delivery_clusters.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[main] Final dataframe saved → {output_csv}")

    print("\n══ Pipeline complete ══\n")


if __name__ == "__main__":
    main()