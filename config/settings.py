"""
config/settings.py
Centralized configuration for the neighbourhood clustering pipeline.
"""

from pathlib import Path
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MAPS_DIR = OUTPUTS_DIR / "maps"

RAW_DATA_FILE = os.path.join(DATA_DIR, "processed", "OR_sample_data.csv")

# ── Expected schema ────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Delivery_ID",
    "Latitude",
    "Longitude",
    "Demand",
    "Depot_Latitude",
    "Depot_Longitude",
]

# ── Feature engineering ────────────────────────────────────────────────────
COORDINATE_FEATURES = ["Latitude", "Longitude"]

# ── DBSCAN ────────────────────────────────────────────────────────────────
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5

# ── K-Means ───────────────────────────────────────────────────────────────
KMEANS_CLUSTER_RANGE = range(3, 7)   # Try k = 3, 4, 5, 6
KMEANS_RANDOM_STATE = 42
KMEANS_INIT = "k-means++"

# ── MIP Optimization ──────────────────────────────────────────────────────
MIP_NUM_CLUSTERS = 4
MIP_VEHICLE_CAPACITY = 400           # Max demand units per cluster
MIP_MAX_DELIVERIES_PER_CLUSTER = 60  # Soft balance constraint
MIP_TIME_LIMIT_SECONDS = 300

# ── Visualization ─────────────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_SIZE_DEFAULT = (10, 6)
MAP_ZOOM_START = 14