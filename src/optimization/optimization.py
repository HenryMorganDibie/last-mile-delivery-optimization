"""
src/models/optimization.py
Mixed-Integer Programming (MIP) optimization via PuLP.

Objective  : Minimise total travel distance across all cluster assignments.
Constraints:
  1. Each delivery point is assigned to exactly one cluster.
  2. Total demand per cluster ≤ vehicle capacity.
  3. Number of deliveries per cluster ≤ max_deliveries_per_cluster.
"""

import numpy as np
import pandas as pd
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    PULP_CBC_CMD,
    value,
)
from config.settings import (
    MIP_NUM_CLUSTERS,
    MIP_VEHICLE_CAPACITY,
    MIP_MAX_DELIVERIES_PER_CLUSTER,
    MIP_TIME_LIMIT_SECONDS,
)


def run_mip_optimization(
    df: pd.DataFrame,
    distances: np.ndarray,
) -> pd.DataFrame:
    """
    Solve the delivery-to-cluster assignment problem with PuLP.

    Parameters
    ----------
    df        : pd.DataFrame — must contain Demand column; index reset to 0…n-1
    distances : np.ndarray of shape (n, n) — pairwise geodesic distances

    Returns
    -------
    pd.DataFrame with Optimized_Cluster column added.
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    k = MIP_NUM_CLUSTERS

    print(f"[MIP] Setting up problem: {n} deliveries → {k} clusters …")

    # ── Decision variables ────────────────────────────────────────────────
    assign = LpVariable.dicts(
        "Assign",
        [(i, j) for i in range(n) for j in range(k)],
        cat="Binary",
    )

    # ── Problem ───────────────────────────────────────────────────────────
    prob = LpProblem("Minimise_Travel_Distance", LpMinimize)

    # Objective: minimise sum of distances for chosen assignments
    prob += lpSum(
        assign[i, j] * distances[i][j]
        for i in range(n)
        for j in range(k)
    )

    # Constraint 1: each delivery point in exactly one cluster
    for i in range(n):
        prob += lpSum(assign[i, j] for j in range(k)) == 1

    # Constraint 2: demand capacity per cluster
    for j in range(k):
        prob += (
            lpSum(assign[i, j] * df.loc[i, "Demand"] for i in range(n))
            <= MIP_VEHICLE_CAPACITY
        )

    # Constraint 3: max deliveries per cluster (balance)
    for j in range(k):
        prob += (
            lpSum(assign[i, j] for i in range(n))
            <= MIP_MAX_DELIVERIES_PER_CLUSTER
        )

    # ── Solve ─────────────────────────────────────────────────────────────
    print(f"[MIP] Solving (time limit={MIP_TIME_LIMIT_SECONDS}s) …")
    prob.solve(PULP_CBC_CMD(timeLimit=MIP_TIME_LIMIT_SECONDS, msg=0))

    # ── Extract assignments ───────────────────────────────────────────────
    optimized_clusters = {}
    for i in range(n):
        for j in range(k):
            if assign[i, j].value() == 1:
                optimized_clusters[i] = j

    df["Optimized_Cluster"] = df.index.map(optimized_clusters)

    unassigned = df["Optimized_Cluster"].isna().sum()
    obj_value = value(prob.objective)
    print(f"[MIP] Objective value : {obj_value:,.1f} m")
    print(f"[MIP] Unassigned      : {unassigned}")

    return df


def mip_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate demand and delivery count by Optimized_Cluster.
    """
    summary = (
        df.groupby("Optimized_Cluster")
        .agg(
            num_deliveries=("Delivery_ID", "count"),
            total_demand=("Demand", "sum"),
        )
        .reset_index()
    )
    print("\n[MIP] Cluster Summary:")
    print(summary.to_string(index=False))
    return summary