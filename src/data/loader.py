"""
src/data/loader.py
Handles loading and basic validation of the delivery dataset.
"""

import pandas as pd
from pathlib import Path
from config.settings import REQUIRED_COLUMNS


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load the delivery dataset from a CSV file and validate its schema.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated dataframe ready for downstream processing.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If required columns are missing or null values are detected.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Place OR_sample_data.csv in the data/ directory."
        )

    df = pd.read_csv(filepath)
    _validate_schema(df)
    _check_nulls(df)

    print(f"[loader] Loaded {len(df)} rows × {df.shape[1]} columns from {filepath.name}")
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _check_nulls(df: pd.DataFrame) -> None:
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        raise ValueError(f"Null values detected:\n{nulls}")


def summarize(df: pd.DataFrame) -> None:
    """Print a quick EDA summary of the loaded dataframe."""
    print("\n── Dataset Summary ──────────────────────────")
    print(f"  Shape        : {df.shape}")
    print(f"  Lat range    : {df['Latitude'].min():.5f} → {df['Latitude'].max():.5f}")
    print(f"  Lon range    : {df['Longitude'].min():.5f} → {df['Longitude'].max():.5f}")
    print(f"  Demand stats :")
    print(df["Demand"].describe().to_string(header=False))
    print("─────────────────────────────────────────────\n")