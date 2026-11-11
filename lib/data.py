from pathlib import Path
import glob
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "absa"
GRAPH_DIR = BASE_DIR / "graph"

ASPECTS = [
    "service",
    "cleanliness",
    "value",
    "location",
    "rooms",
    "sleep_quality",
]

def _parquet_files(path: Path):
    """Find Spark part files while ignoring hidden CRC and metadata files."""
    if path.is_file() and path.suffix == ".parquet":
        return [str(path)]
    if not path.exists():
        return []
    return sorted(glob.glob(str(path / "part-*.parquet")) +
                  glob.glob(str(path / "*.parquet")))

@st.cache_data(show_spinner="Loading Gold data...")
def load_parquet_dataset(dataset_name: str) -> pd.DataFrame:
    path = DATA_DIR / dataset_name
    files = _parquet_files(path)
    if not files:
        raise FileNotFoundError(
            f"No Parquet files found for '{dataset_name}' at {path}. "
            "Check the artifact folder structure."
        )
    return pd.concat(
        [pd.read_parquet(f) for f in files],
        ignore_index=True
    )

@st.cache_data(show_spinner="Loading graph data...")
def load_graph_parquet(filename: str) -> pd.DataFrame:
    path = GRAPH_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing graph artifact: {path}")
    return pd.read_parquet(path)

@st.cache_data(show_spinner="Loading community profiles...")
def load_community_profiles() -> pd.DataFrame:
    path = GRAPH_DIR / "community_profiles.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing graph artifact: {path}")
    return pd.read_csv(path, index_col=0)

def format_number(value):
    return f"{value:,.0f}"

def require_columns(df, columns, dataset_name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing expected columns: {', '.join(missing)}. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
