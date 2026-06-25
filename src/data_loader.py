"""
Data Loader Module
Handles downloading and loading the UCI Online Retail Dataset.
Dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail
"""

import requests
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_FILENAME = "online_retail.xlsx"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
)

FALLBACK_INSTRUCTIONS = """
=============================================================
 DATASET NOT FOUND — Manual Download Required
=============================================================

Please download the UCI Online Retail Dataset manually:

  URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx

  OR visit: https://archive.ics.uci.edu/ml/datasets/Online+Retail

Save the file as:
  {path}

Then re-run the pipeline.
=============================================================
""".format(path=DATASET_PATH)


def download_dataset(force: bool = False) -> bool:
    """
    Attempt to download the UCI Online Retail dataset.
    Returns True if successful, False otherwise.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists() and not force:
        print(f"[DataLoader] Dataset already exists at: {DATASET_PATH}")
        return True

    print(f"[DataLoader] Attempting to download dataset from UCI repository...")
    try:
        response = requests.get(UCI_URL, timeout=60, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(DATASET_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r[DataLoader] Downloading... {pct:.1f}%", end="", flush=True)
        print(f"\n[DataLoader] Download complete: {DATASET_PATH}")
        return True
    except Exception as e:
        print(f"\n[DataLoader] Download failed: {e}")
        print(FALLBACK_INSTRUCTIONS)
        return False


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Online Retail dataset from disk.
    Raises FileNotFoundError with instructions if the file is missing.
    """
    if not DATASET_PATH.exists():
        downloaded = download_dataset()
        if not downloaded:
            raise FileNotFoundError(FALLBACK_INSTRUCTIONS)

    print(f"[DataLoader] Loading dataset from {DATASET_PATH} ...")
    df = pd.read_excel(DATASET_PATH, engine="openpyxl")
    print(f"[DataLoader] Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df


def get_column_info(df: pd.DataFrame) -> None:
    """Print basic column-level information."""
    print("\n[DataLoader] Column Info:")
    print(f"{'Column':<20} {'dtype':<15} {'Non-Null':>10} {'Null%':>8}")
    print("-" * 58)
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        print(f"{col:<20} {str(df[col].dtype):<15} {non_null:>10,} {null_pct:>7.2f}%")
