"""
Data Preprocessing Module
Cleans and prepares the raw Online Retail dataset for analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the UCI Online Retail dataset.

    Steps:
      1. Standardise column names
      2. Drop missing CustomerIDs (anonymous transactions)
      3. Remove cancelled orders (InvoiceNo starting with 'C')
      4. Remove negative/zero Quantity and UnitPrice
      5. Remove known test / adjustment SKUs
      6. Parse and validate InvoiceDate
      7. Add derived columns: TotalPrice, YearMonth, DayOfWeek, Hour
    """
    print("[Preprocessing] Starting data cleaning...")
    original_len = len(df)

    # 1. Standardise column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # 2. Drop rows without a CustomerID
    df.dropna(subset=["CustomerID"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # 3. Remove cancellations
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # 4. Remove non-positive Quantity and UnitPrice
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 5. Remove known test / postage / manual adjustment SKUs
    noise_descriptions = [
        "POSTAGE", "DOTCOM POSTAGE", "MANUAL", "BANK CHARGES",
        "AMAZONFEE", "CRUK COMMISSION", "DISCOUNT", "SAMPLES"
    ]
    df = df[
        ~df["Description"].str.upper().str.strip().isin(noise_descriptions)
        .fillna(False)
    ]

    # 6. Parse InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 7. Derived columns
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    removed = original_len - len(df)
    print(f"[Preprocessing] Removed {removed:,} rows ({removed/original_len*100:.1f}%).")
    print(f"[Preprocessing] Clean dataset: {len(df):,} rows, "
          f"{df['CustomerID'].nunique():,} customers, "
          f"{df['StockCode'].nunique():,} products.")
    return df.reset_index(drop=True)


def get_snapshot_date(df: pd.DataFrame) -> datetime:
    """Return the analysis reference date (one day after last transaction)."""
    return df["InvoiceDate"].max() + pd.Timedelta(days=1)


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) metrics per customer.

    Recency   – days since last purchase (lower = better)
    Frequency – number of unique invoices
    Monetary  – total spend
    """
    snapshot = get_snapshot_date(df)
    print(f"[Preprocessing] RFM snapshot date: {snapshot.date()}")

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    return rfm


def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer richer behavioural features at the customer level.

    Returns a DataFrame indexed by CustomerID with columns:
      - AvgOrderValue, AvgItemsPerOrder
      - UniqueProducts, UniqueCategories (inferred from first word of description)
      - PreferredDayOfWeek, PreferredHour (mode)
      - PurchaseSpan (days between first and last purchase)
      - ReturnRate (cancelled qty / total qty — computed on raw df if passed)
      - CountryCount
    """
    # Average order value
    order_totals = df.groupby(["CustomerID", "InvoiceNo"])["TotalPrice"].sum()
    avg_order_value = order_totals.groupby(level=0).mean().rename("AvgOrderValue")
    avg_items = df.groupby(["CustomerID", "InvoiceNo"])["Quantity"].sum()
    avg_items = avg_items.groupby(level=0).mean().rename("AvgItemsPerOrder")

    # Product diversity
    unique_products = df.groupby("CustomerID")["StockCode"].nunique().rename("UniqueProducts")

    # Infer category from first word of description
    df = df.copy()
    df["Category"] = df["Description"].str.strip().str.split().str[0].str.upper()
    unique_categories = df.groupby("CustomerID")["Category"].nunique().rename("UniqueCategories")

    # Preferred shopping time
    preferred_dow = (
        df.groupby("CustomerID")["DayOfWeek"]
        .agg(lambda x: x.mode().iloc[0] if not x.empty else "Unknown")
        .rename("PreferredDayOfWeek")
    )
    preferred_hour = (
        df.groupby("CustomerID")["Hour"]
        .agg(lambda x: x.mode().iloc[0] if not x.empty else 12)
        .rename("PreferredHour")
    )

    # Purchase span
    span = (
        df.groupby("CustomerID")["InvoiceDate"]
        .agg(lambda x: (x.max() - x.min()).days)
        .rename("PurchaseSpan")
    )

    # Country count (some customers shop across countries)
    country_count = df.groupby("CustomerID")["Country"].nunique().rename("CountryCount")

    # Country (primary)
    primary_country = (
        df.groupby("CustomerID")["Country"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("PrimaryCountry")
    )

    features = pd.concat(
        [avg_order_value, avg_items, unique_products, unique_categories,
         preferred_dow, preferred_hour, span, country_count, primary_country],
        axis=1,
    ).reset_index()

    return features


def merge_customer_profile(rfm: pd.DataFrame, behavioral: pd.DataFrame) -> pd.DataFrame:
    """Merge RFM and behavioural features into a single customer profile."""
    profile = rfm.merge(behavioral, on="CustomerID", how="inner")
    print(f"[Preprocessing] Customer profile: {len(profile):,} customers, "
          f"{profile.shape[1]} features.")
    return profile
