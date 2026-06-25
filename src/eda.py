"""
Exploratory Data Analysis Module
Generates and saves EDA plots for the Online Retail dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "viridis"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, name: str, dpi: int = 150) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved → {path.name}")


def plot_sales_over_time(df: pd.DataFrame) -> None:
    """Monthly revenue trend."""
    monthly = (
        df.groupby("YearMonth")["TotalPrice"]
        .sum()
        .reset_index()
    )
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(monthly["YearMonth"], monthly["TotalPrice"], alpha=0.3, color="#4C72B0")
    ax.plot(monthly["YearMonth"], monthly["TotalPrice"], marker="o", color="#4C72B0", linewidth=2)
    ax.set_title("Monthly Revenue Trend", fontsize=15, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Revenue (£)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save(fig, "01_monthly_revenue.png")


def plot_top_products(df: pd.DataFrame, top_n: int = 15) -> None:
    """Top N products by revenue."""
    top = (
        df.groupby("Description")["TotalPrice"]
        .sum()
        .nlargest(top_n)
        .reset_index()
        .sort_values("TotalPrice")
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["Description"], top["TotalPrice"], color=sns.color_palette("Blues_r", top_n))
    ax.set_title(f"Top {top_n} Products by Revenue", fontsize=15, fontweight="bold")
    ax.set_xlabel("Total Revenue (£)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    plt.tight_layout()
    _save(fig, "02_top_products.png")


def plot_geographic_distribution(df: pd.DataFrame) -> None:
    """Revenue by country (top 10, excluding UK)."""
    country_rev = (
        df[df["Country"] != "United Kingdom"]
        .groupby("Country")["TotalPrice"]
        .sum()
        .nlargest(10)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=country_rev, x="TotalPrice", y="Country",
                palette="Blues_r", ax=ax)
    ax.set_title("Top 10 Countries by Revenue (excl. UK)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Total Revenue (£)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    plt.tight_layout()
    _save(fig, "03_geographic_revenue.png")


def plot_hourly_orders(df: pd.DataFrame) -> None:
    """Orders by hour of day."""
    hourly = df.groupby("Hour")["InvoiceNo"].nunique().reset_index()
    hourly.columns = ["Hour", "OrderCount"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=hourly, x="Hour", y="OrderCount", palette="coolwarm", ax=ax)
    ax.set_title("Orders by Hour of Day", fontsize=15, fontweight="bold")
    ax.set_xlabel("Hour (24h)")
    ax.set_ylabel("Number of Orders")
    plt.tight_layout()
    _save(fig, "04_hourly_orders.png")


def plot_dow_orders(df: pd.DataFrame) -> None:
    """Orders by day of week."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("DayOfWeek")["InvoiceNo"].nunique().reindex(order).reset_index()
    dow.columns = ["DayOfWeek", "OrderCount"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=dow, x="DayOfWeek", y="OrderCount", palette="muted", ax=ax)
    ax.set_title("Orders by Day of Week", fontsize=15, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Orders")
    plt.tight_layout()
    _save(fig, "05_dow_orders.png")


def plot_rfm_distributions(rfm: pd.DataFrame) -> None:
    """Distribution histograms for R, F, M."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("Recency", "Days Since Last Purchase", "#E07B54"),
               ("Frequency", "Number of Orders", "#5B8DB8"),
               ("Monetary", "Total Spend (£)", "#57A464")]
    for ax, (col, label, color) in zip(axes, metrics):
        data = rfm[col].clip(upper=rfm[col].quantile(0.99))
        ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=13, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("Customers")
    fig.suptitle("RFM Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "06_rfm_distributions.png")


def plot_rfm_correlation(rfm: pd.DataFrame) -> None:
    """Correlation heatmap of RFM features."""
    numeric = rfm[["Recency", "Frequency", "Monetary"]].copy()
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("RFM Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "07_rfm_correlation.png")


def plot_spending_boxplots(profile: pd.DataFrame) -> None:
    """Box plots for key numeric features (log-scaled)."""
    cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]
    cols = [c for c in cols if c in profile.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5))
    for ax, col in zip(axes, cols):
        data = profile[col].replace(0, np.nan).dropna()
        ax.boxplot(np.log1p(data), patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", alpha=0.7))
        ax.set_title(f"log({col})", fontsize=11)
        ax.set_xticks([])
    fig.suptitle("Feature Distributions (log-scale)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "08_feature_boxplots.png")


def run_full_eda(df: pd.DataFrame, rfm: pd.DataFrame, profile: pd.DataFrame) -> None:
    """Run all EDA plots."""
    print("[EDA] Generating exploratory analysis plots...")
    plot_sales_over_time(df)
    plot_top_products(df)
    plot_geographic_distribution(df)
    plot_hourly_orders(df)
    plot_dow_orders(df)
    plot_rfm_distributions(rfm)
    plot_rfm_correlation(rfm)
    plot_spending_boxplots(profile)
    print("[EDA] All EDA plots saved.")
