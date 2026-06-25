"""
Recommendation System Module

Implements two complementary strategies:
  1. Segment-Based Recommendations  – products popular within a customer's segment
  2. Similarity-Based Recommendations – products bought by the most similar customers
     (cosine similarity on the customer-product purchase matrix)

Also compares both against a Popularity Baseline.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Utilities ────────────────────────────────────────────────────────────────

def build_customer_product_matrix(df: pd.DataFrame,
                                  min_purchases: int = 5) -> pd.DataFrame:
    """
    Build a customer × product binary matrix.
    Rows = CustomerID, Columns = StockCode, Values = 1 if purchased else 0.
    Only keep products bought by at least `min_purchases` customers.
    """
    # Aggregate purchases
    cp = (
        df.groupby(["CustomerID", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
    )
    matrix = cp.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    fill_value=0,
)

    # Filter sparse products
    product_counts = matrix.sum(axis=0)
    matrix = matrix.loc[:, product_counts >= min_purchases]

    print(f"[Recommendation] Customer-product matrix: "
          f"{matrix.shape[0]:,} customers × {matrix.shape[1]:,} products")
    return matrix


def get_product_descriptions(df: pd.DataFrame) -> dict[str, str]:
    """Return a dict mapping StockCode → Description."""
    return (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .first()
        .to_dict()
    )


# ── Strategy 1: Segment-Based ────────────────────────────────────────────────

def compute_segment_top_products(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    For each segment, find the top_n most popular products by revenue.
    Returns dict {segment_name: DataFrame with StockCode, Description, Revenue, PurchaseCount}.
    """
    merged = df.merge(
        profile[["CustomerID", "Segment"]], on="CustomerID", how="inner"
    )
    segment_recs = {}
    for segment in merged["Segment"].unique():
        seg_df = merged[merged["Segment"] == segment]
        top = (
            seg_df.groupby(["StockCode", "Description"])
            .agg(Revenue=("TotalPrice", "sum"),
                 PurchaseCount=("InvoiceNo", "nunique"))
            .reset_index()
            .sort_values("Revenue", ascending=False)
            .head(top_n)
        )
        segment_recs[segment] = top
    return segment_recs


# ── Strategy 2: Similarity-Based ─────────────────────────────────────────────

class SimilarityRecommender:
    """
    Cosine-similarity recommender over the customer-product matrix.
    For a given customer, find top-N similar customers (excluding self),
    then recommend products those neighbours bought but the target hasn't.
    """

    def __init__(self, matrix: pd.DataFrame, descriptions: dict[str, str]):
        self.matrix = matrix
        self.descriptions = descriptions
        self.similarity_matrix = cosine_similarity(matrix.values)
        self.customer_ids = list(matrix.index)
        self.product_codes = list(matrix.columns)
        self._id_to_idx = {cid: i for i, cid in enumerate(self.customer_ids)}

    def recommend(self, customer_id: str, top_n: int = 10,
                  n_neighbours: int = 20) -> pd.DataFrame:
        """
        Recommend top_n products for a given customer_id.
        Returns DataFrame with StockCode, Description, Score.
        """
        if customer_id not in self._id_to_idx:
            return pd.DataFrame(columns=["StockCode", "Description", "Score"])

        idx = self._id_to_idx[customer_id]
        sim_scores = self.similarity_matrix[idx].copy()
        sim_scores[idx] = -1  # exclude self

        neighbour_idxs = np.argsort(sim_scores)[::-1][:n_neighbours]
        neighbour_weights = sim_scores[neighbour_idxs]

        already_bought = set(
            self.product_codes[j]
            for j, v in enumerate(self.matrix.iloc[idx].values)
            if v > 0
        )

        # Weighted sum of neighbour purchases
        product_scores: dict[str, float] = {}
        for ni, weight in zip(neighbour_idxs, neighbour_weights):
            if weight <= 0:
                continue
            for j, v in enumerate(self.matrix.iloc[ni].values):
                if v > 0:
                    code = self.product_codes[j]
                    if code not in already_bought:
                        product_scores[code] = product_scores.get(code, 0) + weight

        if not product_scores:
            return pd.DataFrame(columns=["StockCode", "Description", "Score"])

        recs = (
            pd.DataFrame(list(product_scores.items()), columns=["StockCode", "Score"])
            .sort_values("Score", ascending=False)
            .head(top_n)
        )
        recs["Description"] = recs["StockCode"].map(self.descriptions)
        return recs[["StockCode", "Description", "Score"]].reset_index(drop=True)


# ── Strategy 3: Popularity Baseline ──────────────────────────────────────────

def compute_popularity_baseline(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Global top-N products by purchase frequency (number of unique buyers)."""
    return (
        df.groupby(["StockCode", "Description"])
        .agg(
            UniqueBuyers=("CustomerID", "nunique"),
            Revenue=("TotalPrice", "sum"),
        )
        .reset_index()
        .sort_values("UniqueBuyers", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_recommendation_coverage(
    segment_recs: dict[str, pd.DataFrame],
    popularity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare catalogue coverage between segment-based and popularity-based recommendations.
    Returns a summary DataFrame.
    """
    pop_set = set(popularity["StockCode"])
    records = []
    for segment, recs in segment_recs.items():
        seg_set = set(recs["StockCode"])
        overlap = len(seg_set & pop_set)
        unique_to_seg = len(seg_set - pop_set)
        records.append({
            "Segment": segment,
            "SegmentRecs": len(seg_set),
            "OverlapWithPopularity": overlap,
            "UniqueToSegment": unique_to_seg,
            "PersonalisationGain%": round(unique_to_seg / len(seg_set) * 100, 1),
        })
    return pd.DataFrame(records).sort_values("PersonalisationGain%", ascending=False)


# ── Visualisations ───────────────────────────────────────────────────────────

def plot_segment_top_products(
    segment_recs: dict[str, pd.DataFrame],
    max_segments: int = 4,
) -> None:
    """Bar charts: top products per segment."""
    segments = list(segment_recs.keys())[:max_segments]
    fig, axes = plt.subplots(1, len(segments), figsize=(7 * len(segments), 6))
    if len(segments) == 1:
        axes = [axes]

    palette = ["#4C72B0", "#2A9D8F", "#E63946", "#E9C46A", "#F4A261", "#457B9D"]
    for ax, seg in zip(axes, segments):
        data = segment_recs[seg].head(8).sort_values("Revenue")
        short_desc = data["Description"].str[:30]
        ax.barh(short_desc, data["Revenue"],
                color=palette[segments.index(seg) % len(palette)], alpha=0.85)
        ax.set_title(seg, fontsize=10, fontweight="bold")
        ax.set_xlabel("Revenue (£)")
    plt.suptitle("Top Products per Customer Segment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "18_segment_top_products.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Recommendation] Saved → {path.name}")


def plot_recommendation_coverage(coverage: pd.DataFrame) -> None:
    """Stacked bar: overlap vs unique recommendations per segment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(coverage))
    ax.bar(x, coverage["OverlapWithPopularity"], label="Overlap w/ Popularity",
           color="#B0C4DE", alpha=0.9)
    ax.bar(x, coverage["UniqueToSegment"], bottom=coverage["OverlapWithPopularity"],
           label="Unique to Segment", color="#4C72B0", alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(coverage["Segment"], rotation=20, ha="right")
    ax.set_ylabel("Number of Products")
    ax.set_title("Recommendation Coverage: Segment vs Popularity", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "19_recommendation_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Recommendation] Saved → {path.name}")


def plot_popularity_baseline(popularity: pd.DataFrame) -> None:
    """Horizontal bar chart for global popularity baseline."""
    data = popularity.copy().sort_values("UniqueBuyers")
    short_desc = data["Description"].str[:35]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(short_desc, data["UniqueBuyers"], color="#A8DADC", edgecolor="white")
    ax.set_title("Global Popularity Baseline – Top Products by Unique Buyers",
                 fontweight="bold")
    ax.set_xlabel("Unique Buyers")
    plt.tight_layout()
    path = FIGURES_DIR / "20_popularity_baseline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Recommendation] Saved → {path.name}")


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_recommendation_pipeline(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    top_n: int = 10,
) -> dict:
    """
    Full recommendation pipeline.
    Returns a dict with keys: segment_recs, popularity, coverage, recommender.
    """
    print("[Recommendation] Building customer-product matrix...")
    matrix = build_customer_product_matrix(df)
    descriptions = get_product_descriptions(df)

    print("[Recommendation] Computing segment-based recommendations...")
    segment_recs = compute_segment_top_products(df, profile, top_n=top_n)

    print("[Recommendation] Computing popularity baseline...")
    popularity = compute_popularity_baseline(df, top_n=top_n)

    print("[Recommendation] Building similarity-based recommender...")
    recommender = SimilarityRecommender(matrix, descriptions)

    coverage = evaluate_recommendation_coverage(segment_recs, popularity)
    print(f"\n[Recommendation] Personalisation Coverage:\n{coverage.to_string(index=False)}\n")

    plot_popularity_baseline(popularity)
    plot_segment_top_products(segment_recs)
    plot_recommendation_coverage(coverage)

    return {
        "segment_recs": segment_recs,
        "popularity": popularity,
        "coverage": coverage,
        "recommender": recommender,
    }
