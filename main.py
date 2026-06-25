"""
Customer Segmentation & Personalized Recommendation System
==========================================================
Main pipeline — run this file to execute the full end-to-end analysis.

Usage:
    python main.py

Dataset:
    The pipeline expects the UCI Online Retail dataset at data/online_retail.xlsx.
    If the file is missing, the loader will attempt an automatic download.
    Manual download URL:
        https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_raw_data, get_column_info
from src.preprocessing import (
    clean_raw_data, compute_rfm,
    compute_behavioral_features, merge_customer_profile,
)
from src.eda import run_full_eda
from src.feature_engineering import prepare_features
from src.pca_analysis import run_pca_pipeline
from src.clustering import run_clustering_pipeline
from src.recommendation import run_recommendation_pipeline
from src.visualization import build_executive_dashboard, generate_business_report


def section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def main() -> None:
    t0 = time.time()

    # ── 1. Load Data ──────────────────────────────────────────
    section("1 / 7  Loading Data")
    df_raw = load_raw_data()
    get_column_info(df_raw)

    # ── 2. Clean & Feature Engineering ───────────────────────
    section("2 / 7  Preprocessing & Feature Engineering")
    df = clean_raw_data(df_raw)
    rfm = compute_rfm(df)
    behavioral = compute_behavioral_features(df)
    profile = merge_customer_profile(rfm, behavioral)

    # ── 3. Exploratory Data Analysis ─────────────────────────
    section("3 / 7  Exploratory Data Analysis")
    run_full_eda(df, rfm, profile)

    # ── 4. Feature Preparation ────────────────────────────────
    section("4 / 7  Feature Preparation & Scaling")
    X_scaled, feature_names, scaler = prepare_features(profile)

    # ── 5. PCA ────────────────────────────────────────────────
    section("5 / 7  PCA – Dimensionality Reduction")
    X_pca, pca_model = run_pca_pipeline(X_scaled, feature_names, variance_threshold=0.85)

    # ── 6. Clustering ─────────────────────────────────────────
    section("6 / 7  Customer Segmentation (Clustering)")
    profile, best_k, segment_map = run_clustering_pipeline(
        X_scaled, X_pca, profile, k_range=range(2, 10)
    )

    # ── 7. Recommendations ────────────────────────────────────
    section("7 / 7  Recommendation System")
    rec_results = run_recommendation_pipeline(df, profile, top_n=10)

    # ── Dashboard & Report ────────────────────────────────────
    section("Generating Dashboard & Business Report")
    dashboard_path = build_executive_dashboard(
        df=df,
        profile=profile,
        X_pca=X_pca,
        segment_recs=rec_results["segment_recs"],
        popularity=rec_results["popularity"],
        coverage=rec_results["coverage"],
    )
    report_path = generate_business_report(
        profile=profile,
        segment_recs=rec_results["segment_recs"],
        coverage=rec_results["coverage"],
        best_k=best_k,
    )

    # ── Demo: similarity recommendations for a sample customer ──
    section("Sample Similarity-Based Recommendations")
    recommender = rec_results["recommender"]
    sample_customers = profile["CustomerID"].sample(3, random_state=42).tolist()
    for cid in sample_customers:
        seg = profile.loc[profile["CustomerID"] == cid, "Segment"].iloc[0]
        recs = recommender.recommend(cid, top_n=5)
        print(f"\nCustomer {cid} (Segment: {seg})")
        if recs.empty:
            print("  No recommendations available (too few common products).")
        else:
            for _, row in recs.iterrows():
                print(f"  ► {row['StockCode']} — {row.get('Description', 'N/A')} "
                      f"(score: {row['Score']:.3f})")

    # ── Summary ────────────────────────────────────────────────
    elapsed = time.time() - t0
    section("Pipeline Complete")
    print(f"  Total runtime  : {elapsed:.1f}s")
    print(f"  Customers      : {len(profile):,}")
    print(f"  Segments       : {best_k}")
    print(f"  Figures        : outputs/figures/")
    print(f"  Dashboard      : {dashboard_path}")
    print(f"  Business report: {report_path}")
    print()


if __name__ == "__main__":
    main()
