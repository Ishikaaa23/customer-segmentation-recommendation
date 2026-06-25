"""
Clustering Module
K-Means and Hierarchical clustering with automated optimal-k selection
and rich business-level cluster interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_LABELS = {
    "high_value": "High-Value Champions",
    "loyal": "Loyal Customers",
    "frequent": "Frequent Shoppers",
    "bargain": "Bargain Seekers",
    "at_risk": "At-Risk Customers",
    "new": "New / Promising",
    "hibernating": "Hibernating",
}

SEGMENT_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#A8DADC",
]


# ── Optimal K Selection ──────────────────────────────────────────────────────

def compute_cluster_metrics(X: np.ndarray, k_range: range) -> pd.DataFrame:
    """Compute inertia, silhouette, and Davies-Bouldin scores for each k."""
    records = []
    print(f"[Clustering] Evaluating k = {list(k_range)} ...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else 0.0
        db = davies_bouldin_score(X, labels) if k > 1 else np.nan
        records.append({"k": k, "inertia": km.inertia_, "silhouette": sil, "davies_bouldin": db})
        print(f"  k={k:2d} | inertia={km.inertia_:>12.1f} | "
              f"silhouette={sil:.4f} | DB={db:.4f}" if not np.isnan(db)
              else f"  k={k:2d} | inertia={km.inertia_:>12.1f}")
    return pd.DataFrame(records)


def plot_cluster_selection(metrics: pd.DataFrame) -> int:
    """
    Plot Elbow, Silhouette, and DB-Index curves.
    Returns the recommended k (highest silhouette).
    """
    best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Elbow
    axes[0].plot(metrics["k"], metrics["inertia"], marker="o", linewidth=2, color="#4C72B0")
    axes[0].set_title("Elbow Method – Inertia", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")

    # Silhouette
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", linewidth=2, color="#2A9D8F")
    axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
    axes[1].set_title("Silhouette Score (higher = better)", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    # Davies-Bouldin
    axes[2].plot(metrics["k"], metrics["davies_bouldin"], marker="o", linewidth=2, color="#E63946")
    axes[2].axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
    axes[2].set_title("Davies-Bouldin Index (lower = better)", fontweight="bold")
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("DB Index")
    axes[2].legend()

    plt.suptitle("Optimal Cluster Count Selection", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "12_cluster_selection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Clustering] Saved → {path.name} | Recommended k={best_k}")
    return best_k


# ── K-Means ──────────────────────────────────────────────────────────────────

def fit_kmeans(X: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    """Fit K-Means and return (model, labels)."""
    km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=500)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    print(f"[Clustering] K-Means k={k} | silhouette={sil:.4f} | DB={db:.4f}")
    return km, labels


def plot_kmeans_scatter(X_pca: np.ndarray, labels: np.ndarray, k: int) -> None:
    """PC1 vs PC2 scatter coloured by cluster."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(k):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=15, alpha=0.6, label=f"Cluster {i}",
                   color=SEGMENT_COLORS[i % len(SEGMENT_COLORS)])
    ax.set_title("K-Means Customer Segments (PCA Space)", fontsize=14, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2)
    plt.tight_layout()
    path = FIGURES_DIR / "13_kmeans_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Clustering] Saved → {path.name}")


# ── Hierarchical Clustering ───────────────────────────────────────────────────

def fit_hierarchical(X: np.ndarray, k: int,
                     method: str = "ward") -> np.ndarray:
    """Fit Agglomerative Clustering and return labels."""
    hc = AgglomerativeClustering(n_clusters=k, linkage=method)
    labels = hc.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"[Clustering] Hierarchical ({method}) k={k} | silhouette={sil:.4f}")
    return labels


def plot_dendrogram(X: np.ndarray, sample_size: int = 500) -> None:
    """Plot a truncated dendrogram on a random sample."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    Z = linkage(X[idx], method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
               leaf_rotation=45, leaf_font_size=9,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(f"Hierarchical Clustering Dendrogram (sample n={len(idx)})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Customer Index")
    ax.set_ylabel("Ward Linkage Distance")
    plt.tight_layout()
    path = FIGURES_DIR / "14_dendrogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Clustering] Saved → {path.name}")


# ── Cluster Analysis & Labelling ─────────────────────────────────────────────

def assign_segment_labels(cluster_stats: pd.DataFrame) -> dict[int, str]:
    """
    Heuristically assign business labels to clusters based on RFM stats.
    Returns a dict mapping cluster_id → segment_name.
    """
    labels_map = {}
    df = cluster_stats.copy()

    # Normalise for comparison
    for col in ["Recency", "Frequency", "Monetary"]:
        if col in df.columns:
            min_v, max_v = df[col].min(), df[col].max()
            df[f"{col}_norm"] = (df[col] - min_v) / (max_v - min_v + 1e-9)

    for idx, row in df.iterrows():
        r = row.get("Recency_norm", 0.5)
        f = row.get("Frequency_norm", 0.5)
        m = row.get("Monetary_norm", 0.5)

        if m > 0.65 and f > 0.65 and r < 0.35:
            label = "High-Value Champions"
        elif f > 0.65 and r < 0.4:
            label = "Loyal Customers"
        elif r > 0.75:
            label = "At-Risk / Hibernating"
        elif m < 0.3 and f > 0.4:
            label = "Bargain Seekers"
        elif r < 0.3 and f < 0.35:
            label = "New / Promising"
        elif f > 0.5:
            label = "Frequent Shoppers"
        else:
            label = "Mid-Tier Customers"

        labels_map[int(idx)] = label
    return labels_map


def analyse_clusters(profile: pd.DataFrame, labels: np.ndarray,
                     label_col: str = "KMeans_Cluster") -> pd.DataFrame:
    """
    Add cluster labels to profile, compute cluster statistics, assign business names.
    Returns the enriched profile DataFrame.
    """
    profile = profile.copy()
    profile[label_col] = labels

    numeric_cols = ["Recency", "Frequency", "Monetary",
                    "AvgOrderValue", "UniqueProducts", "PurchaseSpan"]
    numeric_cols = [c for c in numeric_cols if c in profile.columns]

    stats = profile.groupby(label_col)[numeric_cols].mean()
    stats["CustomerCount"] = profile.groupby(label_col).size()

    segment_map = assign_segment_labels(stats)
    profile["Segment"] = profile[label_col].map(segment_map)

    print("\n[Clustering] Cluster Summary:")
    print(stats.to_string())
    print("\n[Clustering] Segment Labels:")
    for k, v in segment_map.items():
        cnt = stats.loc[k, "CustomerCount"]
        print(f"  Cluster {k} → {v} ({int(cnt):,} customers)")

    return profile, stats, segment_map


# ── Cluster Visualisations ───────────────────────────────────────────────────

def plot_cluster_profiles(stats: pd.DataFrame, segment_map: dict) -> None:
    """Radar/spider charts for each cluster across key metrics."""
    numeric_cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]
    numeric_cols = [c for c in numeric_cols if c in stats.columns]
    n_clusters = len(stats)

    # Normalise 0-1
    norm = stats[numeric_cols].copy()
    for col in numeric_cols:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - mn) / (mx - mn + 1e-9)
    # Recency: lower is better — invert
    if "Recency" in norm.columns:
        norm["Recency"] = 1 - norm["Recency"]

    N = len(numeric_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    cols = min(3, n_clusters)
    rows = (n_clusters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),
                              subplot_kw=dict(polar=True))
    axes = np.array(axes).flatten()

    for i, (cluster_id, row) in enumerate(norm.iterrows()):
        values = row.tolist() + [row.iloc[0]]
        ax = axes[i]
        ax.fill(angles, values,
                color=SEGMENT_COLORS[i % len(SEGMENT_COLORS)], alpha=0.35)
        ax.plot(angles, values,
                color=SEGMENT_COLORS[i % len(SEGMENT_COLORS)], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(numeric_cols, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Cluster {cluster_id}\n{segment_map.get(int(cluster_id), '')}",
            fontsize=10, fontweight="bold", pad=15,
        )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Cluster Profiles – Normalised Feature Radar", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "15_cluster_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Clustering] Saved → {path.name}")


def plot_cluster_distribution(profile: pd.DataFrame, label_col: str = "Segment") -> None:
    """Pie chart showing relative cluster size."""
    counts = profile[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=SEGMENT_COLORS[:len(counts)],
        startangle=140,
        pctdistance=0.8,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Customer Segment Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "16_segment_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Clustering] Saved → {path.name}")


def plot_rfm_by_segment(profile: pd.DataFrame) -> None:
    """Box plots of R, F, M by segment."""
    metrics = [c for c in ["Recency", "Frequency", "Monetary"] if c in profile.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 6))
    for ax, metric in zip(axes, metrics):
        order = profile.groupby("Segment")[metric].median().sort_values().index
        sns.boxplot(data=profile, x="Segment", y=metric, order=order,
                    palette=SEGMENT_COLORS[:profile["Segment"].nunique()], ax=ax)
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        if metric == "Monetary":
            ax.set_ylabel("Total Spend (£)")
    plt.suptitle("RFM Distribution by Customer Segment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "17_rfm_by_segment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Clustering] Saved → {path.name}")


def run_clustering_pipeline(
    X_scaled: np.ndarray,
    X_pca: np.ndarray,
    profile: pd.DataFrame,
    k_range: range = range(2, 11),
) -> pd.DataFrame:
    """
    Full clustering pipeline:
      1. Evaluate k with Elbow / Silhouette / DB
      2. Fit K-Means with best k
      3. Fit Hierarchical clustering
      4. Plot dendrogram, scatter, radar, distribution
      5. Return enriched profile
    """
    metrics = compute_cluster_metrics(X_pca, k_range)
    best_k = plot_cluster_selection(metrics)

    # K-Means
    km_model, km_labels = fit_kmeans(X_pca, best_k)
    plot_kmeans_scatter(X_pca, km_labels, best_k)

    # Hierarchical
    plot_dendrogram(X_pca)
    hc_labels = fit_hierarchical(X_pca, best_k)

    # Analyse and label
    profile, stats, segment_map = analyse_clusters(profile, km_labels, "KMeans_Cluster")

    plot_cluster_profiles(stats, segment_map)
    plot_cluster_distribution(profile)
    plot_rfm_by_segment(profile)

    # Attach hierarchical labels too (for reference)
    profile["HC_Cluster"] = hc_labels

    return profile, best_k, segment_map
