"""
PCA Analysis Module
Applies Principal Component Analysis to reduce feature dimensionality
and visualises explained variance and customer distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def fit_pca(X_scaled: np.ndarray, n_components: int | None = None) -> tuple[PCA, np.ndarray]:
    """
    Fit PCA on scaled data.
    If n_components is None, fit with all components to compute explained variance.
    Returns (fitted_pca, X_pca).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


def select_n_components(pca: PCA, threshold: float = 0.85) -> int:
    """Return the minimum number of PCs that explain `threshold` variance."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.argmax(cumvar >= threshold)) + 1
    print(f"[PCA] {n} components explain ≥{threshold*100:.0f}% variance "
          f"(total: {cumvar[n-1]*100:.1f}%)")
    return n


def plot_explained_variance(pca: PCA, threshold: float = 0.85) -> None:
    """Scree plot and cumulative explained variance."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_selected = select_n_components(pca, threshold)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    ax = axes[0]
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1),
           pca.explained_variance_ratio_ * 100,
           color=sns.color_palette("Blues_r", len(pca.explained_variance_ratio_)))
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Scree Plot – Explained Variance per PC", fontweight="bold")
    ax.set_xticks(range(1, len(pca.explained_variance_ratio_) + 1))

    # Cumulative variance
    ax2 = axes[1]
    ax2.plot(range(1, len(cumvar) + 1), cumvar * 100,
             marker="o", linewidth=2, color="#4C72B0")
    ax2.axhline(threshold * 100, color="red", linestyle="--", alpha=0.7,
                label=f"{threshold*100:.0f}% threshold")
    ax2.axvline(n_selected, color="orange", linestyle="--", alpha=0.7,
                label=f"Selected: {n_selected} PCs")
    ax2.fill_between(range(1, len(cumvar) + 1), cumvar * 100, alpha=0.15, color="#4C72B0")
    ax2.set_xlabel("Number of Principal Components")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title("Cumulative Explained Variance", fontweight="bold")
    ax2.legend()
    ax2.set_ylim(0, 105)

    plt.suptitle("PCA – Dimensionality Reduction Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "09_pca_explained_variance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PCA] Saved → {path.name}")


def plot_pca_biplot(X_pca: np.ndarray, pca: PCA, feature_names: list[str]) -> None:
    """Biplot: PC1 vs PC2 scatter with feature loading vectors."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=10, color="#5B8DB8")

    # Loading vectors (arrows)
    scale = 3.0
    loadings = pca.components_[:2].T
    for i, feat in enumerate(feature_names):
        ax.annotate(
            feat,
            xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
            color="crimson", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA Biplot – PC1 vs PC2 with Feature Loadings", fontweight="bold")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    plt.tight_layout()
    path = FIGURES_DIR / "10_pca_biplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PCA] Saved → {path.name}")


def plot_loading_heatmap(pca: PCA, feature_names: list[str], n_pcs: int = 5) -> None:
    """Heatmap of feature loadings for the top n_pcs components."""
    n_pcs = min(n_pcs, pca.n_components_)
    loadings = pd.DataFrame(
        pca.components_[:n_pcs].T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_pcs)],
    )
    fig, ax = plt.subplots(figsize=(max(8, n_pcs * 1.5), max(6, len(feature_names) * 0.6)))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title(f"PCA Feature Loadings (Top {n_pcs} PCs)", fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "11_pca_loading_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PCA] Saved → {path.name}")


def run_pca_pipeline(
    X_scaled: np.ndarray,
    feature_names: list[str],
    variance_threshold: float = 0.85,
) -> tuple[np.ndarray, PCA]:
    """
    Full PCA pipeline:
      1. Fit PCA with all components
      2. Plot explained variance
      3. Refit with selected n_components
      4. Plot biplot and loading heatmap
    Returns (X_pca_reduced, fitted_pca).
    """
    print("[PCA] Fitting full PCA to assess variance...")
    pca_full, _ = fit_pca(X_scaled)
    plot_explained_variance(pca_full, variance_threshold)
    plot_loading_heatmap(pca_full, feature_names)

    n = select_n_components(pca_full, variance_threshold)
    print(f"[PCA] Refitting with {n} components...")
    pca_reduced, X_pca = fit_pca(X_scaled, n_components=n)
    plot_pca_biplot(X_pca, pca_reduced, feature_names)

    return X_pca, pca_reduced
