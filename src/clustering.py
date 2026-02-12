"""K-Means clustering on sentiment-derived numerical features.

Example:
    python -m src.clustering --features_path artifacts/tweet_features.csv --n_clusters 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster sentiment feature vectors with K-Means.")
    parser.add_argument("--features_path", type=str, required=True, help="Path to tweet_features.csv.")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of K-Means clusters.")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Output directory.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_path)

    # Keep all numeric columns for clustering. This includes class probabilities,
    # sentiment strength, and embedding dimensions emb_0...emb_n.
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric feature columns found for clustering.")

    X = df[numeric_columns].copy()

    # Standardization makes K-Means distance comparisons fair across mixed scales.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=20)
    cluster_ids = kmeans.fit_predict(X_scaled)

    result_df = df.copy()
    result_df["cluster"] = cluster_ids

    # 2D visualization using PCA for easy interpretation.
    pca = PCA(n_components=2, random_state=args.random_state)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_ids, cmap="viridis", alpha=0.8)
    plt.title("K-Means Clusters on Sentiment Features (PCA 2D)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    plot_path = output_dir / "kmeans_scatter.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    clustered_path = output_dir / "tweet_features_clustered.csv"
    result_df.to_csv(clustered_path, index=False)

    print("=== Clustering Summary ===")
    print(result_df[["cluster"]].value_counts().rename("count"))
    print(f"\nClustered features saved to: {clustered_path.resolve()}")
    print(f"Scatter plot saved to      : {plot_path.resolve()}")


if __name__ == "__main__":
    main()
