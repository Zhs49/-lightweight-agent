from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def run_kmeans_with_elbow(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Tuple[KMeans, Dict[int, float], int, float]:
    inertias: Dict[int, float] = {}
    silhouettes: Dict[int, float] = {}
    best_k = None
    best_sil = -1.0
    best_model = None

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)
        inertias[k] = model.inertia_
        sil = silhouette_score(X, labels)
        silhouettes[k] = sil
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = model

    assert best_model is not None and best_k is not None
    return best_model, inertias, best_k, best_sil


def save_elbow_plot(inertias: Dict[int, float], out_png: str, out_svg: str) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    ks = sorted(inertias.keys())
    vals = [inertias[k] for k in ks]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, vals, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_svg)
    plt.close()


def save_pca_scatter(X: np.ndarray, labels: np.ndarray, out_png: str, out_svg: str) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Scatter by Cluster")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_svg)
    plt.close()


def scale_features(df_features: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    feature_cols = list(df_features.select_dtypes(include=[np.number]).columns)
    df_numeric = df_features[feature_cols]
    scaler = StandardScaler()
    if df_numeric.shape[1] == 0:
        return np.zeros((len(df_features), 0)), scaler, feature_cols
    # Impute missing values with median before scaling
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df_numeric.values)
    X = scaler.fit_transform(X_imp)
    return X, scaler, feature_cols


def build_cluster_profiles(df: pd.DataFrame, labels: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
    df_prof = df.copy()
    df_prof["cluster"] = labels
    # Compute mean for numeric and top category for categorical dummies
    summary = df_prof.groupby("cluster")[feature_cols + (['price'] if 'price' in df_prof.columns else [])].mean(numeric_only=True)
    summary = summary.sort_index()
    return summary


