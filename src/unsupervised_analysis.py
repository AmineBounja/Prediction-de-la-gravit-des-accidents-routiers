"""
Analyse non-supervisée des données RTA avec PCA et K-Means.

Ce module effectue une réduction de dimensionnalité (PCA) et un clustering (K-Means)
pour explorer la structure des données sans utiliser les labels.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src import config
from src.utils import load_pickle, print_section_header, ensure_dir

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data():
    """Charge les données d'entraînement pour l'analyse non-supervisée."""
    print_section_header("1️⃣ CHARGEMENT DES DONNÉES")
    
    data_dir = config.PROCESSED_DATA_DIR
    X_train_path = data_dir / 'X_train_processed.csv'
    y_train_path = data_dir / 'y_train_processed.csv'
    
    if not X_train_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {X_train_path}")
        
    X = pd.read_csv(X_train_path)
    y = pd.read_csv(y_train_path).values.ravel()
    
    target_encoder = load_pickle(data_dir / 'target_encoder.pkl')
    
    print(f"  ▸ Données chargées: {X.shape}")
    return X, y, target_encoder

def run_pca_analysis(X, y, target_encoder):
    """
    Effectue une analyse PCA pour réduire la dimensionnalité et visualiser les données.
    
    Génère:
    - Projection 2D avec les vraies classes
    - Courbe de variance cumulative pour déterminer le nombre optimal de composantes
    """
    print_section_header("2️⃣ ANALYSE PCA")
    
    print("  • PCA 2D...")
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)
    
    explained_var = pca_2d.explained_variance_ratio_
    print(f"   Variance expliquée (2D): {sum(explained_var):.2%} {explained_var}")
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Gravité (Vraie)')
    plt.title(f'PCA 2D - Projection des Accidents\nVariance expliquée: {sum(explained_var):.2%}')
    plt.xlabel(f'PC1 ({explained_var[0]:.2%})')
    plt.ylabel(f'PC2 ({explained_var[1]:.2%})')
    
    output_dir = config.RESULTS_DIR / 'unsupervised'
    ensure_dir(output_dir)
    plt.savefig(output_dir / 'pca_2d_ground_truth.png', dpi=300)
    plt.close()
    print("  ▸ Graphique PCA 2D sauvegardé")
    
    print("\n  • Analyse de la variance cumulative...")
    pca_full = PCA()
    pca_full.fit(X)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    # Nombre de composantes pour conserver 95% de la variance
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.axvline(x=n_95, color='r', linestyle='--', label=f'{n_95} Composantes')
    plt.title('Variance Expliquée Cumulative')
    plt.xlabel('Nombre de Composantes')
    plt.ylabel('Variance Cumulative')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_dir / 'pca_variance.png', dpi=300)
    plt.close()
    print(f"  ▸ Graphique variance sauvegardé. 95% de variance avec {n_95} composantes.")
    
    return X_pca_2d

def run_kmeans_analysis(X, X_pca_2d):
    """
    Effectue un clustering K-Means et évalue la qualité avec le silhouette score.
    
    Le silhouette score mesure la cohésion intra-cluster et la séparation inter-cluster.
    Valeurs: -1 (mauvais) à +1 (excellent clustering).
    """
    print_section_header("3️⃣ CLUSTERING K-MEANS")
    
    output_dir = config.RESULTS_DIR / 'unsupervised'
    ensure_dir(output_dir)
    
    print("  • Recherche du nombre optimal de clusters (Elbow Method)...")
    inertias = []
    k_range = range(1, 11)
    
    # Échantillonnage pour accélérer le calcul sur de grands datasets
    if len(X) > 10000:
        X_sample = X.sample(10000, random_state=42)
    else:
        X_sample = X
        
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_sample)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o')
    plt.title('Méthode du Coude (Elbow Method)')
    plt.xlabel('Nombre de Clusters (k)')
    plt.ylabel('Inertie')
    plt.grid(True)
    plt.savefig(output_dir / 'kmeans_elbow.png', dpi=300)
    plt.close()
    print("  ▸ Graphique Elbow sauvegardé")
    
    k = config.KMEANS_N_CLUSTERS
    print(f"\n  • Application de K-Means avec k={k}...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Calcul du silhouette score sur un échantillon si dataset trop grand
    if len(X) > 5000:
        indices = np.random.choice(len(X), 5000, replace=False)
        sil_score = silhouette_score(X.iloc[indices], clusters[indices])
    else:
        sil_score = silhouette_score(X, clusters)
        
    print(f"   Silhouette Score: {sil_score:.4f}")
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'K-Means Clustering (k={k}) sur PCA 2D\nSilhouette: {sil_score:.3f}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    
    plt.savefig(output_dir / 'kmeans_clusters_pca.png', dpi=300)
    plt.close()
    print("  ▸ Graphique Clusters sauvegardé")
    
    return clusters

def main():
    print("📊 DÉMARRAGE DE L'ANALYSE NON-SUPERVISÉE\n")
    
    X, y, target_encoder = load_data()
    
    X_pca_2d = run_pca_analysis(X, y, target_encoder)
    
    clusters = run_kmeans_analysis(X, X_pca_2d)
    
    print("\n✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
    print(f"  • Résultats sauvegardés dans: {config.RESULTS_DIR / 'unsupervised'}")

if __name__ == "__main__":
    main()
