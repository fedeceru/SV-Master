"""
NetworkClusterer

Riduzione dimensionale e clustering dei geni a partire dalla matrice di affinità fusa (SNF).

APPROCCIO:
    Si usa una strategia a due livelli di UMAP:
    1. UMAP a dimensionalità intermedia (default 10D) per creare separazione di densità
       adeguata al clustering con HDBSCAN.
    2. UMAP a 2D esclusivamente per la visualizzazione.

    Il clustering direttamente sulla matrice di distanza precomputata non funziona bene
    con HDBSCAN perché SNF produce un paesaggio di similarità troppo uniforme
    (distanze ravvicinate). UMAP in spazio intermedio amplifica le differenze di densità
    locali, rendendo i cluster rilevabili.

1. CLUSTERING (apply_clustering):
   - UMAP a `cluster_dims` dimensioni dalla matrice di distanza.
   - HDBSCAN sullo spazio intermedio con parametri ragionevoli (min_cluster_size=20).

2. VISUALIZZAZIONE (compute_visualization):
   - Proiezione 2D separata (UMAP, t-SNE, PCA) per la rappresentazione grafica.

3. MARKER GENES (find_marker_genes):
   - Calcolo centroide per cluster nello spazio di embedding 2D.
   - Selezione dei k geni più vicini al centroide come rappresentanti.

4. VALIDAZIONE (_evaluate_quality):
   - Silhouette Score sullo spazio di clustering (non sugli embedding 2D).
"""

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import umap
import hdbscan

class NetworkClusterer:
    def __init__(self, fused_matrix_df, output_dir="../data/results/"):
        self.fused_df = fused_matrix_df
        self.gene_index = fused_matrix_df.index
        self.output_dir = output_dir

        affinity = np.clip(self.fused_df.values, 0, 1)
        self.dist_matrix = 1.0 - affinity
        np.fill_diagonal(self.dist_matrix, 0.0)

        self.cluster_embeddings = None  # high-D for clustering
        self.embeddings = None          # 2D for visualization
        self.labels = None
        self.marker_genes = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def apply_clustering(self, method="hdbscan", n_clusters=5, min_cluster_size=20,
                         min_samples=5, cluster_selection_method='eom',
                         cluster_dims=10, cluster_n_neighbors=15, random_state=42):
        """
        Cluster genes using UMAP intermediate embedding + HDBSCAN.

        Parameters:
            cluster_dims: number of UMAP dimensions for the clustering space.
            cluster_n_neighbors: UMAP n_neighbors for the clustering embedding.
        """
        print(f"[Clustering] UMAP {cluster_dims}D + {method.upper()}")

        # Build intermediate embedding for clustering
        print(f"  [UMAP] Proiezione a {cluster_dims}D (n_neighbors={cluster_n_neighbors})...")
        reducer = umap.UMAP(n_components=cluster_dims, metric="precomputed",
                            n_neighbors=cluster_n_neighbors, random_state=random_state)
        self.cluster_embeddings = reducer.fit_transform(self.dist_matrix)

        if method.lower() == "kmeans":
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
            self.labels = model.fit_predict(self.cluster_embeddings)

        elif method.lower() == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            self.labels = model.fit_predict(self.cluster_embeddings)

        elif method.lower() == "hdbscan":
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=cluster_selection_method,
                metric='euclidean'
            )
            self.labels = model.fit_predict(self.cluster_embeddings)

        else:
            raise ValueError(f"Metodo di clustering '{method}' non supportato.")

        n_found = len(set(self.labels) - {-1})
        n_noise = (self.labels == -1).sum()
        print(f"[OK] Clustering completato: {n_found} moduli, "
              f"{n_noise} geni rumore ({100*n_noise/len(self.labels):.1f}%)")
        return self.labels

    def compute_visualization(self, method="umap", n_components=2, random_state=42, **kwargs):
        """Project to 2D for visualization only (independent of clustering)."""
        print(f"[Visualizzazione] Metodo: {method.upper()} | Componenti: {n_components}")

        if method.lower() == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            self.embeddings = model.fit_transform(self.dist_matrix)

        elif method.lower() == "tsne":
            model = TSNE(n_components=n_components, metric="precomputed",
                         init='random', random_state=random_state, **kwargs)
            self.embeddings = model.fit_transform(self.dist_matrix)

        elif method.lower() == "umap":
            model = umap.UMAP(n_components=n_components, metric="precomputed",
                              random_state=random_state, **kwargs)
            self.embeddings = model.fit_transform(self.dist_matrix)

        else:
            raise ValueError(f"Metodo di visualizzazione '{method}' non supportato.")

        return self.embeddings

    def find_marker_genes(self, k=3):
        """Find the k genes closest to each cluster centroid in 2D embedding space."""
        if self.embeddings is None or self.labels is None:
            raise ValueError("Eseguire clustering e visualizzazione prima di cercare i marker.")

        print(f"[Marker] Identificazione top-{k} geni rappresentativi per cluster")

        self.marker_genes = {}
        genes = np.array(self.gene_index)

        for cluster_id in sorted(set(self.labels) - {-1}):
            mask = self.labels == cluster_id
            cluster_emb = self.embeddings[mask]
            cluster_genes = genes[mask]

            centroid = cluster_emb.mean(axis=0)
            dists = np.linalg.norm(cluster_emb - centroid, axis=1)
            top_idx = np.argsort(dists)[:k]

            self.marker_genes[int(cluster_id)] = [str(g) for g in cluster_genes[top_idx]]

        print(f"[OK] Marker identificati per {len(self.marker_genes)} cluster")
        return self.marker_genes

    def _evaluate_quality(self):
        """Silhouette Score on the clustering embedding (high-D), not on 2D visualization."""
        if self.labels is None or self.cluster_embeddings is None:
            return None

        mask = self.labels != -1
        if len(np.unique(self.labels[mask])) < 2:
            print("[Validazione] Score non calcolabile: meno di 2 cluster validi.")
            return None

        score = silhouette_score(self.cluster_embeddings[mask], self.labels[mask])
        print(f"=== Validazione ===\nSilhouette Score: {score:.4f} "
              f"(su embedding {self.cluster_embeddings.shape[1]}D, escluso rumore)\n")
        return score

    def run_and_save(self, viz_method="umap", clus_method="hdbscan", n_components=2, **kwargs):
        """Run clustering in high-D, then project to 2D for visualization."""
        print("=== Inizio Fase di Clustering ===")

        # Extract clustering-specific params
        n_clusters = kwargs.pop('n_clusters', 5)
        min_size = kwargs.pop('min_cluster_size', 20)
        min_samples = kwargs.pop('min_samples', 5)
        cluster_selection_method = kwargs.pop('cluster_selection_method', 'eom')
        cluster_dims = kwargs.pop('cluster_dims', 10)
        cluster_n_neighbors = kwargs.pop('cluster_n_neighbors', 15)

        # 1. Cluster in high-dimensional UMAP space
        self.apply_clustering(method=clus_method, n_clusters=n_clusters,
                              min_cluster_size=min_size, min_samples=min_samples,
                              cluster_selection_method=cluster_selection_method,
                              cluster_dims=cluster_dims,
                              cluster_n_neighbors=cluster_n_neighbors)

        # 2. Validate on clustering embedding
        self._evaluate_quality()

        # 3. Project to 2D for visualization
        self.compute_visualization(method=viz_method, n_components=n_components, **kwargs)

        # 4. Identify marker genes
        self.find_marker_genes(k=3)

        # 5. Export
        result_df = pd.DataFrame(index=self.gene_index)
        for i in range(n_components):
            result_df[f'Dim_{i+1}'] = self.embeddings[:, i]
        result_df['Cluster'] = self.labels

        output_path = os.path.join(self.output_dir, "gene_clusters_final.csv")
        result_df.to_csv(output_path)
        print(f"[SAVED] Risultati esportati in: {output_path}")

        return result_df
