"""
NetworkClusterer

La classe implementa la fase finale di riduzione della dimensionalità e raggruppamento (clustering)
dei geni a partire dalla matrice di affinità fusa (SNF).

1. PREPARAZIONE (Distanza vs Affinità):
   - Conversione della matrice di affinità fusa (dove 1 indica massima similarità) in una 
     matrice di distanza (D = 1 - Affinità) per la compatibilità con gli algoritmi.
   - Utilizzo di np.clip per garantire la robustezza numerica (range 0-1).

2. RIDUZIONE DELLA DIMENSIONALITÀ (reduce_dimensions):
   - Proiezione dello spazio ad alta dimensionalità in uno spazio latente (2D o 3D).
   - Metodi supportati: 
     * PCA: approccio lineare per la cattura della varianza globale.
     * t-SNE: preserva le strutture locali, ideale per la visualizzazione.
     * UMAP: bilancia strutture locali e globali, standard de facto per dati biologici.

3. CLUSTERING E IDENTIFICAZIONE MODULI (apply_clustering):
   - Identificazione di gruppi di geni funzionalmente correlati basati sulla topologia della rete.
   - Metodi supportati:
     * K-Means: partizionamento basato su centroidi (richiede K fissato).
     * Agglomerative: clustering gerarchico basato su linkage.
     * HDBSCAN: basato sulla densità, identifica cluster di forma arbitraria e gestisce il rumore.

4. VALIDAZIONE E EXPORT (_evaluate_quality & run_and_save):
   - Valutazione tramite Silhouette Score, escludendo il rumore per una metrica pulita.
   - Esportazione dei risultati in CSV con coordinate spaziali e label del cluster.
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
        """Inizializza il clusterer e prepara la matrice di distanza."""
        self.fused_df = fused_matrix_df
        self.gene_index = fused_matrix_df.index
        self.output_dir = output_dir
        
        # Conversione Affinità -> Distanza con clip di sicurezza
        affinity = np.clip(self.fused_df.values, 0, 1)
        self.dist_matrix = 1.0 - affinity
        
        self.embeddings = None
        self.labels = None
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def reduce_dimensions(self, method="umap", n_components=2, random_state=42, **kwargs):
        """Riduce la dimensionalità dei dati per clustering e analisi spaziale."""
        print(f"[Riduzione] Metodo: {method.upper()} | Componenti: {n_components}")
        
        if method.lower() == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            self.embeddings = model.fit_transform(self.dist_matrix)
        
        elif method.lower() == "tsne":
            # t-SNE su matrice di distanza precomputata
            model = TSNE(n_components=n_components, metric="precomputed", 
                         init='random', random_state=random_state, **kwargs)
            self.embeddings = model.fit_transform(self.dist_matrix)
            
        elif method.lower() == "umap":
            # UMAP ottimizzato per matrici di distanza
            model = umap.UMAP(n_components=n_components, metric="precomputed",
                              random_state=random_state, **kwargs)
            self.embeddings = model.fit_transform(self.dist_matrix)
        
        else:
            raise ValueError(f"Metodo di riduzione '{method}' non supportato.")
        
        return self.embeddings

    def apply_clustering(self, method="hdbscan", n_clusters=5, min_cluster_size=15,
                         min_samples=None, cluster_selection_method='eom'):
        """Esegue l'algoritmo di clustering sugli embedding generati."""
        if self.embeddings is None:
            raise ValueError("Errore: Eseguire 'reduce_dimensions' prima del clustering.")

        print(f"[Clustering] Metodo: {method.upper()}")
        
        if method.lower() == "kmeans":
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            self.labels = model.fit_predict(self.embeddings)
            
        elif method.lower() == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            self.labels = model.fit_predict(self.embeddings)
            
        elif method.lower() == "hdbscan":
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=cluster_selection_method,
                metric='euclidean'
            )
            self.labels = model.fit_predict(self.embeddings)
            
        else:
            raise ValueError(f"Metodo di clustering '{method}' non supportato.")
            
        n_found = len(set(self.labels) - {-1})
        print(f"[OK] Clustering completato: {n_found} moduli identificati.")
        return self.labels

    def _evaluate_quality(self):
        """Calcola il Silhouette Score escludendo i punti di rumore (-1)."""
        if self.labels is None: return None
        
        # Maschera per escludere il rumore HDBSCAN
        mask = self.labels != -1
        if len(np.unique(self.labels[mask])) < 2:
            print("[Validazione] Score non calcolabile: meno di 2 cluster validi.")
            return None
            
        score = silhouette_score(self.embeddings[mask], self.labels[mask])
        print(f"=== Validazione ===\nSilhouette Score: {score:.4f} (escluso rumore)\n")
        return score

    def run_and_save(self, dim_method="umap", clus_method="hdbscan", n_components=2, **kwargs):
        """Esegue la pipeline completa e salva il file CSV finale."""
        print("=== Inizio Fase di Clustering ===")
        
        # Gestione parametri opzionali per gli algoritmi
        n_clusters = kwargs.pop('n_clusters', 5)
        min_size = kwargs.pop('min_cluster_size', 15)
        min_samples = kwargs.pop('min_samples', None)
        cluster_selection_method = kwargs.pop('cluster_selection_method', 'eom')

        # 1. Riduzione
        self.reduce_dimensions(method=dim_method, n_components=n_components, **kwargs)
        
        # 2. Clustering
        self.apply_clustering(method=clus_method, n_clusters=n_clusters,
                              min_cluster_size=min_size, min_samples=min_samples,
                              cluster_selection_method=cluster_selection_method)
        
        # 3. Validazione
        self._evaluate_quality()
        
        # 4. Export dei risultati
        result_df = pd.DataFrame(index=self.gene_index)
        for i in range(n_components):
            result_df[f'Dim_{i+1}'] = self.embeddings[:, i]
        result_df['Cluster'] = self.labels
        
        output_path = os.path.join(self.output_dir, "gene_clusters_final.csv")
        result_df.to_csv(output_path)
        print(f"[SAVED] Risultati esportati in: {output_path}")
        
        return result_df