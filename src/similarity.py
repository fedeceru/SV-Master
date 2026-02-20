"""
NetworkBuilder

Costruisce una rete gene x gene unificata a partire dalle quattro viste ontologiche (BP, CC, MF, HPO).

1. MATRICI DI AFFINITÀ (build_affinity):
   - Per ciascuna matrice TF-IDF (gene x termine) si calcola una matrice di affinità gene x gene
     tramite un kernel esponenziale scalato basato sulla distanza coseno.
   - Il parametro K controlla il vicinato locale usato per scalare il kernel.

2. SNF — SIMILARITY NETWORK FUSION (fuse):
   - Le quattro matrici di affinità vengono fuse iterativamente in una rete unica.
   - Ad ogni iterazione le reti vengono rese più simili tra loro, preservando la struttura locale.
   - Il risultato è una matrice gene x gene che riflette la convergenza di evidenze
     da tutte e quattro le viste ontologiche.
"""

import numpy as np
import pandas as pd
import snf
import os


class NetworkBuilder:
    def __init__(self, bp, cc, mf, hpo, output_dir="../data/processed/"):
        self.views = {"BP": bp, "CC": cc, "MF": mf, "HPO": hpo}
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def build_and_fuse(self, K=20, mu=0.5, t=20):
        """
        Build per-view affinity matrices from TF-IDF features using cosine
        distance, then fuse them with SNF.

        Returns the fused (N, N) similarity matrix as a DataFrame.
        """
        print("=== Similarity Network Fusion ===\n")

        gene_index = list(self.views.values())[0].index

        arrays = []
        for name, df in self.views.items():
            arr = df.values.astype(np.float64)
            arrays.append(arr)
            print(f"[{name}] Feature matrix: {arr.shape}")

        print(f"\n[SNF] Building affinity matrices (K={K}, mu={mu}, metric=cosine)...")
        affinities = snf.make_affinity(
            *arrays, metric="cosine", K=K, mu=mu, normalize=True
        )

        print(f"[SNF] Fusing {len(affinities)} networks (t={t} iterations)...")
        fused = snf.snf(affinities, K=K, t=t)

        fused_df = pd.DataFrame(fused, index=gene_index, columns=gene_index)

        fused_df.to_csv(f"{self.output_dir}fused_network.csv")
        print(f"[OK] Fused network: {fused_df.shape}")
        print(f"[SAVED] {self.output_dir}fused_network.csv")

        return fused_df
