"""
DataCleaner 

La classe implementa il processo di raffinamento dei dati biologici (GO/HPO) 

1. BINARIZZAZIONE:
   - Trasformazione di tutti i valori dei dataset (BP, CC, MF, HPO) in formato binario: 
     valore 1 se il gene è annotato con quel termine (valore > 0), altrimenti 0.

2. FILTRAGGIO PER FREQUENZA (filter_terms_by_frequency):
   - Rimozione dei termini (colonne) rari (< 3 geni) o troppo generici (> 20% della popolazione).

3. RIMOZIONE RIDONDANZA (remove_redundant_terms):
   - Calcolo della similarità di Jaccard tra le colonne (threshold >= 0.9).
   - Quando disponibile, si usa la profondità ontologica per scegliere quale termine
     tenere: si preferisce il termine più profondo (più specifico).
   - Rimozione delle righe (geni) rimaste prive di annotazioni dopo il filtraggio.

4. ALLINEAMENTO E SINCRONIZZAZIONE (clean_all):
   - Sincronizzazione dei geni: mantiene i soli geni comuni a tutti i dataset principali (BP, CC, MF, HPO).
   - I file di profondità sono filtrati sulle colonne superstiti nelle matrici principali.
   - I termini GO senza corrispondenza nel file di profondità vengono mantenuti —
     la pesatura a valle (weighting.py) li imputa con la mediana.
"""

import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix

try:
    import torch
    _HAS_TORCH = torch.backends.mps.is_available() or torch.cuda.is_available()
    if _HAS_TORCH:
        _DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
except ImportError:
    _HAS_TORCH = False

class DataCleaner:
    def __init__(self, bp, cc, mf, hpo, d_bp, d_cc, d_mf, output_dir="../data/processed/"):
        self.bp = bp
        self.cc = cc
        self.mf = mf
        self.hpo = hpo
        self.d_bp = d_bp
        self.d_cc = d_cc
        self.d_mf = d_mf
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _binarize(self, df):
        return (df > 0).astype(np.int8)

    def _filter_terms_by_frequency(self, df, min_freq=3, max_prop=0.20):
        n_genes = df.shape[0]
        term_freq = df.sum(axis=0)
        max_freq = max_prop * n_genes
        terms_to_drop = term_freq[(term_freq < min_freq) | (term_freq > max_freq)].index
        return df.drop(columns=terms_to_drop)

    def _remove_redundant_terms(self, df, threshold=0.9, size_tol=0.2, depth=None):
        if _HAS_TORCH:
            return self._remove_redundant_terms_gpu(df, threshold, size_tol, depth)
        return self._remove_redundant_terms_cpu(df, threshold, size_tol, depth)

    def _remove_redundant_terms_gpu(self, df, threshold=0.9, size_tol=0.2, depth=None):
        """GPU-accelerated Jaccard: compute all pairwise intersections via GEMM."""
        cols = df.columns.tolist()
        X = torch.tensor(df.values, dtype=torch.float32, device=_DEVICE)  # (genes, terms)

        col_sums = X.sum(dim=0)  # (terms,)
        intersection = X.T @ X   # (terms, terms) — single GEMM
        union = col_sums.unsqueeze(1) + col_sums.unsqueeze(0) - intersection
        union = union.clamp(min=1.0)
        jaccard = intersection / union

        # Size tolerance mask
        size_ratio = torch.abs(col_sums.unsqueeze(1) - col_sums.unsqueeze(0)) / \
                     torch.max(col_sums.unsqueeze(1), col_sums.unsqueeze(0)).clamp(min=1.0)

        # Find redundant pairs: upper triangle, jaccard >= threshold, size_tol OK
        mask = (jaccard >= threshold) & (size_ratio <= size_tol)
        mask = torch.triu(mask, diagonal=1)

        # Get pairs as CPU lists for greedy dropping
        pairs_i, pairs_j = torch.where(mask)
        pairs_i = pairs_i.cpu().numpy()
        pairs_j = pairs_j.cpu().numpy()

        # Greedy dropping (sequential, but fast — just iterating pairs)
        to_drop = set()
        for pi, pj in zip(pairs_i, pairs_j):
            col_i, col_j = cols[pi], cols[pj]
            if col_i in to_drop or col_j in to_drop:
                continue
            if depth is not None:
                d_i = depth.get(col_i, -1)
                d_j = depth.get(col_j, -1)
                if d_j > d_i:
                    to_drop.add(col_i)
                    continue
            to_drop.add(col_j)

        df_clean = df.drop(columns=list(to_drop))
        df_clean = df_clean.loc[(df_clean != 0).any(axis=1)]
        return df_clean

    def _remove_redundant_terms_cpu(self, df, threshold=0.9, size_tol=0.2, depth=None):
        X = csr_matrix(df.values)
        cols = df.columns.tolist()
        XT = X.T.tocsr()
        
        col_indices = []
        sizes = []  

        for i in range(XT.shape[0]):
            start, end = XT.indptr[i], XT.indptr[i+1]
            col_indices.append(set(XT.indices[start:end]))
            sizes.append(end - start)

        to_drop = set()
        for i in range(len(cols)):
            col_i = cols[i]
            if col_i in to_drop:
                continue
                
            a_idx = col_indices[i]
            a_size = sizes[i]
            
            for j in range(i + 1, len(cols)):
                col_j = cols[j]
                if col_j in to_drop:
                    continue
                    
                b_size = sizes[j]
                
                if abs(a_size - b_size) / max(a_size, b_size) > size_tol:
                    continue
                    
                b_idx = col_indices[j]
                
                intersection = len(a_idx & b_idx)
                union = a_size + b_size - intersection
                
                if union > 0:
                    jaccard = intersection / union
                    if jaccard >= threshold:
                        if depth is not None:
                            d_i = depth.get(col_i, -1)
                            d_j = depth.get(col_j, -1)
                            if d_j > d_i:
                                to_drop.add(col_i)
                                break
                        to_drop.add(col_j)

        df_clean = df.drop(columns=list(to_drop))
        df_clean = df_clean.loc[(df_clean != 0).any(axis=1)]
        return df_clean

    def _depth_series(self, depth_df):
        return depth_df.iloc[0].to_dict()

    def _process_matrix(self, df, depth=None):
        df = self._binarize(df)
        df = self._filter_terms_by_frequency(df)
        df = self._remove_redundant_terms(df, depth=depth)
        return df

    def clean_all(self):
        self.bp = self._process_matrix(self.bp, depth=self._depth_series(self.d_bp))
        self.cc = self._process_matrix(self.cc, depth=self._depth_series(self.d_cc))
        self.mf = self._process_matrix(self.mf, depth=self._depth_series(self.d_mf))
        self.hpo = self._process_matrix(self.hpo)

        common_genes = self.bp.index.intersection(self.cc.index)\
                                    .intersection(self.mf.index)\
                                    .intersection(self.hpo.index)

        self.bp = self.bp.loc[common_genes].sort_index()
        self.cc = self.cc.loc[common_genes].sort_index()
        self.mf = self.mf.loc[common_genes].sort_index()
        self.hpo = self.hpo.loc[common_genes].sort_index()

        self.d_bp = self.d_bp.loc[:, self.d_bp.columns.isin(self.bp.columns)]
        self.d_cc = self.d_cc.loc[:, self.d_cc.columns.isin(self.cc.columns)]
        self.d_mf = self.d_mf.loc[:, self.d_mf.columns.isin(self.mf.columns)]

        self.bp.to_csv(f"{self.output_dir}bp_cleaned.csv")
        self.cc.to_csv(f"{self.output_dir}cc_cleaned.csv")
        self.mf.to_csv(f"{self.output_dir}mf_cleaned.csv")
        self.hpo.to_csv(f"{self.output_dir}hpo_cleaned.csv")
        self.d_bp.to_csv(f"{self.output_dir}d_bp_cleaned.csv")
        self.d_cc.to_csv(f"{self.output_dir}d_cc_cleaned.csv")
        self.d_mf.to_csv(f"{self.output_dir}d_mf_cleaned.csv")

        print(f"[OK] BP data cleaned: {self.bp.shape}")
        print(f"[OK] CC data cleaned: {self.cc.shape}")
        print(f"[OK] MF data cleaned: {self.mf.shape}")
        print(f"[OK] HPO data cleaned: {self.hpo.shape}")
        print(f"[OK] DepthBP data cleaned: {self.d_bp.shape}")
        print(f"[OK] DepthCC data cleaned: {self.d_cc.shape}")
        print(f"[OK] DepthMF data cleaned: {self.d_mf.shape}")

        return self.bp, self.cc, self.mf, self.hpo, self.d_bp, self.d_cc, self.d_mf