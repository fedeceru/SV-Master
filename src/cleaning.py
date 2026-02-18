"""
DESCRIZIONE DELLA CLASSE DATACLEANER (VERSIONE CORRETTA)

La classe DataCleaner implementa il processo di raffinamento dei dati biologici (GO/HPO) 
seguendo esattamente i passaggi del notebook Final_WorkFlow, estendendoli a tutti i dataset
caricati dal DataLoader:

1. BINARIZZAZIONE:
   - Trasformazione di tutti i valori dei dataset (BP, CC, MF, HPO) in formato binario: 
     valore 1 se il gene è annotato con quel termine (valore > 0), altrimenti 0.

2. FILTRAGGIO PER FREQUENZA (filter_terms_by_frequency):
   - Rimozione dei termini (colonne) rari (< 3 geni) o troppo generici (> 20% della popolazione).

3. RIMOZIONE RIDONDANZA (remove_redundant_terms):
   - Calcolo della similarità di Jaccard tra le colonne (threshold >= 0.9).
   - Rimozione delle righe (geni) rimaste prive di annotazioni dopo il filtraggio.

4. ALLINEAMENTO E SINCRONIZZAZIONE (clean_all):
   - Sincronizzazione dei geni: mantiene i soli geni comuni a tutti i dataset principali (BP, CC, MF, HPO).
   - SINCRONIZZAZIONE DEPTH (CORRETTA): I file di profondità sono filtrati sulle COLONNE. 
     Vengono mantenute solo le colonne che corrispondono ai termini GO superstiti nelle matrici principali.
"""

import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix

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
        return df.applymap(lambda x: 1 if x > 0 else 0)

    def _filter_terms_by_frequency(self, df, min_freq=3, max_prop=0.20):
        n_genes = df.shape[0]
        term_freq = df.sum(axis=0)
        max_freq = max_prop * n_genes
        terms_to_drop = term_freq[(term_freq < min_freq) | (term_freq > max_freq)].index
        return df.drop(columns=terms_to_drop)

    def _remove_redundant_terms(self, df, threshold=0.9, size_tol=0.2):
        X = csr_matrix(df.values)
        cols = df.columns.tolist()
        XT = X.T.tocsr()
        
        col_indices = []
        for i in range(XT.shape[0]):
            start, end = XT.indptr[i], XT.indptr[i+1]
            col_indices.append(XT.indices[start:end])

        to_drop = set()
        for i in range(len(cols)):
            if cols[i] in to_drop:
                continue
            a_idx, a_size = col_indices[i], len(col_indices[i])
            for j in range(i + 1, len(cols)):
                if cols[j] in to_drop:
                    continue
                b_idx, b_size = col_indices[j], len(col_indices[j])
                if abs(a_size - b_size) / max(a_size, b_size) > size_tol:
                    continue
                intersection = len(np.intersect1d(a_idx, b_idx, assume_unique=True))
                union = a_size + b_size - intersection
                jaccard = intersection / union if union > 0 else 0
                if jaccard >= threshold:
                    to_drop.add(cols[j])

        df_clean = df.drop(columns=list(to_drop))
        df_clean = df_clean.loc[(df_clean != 0).any(axis=1)]
        return df_clean

    def _process_matrix(self, df):
        df = self._binarize(df)
        df = self._filter_terms_by_frequency(df)
        df = self._remove_redundant_terms(df)
        return df

    def clean_all(self):
        self.bp = self._process_matrix(self.bp)
        self.cc = self._process_matrix(self.cc)
        self.mf = self._process_matrix(self.mf)
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