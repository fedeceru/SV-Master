"""
DataWeighter

La classe implementa la pesatura depth-aware TF-IDF sulle matrici binarie gene-termine (GO/HPO).

1. ALLINEAMENTO DEPTH (_get_depth_weights):
   - I file di profondità potrebbero non coprire tutti i termini presenti nella matrice di annotazione.
   - Termini presenti in entrambi: si usa direttamente la profondità nell'ontologia.
   - Termini mancanti dal file di profondità: imputati con la **mediana** delle profondità note.
     La mediana è un default neutro — evita di gonfiare artificialmente la specificità (profondità alta)
     o di trattare gli sconosciuti come generici (profondità 0/1).
     Questi termini mancanti derivano tipicamente da percorsi di propagazione ("propT")
     non enumerati nel file di profondità.
   - HPO non ha file di profondità, quindi riceve solo TF-IDF senza pesatura depth.

2. PESATURA PER PROFONDITÀ (_apply_depth_weighting):
   - Ogni annotazione binaria viene moltiplicata per log(depth + 1), così i termini
     più profondi (più specifici) contribuiscono di più rispetto a quelli superficiali (generici).
   - Si usa log(1+d) anziché la profondità grezza per comprimere l'intervallo (0–13)
     ed evitare di penalizzare eccessivamente i termini superficiali o amplificare troppo quelli profondi.

3. TF-IDF (_apply_tfidf):
   - TF (Term Frequency): normalizzazione per riga — il peso di ciascun termine viene diviso
     per il peso totale del gene, producendo una distribuzione per gene.
   - IDF (Inverse Document Frequency): log(N / df_t), dove N è il numero di geni e df_t è
     il numero di geni annotati con il termine t. Penalizza i termini condivisi da molti geni.
   - Peso finale = TF * IDF, un punteggio di importanza per gene e per termine che tiene conto
     sia della specificità ontologica (depth) sia della rarità statistica (IDF).
"""

import pandas as pd
import numpy as np
import os


class DataWeighter:
    def __init__(self, bp, cc, mf, hpo, d_bp, d_cc, d_mf, output_dir="../data/processed/"):
        self.bp = bp.copy()
        self.cc = cc.copy()
        self.mf = mf.copy()
        self.hpo = hpo.copy()
        self.d_bp = d_bp
        self.d_cc = d_cc
        self.d_mf = d_mf
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_depth_weights(self, df, depth_df):
        """
        Build a depth-weight vector aligned to the columns of df.

        For each term in df.columns:
          - If present in depth_df -> weight = log(depth + 1)
          - If missing from depth_df -> weight = log(median_depth + 1)

        Returns a pd.Series indexed by df.columns.
        """
        depth_values = depth_df.iloc[0]  # single-row DataFrame -> Series

        # Align: reindex to the annotation matrix columns
        aligned = depth_values.reindex(df.columns)

        n_missing = aligned.isna().sum()
        n_total = len(aligned)

        if n_missing > 0:
            median_depth = depth_values.median()
            aligned = aligned.fillna(median_depth)
            print(f"  [DEPTH] {n_missing}/{n_total} terms missing from depth file -> imputed with median depth ({median_depth:.1f})")
        else:
            print(f"  [DEPTH] All {n_total} terms matched in depth file")

        weights = np.log(aligned + 1)
        return weights

    def _apply_depth_weighting(self, df, depth_df):
        """
        Multiply each term's binary annotation by its depth weight.
        Result: genes annotated with deeper terms get higher raw values.
        """
        weights = self._get_depth_weights(df, depth_df)
        return df.mul(weights, axis=1)

    def _apply_tfidf(self, df):
        """
        Compute TF-IDF on a (possibly depth-weighted) genexterm matrix.

        TF: normalize each gene (row) by its total weight, so that genes with
            many annotations aren't artificially inflated.
        IDF: log(N / df_t) — penalizes terms that appear across many genes.
        """
        n_genes = df.shape[0]

        # TF: row-wise normalization
        row_sums = df.sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # safety: avoid division by zero
        tf = df.div(row_sums, axis=0)

        # IDF: based on presence (non-zero entries), not on the weighted values
        doc_freq = (df > 0).sum(axis=0)
        doc_freq = doc_freq.replace(0, 1)  # safety
        idf = np.log(n_genes / doc_freq)

        tfidf = tf.mul(idf, axis=1)
        return tfidf

    def transform_all(self):
        """
        Run the full depth-weighted TF-IDF pipeline on all four matrices.

        GO matrices (BP, CC, MF) get depth weighting + TF-IDF.
        HPO matrix gets plain TF-IDF (no depth file available).
        """
        print("=== Depth-Weighted TF-IDF ===\n")

        # --- BP ---
        print("[BP] Applying depth weighting...")
        self.bp = self._apply_depth_weighting(self.bp, self.d_bp)
        print("[BP] Applying TF-IDF...")
        self.bp = self._apply_tfidf(self.bp)
        print(f"[OK] BP transformed: {self.bp.shape}\n")

        # --- CC ---
        print("[CC] Applying depth weighting...")
        self.cc = self._apply_depth_weighting(self.cc, self.d_cc)
        print("[CC] Applying TF-IDF...")
        self.cc = self._apply_tfidf(self.cc)
        print(f"[OK] CC transformed: {self.cc.shape}\n")

        # --- MF ---
        print("[MF] Applying depth weighting...")
        self.mf = self._apply_depth_weighting(self.mf, self.d_mf)
        print("[MF] Applying TF-IDF...")
        self.mf = self._apply_tfidf(self.mf)
        print(f"[OK] MF transformed: {self.mf.shape}\n")

        # --- HPO (no depth file -> plain TF-IDF) ---
        print("[HPO] No depth file available — applying plain TF-IDF...")
        self.hpo = self._apply_tfidf(self.hpo)
        print(f"[OK] HPO transformed: {self.hpo.shape}\n")

        # Save
        self.bp.to_csv(f"{self.output_dir}bp_tfidf.csv")
        self.cc.to_csv(f"{self.output_dir}cc_tfidf.csv")
        self.mf.to_csv(f"{self.output_dir}mf_tfidf.csv")
        self.hpo.to_csv(f"{self.output_dir}hpo_tfidf.csv")
        print(f"[SAVED] All TF-IDF matrices written to {self.output_dir}")

        return self.bp, self.cc, self.mf, self.hpo
