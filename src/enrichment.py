"""
GeneEnricher

La classe implementa la fase di arricchimento funzionale dei cluster genici,
integrando annotazioni da MyGene.info e analisi di enrichment tramite g:Profiler.

1. ANNOTAZIONE GENICA (fetch_gene_annotations):
   - Interroga l'API batch di MyGene.info per tradurre gli Entrez ID in simboli genici,
     nomi e descrizioni funzionali.
   - Gestisce i geni non trovati imputando il simbolo con l'Entrez ID stesso.
   - Elaborazione a batch di 1000 ID per rispettare i limiti dell'API.

2. ENRICHMENT DEI CLUSTER (fetch_cluster_enrichment):
   - Per ogni cluster non-rumore (cluster != -1), esegue un'analisi di over-representation
     tramite g:Profiler sulle ontologie GO (BP, CC, MF), KEGG e Reactome.
   - Conserva solo i top 10 termini più significativi per cluster (ordinati per p-value).
   - Cluster con meno di 3 geni vengono saltati (enrichment non significativo).

3. COSTRUZIONE RETE k-NN (build_knn_edges):
   - Costruisce una edge list k-NN dalla matrice di similarità fusa (SNF)
     per la visualizzazione della rete genica.
   - Per ogni gene vengono selezionati i K vicini più simili, producendo
     un grafo sparso adatto alla visualizzazione.

4. ORCHESTRAZIONE (run):
   - Esegue la pipeline completa: annotazione, enrichment, k-NN, salvataggio JSON.
"""

import pandas as pd
import numpy as np
import os
import json
import time
import requests


class GeneEnricher:
    def __init__(self, clusters_path="data/results/gene_clusters_final.csv",
                 fused_path="data/processed/fused_network.csv",
                 output_dir="data/results/",
                 marker_genes=None):
        self.clusters_path = clusters_path
        self.fused_path = fused_path
        self.output_dir = output_dir
        self.marker_genes = marker_genes or {}

        self.clusters_df = pd.read_csv(clusters_path, index_col=0)
        self.clusters_df.index = self.clusters_df.index.astype(str)

        self.gene_annotations = {}
        self.cluster_enrichment = {}

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[OK] Cluster caricati: {self.clusters_df.shape[0]} geni, "
              f"{self.clusters_df['Cluster'].nunique()} cluster unici")

    def fetch_gene_annotations(self):
        """
        Interroga MyGene.info in batch per ottenere simbolo, nome e sommario
        di ogni gene (Entrez ID). Salva in self.gene_annotations.
        """
        print("\n=== Annotazione Genica (MyGene.info) ===\n")

        all_ids = list(self.clusters_df.index)
        batch_size = 1000
        batches = [all_ids[i:i + batch_size] for i in range(0, len(all_ids), batch_size)]

        print(f"[INFO] {len(all_ids)} geni da annotare in {len(batches)} batch")

        found = 0
        not_found = 0

        for i, batch in enumerate(batches):
            ids_str = ",".join(batch)
            response = requests.post(
                "https://mygene.info/v3/gene",
                data={"ids": ids_str, "fields": "symbol,name,summary", "species": "human"},
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            results = response.json()

            for entry in results:
                gene_id = str(entry.get("_id", entry.get("query", "")))
                if entry.get("notfound", False):
                    not_found += 1
                    self.gene_annotations[gene_id] = {
                        "symbol": gene_id,
                        "name": "N/A",
                        "summary": "N/A"
                    }
                else:
                    found += 1
                    self.gene_annotations[gene_id] = {
                        "symbol": entry.get("symbol", gene_id),
                        "name": entry.get("name", "N/A"),
                        "summary": entry.get("summary", "N/A")
                    }

            print(f"  Batch {i + 1}/{len(batches)} completato")

        print(f"\n[OK] Annotazione completata: {found} trovati, {not_found} non trovati")
        return self.gene_annotations

    def fetch_cluster_enrichment(self):
        """
        Per ogni cluster non-rumore, esegue enrichment tramite g:Profiler.
        Conserva i top 10 termini per cluster, ordinati per p-value.
        """
        print("\n=== Enrichment dei Cluster (g:Profiler) ===\n")

        cluster_labels = self.clusters_df["Cluster"]
        unique_clusters = sorted([c for c in cluster_labels.unique() if c != -1])

        print(f"[INFO] {len(unique_clusters)} cluster da analizzare (escluso rumore)")

        skipped = 0
        failed = 0
        enriched = 0

        for idx, cluster_id in enumerate(unique_clusters):
            gene_ids = list(cluster_labels[cluster_labels == cluster_id].index)

            if len(gene_ids) < 3:
                skipped += 1
                continue

            # Converti Entrez ID in simboli genici per g:Profiler
            symbols = []
            for gid in gene_ids:
                ann = self.gene_annotations.get(gid, {})
                symbols.append(ann.get("symbol", gid))

            try:
                response = requests.post(
                    "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
                    json={
                        "organism": "hsapiens",
                        "query": symbols,
                        "sources": ["GO:BP", "GO:CC", "GO:MF", "KEGG", "REAC"],
                        "user_threshold": 0.05,
                        "no_evidences": True,
                        "all_results": False
                    }
                )
                data = response.json()
                results = data.get("result", [])

                terms = []
                for r in results:
                    terms.append({
                        "source": r.get("source", ""),
                        "term_id": r.get("native", ""),
                        "term_name": r.get("name", ""),
                        "p_value": r.get("p_value", 1.0),
                        "gene_count": r.get("intersection_size", 0),
                        "term_size": r.get("term_size", 0)
                    })

                # Top 10 per p-value
                terms.sort(key=lambda x: x["p_value"])
                self.cluster_enrichment[str(cluster_id)] = terms[:10]
                enriched += 1

            except Exception as e:
                failed += 1
                self.cluster_enrichment[str(cluster_id)] = []
                print(f"  [ERRORE] Cluster {cluster_id}: {e}")

            if (idx + 1) % 50 == 0:
                print(f"  Progresso: {idx + 1}/{len(unique_clusters)} cluster elaborati")

            time.sleep(0.5)

        print(f"\n[OK] Enrichment completato: {enriched} arricchiti, "
              f"{skipped} saltati (<3 geni), {failed} falliti")
        return self.cluster_enrichment

    def build_knn_edges(self, k=15):
        """
        Costruisce una edge list k-NN dalla matrice di similarità fusa
        per la visualizzazione della rete genica.
        """
        print("\n=== Costruzione Rete k-NN ===\n")

        fused_df = pd.read_csv(self.fused_path, index_col=0)
        fused_df.index = fused_df.index.astype(str)
        fused_df.columns = fused_df.columns.astype(str)

        print(f"[INFO] Matrice fusa caricata: {fused_df.shape}")

        edges = []
        genes = fused_df.index.tolist()

        for gene in genes:
            row = fused_df.loc[gene].drop(gene)
            top_k = row.nlargest(k)
            for neighbor, weight in top_k.items():
                edges.append({
                    "source": gene,
                    "target": neighbor,
                    "weight": weight
                })

        edges_df = pd.DataFrame(edges)
        output_path = os.path.join(self.output_dir, "knn_edges.csv")
        edges_df.to_csv(output_path, index=False)

        print(f"[OK] {len(edges_df)} archi generati (k={k})")
        print(f"[SAVED] Edge list esportata in: {output_path}")
        return edges_df

    def run(self, k=15):
        """Esegue la pipeline completa di arricchimento funzionale."""
        print("=== Enrichment Pipeline ===\n")

        # 1. Annotazione genica
        self.fetch_gene_annotations()

        # 2. Enrichment dei cluster
        self.fetch_cluster_enrichment()

        # 3. Rete k-NN
        self.build_knn_edges(k=k)

        # 4. Assemblaggio e salvataggio JSON
        print("\n=== Salvataggio Risultati ===\n")

        # Build per-gene entries with coordinates + cluster + annotation
        genes_output = {}
        for gene_id in self.clusters_df.index:
            row = self.clusters_df.loc[gene_id]
            ann = self.gene_annotations.get(gene_id, {})
            gene_entry = {
                "symbol": ann.get("symbol", gene_id),
                "name": ann.get("name", "N/A"),
                "summary": ann.get("summary", "N/A"),
                "cluster": int(row["Cluster"]),
                "x": float(row["Dim_1"]),
                "y": float(row["Dim_2"]),
            }
            genes_output[gene_id] = gene_entry

        # Build per-cluster entries with enrichment + markers
        cluster_labels = self.clusters_df["Cluster"]
        clusters_output = {}
        for cluster_id in sorted(cluster_labels.unique()):
            if cluster_id == -1:
                continue
            cid = str(cluster_id)
            size = int((cluster_labels == cluster_id).sum())
            markers = self.marker_genes.get(int(cluster_id),
                                            self.marker_genes.get(cid, []))
            clusters_output[cid] = {
                "size": size,
                "markers": markers,
                "enrichment": self.cluster_enrichment.get(cid, [])
            }

        output = {
            "genes": genes_output,
            "clusters": clusters_output
        }

        output_path = os.path.join(self.output_dir, "enriched_data.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[SAVED] Dati arricchiti esportati in: {output_path}")
        print(f"\n=== Pipeline Completata ===")
        print(f"  Geni annotati:    {len(genes_output)}")
        print(f"  Cluster arricchiti: {len(clusters_output)}")
        print(f"  Archi k-NN:       {os.path.join(self.output_dir, 'knn_edges.csv')}")
