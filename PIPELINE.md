# Pipeline — Semantic-Aware Genomic Data Integration

## Dati di partenza

Matrici binarie gene x termine ontologico provenienti da quattro sorgenti:

- **BP** (Biological Process), **CC** (Cellular Component), **MF** (Molecular Function) — Gene Ontology
- **HPO** — Human Phenotype Ontology

Ogni riga è un gene (Entrez ID), ogni colonna è un termine ontologico (GO/HPO).
Si dispone inoltre di file di **profondità** per BP, CC e MF, che indicano la posizione di ciascun termine nella gerarchia dell'ontologia (più profondo = più specifico).

---

## Step 1 — Pulizia e filtraggio (`src/cleaning.py`)

**Obiettivo:** rimuovere rumore, ridondanza e termini non informativi dalle matrici grezze.

- **Binarizzazione:** i valori di propagazione vengono convertiti in 0/1.
- **Filtraggio per frequenza:** si eliminano i termini rari (< 3 geni) perché statisticamente non affidabili, e quelli troppo generici (> 20% dei geni) perché non discriminanti.
- **Rimozione ridondanza (Jaccard ≥ 0.7):** coppie di termini con alta sovrapposizione di geni annotati sono ridondanti — quando disponibile la profondità ontologica, si mantiene il termine più profondo (più specifico) e si rimuove l'antenato più generico.
- **Sincronizzazione:** si mantengono solo i geni presenti in tutti e quattro i dataset, e i file di profondità vengono filtrati sulle colonne superstiti.

**Perché:** matrici grezze con migliaia di termini rumorosi o ridondanti producono distanze e clustering inaffidabili. La pulizia è prerequisito per ogni analisi a valle.
---

## Step 2 — Pesatura depth-aware TF-IDF (`src/weighting.py`)

**Obiettivo:** trasformare le matrici binarie in rappresentazioni continue che riflettano sia la specificità ontologica sia la rarità statistica di ciascun termine.

- **Pesatura per profondità:** ogni annotazione viene moltiplicata per `log(depth + 1)`. I termini più profondi nell'ontologia (più specifici biologicamente) ricevono un peso maggiore. Si usa il logaritmo per comprimere l'intervallo (0–13) ed evitare distorsioni eccessive.
- **Gestione termini senza profondità:** alcuni termini presenti nelle matrici non hanno corrispondenza nel file di profondità (derivano da percorsi di propagazione). Vengono imputati con la **mediana** delle profondità note — un valore neutro che non gonfia né penalizza artificialmente.
- **TF-IDF:** normalizzazione per riga (TF) seguita da pesatura inversa per frequenza documentale (IDF). In questo modo termini condivisi da molti geni vengono ulteriormente penalizzati, anche se hanno superato il filtro di frequenza.
- **HPO** non ha file di profondità, quindi riceve solo TF-IDF senza pesatura depth.

**Perché:** una matrice binaria tratta tutti i termini come ugualmente importanti. La pesatura depth-aware è il principale differenziatore di questa pipeline — sfrutta la struttura gerarchica dell'ontologia per dare più peso ai termini biologicamente più informativi.

---

## Step 3 — Matrici di similarità e SNF (`src/similarity.py`)

**Obiettivo:** costruire una rete gene x gene unificata a partire dalle quattro viste ontologiche.

- **Matrici di affinità:** per ciascuna matrice TF-IDF si costruisce una matrice di affinità gene x gene tramite un kernel esponenziale scalato basato sulla **distanza coseno** (K=20 vicini, mu=0.5). La distanza coseno è la scelta naturale per vettori TF-IDF sparsi ad alta dimensionalità.
- **SNF (Similarity Network Fusion):** algoritmo iterativo (t=20 iterazioni) che fonde le quattro reti di affinità in una rete unica. Ad ogni iterazione le reti vengono rese più simili tra loro, preservando la **struttura locale** — geni che sono vicini in più viste ontologiche vengono avvicinati nella rete fusa.

**Perché:** ogni ontologia cattura un aspetto diverso della funzione genica. La fusione tramite SNF permette di integrare queste informazioni complementari senza forzare un'unica rappresentazione a priori, ottenendo una rete che riflette la convergenza di evidenze multiple.

---

## Step 4 — Clustering e visualizzazione (`src/clustering.py`)

**Obiettivo:** identificare gruppi funzionali di geni nella rete fusa.

- **Strategia dual-UMAP:** si usa UMAP a dimensionalità intermedia (10D) per creare un embedding di clustering con separazione di densità adeguata, e una proiezione 2D separata esclusivamente per la visualizzazione. La rete fusa SNF produce un paesaggio di similarità troppo uniforme per HDBSCAN diretto; UMAP 10D amplifica le differenze di densità locali preservando la struttura complessiva.
- **HDBSCAN (min_cluster_size=20):** clustering density-based sullo spazio 10D. Produce ~60 cluster biologicamente significativi (media ~50 geni) anziché centinaia di micro-cluster.
- **Marker genes:** per ogni cluster si calcola il centroide nello spazio 2D e si identificano i 3 geni più vicini come rappresentanti del modulo.
- **Validazione:** il Silhouette Score viene calcolato sullo spazio di clustering 10D (non sugli embedding 2D), fornendo una metrica più affidabile.

**Perché:** clusterizzare su una proiezione 2D (UMAP) scarta troppa informazione topologica e produce frammentazione eccessiva (~370 cluster). Clusterizzare direttamente sulla matrice di distanza non funziona bene con HDBSCAN perché SNF produce distanze troppo uniformi. Lo spazio intermedio 10D è il compromesso ottimale.

---

## Step 5 — Arricchimento con database esterni (`src/enrichment.py`)

**Obiettivo:** dare significato biologico ai cluster identificati, traducendo gli ID numerici in informazioni interpretabili.

- **MyGene.info (batch API):** traduce i ~4 153 Entrez ID in simboli genici, nomi e descrizioni funzionali. Le query vengono inviate in batch (max 1 000 ID per richiesta) per rispettare i limiti dell'API; non è necessaria una chiave di autenticazione.
- **g:Profiler (REST API):** per ogni cluster non-rumore si esegue un'analisi di enrichment funzionale sulle fonti GO (BP, CC, MF), KEGG e Reactome. I risultati vengono filtrati per significatività (p-value aggiustato < 0.05) e ordinati per rilevanza.
- **Export:** tutti i dati vengono salvati in `data/results/enriched_data.json`, un JSON unificato che contiene per ogni gene: simbolo, nome, sommario, cluster, coordinate UMAP (x/y); e per ogni cluster: dimensione, marker genes, e i top 10 termini di enrichment.

**Perché:** un cluster di numeri non è interpretabile. L'arricchimento con database esterni è il passaggio che trasforma risultati computazionali in conoscenza biologica — permette di capire *cosa* fanno i geni raggruppati insieme e validare la coerenza funzionale dei cluster.

---

## Step 6 — Web App interattiva (`webapp/`)

TODO: Inserire info della webapp

---

## Esecuzione

```bash
# Pipeline completa (da dati grezzi)
python run_pipeline.py

# Solo clustering + enrichment (da rete fusa pre-calcolata)
python run_pipeline.py --from-fused
```