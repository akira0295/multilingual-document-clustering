# Multilingual Document Clustering: Decision and Change Log

**Project:** Multilingual Document Clustering using NLP (English, Hindi, Telugu)
**Course:** PROJ2999 Capstone, GITAM School of CSE, Hyderabad
**Team:** Vaishnavi G, Akira Nandan Reddy, Harsha Vardhan, Dhruv Chowdary (Guide: Jyothi B)
**Base paper:** Katta & Dhanalakshmi (2025), *Multilingual Text Document Clustering and Classification*, IJRASET
**Log last updated:** 29 October 2026

Status markers used below: **Decided** (made and applied), **Proposed** (recommended, awaiting team confirmation), **Pending** (not yet done).

---

## 1. Where the project stands (summary)

| Area | Status |
|---|---|
| Literature survey, problem statement, objectives | Done |
| Environment setup (Python, libraries, model test) | Done |
| Dataset: pilot (750 docs) and final (2,250 docs) built and balanced | Done |
| Language detection and preprocessing | Done (100% label agreement) |
| PDF text extraction | Done (tested on one PDF per language) |
| Embeddings (MiniLM and LaBSE) | Done |
| Joint clustering and evaluation (Silhouette, ARI, NMI) | Done: KMeans on both models; Agglomerative (average and complete linkage) on LaBSE |
| Telugu problem diagnosed | Done: model quality, not truncation or preprocessing |
| UMAP visualization | Done (`results/umap_labse.png`) |
| Cross-language semantic search | Built and evaluated; same-language bias found, merge option pending |
| Final model choice confirmed by team | Proposed (LaBSE) |
| Design diagrams (DFD L0/L1, use-case, activity) | Pending |
| Written methodology and justification tables | In progress |

**Headline result:** with the base paper's model (MiniLM), Telugu documents did not cluster by topic (Telugu ARI 0.13). Switching to LaBSE raised Telugu ARI to about 0.61 and overall ARI from 0.29 to about 0.47 to 0.49 (KMeans), with clusters mixing all three languages by topic.

---

## 2. Chronological decision log

### Phase 1: Planning and literature

| # | Decision | Alternatives considered | Justification | Status |
|---|---|---|---|---|
| D1 | Use Katta & Dhanalakshmi (2025) as the base paper | Schneider 2024 (LLM clustering), Mechergui 2024 (MultiHLDA + semantic graphs), Sun 2024 (graph + contrastive), Pavlyshenko 2026 (English-Ukrainian) | It directly matches the scope: English, Hindi, Telugu documents, multilingual embeddings, clustering and semantic search. The other papers are heavier (LLMs, semantic graphs, multi-view contrastive learning) or cover other language pairs. | Decided |
| D2 | Define the project as an extension of the base paper, not a reproduction | Reproduce the paper as is | The base paper evaluated only 15 documents (5 per language) with per-language silhouette scores. The project adds a larger balanced labeled dataset, joint clustering, multiple metrics (Silhouette, ARI, NMI, cosine similarity) and semantic search. | Decided |
| D3 | Cluster all three languages together in one shared space | Cluster each language separately | A shared space is the project's central goal and the problem statement's point: documents on the same topic should group across languages. | Decided |
| D4 | Keep the original pipeline shape: language detection, preprocessing, multilingual MiniLM embeddings, Agglomerative Clustering with cosine distance, UMAP, semantic search | Replace with more complex architectures | Keeps the pipeline simple and practical (research gap 3) and keeps results comparable with the base paper. | Decided (later modified, see D14 to D17) |

### Phase 2: Environment

| # | Decision | Justification | Status |
|---|---|---|---|
| D5 | Python with pandas, numpy, scikit-learn, sentence-transformers, torch, langdetect, nltk, matplotlib, seaborn, umap-learn, pymupdf | Covers data handling, embeddings, clustering, language detection, visualization and PDF extraction. A model load test passed with 384-dimensional embeddings. | Decided |
| D6 | Ignore the urllib3/LibreSSL warning | It is a warning, not an error, and imports succeeded. Avoids unnecessary package downgrades. | Decided |
| D7 | Switch `import fitz` to `import pymupdf` | `fitz` import is deprecated. Same library, no warning. | Decided |
| D8 | Run all scripts from the project root as `python3 src/<script>.py` | An earlier run failed because files sat in `src/` while the command ran from the root. A consistent root-relative convention keeps data paths (`data/processed/...`) working. | Decided |

### Phase 3: Dataset selection

| # | Decision | Alternatives / what happened | Justification | Status |
|---|---|---|---|---|
| D9 | Use an existing public dataset instead of hand-building a 45-document spreadsheet | Original plan was a 45-document pilot (15 per language) collected manually | Saves collection time and gives far more data with labels. | Decided (changed from original plan) |
| D10 | Tried IndicCorpV2 (Hugging Face) first | Hugging Face `ai4bharat/IndicCorpV2` | Large Indic corpus. Read with streaming, because the full corpus is about 275 GB. | Demoted to sanity check |
| D11 | **Main dataset: L3Cube-IndicNews (LPC version, test split)** | IndicCorpV2 (no English, no labels), AG News (English only, 3 topics), `telugu_news` (could not load: script-only repository, no data files) | English, Hindi and Telugu share a file format and labeling scheme, and five topics exist in all three languages. Paragraph lengths are comparable across languages. License is CC BY-NC-SA 4.0, fine for research. | Decided |
| D12 | Use five shared topics: business, education, entertainment, sports, technology | Use all source labels (12/11/10 per language) | Only these five exist in all three languages, so the dataset is balanced and comparable. Label mapping: Hindi `khel` becomes sports, `technology-news` becomes technology, Telugu `tech` becomes technology. | Decided |
| D13 | Cleaning rules: drop missing text/labels, drop documents under 200 characters, drop exact duplicates, fixed random seed 42 | Keep short documents | Short texts carry too little meaning for semantic comparison. The fixed seed keeps the sample reproducible. | Decided |
| D13a | Keep IndicCorpV2 (400 passages: 200 Hindi, 200 Telugu) and AG News only as side artifacts | Delete them | IndicCorpV2 served as an embedding sanity check and Hindi-to-Telugu nearest-neighbor test. AG News (English, 300 docs) was superseded because L3Cube English has longer paragraphs matching Hindi and Telugu. Neither is part of the evaluated dataset. | Decided |

### Phase 4: Pilot (750 documents)

| # | Decision / change | Justification and result | Status |
|---|---|---|---|
| D13b | Pilot size: 50 documents per topic per language = 750 | Large enough to debug the pipeline. The pilot is for debugging and not for strong claims. | Decided |
| D13c | Embedding for all pilot steps: `paraphrase-multilingual-MiniLM-L12-v2`, normalized (384-d) | The base paper's model. Normalization makes dot product equal cosine similarity. | Decided |
| D13d | Topic-centroid cosine analysis before clustering | Same-topic centroids across languages averaged 0.579 versus 0.340 for different topics. English-Hindi was strongest (0.784), Telugu pairs weaker (about 0.47 to 0.49). This was the first sign of a Telugu gap. | Done |
| D13e | First pilot clustering: Agglomerative, 5 clusters, cosine, **average** linkage | Ward linkage cannot use cosine distance. Result: Silhouette 0.133, ARI 0.104, NMI 0.177. One cluster held 481 of 750 documents, including 245 of 250 Telugu documents. Telugu ARI was 0.0002. | Done |

### Phase 5: Script audit and cleanup

| # | Decision | Justification | Status |
|---|---|---|---|
| D13f | Keep `build_dataset.py`, `pilot_similarity.py`, `embed_sample.py`, `sample_data.py`. Delete `load_dataset.py`. Archive `build_english.py`. Delete or archive `build_telugu.py`. | The last three are leftovers or depend on files that were never created. | Decided (cleanup to do) |
| D13g | Add `os.makedirs(..., exist_ok=True)` for data folders | The dataset script fails if folders are missing. | Decided |

### Phase 6: Section A (Dataset)

| # | Decision | Justification | Status |
|---|---|---|---|
| D14 | Accepted input formats: `.txt`, `.pdf`, `.csv` | `.pdf` matches the base paper, `.csv` matches the working pipeline, `.txt` is the simplest single-document input. | Decided |
| D15 | PDF extraction with PyMuPDF. Tested on one browser-generated PDF per language: English, Hindi and Telugu all extracted as readable Unicode text. | Fast, offline, installed already. Known limits: scanned PDFs need OCR (out of scope), and some Indic PDFs duplicate combining marks. | Decided |
| D16 | **Final dataset: 2,250 documents, 150 per topic per language** (`final_dataset.csv`) | Availability counts after cleaning (smallest cell is Telugu technology, 168) capped the balanced size. N=150 gives three times the pilot, leaves a margin in the smallest cell, and embeds quickly on a laptop. The pilot files stay untouched so pilot results remain reproducible. | Decided |

Availability after cleaning (source test split):

| Topic | English | Hindi | Telugu |
|---|---|---|---|
| business | 428 | 367 | 237 |
| education | 394 | 342 | 189 |
| entertainment | 323 | 355 | 239 |
| sports | 321 | 346 | 283 |
| technology | 299 | 354 | 168 |

### Phase 7: Section B (Technologies)

| # | Decision | Justification | Status |
|---|---|---|---|
| D17 | Preprocessing: Unicode NFC normalization, URL and email removal, whitespace cleanup, extra combining-mark cleanup for PDF input only. Language detection with langdetect (seed fixed). **No** lowercasing, stopword removal or stemming. | Transformer embeddings use natural word order and function words. Indic stopword lists can remove meaningful words. Result: langdetect agreed with the language label on 100% of 2,250 documents, and no document became empty. The agreement is a label verification, not a hard test, since the three scripts are very distinct. | Decided |
| D18 | Extract, clean, then detect language, combined in one `load_document` function | Keeps the order consistent for any input file. Test PDFs were detected correctly, and doubled marks (for example in Hindi and Telugu words) were corrected. | Decided |
| D19 | Leave NLTK out of the pipeline and the report's library list | Nothing in the pipeline needs it, and its sentence splitter is built for English. Claiming an unused library would be inaccurate. | Proposed |
| D20 | Replace the Indic NLP Toolkit with NFC normalization plus the combining-mark cleanup | Fewer dependencies and a simpler pipeline. The cleanup targets a problem actually observed. An optional A/B test with the Toolkit could be reported later. | Proposed |
| D21 | Similarity measure: cosine | Standard for sentence embeddings and ignores vector length. | Decided |
| D22 | Metrics: Silhouette (cosine), ARI, NMI, plus per-language ARI and cluster-versus-language crosstabs | Silhouette needs no labels. ARI and NMI compare clusters with topic labels. Per-language ARI exposed the Telugu problem. Labels are used only for evaluation, never to form clusters. | Decided |
| D23 | Visualization: UMAP, colored once by topic and once by language | Used in the base paper. The plot should show languages mixed within topics. | Decided (not yet built) |

### Phase 8: Telugu problem and experiments

| # | Finding or decision | Evidence | Status |
|---|---|---|---|
| D24 | **Diagnosis step 1: token truncation.** MiniLM's `max_seq_length` is 128 tokens. | 95.6% of English, 87.2% of Hindi and 95.7% of Telugu documents exceed it. Median lengths: 244, 192.5 and 246.5 tokens. Telugu is not truncated more than English, so truncation cannot explain the Telugu gap alone. | Done |
| D25 | **Add chunking** (split each document into token windows, embed each, average the vectors). The first chunker split on sentence punctuation and produced only 1.14 chunks per document, so it did not really test chunking. It was replaced with a token-window chunker (2.57 chunks per document). | With MiniLM and KMeans, chunking raised ARI from 0.264 to 0.294 and NMI from 0.342 to 0.380. Truncation mattered, though modestly. | Decided |
| D26 | **Add KMeans (k=5) as a comparison to Agglomerative.** | With MiniLM, average-linkage Agglomerative gave ARI about 0.11 versus about 0.26 to 0.29 for KMeans. Average linkage tended to produce one giant cluster. | Decided |
| D27 | **Compare MiniLM with LaBSE** (same pipeline, KMeans, chunked, three seeds). | See the results table below. LaBSE raised Telugu ARI from 0.13 to about 0.61 and clustered by topic across languages. MiniLM clusters followed language (540 of 750 Telugu documents in one cluster holding every topic). | Done |
| D28 | **Adopt LaBSE as the primary embedding model and keep MiniLM as the base-paper baseline.** | Same pipeline, only the model changed, and the improvement is large. | Proposed (confirm with team and guide) |
| D29 | **Main reported clustering: KMeans (k=5) on LaBSE embeddings. Agglomerative with cosine distance and complete linkage is reported as the base-paper-style comparison. Average linkage is reported as a failed setting.** | On the same LaBSE embeddings, average-linkage Agglomerative collapsed into essentially one cluster (ARI 0.000, NMI 0.006), complete linkage gave ARI 0.318, and KMeans gave ARI 0.465 to 0.491. Average linkage also failed with MiniLM, so the cause is the linkage, not the model. | Decided |
| D31 | **Semantic search: embed the query with LaBSE and rank all 2,250 documents by cosine similarity (`src/semantic_search.py`).** Evaluated with `src/eval_search.py` using documents as queries and topic labels as relevance. | Simple, matches the base paper's search step, and reuses the saved embeddings. Document-as-query evaluation gives numbers instead of judging from a few example queries. | Decided |
| D32 | **Add a per-language retrieval option: take the top results from each language separately and merge them.** | Plain ranking returns mostly one language (section 3.6), so a user searching in one language rarely sees the others. Per-language retrieval guarantees results from all three. | Proposed (not yet built) |
| D30 | **Treat the English result as label overlap, to be verified.** English ARI under LaBSE is about 0.29 to 0.31 because business and technology documents merge into one cluster (see section 3.4). | English technology news often overlaps with business news (company earnings, product launches). A sample of the English technology documents in the mixed cluster should be read before this is stated as fact. | Hypothesis |

---

## 3. Experimental results

### 3.1 Pilot (750 documents, MiniLM, Agglomerative average linkage)

| Silhouette | ARI | NMI | ARI English | ARI Hindi | ARI Telugu |
|---|---|---|---|---|---|
| 0.133 | 0.104 | 0.177 | 0.183 | 0.317 | 0.0002 |

### 3.2 Final dataset (2,250 documents)

| Setup | Silhouette | ARI | NMI | ARI En | ARI Hi | ARI Te |
|---|---|---|---|---|---|---|
| MiniLM, truncated, Agglomerative | 0.133 | 0.113 | 0.180 | 0.210 | 0.338 | 0.000 |
| MiniLM, truncated, KMeans | 0.240 | 0.264 | 0.342 | 0.257 | 0.531 | 0.100 |
| MiniLM, chunked, Agglomerative | 0.138 | 0.114 | 0.182 | 0.229 | 0.337 | 0.000 |
| MiniLM, chunked, KMeans | 0.256 | 0.294 | 0.380 | 0.277 | 0.555 | 0.130 |
| **LaBSE, chunked, KMeans (seeds 0 to 2, mean)** | **0.109** | **about 0.487** | **about 0.533** | **about 0.311** | **about 0.579** | **about 0.613** |
| LaBSE, chunked, KMeans (seed 42) | 0.108 | 0.465 | 0.521 | 0.294 | 0.554 | 0.582 |
| LaBSE, chunked, Agglomerative, complete linkage | 0.052 | 0.318 | 0.396 | 0.224 | 0.370 | 0.434 |
| LaBSE, chunked, Agglomerative, average linkage | 0.033 | 0.000 | 0.006 | 0.000 | 0.000 | 0.000 |

LaBSE used a 256-token limit (1.33 chunks per document); MiniLM used 128 tokens (2.57 chunks per document). KMeans ARI for LaBSE ranged from 0.465 (seed 42) to 0.491 (seed 0), so the result is reported as a range of about 0.47 to 0.49.

### 3.3 What the cluster crosstabs showed

- **MiniLM:** one cluster held 540 Telugu and 1 English document, and the Telugu-only table showed every topic inside it. The clusters followed language and source style, not topic.
- **LaBSE:** every cluster contained all three languages. In the Telugu-only table each cluster was dominated by one topic (technology 119, education 137, sports 135, business 79, entertainment 147).

### 3.4 English diagnostic (LaBSE, KMeans, English documents only)

| Cluster | business | education | entertainment | sports | technology |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 114 | 3 |
| 1 | 2 | 81 | 1 | 1 | 4 |
| 2 | 140 | 61 | 39 | 28 | 123 |
| 3 | 0 | 0 | 0 | 0 | 12 |
| 4 | 7 | 8 | 110 | 7 | 8 |

Sports, education and entertainment each form a clean cluster. Cluster 2 is a mixed cluster dominated by business (140) and technology (123), with education spilling in (61). English's lower ARI therefore comes mainly from business and technology merging. This is a hypothesis about topic overlap in the source labels (D30).

### 3.5 UMAP visualization (LaBSE, `results/umap_labse.png`)

Two panels, colored by topic and by language (UMAP, cosine metric, seed 42). Reading from the saved image:
- Education, sports and entertainment form recognizable regions, and technology forms a band at the top. Business is the most spread out and overlaps technology.
- All three languages appear inside each topic region, so documents are placed by topic and not by language, in contrast with MiniLM, where Telugu formed its own block.
- The central region is mostly English and mixes business, technology and some education, consistent with the English diagnostic (section 3.4).
- A small group of Hindi business documents sits apart from the main business area, and a few isolated points lie far from everything else. These were not inspected.
- UMAP is qualitative. The layout depends on its settings, so the plot illustrates the metric results and does not replace them.

### 3.6 Semantic search results (LaBSE, 2,250 documents)

**Example queries** (cricket match result, in English, Hindi and Telugu; the header of the first query was cut off in the recorded output and is taken to be the English query):

| Query language | Topics in top 10 | Languages in top 10 |
|---|---|---|
| English | sports 9, education 1 | Hindi 7, Telugu 2, English 1 |
| Hindi | sports 9, education 1 | Hindi 9, Telugu 1 |
| Telugu | sports 10 | Hindi 8, Telugu 2 |

Topic retrieval was correct, and one Hindi document (HI_SPORTS_136) ranked first for all three queries (a "hub" document). English documents were almost absent for short text queries.

**Document-as-query evaluation** (each document queries the collection, itself excluded, K=10, chance precision is 0.2):

Precision@10 (rows = query language, columns = searched language):

| | English | Hindi | Telugu |
|---|---|---|---|
| English | 0.682 | 0.651 | 0.686 |
| Hindi | 0.675 | 0.860 | 0.785 |
| Telugu | 0.794 | 0.817 | 0.896 |

Share of the top 10 by language when searching all documents (chance is about 0.33 each):

| | English | Hindi | Telugu |
|---|---|---|---|
| English query | 0.896 | 0.047 | 0.057 |
| Hindi query | 0.065 | 0.788 | 0.147 |
| Telugu query | 0.147 | 0.281 | 0.572 |

Findings:
- Cross-language topic retrieval works: precision@10 is 0.65 to 0.82 across languages, far above chance (0.2), though below same-language retrieval for English and Hindi.
- Telugu has the best same-language precision (0.896). English has the lowest (0.682), consistent with business and technology overlap in the English source.
- Results strongly favour the query's own language (English 90%, Hindi 79%, Telugu 57% of the top 10), so unmodified ranking returns few cross-language results even though cross-language matches are good. This motivates D32.
- Short text queries behaved differently from document queries: English documents were nearly absent for the English query, while Hindi dominated. Possible causes, not yet tested: length mismatch between short queries and long averaged-chunk document vectors, and a residual language effect in LaBSE.
- Relevance here is the topic label, which is coarse. There are no human relevance judgements.

---

## 4. Change log (what changed from the original plan, and why)

| Change | Original plan | New approach | Reason |
|---|---|---|---|
| Dataset source | Hand-collected 45-document spreadsheet | Public dataset (L3Cube-IndicNews) | Faster, larger, labeled, consistent across languages |
| Main corpus | IndicCorpV2 | L3Cube-IndicNews | IndicCorpV2 has no English and no labels |
| Dataset size | 45, then 750 (pilot) | 2,250 (final) | Balanced and limited by the smallest cell (Telugu technology, 168) |
| Clustering algorithm | Agglomerative, average linkage (base paper style) | KMeans for results, Agglomerative with complete linkage kept as comparison | Average linkage produced one giant cluster with both models (LaBSE ARI 0.000); complete linkage gave ARI 0.318 and KMeans 0.465 to 0.491 |
| Text handling | Truncate at model limit | Chunk and average embeddings | 87% to 96% of documents exceeded 128 tokens |
| Embedding model | MiniLM only | LaBSE proposed as primary, MiniLM as baseline | MiniLM failed on Telugu |
| Libraries | NLTK and Indic NLP Toolkit (base paper) | NFC cleanup and langdetect, NLTK and Toolkit proposed to be dropped | Simpler, and transformer embeddings need natural text |
| Imports | `fitz` | `pymupdf` | Deprecation warning |

---

## 5. Limitations and caveats to state in the report

1. The pilot (750) and final (2,250) datasets are drawn from news articles of different outlets per language, so source style may influence embeddings.
2. KMeans results vary slightly with the random seed (LaBSE ARI 0.465 to 0.491 across seeds 0, 1, 2 and 42), so report a range or mean. Seeds test initialization only, not variation from data sampling.
3. LaBSE and MiniLM use different token limits and chunk counts, so the comparison is not perfectly controlled. The size of the Telugu gain makes the model the main driver.
4. Silhouette is lower for LaBSE (0.109 against 0.256). Silhouette measures compactness in the embedding space, and MiniLM's compact clusters follow language. Report all three metrics together.
5. The model comparison used the topic labels for evaluation, so results are tuned on this dataset. Labels were never used to form clusters.
6. ARI of about 0.49 is moderate. Several news topics overlap in nature, so some confusion is expected.
7. English became the weakest language under LaBSE (ARI about 0.29 to 0.31). The diagnostic points to business and technology merging into one cluster, but this is a hypothesis until a sample of those documents is read (D30).
8. Results are not directly comparable to the base paper's numbers, because the datasets differ (news paragraphs versus PDFs) and the base paper reported per-language silhouettes on 15 documents.
9. Scanned PDFs are out of scope (no OCR). Some Indic PDFs may produce split words at line breaks.
10. Search is evaluated with documents as queries and topic labels as relevance, which favours same-language neighbours and treats relevance as topic match only. Short user queries behaved differently (English documents rarely returned) and have been tried on only one topic (cricket).
11. Plain similarity ranking returns mostly the query's own language, and some "hub" documents rank highly for many queries.

---

## 6. Remaining work (in priority order)

1. Confirm the model decision (D28) and the library decisions (D19, D20) with the team.
2. Build the per-language retrieval option (D32), try short queries in each language across all five topics, and rerun the search evaluation.
3. Read a sample of English technology and business documents from the mixed cluster to confirm or reject the label-overlap hypothesis (D30).
4. Check the isolated points in the UMAP plot for corrupted or unusual text.
5. Generate `requirements.txt` (`pip freeze`) and write the README.
6. Finish the remaining design and documentation items: DFD Level 0 and Level 1, use-case and activity diagrams, written methodology (Section C), and the justification table.
7. Clean up leftover scripts (D13f) and optionally move to a larger train-split dataset if more Telugu data is wanted.
8. Update the Notion to-do board and the script-audit spreadsheet to match this log.

Completed since the last version of this log: the UMAP plot, semantic search and its evaluation, in addition to Agglomerative Clustering with LaBSE and the English diagnostic.

---

## 7. Checklist status

**Section A (Dataset): complete.** Dataset, source, languages, formats, size (2,250) and ground-truth labels are all recorded.

**Section B (Technologies): about 95%.**
- Done: language, embedding model, clustering algorithm, extraction, preprocessing, similarity, metrics, visualization choice, libraries list.
- Remaining: generate `requirements.txt`, confirm D19 and D20, finalize the justification table with LaBSE as the chosen model.

**Section C (Methodology): in progress.** Implemented so far: loading, extraction, preprocessing, embeddings, similarity and clustering with evaluation. Still to write up or build: final output and visualization (UMAP, search) and the written explanations.

**Design and pilot items:** dataset source and topics confirmed, environment set up, tools table prepared, and workflow and architecture drawn. DFDs and use-case/activity diagrams are still open. The small pilot implementation is effectively done, with observations recorded above.

---

## 8. Report-ready summary sentences

- "A balanced dataset of 2,250 news documents (150 per topic per language across five topics) was built from L3Cube-IndicNews and used for evaluation, with labels used only to evaluate clusters."
- "Language labels were verified with langdetect (100% agreement). Preprocessing applies Unicode NFC normalization and URL and whitespace cleanup; lowercasing, stopword removal and stemming are not used, because transformer embeddings rely on natural text."
- "Over 87% of documents exceed the 128-token limit of paraphrase-multilingual-MiniLM-L12-v2, so documents were split into token windows and chunk embeddings averaged, which improved ARI from 0.264 to 0.294."
- "With MiniLM, Telugu documents clustered by language and source rather than by topic (Telugu ARI 0.13). With LaBSE and KMeans, Telugu ARI rose to about 0.58 to 0.61, overall ARI to about 0.47 to 0.49 and NMI to about 0.52 to 0.53, and clusters mixed all three languages by topic."
- "On the same LaBSE embeddings, Agglomerative clustering with cosine distance gave ARI 0.318 with complete linkage and collapsed to a single cluster with average linkage (ARI 0.000), so KMeans was used for the main results."
- "A UMAP projection of the LaBSE embeddings shows topic regions that contain all three languages, in contrast to MiniLM, where Telugu documents formed a separate block."
- "Cross-language semantic search over 2,250 documents achieved precision@10 of 0.65 to 0.82 across language pairs (chance 0.2), but results favoured the query's own language, which motivates retrieving per language and merging."
- "English was the weakest language (ARI about 0.29 to 0.31) because business and technology documents merged into one cluster, which suggests topic overlap in the English source labels."
- "Silhouette was lower for LaBSE than MiniLM (0.109 against 0.256), because MiniLM's compact clusters followed language; ARI and NMI measure agreement with topic and are the more relevant metrics for this goal."