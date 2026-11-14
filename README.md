# Assignment 3 - Search Engine (Milestone 2)

## Overview

This repository contains the indexer and search component for Assignment 3 (developer option).

- **Indexer** reads the crawler's JSON pages, extracts alphanumeric tokens, applies Porter stemming, and builds a disk-based inverted index of document IDs and term frequencies.
- **Search** prompts for a query, stems it the same way, reads only the needed posting lists from disk, keeps documents that contain **every** query term (Boolean AND), and ranks them by tf-idf cosine similarity.

## Files

- `indexer.py` - builds the inverted index, lexicon and document table
- `index_retriever.py` - `IndexRetriever`: seeks into the index file for a term's postings
- `document_retriever.py` - `DocRetriever`: loads doc_id -> URL and doc_id -> norm
- `query_processor.py` - query tokenizing/stemming and Boolean AND intersection
- `search.py` - ranking (`search()`) and the console search interface
- `test_search.py` - self-contained tests on a tiny generated corpus
- `test_index_retriever.py` - tests against the real index (skipped if absent)
- `requirements.txt` - Python dependencies
- `.gitignore` - keeps the corpus and generated index out of git

## Requirements

Python 3 is required.

```bash
python -m pip install -r requirements.txt
```

## Building the index

After extracting the developer corpus, run:

```bash
python indexer.py developer --output index_data --flush-docs 5000 --clean
```

Replace `developer` with the corpus path and `index_data` with any output directory (it can live outside the repo).

The indexer:

1. Reads JSON pages from the corpus and removes URL fragments with `urldefrag`.
2. Parses HTML with BeautifulSoup (tolerates broken HTML).
3. Extracts lowercase alphanumeric tokens and applies Porter stemming. Stop words are **not** removed.
4. Identifies important text in `title`, `h1`, `h2`, `h3`, `b`, and `strong` elements (not yet used in ranking).
5. Flushes a partial index to disk every `--flush-docs` documents (11 partials for the developer corpus at 5000), then merges them into `index.jsonl`.
6. Records each term's byte offset in `lexicon.json` and each document's length norm in `documents.jsonl`.

### Generated files

```text
index_data/
├── index.jsonl       {"term": "...", "postings": [[doc_id, tf], ...]} per line, sorted by term
├── lexicon.json      {"term": byte_offset_in_index.jsonl, ...}
├── documents.jsonl   {"doc_id": 0, "url": "...", "norm": 4.12} per line
├── metadata.json     document/token counts and index size for reports
└── partials/         partial indexes written during construction
```

The generated index is intentionally not included in GitHub. It is recreated from the provided corpus.

## Searching

```bash
python search.py --index-dir index_data
```

This loads the lexicon and document table once and then shows a `Search>` prompt. Type a query, or `exit` to quit. Each query prints the number of matching documents, the top results (URL, score, doc id), and the response time in ms.

Options:

- `--index-dir PATH` - directory produced by `indexer.py` (default `index_data`)
- `--top-k N` - number of results to show (default 5)
- `--query "text"` - run a single query and exit

### How retrieval works

1. The query is tokenized and stemmed exactly like the documents.
2. For each term, `IndexRetriever` looks up its byte offset in the lexicon, seeks into `index.jsonl`, and parses that one line. The inverted index itself is never loaded into memory.
3. If any term is missing, there are no results. Otherwise, the posting lists are intersected (Boolean AND), shortest first.
4. Matching documents are ranked by cosine similarity using the lnc.ltc scheme:
   - query term weight = `(1 + log10 tf_q) * log10(N / df)`, normalized by query length
   - document term weight = `(1 + log10 tf_d)`, normalized by the document's precomputed `norm`

## Tests

```bash
python -m pytest
```

To also run the tests against a real index stored elsewhere, set `INDEX_DIR` to that directory first.

## Reused Assignment 2 Components

Only general techniques from Assignment 2 were reused:

- BeautifulSoup for HTML parsing
- regular-expression token extraction
- URL fragment removal

The Assignment 2 crawler/frontier/worker implementation and stop-word filtering were not copied into this project.
