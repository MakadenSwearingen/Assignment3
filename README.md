# Assignment 3 - Milestone 1

## Overview

This repository contains the Milestone 1 implementation for Assignment 3.

The program builds an inverted index from the Assignment 3 Developer corpus. It reads the crawler's JSON pages, extracts alphanumeric tokens, applies Porter stemming, and stores document IDs and term frequencies in a disk-based inverted index.

## Files

- `indexer.py` - builds the inverted index
- `requirements.txt` - Python dependencies
- `.gitignore` - prevents the large corpus and generated index from being committed

## Requirements

Python 3 is required.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Running

After extracting the Developer corpus, run:

```bash
python3 indexer.py developer --output index_data --flush-docs 5000 --clean
```

If the corpus directory has another name or location, replace `developer` with that path.

## Milestone 1 Behavior

The indexer:

1. Reads JSON pages from the corpus.
2. Removes URL fragments with `urldefrag`.
3. Parses HTML with BeautifulSoup.
4. Extracts alphanumeric tokens.
5. Converts tokens to lowercase.
6. Applies Porter stemming.
7. Does not remove stop words.
8. Identifies important text in `title`, `h1`, `h2`, `h3`, `b`, and `strong` elements.
9. Builds partial inverted indexes on disk.
10. Merges the partial indexes into `index_data/index.jsonl`.
11. Reports the number of indexed documents, unique tokens, and index size in KB.

The posting format used for Milestone 1 is:

```text
[document_id, term_frequency]
```

## Generated Files

Running the program creates:

```text
index_data/
├── index.jsonl
├── documents.jsonl
├── metadata.json
└── partials/
```

The generated index is intentionally not included in GitHub. It is recreated from the provided corpus.

## Reused Assignment 2 Components

Only general techniques from Assignment 2 were reused:

- BeautifulSoup for HTML parsing
- regular-expression token extraction
- URL fragment removal

The Assignment 2 crawler/frontier/worker implementation and stop-word filtering were not copied into this project.

## Milestone 1 Report

The PDF report should include at least:

- Number of indexed documents
- Number of unique tokens
- Total index size in KB
