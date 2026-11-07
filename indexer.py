#!/usr/bin/env python3


import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from urllib.parse import urldefrag

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
IMPORTANT_TAGS = ("title", "h1", "h2", "h3", "b", "strong")


def tokenize(text):
    """Return lowercase alphanumeric tokens from text."""
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def stem_tokens(tokens, stemmer):
    """Stem a sequence of tokens with Porter stemming."""
    return [stemmer.stem(token) for token in tokens]


def parse_page(page_path, stemmer):
    """
    Read one crawler JSON file and return:
        url, normal text token counts, important token counts

    The crawler JSON format contains url, content, and encoding.
    Broken HTML is handled by BeautifulSoup
    """
    try:
        with page_path.open("r", encoding="utf-8", errors="replace") as f:
            page = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    url = page.get("url")
    content = page.get("content", "")

    if not isinstance(url, str):
        return None

    # Assignment 3 says URL fragments should be ignored.
    url = urldefrag(url)[0]

    if isinstance(content, bytes):
        content = content.decode(page.get("encoding") or "utf-8", errors="replace")
    elif not isinstance(content, str):
        content = str(content)

    # Reused from Assignment 2: BeautifulSoup is used to parse crawler HTML.
    soup = BeautifulSoup(content, "html.parser")

    # All visible/page text is indexed. Stop words are intentionally NOT removed.
    normal_text = soup.get_text(" ", strip=True)
    normal_tokens = stem_tokens(tokenize(normal_text), stemmer)

    # Important text receives separate counts so later milestones can weight it.
    important_text_parts = []
    for tag in IMPORTANT_TAGS:
        for element in soup.find_all(tag):
            important_text_parts.append(element.get_text(" ", strip=True))

    important_tokens = stem_tokens(
        tokenize(" ".join(important_text_parts)), stemmer
    )

    return url, Counter(normal_tokens), Counter(important_tokens)


def find_json_files(corpus_dir):
    """Find JSON documents recursively in the corpus directory."""
    return sorted(
        path for path in corpus_dir.rglob("*.json") if path.is_file()
    )


def write_partial_index(index, output_path):
    """
    Write one partial inverted index as JSON Lines.

    Each line:
      {"term": "...", "postings": [[doc_id, tf], ...]}
    """
    with output_path.open("w", encoding="utf-8") as f:
        for term in sorted(index):
            postings = [
                [doc_id, tf]
                for doc_id, tf in sorted(index[term].items())
            ]
            f.write(
                json.dumps(
                    {"term": term, "postings": postings},
                    separators=(",", ":"),
                )
                + "\n"
            )


def merge_partials(partial_paths, final_path):
    """
    Merge all partial JSONL indexes into one sorted JSONL index.

    This merge is intentionally performed term by term so the final index
    remains a filesystem artifact instead of needing the whole final index
    to be loaded into memory
    """
    merged_terms = defaultdict(dict)

    for partial_path in partial_paths:
        with partial_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                term = record["term"]

                for doc_id, tf in record["postings"]:
                    merged_terms[term][int(doc_id)] = (
                        merged_terms[term].get(int(doc_id), 0) + int(tf)
                    )

    with final_path.open("w", encoding="utf-8") as f:
        for term in sorted(merged_terms):
            postings = [
                [doc_id, tf]
                for doc_id, tf in sorted(merged_terms[term].items())
            ]
            f.write(
                json.dumps(
                    {"term": term, "postings": postings},
                    separators=(",", ":"),
                )
                + "\n"
            )

    return len(merged_terms)


def index_corpus(corpus_dir, output_dir, flush_docs=5000, clean=False):
    """Build the Milestone 1 inverted index."""
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)

    if not corpus_dir.exists() or not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"Corpus directory does not exist: {corpus_dir}"
        )

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(corpus_dir)

    if not json_files:
        raise RuntimeError(f"No JSON files found in {corpus_dir}")

    stemmer = PorterStemmer()

    # term -> doc_id -> term frequency
    in_memory_index = defaultdict(dict)

    documents_path = output_dir / "documents.jsonl"
    partial_paths = []

    document_count = 0
    skipped_count = 0
    partial_number = 0

    start = perf_counter()

    with documents_path.open("w", encoding="utf-8") as documents_file:
        for page_path in json_files:
            parsed = parse_page(page_path, stemmer)

            if parsed is None:
                skipped_count += 1
                continue

            url, term_counts, _important_counts = parsed

            # Every successfully parsed page gets a numeric document ID.
            doc_id = document_count

            documents_file.write(
                json.dumps(
                    {"doc_id": doc_id, "url": url},
                    separators=(",", ":"),
                )
                + "\n"
            )

            for term, tf in term_counts.items():
                in_memory_index[term][doc_id] = tf

            document_count += 1

            if document_count % flush_docs == 0:
                partial_path = partial_dir / (
                    f"partial_{partial_number:04d}.jsonl"
                )
                write_partial_index(in_memory_index, partial_path)
                partial_paths.append(partial_path)

                print(
                    f"Flushed partial index "
                    f"{partial_path.name} after {document_count} documents"
                )

                in_memory_index.clear()
                partial_number += 1

    # Flush the final non-empty in-memory index.
    if in_memory_index:
        partial_path = partial_dir / f"partial_{partial_number:04d}.jsonl"
        write_partial_index(in_memory_index, partial_path)
        partial_paths.append(partial_path)

        print(
            f"Flushed partial index "
            f"{partial_path.name} after {document_count} documents"
        )

    final_index_path = output_dir / "index.jsonl"
    unique_tokens = merge_partials(partial_paths, final_index_path)

    # Calculate the final index size in KB.
    index_size_kb = final_index_path.stat().st_size / 1024.0

    elapsed = perf_counter() - start

    metadata = {
        "milestone": 1,
        "documents_indexed": document_count,
        "documents_skipped": skipped_count,
        "unique_tokens": unique_tokens,
        "index_size_kb": round(index_size_kb, 2),
        "partial_indexes": len(partial_paths),
        "flush_docs": flush_docs,
        "index_file": str(final_index_path),
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("===== MILESTONE 1 COMPLETE =====")
    print(f"Indexed documents    : {document_count}")
    print(f"Skipped documents    : {skipped_count}")
    print(f"Unique stemmed tokens: {unique_tokens}")
    print(f"Index size (KB)      : {index_size_kb:.2f}")
    print(f"Partial indexes      : {len(partial_paths)}")
    print(f"Processing time (s)  : {elapsed:.2f}")
    print(f"Final index          : {final_index_path}")
    print()

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build the Assignment 3 Milestone 1 inverted index."
    )
    parser.add_argument(
        "corpus",
        help="Path to the extracted Developer corpus directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="index_data",
        help="Directory for generated index files (default: index_data)",
    )
    parser.add_argument(
        "--flush-docs",
        type=int,
        default=5000,
        help="Flush after this many documents (default: 5000)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the existing output directory before indexing",
    )

    args = parser.parse_args()

    if args.flush_docs <= 0:
        parser.error("--flush-docs must be greater than zero")

    index_corpus(
        corpus_dir=args.corpus,
        output_dir=args.output,
        flush_docs=args.flush_docs,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
