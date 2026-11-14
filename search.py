#!/usr/bin/env python3
"""
Console search interface for the Assignment 3 search engine.

Usage:
    python search.py                       # interactive Search> prompt
    python search.py --query "acm"         # run one query and exit
    python search.py --index-dir PATH      # use an index built elsewhere
"""

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from document_retriever import DocRetriever
from index_retriever import IndexRetriever
from query_processor import boolean_and, process_query


def search(query, retriever, doc_urls, doc_norms, num_docs, top_k=5):
    """
    Boolean AND retrieval ranked by tf-idf cosine similarity (lnc.ltc).

    Args:
        query:     raw user search string
        retriever: IndexRetriever reading postings from disk
        doc_urls:  {doc_id: url}
        doc_norms: {doc_id: lnc vector length} from documents.jsonl
        num_docs:  total number of documents in the corpus (N)
        top_k:     number of results to return

    Returns:
        (total_matches, [(score, doc_id, url), ...]) best first.
    """
    # 1. Process query into stemmed tokens and count term frequencies
    query_tf = Counter(process_query(query))
    if not query_tf:
        return 0, []

    unique_terms = list(query_tf)

    # 2. Fetch posting lists from disk ([] if any term is not indexed)
    posting_lists = retriever.get_postings_for_terms(unique_terms)
    if not posting_lists:
        return 0, []

    # 3. Boolean AND: only documents containing every query term
    candidate_ids = set(boolean_and(posting_lists))
    if not candidate_ids:
        return 0, []

    # 4. Cosine similarity.
    #    Query weight: (1 + log10 tf) * idf          (ltc)
    #    Doc weight:   (1 + log10 tf) / doc_norm     (lnc)
    dot_products = defaultdict(float)
    query_sq_sum = 0.0
    # Used only if the index has no stored norms (built by older code).
    fallback_doc_sq_sum = defaultdict(float)

    for term, postings in zip(unique_terms, posting_lists):
        idf = math.log10(num_docs / len(postings))
        q_weight = (1 + math.log10(query_tf[term])) * idf
        query_sq_sum += q_weight ** 2

        for doc_id, tf in postings:
            if doc_id in candidate_ids:
                d_weight = 1 + math.log10(tf)
                dot_products[doc_id] += q_weight * d_weight
                fallback_doc_sq_sum[doc_id] += d_weight ** 2

    # A term in every document has idf 0; fall back to plain tf matching so
    # such queries still return ranked results instead of nothing.
    query_norm = math.sqrt(query_sq_sum) or 1.0

    scored = []
    for doc_id in candidate_ids:
        d_norm = doc_norms.get(doc_id) or math.sqrt(fallback_doc_sq_sum[doc_id])
        if d_norm > 0:
            scored.append((dot_products[doc_id] / (query_norm * d_norm), doc_id))

    # 5. Highest score first; ties go to the lower doc_id for stable output.
    #    The corpus has some pages crawled more than once (same URL once the
    #    #fragment is dropped), so only the best copy of each URL is kept.
    scored.sort(key=lambda s: (-s[0], s[1]))
    results = []
    seen_urls = set()
    for score, doc_id in scored:
        url = doc_urls.get(doc_id, "<missing URL>")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append((score, doc_id, url))
        if len(results) == top_k:
            break

    return len(candidate_ids), results


def print_results(query, total, results, elapsed_ms):
    print(f"\n{total} matching document(s) for '{query}' ({elapsed_ms:.1f} ms)")
    for rank, (score, doc_id, url) in enumerate(results, start=1):
        print(f"  {rank}. {url}  (score {score:.4f}, doc {doc_id})")
    print()


def run_query(query, retriever, doc_urls, doc_norms, top_k):
    start = perf_counter()
    total, results = search(
        query, retriever, doc_urls, doc_norms, len(doc_urls), top_k
    )
    elapsed_ms = (perf_counter() - start) * 1000
    print_results(query, total, results, elapsed_ms)


def main():
    parser = argparse.ArgumentParser(description="UCI ICS search engine")
    parser.add_argument(
        "--index-dir",
        default="index_data",
        help="Directory produced by indexer.py (default: index_data)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Results to show (default: 5)"
    )
    parser.add_argument(
        "--query", help="Run a single query and exit instead of prompting"
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    index_path = index_dir / "index.jsonl"
    docs_path = index_dir / "documents.jsonl"
    if not index_path.exists() or not docs_path.exists():
        sys.exit(
            f"No index found in '{index_dir}'. Build one first:\n"
            f"  python indexer.py <corpus_dir> --output {index_dir}"
        )

    # Load everything that stays in memory once, before the first query.
    print("Loading index...")
    start = perf_counter()
    doc_urls, doc_norms = DocRetriever(docs_path).load()
    with IndexRetriever(index_path) as retriever:
        print(
            f"Ready: {len(doc_urls)} documents, {len(retriever.lexicon)} terms "
            f"({perf_counter() - start:.1f} s)\n"
        )

        if args.query is not None:
            run_query(args.query, retriever, doc_urls, doc_norms, args.top_k)
            return

        print("UCI Search Engine")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                query = input("Search> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if query.lower() == "exit":
                break

            if not query:
                continue

            run_query(query, retriever, doc_urls, doc_norms, args.top_k)


if __name__ == "__main__":
    main()
