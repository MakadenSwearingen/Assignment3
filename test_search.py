"""
Self-contained tests: build a tiny corpus in a temp dir, index it, search it.
"""

import json

import pytest

from document_retriever import DocRetriever
from index_retriever import IndexRetriever
from indexer import index_corpus
from query_processor import boolean_and, process_query
from search import search


PAGES = {
    "a.json": (
        "https://example.com/ml#frag",
        "<html><title>Machine Learning</title>"
        "<body>machine learning machine learning research</body></html>",
    ),
    "b.json": (
        "https://example.com/ml-intro",
        "<html><body>An intro to machine learning, plus a lot of other "
        "words about cooking, gardening, travel and music.</body></html>",
    ),
    "c.json": (
        "https://example.com/acm",
        "<html><body><b>ACM</b> student chapter meetings</body></html>",
    ),
}


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    corpus = tmp_path_factory.mktemp("corpus")
    domain = corpus / "example_com"
    domain.mkdir()
    for name, (url, html) in PAGES.items():
        (domain / name).write_text(
            json.dumps({"url": url, "content": html, "encoding": "utf-8"}),
            encoding="utf-8",
        )

    out = tmp_path_factory.mktemp("index")
    # flush_docs=1 forces one partial index per document, so the merge
    # path is exercised with several partials.
    meta = index_corpus(corpus, out, flush_docs=1)
    assert meta["partial_indexes"] == 3
    return out


@pytest.fixture(scope="module")
def loaded(index_dir):
    urls, norms = DocRetriever(index_dir / "documents.jsonl").load()
    with IndexRetriever(index_dir / "index.jsonl") as retriever:
        yield retriever, urls, norms


def test_process_query_stems_and_lowercases():
    assert process_query("Machine Learning") == ["machin", "learn"]
    assert process_query("ACM") == ["acm"]


def test_boolean_and():
    a = [[1, 2], [3, 1], [5, 1]]
    b = [[3, 4], [5, 2]]
    c = [[5, 1]]
    assert boolean_and([a, b]) == [3, 5]
    assert boolean_and([a, b, c]) == [5]
    assert boolean_and([a, [[9, 1]]]) == []
    assert boolean_and([]) == []


def test_lexicon_written_and_seekable(index_dir, loaded):
    retriever, _, _ = loaded
    assert (index_dir / "lexicon.json").exists()
    # Docs are numbered in sorted file order: a.json=0, b.json=1, c.json=2.
    # a.json has "machine" once in the title and twice in the body.
    assert retriever.get_postings("machin") == [[0, 3], [1, 1]]
    assert retriever.get_postings("doesnotexist") == []


def test_lexicon_rebuilt_when_missing(index_dir):
    (index_dir / "lexicon.json").unlink()
    with IndexRetriever(index_dir / "index.jsonl") as retriever:
        assert retriever.get_postings("acm")
    assert (index_dir / "lexicon.json").exists()


def test_doc_retriever_maps_ids_to_urls(loaded):
    _, urls, norms = loaded
    assert len(urls) == 3
    assert set(urls) == set(norms) == {0, 1, 2}
    # URL fragments are removed by the indexer
    assert "https://example.com/ml" in urls.values()
    assert all(n > 0 for n in norms.values())


def test_search_and_semantics_and_ranking(loaded):
    retriever, urls, norms = loaded

    total, results = search(
        "machine learning", retriever, urls, norms, len(urls)
    )
    assert total == 2
    # The short, focused page should outrank the long, diluted one.
    assert results[0][2] == "https://example.com/ml"
    assert results[0][0] >= results[1][0]

    total, results = search("acm", retriever, urls, norms, len(urls))
    assert total == 1
    assert results[0][2] == "https://example.com/acm"

    # AND: no page has both terms
    assert search("acm learning", retriever, urls, norms, len(urls)) == (0, [])
    assert search("", retriever, urls, norms, len(urls)) == (0, [])


def test_duplicate_urls_collapsed(loaded):
    retriever, urls, norms = loaded
    # Pretend docs 0 and 1 were the same page crawled twice.
    dup_urls = dict(urls)
    dup_urls[1] = dup_urls[0]
    total, results = search(
        "machine learning", retriever, dup_urls, norms, len(dup_urls)
    )
    assert total == 2
    assert [url for _s, _d, url in results] == [dup_urls[0]]


def test_top_k_limits_results(loaded):
    retriever, urls, norms = loaded
    _, results = search(
        "machine", retriever, urls, norms, len(urls), top_k=1
    )
    assert len(results) == 1
