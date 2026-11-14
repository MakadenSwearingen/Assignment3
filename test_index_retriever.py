"""
Tests against the real index built from the developer corpus.

Point INDEX_DIR at the index directory if it is not ./index_data, e.g.
    INDEX_DIR=C:/Users/you/CS121_data/index_data python -m pytest
Skipped automatically when no index exists.
"""

import os
from pathlib import Path

import pytest

from index_retriever import IndexRetriever


INDEX_PATH = Path(os.environ.get("INDEX_DIR", "index_data")) / "index.jsonl"

pytestmark = pytest.mark.skipif(
    not INDEX_PATH.exists(), reason=f"no index at {INDEX_PATH}"
)


@pytest.fixture(scope="module")
def retriever():
    with IndexRetriever(INDEX_PATH) as r:
        yield r


def test_existing_term(retriever):
    postings = retriever.get_postings("machin")

    assert postings is not None
    assert len(postings) > 0

    for posting in postings:
        assert len(posting) == 2
        assert isinstance(posting[0], int)
        assert isinstance(posting[1], int)


def test_nonexistent_term(retriever):
    postings = retriever.get_postings("thisworddefinitelydoesnotexist")

    assert postings == []


def test_multiple_terms(retriever):
    terms = ["machin", "learn"]

    posting_lists = retriever.get_postings_for_terms(terms)

    assert len(posting_lists) == 2

    for postings in posting_lists:
        assert isinstance(postings, list)


def test_missing_term_returns_no_results(retriever):
    terms = ["machin", "thisworddefinitelydoesnotexist"]

    posting_lists = retriever.get_postings_for_terms(terms)

    assert posting_lists == []
