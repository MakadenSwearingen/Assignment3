from index_retriever import IndexRetriever


INDEX_PATH = "index_data/index.jsonl"


def test_existing_term():
    retriever = IndexRetriever(INDEX_PATH)

    postings = retriever.get_postings("machin")

    assert postings is not None
    assert len(postings) > 0

    for posting in postings:
        assert len(posting) == 2
        assert isinstance(posting[0], int)
        assert isinstance(posting[1], int)


def test_nonexistent_term():
    retriever = IndexRetriever(INDEX_PATH)

    postings = retriever.get_postings("thisworddefinitelydoesnotexist")

    assert postings == []


def test_multiple_terms():
    retriever = IndexRetriever(INDEX_PATH)

    terms = ["machin", "learn"]

    posting_lists = retriever.get_postings_for_terms(terms)

    assert len(posting_lists) == 2

    for postings in posting_lists:
        assert isinstance(postings, list)


def test_missing_term_returns_no_results():
    retriever = IndexRetriever(INDEX_PATH)

    terms = ["machin", "thisworddefinitelydoesnotexist"]

    posting_lists = retriever.get_postings_for_terms(terms)

    assert posting_lists == []
