from nltk.stem import PorterStemmer

from indexer import tokenize, stem_tokens

# Same tokenizer and stemmer as the indexer, so query terms match index terms.
_STEMMER = PorterStemmer()


def process_query(query):
    """Turn a raw query string into a list of lowercase, stemmed tokens."""
    return stem_tokens(tokenize(query), _STEMMER)


def boolean_and(posting_lists):
    """
    Return the sorted doc_ids that appear in every posting list.

    Intersects starting from the shortest list so the working set stays
    as small as possible.
    """
    if not posting_lists:
        return []

    ordered = sorted(posting_lists, key=len)

    # Extract document IDs from the shortest posting list
    doc_ids = {doc_id for doc_id, _tf in ordered[0]}

    # Intersect with each remaining posting list
    for postings in ordered[1:]:
        doc_ids &= {doc_id for doc_id, _tf in postings}

        # Stop early once nothing can match
        if not doc_ids:
            return []

    return sorted(doc_ids)
