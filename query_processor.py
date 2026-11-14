from nltk.stem import PorterStemmer
from indexer import tokenize, stem_tokens

def process_query(query):
    stemmer = PorterStemmer()
    
    tokenized = tokenize(query)
    stemmed_tokens = stem_tokens(tokenized, stemmer)
    
    return stemmed_tokens


def boolean_and(posting_lists):
    if not posting_lists:
        return []

    # Extract document IDs from the first posting list
    doc_ids = {doc_id for doc_id, tf in posting_lists[0]}

    # Iterate through the remaining posting lists
    for x in range(1, len(posting_lists)):
        current_ids = {doc_id for doc_id, tf in posting_lists[x]}

        # Find the intersection of document IDs
        doc_ids = doc_ids & current_ids

        # Check if empty
        if not doc_ids:
            return []
    
    return sorted(doc_ids)