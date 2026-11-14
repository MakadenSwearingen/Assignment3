from nltk.stem import PorterStemmer
from indexer import tokenize, stem_tokens

def process_query(query):
    stemmer = PorterStemmer()
    
    tokenized = tokenize(query)
    stemmed_tokens = stem_tokens(tokenized, stemmer)
    
    return stemmed_tokens
