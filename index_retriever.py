import json
from pathlib import Path


class IndexRetriever:
    """
    Reads posting lists from the on-disk inverted index (index.jsonl).

    Only the lexicon (term -> byte offset) is kept in memory. Each lookup
    seeks straight to the term's line in the index file and parses just that
    line, so the inverted index itself is never loaded into memory.
    """

    def __init__(self, index_path, lexicon_path=None):
        self.index_path = Path(index_path)
        if lexicon_path is None:
            lexicon_path = self.index_path.with_name("lexicon.json")
        self.lexicon_path = Path(lexicon_path)

        self.lexicon = self._load_lexicon()
        # Kept open for the lifetime of the retriever; binary mode so the
        # stored byte offsets can be used directly with seek().
        self._index_file = self.index_path.open("rb")

    def _load_lexicon(self):
        """Load term -> offset map, building it once if it does not exist."""
        if self.lexicon_path.exists():
            with self.lexicon_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback for indexes built before lexicon.json existed: scan the
        # index once, record where each line starts, and save the result.
        lexicon = {}
        with self.index_path.open("rb") as f:
            offset = f.tell()
            for line in iter(f.readline, b""):
                lexicon[json.loads(line)["term"]] = offset
                offset = f.tell()

        with self.lexicon_path.open("w", encoding="utf-8") as f:
            json.dump(lexicon, f, separators=(",", ":"))

        return lexicon

    def get_postings(self, term):
        """Return [[doc_id, tf], ...] for term, or [] if it is not indexed."""
        offset = self.lexicon.get(term)
        if offset is None:
            return []

        self._index_file.seek(offset)
        record = json.loads(self._index_file.readline())
        return record["postings"]

    def get_postings_for_terms(self, terms):
        """
        Return one posting list per term, in order.

        Returns [] if any term is missing, since an AND query can then
        have no matching documents.
        """
        posting_lists = []

        for term in terms:
            postings = self.get_postings(term)

            if not postings:
                return []

            posting_lists.append(postings)

        return posting_lists

    def close(self):
        self._index_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
