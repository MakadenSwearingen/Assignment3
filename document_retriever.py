import json


class DocRetriever:
    """Reads the document table (documents.jsonl) written by the indexer."""

    def __init__(self, docs_path):
        self.docs_path = docs_path

    def load(self):
        """
        Read documents.jsonl once and return (urls, norms):
            urls:  {doc_id: url}
            norms: {doc_id: lnc vector length}  (empty for old indexes
                   built before norms were stored)
        """
        urls = {}
        norms = {}

        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                doc_id = int(record["doc_id"])
                urls[doc_id] = record["url"]
                if "norm" in record:
                    norms[doc_id] = record["norm"]

        return urls, norms

    def get_urls(self):
        """Return {doc_id: url}."""
        return self.load()[0]

    def get_norms(self):
        """Return {doc_id: norm}."""
        return self.load()[1]
