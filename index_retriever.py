import json


class IndexRetriever:

    def __init__(self, index_path):
        self.index_path = index_path

    def get_postings(self, term):
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)

                if record["term"] == term:
                    return record["postings"]

        return []

    def get_postings_for_terms(self, terms):
        posting_lists = []

        for term in terms:
            postings = self.get_postings(term)

            if not postings:
                return []

            posting_lists.append(postings)

        return posting_lists
