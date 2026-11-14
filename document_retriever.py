import json

class DocRetriever:

    def __init__(self, docs_path):
        self.docs_path = docs_path

    def get_urls(self): # load doc json onto disk and return information as dictionary
        doc_links = {}
        
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                temp = json.loads(line)
                doc_links.update(temp)

        return doc_links
