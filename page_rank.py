import json
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag
from indexer import find_json_files


def PageRank(corpus_dir, documents_path, damping=0.85, iterations=50):
    # Map URLs to document IDs
    url_to_id = {}
    doc_to_id = {}

    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            document = json.loads(line)
            url = document["url"]
            doc_id = document["doc_id"]

            if url not in url_to_id:
                url_to_id[url] = doc_id

            doc_to_id[doc_id] = url_to_id[url]

    # Initialize the link graph
    link_graph = {doc_id: set() for doc_id in url_to_id.values()}

    # Extract hyperlinks and build the graph
    for page_path in find_json_files(Path(corpus_dir)):
        try:
            with page_path.open("r", encoding="utf-8", errors="replace") as f:
                page = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(page.get("url"), str):
            continue

        url = urldefrag(page["url"])[0]

        if url not in url_to_id:
            continue

        doc_id = url_to_id[url]
        soup = BeautifulSoup(str(page.get("content", "")), "html.parser")

        for element in soup.find_all("a", href=True):
            target_url = urljoin(url, element["href"])
            target_url = urldefrag(target_url)[0]

            if target_url in url_to_id:
                target_id = url_to_id[target_url]

                # Ignore self-links
                if target_id != doc_id:
                    link_graph[doc_id].add(target_id)

    # TODO:
    # Initialize PageRank scores
    # Iteratively calculate PageRank
    # Return final scores for all document IDs