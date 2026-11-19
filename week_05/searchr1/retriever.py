"""Search engine clients.

HTTPRetriever talks to Search-R1's `retrieval_server.py` (E5 + 2018 Wikipedia,
top-3 passages) -- this is the paper's setting.
BM25Retriever uses pyserini's prebuilt `wikipedia-dpr-100w` index (same 2018
Wikipedia passages, BM25 instead of E5). Cheaper, but a deviation from the paper.
"""
import json
import time


def passages_to_string(docs) -> str:
    """Same formatting as Search-R1's _passages2string."""
    out = []
    for i, doc in enumerate(docs):
        content = doc["contents"] if isinstance(doc, dict) else str(doc)
        lines = content.split("\n")
        title, text = lines[0].strip('"'), "\n".join(lines[1:])
        out.append(f"Doc {i + 1}(Title: {title}) {text}")
    return "\n".join(out) + "\n"


class HTTPRetriever:
    def __init__(self, url: str = "http://127.0.0.1:8000/retrieve", topk: int = 3, retries: int = 5):
        self.url, self.topk, self.retries = url, topk, retries
        self.cache: dict[str, str] = {}

    def _call(self, queries):
        import requests

        payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        for attempt in range(self.retries):
            try:
                r = requests.post(self.url, json=payload, timeout=120)
                r.raise_for_status()
                return r.json()["result"]
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def search(self, queries: list[str]) -> list[str]:
        todo = [q for q in dict.fromkeys(queries) if q not in self.cache]
        if todo:
            for q, res in zip(todo, self._call(todo)):
                docs = [item["document"] if "document" in item else item for item in res]
                self.cache[q] = passages_to_string(docs)
        return [self.cache[q] for q in queries]


class BM25Retriever:
    def __init__(self, index: str = "wikipedia-dpr-100w", topk: int = 3):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher.from_prebuilt_index(index)
        self.topk = topk
        self.cache: dict[str, str] = {}

    def search(self, queries: list[str]) -> list[str]:
        out = []
        for q in queries:
            if q not in self.cache:
                hits = self.searcher.search(q, k=self.topk)
                docs = [{"contents": json.loads(self.searcher.doc(h.docid).raw())["contents"]} for h in hits]
                self.cache[q] = passages_to_string(docs)
            out.append(self.cache[q])
        return out


def make_retriever(kind: str, url: str, topk: int):
    if kind == "http":
        return HTTPRetriever(url=url, topk=topk)
    if kind == "bm25":
        return BM25Retriever(topk=topk)
    raise ValueError(kind)
