import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant

from vernadskiy_boltalka.config import config
from vernadskiy_boltalka.embeddings_core import get_langchain_embeddings


def _collection_nonempty(client: Any, name: str) -> bool:
    if not name:
        return False
    try:
        info = client.get_collection(name)
        return bool(getattr(info, "points_count", 0) or getattr(info, "vectors_count", 0))
    except Exception:
        return False


def _dedupe_docs(docs: list[Document], limit: int) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        key = hashlib.sha256((d.page_content or "")[:2000].encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
        if len(out) >= limit:
            break
    return out


def _active_collection_names(client: Any) -> list[str]:
    names: list[str] = []
    for c in (config.RAG_GRAPH_COLLECTION, config.RAG_DOCUMENTS_COLLECTION):
        if c and _collection_nonempty(client, c) and c not in names:
            names.append(c)
    if not names and config.COLLECTION and _collection_nonempty(client, config.COLLECTION):
        return [config.COLLECTION]
    return names


class ExpertRAGRetriever:
    def __init__(self, k: int) -> None:
        self.k = k
        self._emb = get_langchain_embeddings()
        self._client = config.vector_db.client

    def invoke(self, query: str) -> list[Document]:
        cols = _active_collection_names(self._client)
        if not cols:
            return []
        per = max(1, self.k // len(cols))
        acc: list[Document] = []
        for name in cols:
            store = Qdrant(
                client=self._client,
                collection_name=name,
                embeddings=self._emb,
            )
            r = store.as_retriever(search_kwargs={"k": per})
            acc.extend(r.invoke(query))
        return _dedupe_docs(acc, self.k)


_cached: tuple[int, ExpertRAGRetriever] | None = None


def get_expert_retriever(k: int = 6) -> ExpertRAGRetriever:
    global _cached
    if _cached is not None and _cached[0] == k:
        return _cached[1]
    r = ExpertRAGRetriever(k=k)
    _cached = (k, r)
    return r


def invalidate_expert_retriever_cache() -> None:
    global _cached
    _cached = None
