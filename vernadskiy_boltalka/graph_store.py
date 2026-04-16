import json
import os

from langchain_community.vectorstores import Qdrant
from qdrant_client.models import Distance, VectorParams

from vernadskiy_boltalka.config import config
from vernadskiy_boltalka.embeddings_core import get_langchain_embeddings
from vernadskiy_boltalka.paths import project_root
from vernadskiy_boltalka.rag_retriever import get_expert_retriever, invalidate_expert_retriever_cache


def _load_graph() -> dict:
    path = os.path.join(project_root(), "data", "vernadsky_graph.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _graph_to_texts(graph: dict) -> list[tuple[str, dict]]:
    texts = []
    for edge in graph["edges"]:
        source = edge.get("source", "")
        target = edge.get("target", "")
        quote = edge.get("quote", "")
        relation = edge.get("relation", "")
        text = f"Концепт: {source}. Отношение к {target}: {relation}. Цитата Вернадского: {quote}"
        texts.append((text, {"source": source, "target": target, "quote": quote}))
    for node in graph["nodes"]:
        if node.get("type") == "concept":
            texts.append(
                (
                    f"Понятие Вернадского: {node['id']}",
                    {"id": node["id"], "type": node["type"]},
                )
            )
    return texts


def build_index():
    graph = _load_graph()
    texts_meta = _graph_to_texts(graph)
    texts = [t[0] for t in texts_meta]
    metadatas = [t[1] for t in texts_meta]
    embeddings = get_langchain_embeddings()
    client = config.vector_db.client
    coll = config.RAG_GRAPH_COLLECTION
    try:
        client.delete_collection(coll)
    except Exception:
        pass
    dim = len(embeddings.embed_query("probe"))
    client.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    store = Qdrant(
        client=client,
        collection_name=coll,
        embeddings=embeddings,
    )
    store.add_texts(texts, metadatas=metadatas)
    invalidate_expert_retriever_cache()
    return True


def get_retriever(k: int = 6):
    return get_expert_retriever(k=k)
