import json
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import Distance, VectorParams

from vernadskiy_boltalka.config import config
from vernadskiy_boltalka.paths import project_root


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


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
    )


def build_index():
    graph = _load_graph()
    texts_meta = _graph_to_texts(graph)
    texts = [t[0] for t in texts_meta]
    metadatas = [t[1] for t in texts_meta]
    embeddings = _get_embeddings()
    client = config.vector_db.client
    try:
        client.delete_collection(config.vector_db.COLLECTION)
    except Exception:
        pass
    dim = len(embeddings.embed_query("x"))
    client.create_collection(
        collection_name=config.vector_db.COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    store = Qdrant(
        client=client,
        collection_name=config.vector_db.COLLECTION,
        embeddings=embeddings,
    )
    store.add_texts(texts, metadatas=metadatas)
    return True


def get_retriever(k: int = 4):
    embeddings = _get_embeddings()
    client = config.vector_db.client
    store = Qdrant(
        client=client,
        collection_name=config.vector_db.COLLECTION,
        embeddings=embeddings,
    )
    return store.as_retriever(search_kwargs={"k": k})
