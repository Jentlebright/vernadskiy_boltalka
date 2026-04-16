import argparse
import os
import sys
import time
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vernadskiy_boltalka.config import config
from vernadskiy_boltalka.embeddings_core import get_langchain_embeddings
from vernadskiy_boltalka.paths import project_root
from vernadskiy_boltalka.rag_pipeline.llm_chunker import chunk_corpus_with_llm
from vernadskiy_boltalka.rag_pipeline.preprocessing_data import Processor
from vernadskiy_boltalka.rag_retriever import invalidate_expert_retriever_cache


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    emb = get_langchain_embeddings()
    return emb.embed_documents(texts)


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    recreate: bool = False,
) -> None:
    if client.collection_exists(collection):
        if recreate:
            client.delete_collection(collection)
        else:
            return
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    vectors: list[list[float]],
    chunks: list[dict],
    topic: Optional[str] = None,
) -> None:
    if not vectors:
        return
    points = []
    for vec, ch in zip(vectors, chunks):
        payload = {
            "text": ch["text"],
            "chunk": ch["text"],
            "source": ch["meta"]["source"],
            **ch["meta"],
        }
        if topic:
            payload["topic"] = topic
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
    client.upsert(collection_name=collection, points=points)
    print(f"Чанки загружены: {len(points)}")


def build_collection(
    chunks: list[dict],
    collection_name: str,
    client: QdrantClient,
    recreate: bool = False,
    topic: Optional[str] = None,
    model_name: str | None = None,
) -> None:
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, model_name=model_name)
    if not vectors:
        return

    vector_size = len(vectors[0])
    ensure_collection(client, collection_name, vector_size, recreate=recreate)
    upsert_chunks(client, collection_name, vectors, chunks, topic=topic)


def _get_qdrant_client() -> QdrantClient:
    return config.vector_db.client


def run(
    data_dir: str,
    recreate: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_llm_chunks: bool = False,
) -> None:
    if use_llm_chunks:
        proc = Processor()
        pairs = proc.load_from_dir(data_dir)
        if not pairs:
            sys.exit(1)
        chunks, _ = chunk_corpus_with_llm(pairs)
    else:
        processor = Processor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = processor.process_dir(data_dir)
    if not chunks:
        sys.exit(1)

    print(f"Чанков: {len(chunks)}")
    client = _get_qdrant_client()
    collection = config.RAG_DOCUMENTS_COLLECTION
    print(f"Коллекция: {collection}")

    build_collection(chunks=chunks, collection_name=collection, client=client, recreate=recreate)
    invalidate_expert_retriever_cache()
    print("Готово.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предобработка текстов → эмбеддинги → Qdrant")
    parser.add_argument("--data-dir", type=str, default=None, help="Папка с PDF/EPUB/DOCX (по умолч. vernadskiy_data)")
    parser.add_argument("--recreate", action="store_true", help="Пересоздать коллекцию")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--llm-chunks", action="store_true", help="Семантические чанки и overlap через LLM")
    args = parser.parse_args()

    root = project_root()
    data_dir = args.data_dir or os.path.join(root, "vernadskiy_data")
    run(
        data_dir=data_dir,
        recreate=args.recreate,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_llm_chunks=args.llm_chunks,
    )
