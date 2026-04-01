import argparse
import os
import sys
import time
import uuid
from typing import Optional

import httpx
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vernadskiy_boltalka.config import config
from vernadskiy_boltalka.paths import project_root
from vernadskiy_boltalka.rag_pipeline.preprocessing_data import Processor


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    if not texts:
        return []

    cfg = config.embedding_model

    if cfg and cfg.EMB_API_BASE:
        model = model_name or cfg.EMB_MODEL
        if not model:
            return []

        api_key = cfg.EMB_API_KEY

        http_client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        client = OpenAI(
            api_key=api_key if api_key else None,
            base_url=cfg.EMB_API_BASE,
            http_client=http_client,
        )

        embeddings = []
        for idx, text in enumerate(texts):
            if idx > 0 and idx % 10 == 0:
                print(f"Обработано {idx}/{len(texts)} текстов...")

            max_retries, retry_delay = 3, 1.0
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(model=model, input=text)
                    if response.data and len(response.data) > 0:
                        embeddings.append(list(response.data[0].embedding))
                        break
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        break
                except Exception as e:
                    print(f"Ошибка для текста {idx + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        break
        return embeddings


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


def run(data_dir: str, recreate: bool = False, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
    processor = Processor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.process_dir(data_dir)
    if not chunks:
        sys.exit(1)

    print(f"Чанков: {len(chunks)}")
    client = _get_qdrant_client()
    collection = config.vector_db.COLLECTION
    print(f"Коллекция: {collection}")

    build_collection(chunks=chunks, collection_name=collection, client=client, recreate=recreate)
    print("Готово.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предобработка текстов → эмбеддинги → Qdrant")
    parser.add_argument("--data-dir", type=str, default=None, help="Папка с PDF/EPUB/DOCX (по умолч. vernadskiy_data)")
    parser.add_argument("--recreate", action="store_true", help="Пересоздать коллекцию")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    root = project_root()
    data_dir = args.data_dir or os.path.join(root, "vernadskiy_data")
    run(data_dir=data_dir, recreate=args.recreate, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
