import random
import sys
import time
from collections.abc import Callable
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from vernadskiy_boltalka.config import config


class _RetryingEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, max_attempts: int = 12, base_delay: float = 2.0) -> None:
        self._inner = inner
        self._max_attempts = max_attempts
        self._base_delay = base_delay

    def _run(self, fn: Callable[[], Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if not self._retriable(e) or attempt == self._max_attempts - 1:
                    raise
                delay = min(self._base_delay * (2**attempt) + random.uniform(0, 2.0), 90.0)
                print(
                    f"эмбеддинги: сбой ({type(e).__name__}), повтор {attempt + 2}/{self._max_attempts} через {delay:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _retriable(e: Exception) -> bool:
        name = type(e).__name__
        if name in (
            "InternalServerError",
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "APIStatusError",
        ):
            return True
        s = str(e).lower()
        for token in ("502", "503", "504", "429", "timeout", "connection", "temporarily"):
            if token in s:
                return True
        return False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run(lambda: self._inner.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._run(lambda: self._inner.embed_query(text))


def get_langchain_embeddings() -> Embeddings:
    emb = config.embedding_model
    if emb.EMB_API_BASE and emb.EMB_MODEL:
        inner: Embeddings = OpenAIEmbeddings(
            model=emb.EMB_MODEL,
            openai_api_key=emb.EMB_API_KEY or None,
            openai_api_base=emb.EMB_API_BASE,
        )
        return _RetryingEmbeddings(inner)
    inner = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
    )
    return _RetryingEmbeddings(inner)
