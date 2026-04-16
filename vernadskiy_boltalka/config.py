import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class EmbeddingsConfig(BaseModel):
    EMB_MODEL: str = ""
    EMB_API_BASE: str = ""
    EMB_API_KEY: str = ""


class VectorDBConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_PATH: str = ""
    COLLECTION: str = "vernadsky_rag"

    @property
    def client(self) -> "QdrantClient":
        from qdrant_client import QdrantClient

        from vernadskiy_boltalka.paths import project_root

        if self.QDRANT_PATH:
            return QdrantClient(path=self.QDRANT_PATH)
        if self.QDRANT_URL:
            return QdrantClient(
                url=self.QDRANT_URL,
                api_key=self.QDRANT_API_KEY or None,
                check_compatibility=False,
            )
        return QdrantClient(path=os.path.join(project_root(), "qdrant_data"))


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    QDRANT_URL: str = Field(default="")
    QDRANT_API_KEY: str = Field(default="")
    QDRANT_PATH: str = Field(default="")
    COLLECTION: str = Field(default="vernadsky_rag")
    RAG_GRAPH_COLLECTION: str = Field(default="vernadsky_rag_graph")
    RAG_DOCUMENTS_COLLECTION: str = Field(default="vernadsky_rag_docs")

    BOTHUB_BASE_URL: str = Field(default="")
    BOTHUB_API_KEY: str = Field(default="")
    BOTHUB_MODEL: str = Field(default="gemini-2.0-flash-lite-001")
    USE_BOTHUB: bool = Field(default=False)

    OLLAMA_MODEL: str = Field(default="")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")

    USE_QWEN: bool = Field(default=False)
    QWEN_RUADAPT_BASE_URL: str = Field(default="")
    QWEN_RUADAPT_API_KEY: str = Field(default="")
    QWEN_MODEL: str = Field(default="")

    USE_VSEGPT: bool = Field(default=False)
    VSEGPT_API_URL: str = Field(default="")
    VSEGPT_API_KEY: str = Field(default="")
    VSEGPT_MODEL: str = Field(default="")

    OPENAI_API_KEY: str = Field(default="")
    MODEL: str = Field(default="")

    EMB_MODEL: str = Field(default="")
    EMB_API_BASE: str = Field(default="")
    EMB_API_KEY: str = Field(default="")

    @property
    def vector_db(self) -> VectorDBConfig:
        return VectorDBConfig(
            QDRANT_URL=self.QDRANT_URL,
            QDRANT_API_KEY=self.QDRANT_API_KEY,
            QDRANT_PATH=self.QDRANT_PATH,
            COLLECTION=self.COLLECTION,
        )

    @property
    def embedding_model(self) -> EmbeddingsConfig:
        return EmbeddingsConfig(
            EMB_MODEL=self.EMB_MODEL,
            EMB_API_BASE=self.EMB_API_BASE,
            EMB_API_KEY=self.EMB_API_KEY,
        )


config = Config()


def load_config() -> Config:
    global config
    config = Config()
    return config
