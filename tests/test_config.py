from vernadskiy_boltalka.config import Config, VectorDBConfig


def test_vector_db_config_client_factory():
    v = VectorDBConfig(QDRANT_URL="http://localhost:6333", QDRANT_API_KEY="", COLLECTION="t")
    c = v.client
    assert c is not None


def test_config_has_collection():
    c = Config()
    assert c.COLLECTION
    assert c.vector_db.COLLECTION == c.COLLECTION
    assert c.RAG_GRAPH_COLLECTION
    assert c.RAG_DOCUMENTS_COLLECTION
