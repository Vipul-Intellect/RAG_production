from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag_app.embeddings.embedder import EXPECTED_EMBEDDING_DIMENSION
from rag_app.observability.logger import logger
from rag_app.storage.qdrant_connection import get_qdrant_client

PARENT_CHUNKS_COLLECTION = "parent_chunks"
CHILD_CHUNKS_COLLECTION = "child_chunks"
REQUIRED_COLLECTIONS = (PARENT_CHUNKS_COLLECTION, CHILD_CHUNKS_COLLECTION)
QDRANT_VECTOR_DISTANCE = Distance.COSINE


def ensure_qdrant_collections(client: QdrantClient | None = None) -> None:
    """Create required Qdrant collections when missing, without recreating existing ones."""
    qdrant_client = client or get_qdrant_client()
    logger.info("Qdrant collection initialization started.")

    for collection_name in REQUIRED_COLLECTIONS:
        try:
            if not qdrant_client.collection_exists(collection_name):
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=EXPECTED_EMBEDDING_DIMENSION,
                        distance=QDRANT_VECTOR_DISTANCE,
                    ),
                )
                logger.info("Qdrant collection created: %s", collection_name)
            else:
                logger.info("Qdrant collection already exists: %s", collection_name)

            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema="keyword",
            )
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="status",
                field_schema="keyword",
            )
        except Exception as exc:
            logger.error(
                "Qdrant collection initialization failed for %s: %s",
                collection_name,
                exc.__class__.__name__,
            )
            raise RuntimeError(
                f"Qdrant collection initialization failed for {collection_name}"
            ) from exc
