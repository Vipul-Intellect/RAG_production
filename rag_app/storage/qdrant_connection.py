import os
from dataclasses import dataclass

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rag_app.observability.logger import logger

QDRANT_URL_ENV = "QDRANT_URL"
QDRANT_API_KEY_ENV = "QDRANT_API_KEY"


@dataclass(frozen=True)
class QdrantConfig:
    url: str
    api_key: str


def load_qdrant_config() -> QdrantConfig:
    """Load required Qdrant credentials from environment variables."""
    load_dotenv()

    qdrant_url = os.getenv(QDRANT_URL_ENV)
    qdrant_api_key = os.getenv(QDRANT_API_KEY_ENV)

    if not qdrant_url:
        raise ValueError(f"Missing required environment variable: {QDRANT_URL_ENV}")
    if not qdrant_api_key:
        raise ValueError(f"Missing required environment variable: {QDRANT_API_KEY_ENV}")

    logger.info("Qdrant configuration loaded from environment.")
    return QdrantConfig(url=qdrant_url, api_key=qdrant_api_key)


def get_qdrant_client(config: QdrantConfig | None = None) -> QdrantClient:
    """Create a Qdrant client from environment-based configuration."""
    qdrant_config = config or load_qdrant_config()
    logger.info("Creating Qdrant client.")
    return QdrantClient(
        url=qdrant_config.url,
        api_key=qdrant_config.api_key,
    )


def verify_qdrant_connection(client: QdrantClient | None = None) -> bool:
    """Verify Qdrant is reachable without creating collections or writing data."""
    try:
        qdrant_client = client or get_qdrant_client()
        qdrant_client.get_collections()
        logger.info("Qdrant connection verified successfully.")
        return True
    except Exception as exc:
        logger.error("Qdrant connection verification failed: %s", exc.__class__.__name__)
        return False
