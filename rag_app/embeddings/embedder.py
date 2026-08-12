from typing import Any
from typing import TypedDict
from time import perf_counter

from rag_app.chunking.chunker import ChunkGroup
from langchain_core.documents import Document
from rag_app.observability.logger import logger

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_EMBEDDING_DIMENSION = 384


class EmbeddedDocument(TypedDict):
    document: Document
    embedding: list[float]


class EmbeddedChunkGroup(TypedDict):
    parent: EmbeddedDocument
    children: list[EmbeddedDocument]


def _load_sentence_transformer(model_name: str) -> Any:
    """Load the local embedding model only when an embedding model is created."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class BGEEmbeddingModel:
    """Small wrapper around the locked local BGE embedding model."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    ) -> None:
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self._model = _load_sentence_transformer(model_name)
        self.actual_dimension = self._model.get_sentence_embedding_dimension()
        if self.actual_dimension != self.expected_dimension:
            raise ValueError(
                f"Embedding model dimension mismatch for {self.model_name}: "
                f"expected {self.expected_dimension}, got {self.actual_dimension}"
            )

    def embed_text(self, text: str) -> list[float]:
        """Generate one embedding and validate the expected vector dimension."""
        if not isinstance(text, str):
            raise TypeError("Embedding input must be a string.")

        vector = self._model.encode(text, normalize_embeddings=True)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        embedding = [float(value) for value in vector]
        if len(embedding) != self.expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dimension}, got {len(embedding)}"
            )

        return embedding


def embed_chunks(
    chunk_groups: list[ChunkGroup],
    embedding_model: BGEEmbeddingModel | None = None,
) -> list[EmbeddedChunkGroup]:
    """Embed each Parent/Child group while keeping vectors separate from metadata."""
    start_time = perf_counter()
    model = embedding_model or BGEEmbeddingModel()
    parents_received = len(chunk_groups)
    children_received = sum(len(group["children"]) for group in chunk_groups)
    parents_embedded = 0
    children_embedded = 0
    failed_groups = 0

    logger.info("Starting embedding phase.")
    logger.info(
        "CONFIG Embedding: model_name=%s expected_dimension=%s",
        model.model_name,
        model.expected_dimension,
    )

    embedded_groups: list[EmbeddedChunkGroup] = []
    for chunk_group in chunk_groups:
        parent = chunk_group["parent"]
        children = chunk_group["children"]
        source_file = (
            parent.metadata.get("source_file")
            or parent.metadata.get("source")
            or "unknown"
        )
        document_id = parent.metadata.get("document_id")

        try:
            parent_embedding = model.embed_text(parent.page_content)
            embedded_children = [
                {
                    "document": child,
                    "embedding": model.embed_text(child.page_content),
                }
                for child in children
            ]
            embedded_groups.append(
                {
                    "parent": {
                        "document": parent,
                        "embedding": parent_embedding,
                    },
                    "children": embedded_children,
                }
            )
            parents_embedded += 1
            children_embedded += len(embedded_children)
        except Exception as exc:
            failed_groups += 1
            if document_id:
                logger.error(
                    "Embedding failed for chunk group source_file=%r document_id=%r reason=%r",
                    source_file,
                    document_id,
                    exc,
                )
            else:
                logger.error(
                    "Embedding failed for chunk group source_file=%r reason=%r",
                    source_file,
                    exc,
                )

    duration_seconds = perf_counter() - start_time
    logger.info("Embedding phase completed.")
    logger.info(
        "STATISTICS Embedding: parents_received=%s parents_embedded=%s children_received=%s children_embedded=%s failed_groups=%s model_name=%s dimension=%s duration=%.3fs",
        parents_received,
        parents_embedded,
        children_received,
        children_embedded,
        failed_groups,
        model.model_name,
        model.expected_dimension,
        duration_seconds,
    )

    return embedded_groups
