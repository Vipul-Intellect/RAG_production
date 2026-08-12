from time import perf_counter
from typing import Any, TypedDict

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_app.embeddings.embedder import BGEEmbeddingModel, EXPECTED_EMBEDDING_DIMENSION
from rag_app.observability.logger import logger
from rag_app.storage.qdrant_collections import (
    CHILD_CHUNKS_COLLECTION,
    PARENT_CHUNKS_COLLECTION,
)
from rag_app.storage.qdrant_connection import get_qdrant_client
from rag_app.storage.qdrant_storage import (
    ACTIVE_STATUS,
    PAYLOAD_CONTENT_FIELD,
    build_parent_point_id,
)


class RetrievedChild(TypedDict):
    point_id: str
    score: float
    content: str | None
    child_id: Any
    parent_id: Any
    document_id: Any
    document_checksum: Any
    chunking_config_hash: Any
    parent_index: Any
    child_index: Any
    status: Any
    metadata: dict[str, Any]


class RetrievedParent(TypedDict):
    point_id: str
    content: str | None
    parent_id: Any
    document_id: Any
    document_checksum: Any
    chunking_config_hash: Any
    parent_index: Any
    status: Any
    metadata: dict[str, Any]


class RetrievalResult(TypedDict):
    query: str
    requested_k: int
    retrieved_children: list[RetrievedChild]
    corresponding_parents: list[RetrievedParent]
    retrieved_child_count: int
    retrieved_parent_count: int
    duration_seconds: float


def _active_filter() -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="status",
                match=MatchValue(value=ACTIVE_STATUS),
            )
        ]
    )


def _query_response_points(response: Any) -> list[Any]:
    return list(getattr(response, "points", response))


def _payload(record: Any) -> dict[str, Any]:
    return dict(getattr(record, "payload", None) or {})


def _score(record: Any) -> float:
    return float(getattr(record, "score", 0.0))


def _point_id(record: Any) -> str:
    return str(getattr(record, "id"))


def _to_retrieved_child(record: Any) -> RetrievedChild:
    payload = _payload(record)
    return {
        "point_id": _point_id(record),
        "score": _score(record),
        "content": payload.get(PAYLOAD_CONTENT_FIELD),
        "child_id": payload.get("child_id"),
        "parent_id": payload.get("parent_id"),
        "document_id": payload.get("document_id"),
        "document_checksum": payload.get("document_checksum"),
        "chunking_config_hash": payload.get("chunking_config_hash"),
        "parent_index": payload.get("parent_index"),
        "child_index": payload.get("child_index"),
        "status": payload.get("status"),
        "metadata": payload,
    }


def _to_retrieved_parent(record: Any) -> RetrievedParent:
    payload = _payload(record)
    return {
        "point_id": _point_id(record),
        "content": payload.get(PAYLOAD_CONTENT_FIELD),
        "parent_id": payload.get("parent_id"),
        "document_id": payload.get("document_id"),
        "document_checksum": payload.get("document_checksum"),
        "chunking_config_hash": payload.get("chunking_config_hash"),
        "parent_index": payload.get("parent_index"),
        "status": payload.get("status"),
        "metadata": payload,
    }


def _parent_point_id_from_child(child: RetrievedChild) -> str | None:
    required_values = (
        child["document_id"],
        child["document_checksum"],
        child["chunking_config_hash"],
        child["parent_index"],
    )
    if any(value is None for value in required_values):
        return None

    parent_document = Document(
        page_content="",
        metadata={
            "document_id": child["document_id"],
            "document_checksum": child["document_checksum"],
            "chunking_config_hash": child["chunking_config_hash"],
            "parent_index": child["parent_index"],
        },
    )
    return build_parent_point_id(parent_document)


def retrieve_context(
    query: str,
    k: int,
    embedding_model: BGEEmbeddingModel | None = None,
    client: QdrantClient | None = None,
) -> RetrievalResult:
    """Retrieve ACTIVE child chunks and their corresponding parent context."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Retrieval query must be a non-empty string.")
    if k <= 0:
        raise ValueError("Retrieval k must be greater than 0.")

    start_time = perf_counter()
    model = embedding_model or BGEEmbeddingModel()
    qdrant_client = client or get_qdrant_client()

    logger.info("Retrieval started. k=%s", k)
    query_vector = model.embed_text(query)
    if len(query_vector) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Query embedding dimension mismatch: expected {EXPECTED_EMBEDDING_DIMENSION}, got {len(query_vector)}"
        )

    response = qdrant_client.query_points(
        collection_name=CHILD_CHUNKS_COLLECTION,
        query=query_vector,
        query_filter=_active_filter(),
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    child_records = _query_response_points(response)
    retrieved_children = [
        child
        for child in (_to_retrieved_child(record) for record in child_records)
        if child["status"] == ACTIVE_STATUS
    ]

    parent_point_ids = []
    child_parent_ids_by_point_id: dict[str, set[Any]] = {}
    for child in retrieved_children:
        parent_point_id = _parent_point_id_from_child(child)
        if parent_point_id is None:
            logger.warning(
                "Retrieved child is missing parent lookup metadata. child_point_id=%s",
                child["point_id"],
            )
            continue
        parent_point_ids.append(parent_point_id)
        child_parent_ids_by_point_id.setdefault(parent_point_id, set()).add(
            child["parent_id"]
        )

    unique_parent_point_ids = list(dict.fromkeys(parent_point_ids))

    parent_records = []
    if unique_parent_point_ids:
        parent_records = qdrant_client.retrieve(
            collection_name=PARENT_CHUNKS_COLLECTION,
            ids=unique_parent_point_ids,
            with_payload=True,
            with_vectors=False,
        )

    corresponding_parents = []
    for record in parent_records:
        parent = _to_retrieved_parent(record)
        expected_parent_ids = child_parent_ids_by_point_id.get(parent["point_id"], set())
        if parent["parent_id"] not in expected_parent_ids:
            logger.warning(
                "Retrieved parent_id mismatch. parent_point_id=%s",
                parent["point_id"],
            )
            continue
        corresponding_parents.append(parent)

    if retrieved_children and len(corresponding_parents) < len(unique_parent_point_ids):
        logger.warning(
            "One or more parent chunks were missing during retrieval. children=%s parents=%s",
            len(retrieved_children),
            len(corresponding_parents),
        )

    duration_seconds = perf_counter() - start_time
    if not retrieved_children:
        logger.info("Retrieval completed with no ACTIVE child results.")
    logger.info(
        "Retrieval completed. child_results=%s parent_results=%s duration=%.3fs",
        len(retrieved_children),
        len(corresponding_parents),
        duration_seconds,
    )

    return {
        "query": query,
        "requested_k": k,
        "retrieved_children": retrieved_children,
        "corresponding_parents": corresponding_parents,
        "retrieved_child_count": len(retrieved_children),
        "retrieved_parent_count": len(corresponding_parents),
        "duration_seconds": duration_seconds,
    }
