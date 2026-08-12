from typing import Any
from time import perf_counter
from uuid import UUID, uuid5

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from rag_app.embeddings.embedder import (
    EXPECTED_EMBEDDING_DIMENSION,
    EmbeddedChunkGroup,
    EmbeddedDocument,
)
from rag_app.observability.logger import logger
from rag_app.storage.qdrant_collections import (
    CHILD_CHUNKS_COLLECTION,
    PARENT_CHUNKS_COLLECTION,
)
from rag_app.storage.qdrant_connection import get_qdrant_client

QDRANT_POINT_NAMESPACE = UUID("2c02eac1-6d6d-4b9e-9e8f-6a8f62564b29")
PAYLOAD_CONTENT_FIELD = "content"
PARENT_POINT_ID_FIELDS = (
    "document_id",
    "document_checksum",
    "chunking_config_hash",
    "parent_index",
)
CHILD_POINT_ID_FIELDS = (
    "document_id",
    "document_checksum",
    "chunking_config_hash",
    "parent_index",
    "child_index",
)
GENERATION_ID_FIELDS = ("document_id", "document_checksum", "chunking_config_hash")
ACTIVE_STATUS = "ACTIVE"
INACTIVE_STATUS = "INACTIVE"


def _require_metadata(document: Document, field_name: str) -> Any:
    if field_name not in document.metadata:
        raise ValueError(f"Missing required metadata field for Qdrant point ID: {field_name}")
    return document.metadata[field_name]


def build_parent_point_id(document: Document) -> str:
    """Build deterministic Qdrant point ID for a Parent chunk."""
    values = {
        field_name: _require_metadata(document, field_name)
        for field_name in PARENT_POINT_ID_FIELDS
    }
    point_key = (
        "parent:"
        + str(values["document_id"])
        + ":"
        + str(values["document_checksum"])
        + ":"
        + str(values["chunking_config_hash"])
        + ":"
        + str(values["parent_index"])
    )
    return str(uuid5(QDRANT_POINT_NAMESPACE, point_key))


def build_child_point_id(document: Document) -> str:
    """Build deterministic Qdrant point ID for a Child chunk."""
    values = {
        field_name: _require_metadata(document, field_name)
        for field_name in CHILD_POINT_ID_FIELDS
    }
    point_key = (
        "child:"
        + str(values["document_id"])
        + ":"
        + str(values["document_checksum"])
        + ":"
        + str(values["chunking_config_hash"])
        + ":"
        + str(values["parent_index"])
        + ":"
        + str(values["child_index"])
    )
    return str(uuid5(QDRANT_POINT_NAMESPACE, point_key))


def _validate_vector(vector: list[float], label: str) -> None:
    if not isinstance(vector, list):
        raise TypeError(f"{label} vector must be a list of floats.")
    if len(vector) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{label} vector dimension mismatch: expected {EXPECTED_EMBEDDING_DIMENSION}, got {len(vector)}"
        )
    if not all(isinstance(value, float) for value in vector):
        raise TypeError(f"{label} vector must contain only float values.")


def _build_payload(document: Document, status: str | None = None) -> dict[str, Any]:
    payload = dict(document.metadata)
    if status is not None:
        payload["status"] = status
    payload[PAYLOAD_CONTENT_FIELD] = document.page_content
    return payload


def _generation_identity(document: Document) -> tuple[str, str, str]:
    return tuple(
        str(_require_metadata(document, field_name))
        for field_name in GENERATION_ID_FIELDS
    )


def _payload_generation_matches(payload: dict[str, Any], generation: tuple[str, str, str]) -> bool:
    return (
        str(payload.get("document_id")) == generation[0]
        and str(payload.get("document_checksum")) == generation[1]
        and str(payload.get("chunking_config_hash")) == generation[2]
    )


def _parent_point(
    embedded_parent: EmbeddedDocument,
    status: str = ACTIVE_STATUS,
) -> PointStruct:
    document = embedded_parent["document"]
    vector = embedded_parent["embedding"]
    _validate_vector(vector, "Parent")
    _require_metadata(document, "parent_id")

    return PointStruct(
        id=build_parent_point_id(document),
        vector=vector,
        payload=_build_payload(document, status),
    )


def _child_point(
    embedded_child: EmbeddedDocument,
    parent_document: Document,
    status: str = ACTIVE_STATUS,
) -> PointStruct:
    document = embedded_child["document"]
    vector = embedded_child["embedding"]
    _validate_vector(vector, "Child")

    parent_id = _require_metadata(parent_document, "parent_id")
    child_parent_id = _require_metadata(document, "parent_id")
    if child_parent_id != parent_id:
        raise ValueError("Child parent_id does not match Parent parent_id.")

    return PointStruct(
        id=build_child_point_id(document),
        vector=vector,
        payload=_build_payload(document, status),
    )


def _active_document_filter(document_id: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            FieldCondition(key="status", match=MatchValue(value=ACTIVE_STATUS)),
        ]
    )


def _active_old_point_ids(
    client: QdrantClient,
    collection_name: str,
    document_id: str,
    current_generation: tuple[str, str, str],
) -> list[str]:
    old_point_ids: list[str] = []
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=_active_document_filter(document_id),
            with_payload=True,
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for record in records:
            payload = record.payload or {}
            if not _payload_generation_matches(payload, current_generation):
                old_point_ids.append(str(record.id))
        if offset is None:
            return old_point_ids


def _active_current_point_ids(
    client: QdrantClient,
    collection_name: str,
    document_id: str,
    current_generation: tuple[str, str, str],
) -> list[str]:
    current_point_ids: list[str] = []
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=_active_document_filter(document_id),
            with_payload=True,
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for record in records:
            payload = record.payload or {}
            if _payload_generation_matches(payload, current_generation):
                current_point_ids.append(str(record.id))
        if offset is None:
            return current_point_ids


def _collection_count(client: QdrantClient, collection_name: str) -> int:
    return int(client.count(collection_name=collection_name, exact=True).count)


def _verify_points_stored(
    client: QdrantClient,
    collection_name: str,
    point_ids: list[str],
) -> None:
    if not point_ids:
        return
    records = client.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_payload=False,
        with_vectors=False,
    )
    if len(records) != len(point_ids):
        raise RuntimeError(
            f"Qdrant verification failed for {collection_name}: expected {len(point_ids)}, got {len(records)}"
        )


def _set_point_status(
    client: QdrantClient,
    collection_name: str,
    point_ids: list[str],
    status: str,
) -> None:
    if not point_ids:
        return
    client.set_payload(
        collection_name=collection_name,
        payload={"status": status},
        points=point_ids,
    )


def _activate_generation(
    client: QdrantClient,
    parent_point_ids: list[str],
    child_point_ids: list[str],
) -> None:
    logger.info(
        "Activating new Qdrant generation. parent_points=%s child_points=%s",
        len(parent_point_ids),
        len(child_point_ids),
    )
    _set_point_status(
        client,
        PARENT_CHUNKS_COLLECTION,
        parent_point_ids,
        ACTIVE_STATUS,
    )
    _set_point_status(
        client,
        CHILD_CHUNKS_COLLECTION,
        child_point_ids,
        ACTIVE_STATUS,
    )


def _deactivate_generation(
    client: QdrantClient,
    parent_point_ids: list[str],
    child_point_ids: list[str],
) -> None:
    logger.info(
        "Deactivating old Qdrant generation. parent_points=%s child_points=%s",
        len(parent_point_ids),
        len(child_point_ids),
    )
    _set_point_status(
        client,
        PARENT_CHUNKS_COLLECTION,
        parent_point_ids,
        INACTIVE_STATUS,
    )
    _set_point_status(
        client,
        CHILD_CHUNKS_COLLECTION,
        child_point_ids,
        INACTIVE_STATUS,
    )


def _restore_generation_status(
    client: QdrantClient,
    new_parent_point_ids: list[str],
    new_child_point_ids: list[str],
    old_parent_point_ids: list[str],
    old_child_point_ids: list[str],
) -> None:
    logger.info(
        "Qdrant lifecycle recovery attempted. new_parent_points=%s new_child_points=%s old_parent_points=%s old_child_points=%s",
        len(new_parent_point_ids),
        len(new_child_point_ids),
        len(old_parent_point_ids),
        len(old_child_point_ids),
    )
    recovery_errors: list[str] = []

    recovery_steps = (
        (PARENT_CHUNKS_COLLECTION, new_parent_point_ids, INACTIVE_STATUS),
        (CHILD_CHUNKS_COLLECTION, new_child_point_ids, INACTIVE_STATUS),
        (PARENT_CHUNKS_COLLECTION, old_parent_point_ids, ACTIVE_STATUS),
        (CHILD_CHUNKS_COLLECTION, old_child_point_ids, ACTIVE_STATUS),
    )

    for collection_name, point_ids, status in recovery_steps:
        try:
            _set_point_status(client, collection_name, point_ids, status)
        except Exception as exc:
            recovery_errors.append(f"{collection_name}:{status}:{exc.__class__.__name__}")

    if recovery_errors:
        logger.error(
            "Qdrant lifecycle recovery failed. errors=%s",
            ",".join(recovery_errors),
        )
    else:
        logger.info("Qdrant lifecycle recovery succeeded.")


def store_embedded_chunks(
    embedded_groups: list[EmbeddedChunkGroup],
    client: QdrantClient | None = None,
) -> dict[str, int]:
    """Store embedded Parent and Child chunks in their locked Qdrant collections."""
    start_time = perf_counter()
    qdrant_client = client or get_qdrant_client()
    parent_points: list[PointStruct] = []
    child_points: list[PointStruct] = []
    generation_by_document_id: dict[str, tuple[str, str, str]] = {}

    logger.info("Starting Qdrant point storage.")

    for embedded_group in embedded_groups:
        parent_document = embedded_group["parent"]["document"]
        generation = _generation_identity(parent_document)
        document_id = generation[0]
        generation_by_document_id[document_id] = generation

        parent_point = _parent_point(embedded_group["parent"], INACTIVE_STATUS)
        child_points.extend(
            _child_point(embedded_child, parent_document, INACTIVE_STATUS)
            for embedded_child in embedded_group["children"]
        )
        parent_points.append(parent_point)

    parent_point_ids = [str(point.id) for point in parent_points]
    child_point_ids = [str(point.id) for point in child_points]
    old_parent_point_ids: list[str] = []
    old_child_point_ids: list[str] = []
    active_current_parent_point_ids: list[str] = []
    active_current_child_point_ids: list[str] = []

    try:
        before_parent_count = _collection_count(qdrant_client, PARENT_CHUNKS_COLLECTION)
        before_child_count = _collection_count(qdrant_client, CHILD_CHUNKS_COLLECTION)

        for document_id, generation in generation_by_document_id.items():
            old_parent_point_ids.extend(
                _active_old_point_ids(
                    qdrant_client,
                    PARENT_CHUNKS_COLLECTION,
                    document_id,
                    generation,
                )
            )
            old_child_point_ids.extend(
                _active_old_point_ids(
                    qdrant_client,
                    CHILD_CHUNKS_COLLECTION,
                    document_id,
                    generation,
                )
            )
            active_current_parent_point_ids.extend(
                _active_current_point_ids(
                    qdrant_client,
                    PARENT_CHUNKS_COLLECTION,
                    document_id,
                    generation,
                )
            )
            active_current_child_point_ids.extend(
                _active_current_point_ids(
                    qdrant_client,
                    CHILD_CHUNKS_COLLECTION,
                    document_id,
                    generation,
                )
            )

        same_generation_point_ids = set(active_current_parent_point_ids) | set(
            active_current_child_point_ids
        )
        for point in parent_points:
            if str(point.id) in same_generation_point_ids:
                point.payload["status"] = ACTIVE_STATUS
        for point in child_points:
            if str(point.id) in same_generation_point_ids:
                point.payload["status"] = ACTIVE_STATUS

        logger.info(
            "Qdrant generation storage started. documents=%s parents_received=%s children_received=%s old_active_parent_points=%s old_active_child_points=%s same_generation_parent_points=%s same_generation_child_points=%s",
            len(generation_by_document_id),
            len(parent_points),
            len(child_points),
            len(old_parent_point_ids),
            len(old_child_point_ids),
            len(active_current_parent_point_ids),
            len(active_current_child_point_ids),
        )

        if parent_points:
            qdrant_client.upsert(
                collection_name=PARENT_CHUNKS_COLLECTION,
                points=parent_points,
            )
        if child_points:
            qdrant_client.upsert(
                collection_name=CHILD_CHUNKS_COLLECTION,
                points=child_points,
            )

        _verify_points_stored(qdrant_client, PARENT_CHUNKS_COLLECTION, parent_point_ids)
        _verify_points_stored(qdrant_client, CHILD_CHUNKS_COLLECTION, child_point_ids)
        logger.info(
            "Qdrant generation verified. parent_points=%s child_points=%s",
            len(parent_point_ids),
            len(child_point_ids),
        )

        try:
            _activate_generation(qdrant_client, parent_point_ids, child_point_ids)
        except Exception:
            _restore_generation_status(
                qdrant_client,
                parent_point_ids,
                child_point_ids,
                old_parent_point_ids + active_current_parent_point_ids,
                old_child_point_ids + active_current_child_point_ids,
            )
            raise

        try:
            _deactivate_generation(
                qdrant_client,
                old_parent_point_ids,
                old_child_point_ids,
            )
        except Exception:
            _restore_generation_status(
                qdrant_client,
                parent_point_ids,
                child_point_ids,
                old_parent_point_ids + active_current_parent_point_ids,
                old_child_point_ids + active_current_child_point_ids,
            )
            raise

        after_parent_count = _collection_count(qdrant_client, PARENT_CHUNKS_COLLECTION)
        after_child_count = _collection_count(qdrant_client, CHILD_CHUNKS_COLLECTION)
    except Exception as exc:
        logger.error(
            "Qdrant generation storage failed. documents=%s parents_received=%s children_received=%s reason=%s",
            len(generation_by_document_id),
            len(parent_points),
            len(child_points),
            exc.__class__.__name__,
        )
        raise

    duration_seconds = perf_counter() - start_time
    logger.info(
        "Qdrant point storage completed. parents=%s children=%s before_parent_count=%s before_child_count=%s after_parent_count=%s after_child_count=%s duration=%.3fs",
        len(parent_points),
        len(child_points),
        before_parent_count,
        before_child_count,
        after_parent_count,
        after_child_count,
        duration_seconds,
    )
    return {
        "parents_stored": len(parent_points),
        "children_stored": len(child_points),
        "before_parent_count": before_parent_count,
        "before_child_count": before_child_count,
        "after_parent_count": after_parent_count,
        "after_child_count": after_child_count,
    }
