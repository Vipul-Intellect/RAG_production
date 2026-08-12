import hashlib
import json
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.observability.logger import logger


class ChunkGroup(TypedDict):
    parent: Document
    children: list[Document]


REQUIRED_LOADING_METADATA = {
    "document_id",
    "document_checksum",
    "source_file",
    "source_path",
    "domain",
    "document_type",
    "loader_used",
    "page_number",
    "character_count",
    "is_empty_page",
    "ingested_at",
}


def _content_checksum(text: str) -> str:
    """Return SHA-256 checksum for chunk text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunking_config_hash(
    splitter_name: str,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> str:
    """Return a stable SHA-256 fingerprint of the actual chunking config."""
    canonical_config = {
        "child_chunk_overlap": child_chunk_overlap,
        "child_chunk_size": child_chunk_size,
        "parent_chunk_overlap": parent_chunk_overlap,
        "parent_chunk_size": parent_chunk_size,
        "splitter": splitter_name,
    }
    canonical_text = json.dumps(
        canonical_config,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def _ensure_loading_metadata(chunk: Document) -> None:
    """Ensure required Loading metadata exists before adding chunk metadata."""
    missing_fields = REQUIRED_LOADING_METADATA - set(chunk.metadata)
    if missing_fields:
        raise ValueError(
            f"Chunk is missing required Loading metadata: {sorted(missing_fields)}"
        )


def _apply_common_chunk_metadata(chunk: Document, chunking_config_hash: str) -> None:
    """Add finalized chunk metadata that is common to parent and child chunks."""
    _ensure_loading_metadata(chunk)
    page_number = chunk.metadata.get("page_number")
    chunk.metadata.update(
        {
            "chunking_config_hash": chunking_config_hash,
            "page_start": page_number,
            "page_end": page_number,
            "status": "ACTIVE",
            "version": chunk.metadata.get("version"),
            "updated_at": chunk.metadata.get("updated_at"),
            "indexed_at": chunk.metadata.get("indexed_at"),
        }
    )


def _log_document_outcome(
    source_file: str,
    document_id: str | None,
    outcome: str,
    reason: Exception | None = None,
) -> None:
    """Log document outcome without fabricating unavailable identifiers."""
    if outcome == "FAILED":
        if document_id and reason:
            logger.error(
                "Document chunking failed source_file=%r document_id=%r status=FAILED reason=%r",
                source_file,
                document_id,
                reason,
            )
        elif document_id:
            logger.error(
                "Document chunking failed source_file=%r document_id=%r status=FAILED",
                source_file,
                document_id,
            )
        elif reason:
            logger.error(
                "Document chunking failed source_file=%r status=FAILED reason=%r",
                source_file,
                reason,
            )
        else:
            logger.error(
                "Document chunking failed source_file=%r status=FAILED",
                source_file,
            )
        return

    if document_id:
        logger.info(
            "Document chunking completed source_file=%r document_id=%r status=%s",
            source_file,
            document_id,
            outcome,
        )
    else:
        logger.info(
            "Document chunking completed source_file=%r status=%s",
            source_file,
            outcome,
        )


def _log_parent_failure(
    source_file: str,
    document_id: str | None,
    parent_index: int,
    parent_id: str | None,
    reason: Exception,
) -> None:
    """Log parent failure without logging chunk content."""
    if document_id and parent_id:
        logger.error(
            "Parent chunking failed source_file=%r document_id=%r parent_index=%s parent_id=%r reason=%r",
            source_file,
            document_id,
            parent_index,
            parent_id,
            reason,
        )
    elif document_id:
        logger.error(
            "Parent chunking failed source_file=%r document_id=%r parent_index=%s reason=%r",
            source_file,
            document_id,
            parent_index,
            reason,
        )
    elif parent_id:
        logger.error(
            "Parent chunking failed source_file=%r parent_index=%s parent_id=%r reason=%r",
            source_file,
            parent_index,
            parent_id,
            reason,
        )
    else:
        logger.error(
            "Parent chunking failed source_file=%r parent_index=%s reason=%r",
            source_file,
            parent_index,
            reason,
        )


def create_parent_child_chunks(
    documents: list[Document],
    parent_chunk_size: int = 1900,
    parent_chunk_overlap: int = 300,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 80,
) -> list[ChunkGroup]:
    """Create parent-child chunks while preserving existing document metadata."""
    start_time = perf_counter()
    documents_received = len(documents)
    parent_chunks_created = 0
    child_chunks_created = 0
    splitter_name = RecursiveCharacterTextSplitter.__name__
    config_hash = _chunking_config_hash(
        splitter_name,
        parent_chunk_size,
        parent_chunk_overlap,
        child_chunk_size,
        child_chunk_overlap,
    )

    logger.info("Starting chunking.")
    logger.info(
        "CONFIG Chunking: parent_chunk_size=%s parent_chunk_overlap=%s child_chunk_size=%s child_chunk_overlap=%s splitter=%s",
        parent_chunk_size,
        parent_chunk_overlap,
        child_chunk_size,
        child_chunk_overlap,
        splitter_name,
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    chunk_hierarchy: list[ChunkGroup] = []

    for document in documents:
        document_successful_parent_count = 0
        document_failed_parent_count = 0
        source_file = (
            document.metadata.get("source_file")
            or document.metadata.get("source")
            or "unknown"
        )
        document_id = document.metadata.get("document_id")

        try:
            parent_chunks = parent_splitter.split_documents([document])
        except Exception as exc:
            _log_document_outcome(source_file, document_id, "FAILED", exc)
            continue

        for parent_index, parent_chunk in enumerate(parent_chunks):
            parent_id = None

            try:
                parent_id = str(uuid4())
                parent_chunk.metadata["parent_id"] = parent_id

                # Each parent is split independently so its children remain associated with it.
                child_chunks = child_splitter.split_documents([parent_chunk])
                parent_chunk.metadata["parent_index"] = parent_index
                parent_chunk.metadata["parent_checksum"] = _content_checksum(
                    parent_chunk.page_content
                )
                _apply_common_chunk_metadata(parent_chunk, config_hash)

                for child_index, child_chunk in enumerate(child_chunks):
                    child_chunk.metadata["parent_id"] = parent_id
                    child_chunk.metadata["child_id"] = str(uuid4())
                    child_chunk.metadata["child_index"] = child_index
                    child_chunk.metadata["child_checksum"] = _content_checksum(
                        child_chunk.page_content
                    )
                    _apply_common_chunk_metadata(child_chunk, config_hash)

                chunk_hierarchy.append(
                    {
                        "parent": parent_chunk,
                        "children": child_chunks,
                    }
                )
                document_successful_parent_count += 1
                parent_chunks_created += 1
                child_chunks_created += len(child_chunks)
            except Exception as exc:
                document_failed_parent_count += 1
                _log_parent_failure(
                    source_file,
                    document_id,
                    parent_index,
                    parent_id,
                    exc,
                )

        if document_successful_parent_count > 0 and document_failed_parent_count == 0:
            _log_document_outcome(source_file, document_id, "FULL_SUCCESS")
        elif document_successful_parent_count > 0 and document_failed_parent_count > 0:
            _log_document_outcome(source_file, document_id, "PARTIAL_SUCCESS")
        else:
            _log_document_outcome(source_file, document_id, "FAILED")

    duration_seconds = perf_counter() - start_time
    logger.info("Chunking completed.")
    logger.info(
        "STATISTICS Chunking: documents_received=%s parent_chunks_created=%s child_chunks_created=%s duration=%.3fs",
        documents_received,
        parent_chunks_created,
        child_chunks_created,
        duration_seconds,
    )

    return chunk_hierarchy
