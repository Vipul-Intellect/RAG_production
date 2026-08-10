import hashlib
from typing import TypedDict
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkGroup(TypedDict):
    parent: Document
    children: list[Document]


def _content_checksum(text: str) -> str:
    """Return SHA-256 checksum for chunk text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_parent_child_chunks(
    documents: list[Document],
    parent_chunk_size: int = 1900,
    parent_chunk_overlap: int = 300,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 80,
) -> list[ChunkGroup]:
    """Create parent-child chunks while preserving existing document metadata."""

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
        parent_chunks = parent_splitter.split_documents([document])

        for parent_chunk in parent_chunks:
            parent_id = str(uuid4())
            parent_chunk.metadata["parent_id"] = parent_id

            # Each parent is split independently so its children remain associated with it.
            child_chunks = child_splitter.split_documents([parent_chunk])
            parent_chunk.metadata["parent_checksum"] = _content_checksum(
                parent_chunk.page_content
            )

            for child_chunk in child_chunks:
                child_chunk.metadata["parent_id"] = parent_id
                child_chunk.metadata["child_id"] = str(uuid4())
                child_chunk.metadata["child_checksum"] = _content_checksum(
                    child_chunk.page_content
                )

            chunk_hierarchy.append(
                {
                    "parent": parent_chunk,
                    "children": child_chunks,
                }
            )

    return chunk_hierarchy
