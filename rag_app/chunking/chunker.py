from typing import TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkGroup(TypedDict):
    parent: Document
    children: list[Document]


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
            # Each parent is split independently so its children remain associated with it.
            child_chunks = child_splitter.split_documents([parent_chunk])

            chunk_hierarchy.append(
                {
                    "parent": parent_chunk,
                    "children": child_chunks,
                }
            )

    return chunk_hierarchy