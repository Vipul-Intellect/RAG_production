# Cost-Efficient RAG Application

A production-oriented RAG pipeline using Qdrant Cloud, hierarchical
Parent/Child chunking, local BGE embeddings, and Gemini for grounded
answer generation.

## Architecture

PDF / HTML / MD
→ Ingestion + metadata
→ Parent/Child chunking
→ BGE-small-en-v1.5 (384D)
→ Qdrant
→ ACTIVE top-k retrieval
→ Parent recovery
→ Gemini
→ Grounded answer + citations

## Key Features

- PDF/HTML/MD ingestion
- Configurable chunk size and overlap
- Parent/Child hierarchical retrieval
- `BAAI/bge-small-en-v1.5` embeddings (384D)
- Qdrant Cloud with cosine similarity
- ACTIVE/INACTIVE generation lifecycle
- Deterministic UUID5 Point IDs
- Idempotent re-ingestion with no duplicate vectors
- Document checksum and chunking-config hashing
- Metadata filtering
- Grounded Gemini generation
- No-context protection against hallucination
- Retrieval and generation latency/token logging

## Point Identity

Parent:
`UUID5(namespace, document_id + document_checksum + chunking_config_hash + parent_index)`

Child:
`UUID5(namespace, document_id + document_checksum + chunking_config_hash + parent_index + child_index)`

The same corpus and configuration therefore produce the same Point IDs,
allowing Qdrant upsert to remain idempotent.

## Configuration

Set these environment variables:

```env
QDRANT_URL=
QDRANT_API_KEY=
GEMINI_API_KEY=
GEMINI_MODEL=
