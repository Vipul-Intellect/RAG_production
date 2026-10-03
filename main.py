import os
import sys
from dotenv import load_dotenv

from rag_app.ingestion.loader import load_documents
from rag_app.chunking.chunker import create_parent_child_chunks
from rag_app.embeddings.embedder import embed_chunks
from rag_app.storage.qdrant_collections import ensure_qdrant_collections
from rag_app.storage.qdrant_storage import store_embedded_chunks
from rag_app.retrieval.retriever import retrieve_context
from rag_app.generation.generator import generate_answer

def main():
    print("--- RAG Production Demo ---")
    load_dotenv()
    
    required_envs = ["QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"]
    missing = [env for env in required_envs if not os.getenv(env)]
    if missing:
        print(f"ERROR: Missing environment variables: {missing}")
        sys.exit(1)
        
    print("\n1. INGESTION")
    try:
        documents, summary = load_documents(pdf_dir="data/pdf")
        print("Ingestion Statistics:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)
        
    if not documents:
        print("No documents loaded. Exiting.")
        sys.exit(0)
        
    print("\n2. CHUNKING")
    try:
        chunk_groups = create_parent_child_chunks(documents)
        parent_count = len(chunk_groups)
        child_count = sum(len(g["children"]) for g in chunk_groups)
        print(f"Created {parent_count} parent chunks and {child_count} child chunks.")
    except Exception as e:
        print(f"Chunking failed: {e}")
        sys.exit(1)
        
    print("\n3. EMBEDDINGS")
    try:
        embedded_groups = embed_chunks(chunk_groups)
        print("Embedding generated successfully.")
    except Exception as e:
        print(f"Embedding failed: {e}")
        sys.exit(1)
        
    print("\n4. QDRANT STORAGE")
    try:
        ensure_qdrant_collections()
        storage_result = store_embedded_chunks(embedded_groups)
        print("Qdrant Indexing Result:")
        for k, v in storage_result.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Qdrant storage failed: {e}")
        sys.exit(1)
        
    print("\n5. RETRIEVAL")
    demo_query = "What is the Human Capital Strategy?"
    print(f"Query: '{demo_query}'")
    try:
        retrieval_result = retrieve_context(query=demo_query, k=5)
        print(f"Retrieved child count: {retrieval_result['retrieved_child_count']}")
        print(f"Retrieved parent count: {retrieval_result['retrieved_parent_count']}")
        print(f"Retrieval duration: {retrieval_result['duration_seconds']:.3f}s")
    except Exception as e:
        print(f"Retrieval failed: {e}")
        sys.exit(1)
        
    print("\n6. GENERATION")
    try:
        gen_result = generate_answer(retrieval_result)
        print(f"\nAnswer:\n{gen_result['answer']}\n")
        print("Citations:")
        for c in gen_result['citations']:
            print(f"  [{c['label']}] File: {c.get('source_file')}, Page: {c.get('page_start')}")
        print(f"\nGeneration timing: {gen_result['duration_seconds']:.3f}s")
        print(f"Citation validation passed: {gen_result['citation_validation_passed']}")
    except Exception as e:
        print(f"Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
