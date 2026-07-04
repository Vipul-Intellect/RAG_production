from rag_app.ingestion.loader import load_documents

documents, summary = load_documents()

print("\n===== Loading Summary =====")
print(summary)

print(f"\nTotal Documents: {len(documents)}")

if documents:
    print("\n===== First Document Metadata =====")
    print(documents[0].metadata)

    print("\n===== First Document Content =====")
    print(documents[0].page_content[:500])