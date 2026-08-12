import csv
from datetime import UTC, datetime
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,

)
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_core.documents import Document

from rag_app.observability.logger import logger

SUPPORTED_PDF_LOADERS = {"pymupdf", "pdfplumber"}
SUPPORTED_FILE_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".html", ".md"}


def get_supported_files(data_dir: str | Path) -> list[Path]:
    """Validate the data folder and return supported files in a stable order."""
    data_path = Path(data_dir)
    logger.info("Validating document directory: %s", data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Document directory does not exist: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {data_path}")

    supported_files = sorted(
        file
        for file in data_path.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_FILE_EXTENSIONS
    )
    if not supported_files:
        raise FileNotFoundError(f"No supported files found in: {data_path}")

    logger.info("Discovered %s supported files.", len(supported_files))
    return supported_files


def load_with_pymupdf(pdf_dir: str | Path) -> list[Document]:
    """Primary loader: fast page-level PDF loading with table markdown support."""
    logger.info("Starting PDF loading using PyMuPDFLoader.")
    documents: list[Document] = []
    for file_path in sorted(Path(pdf_dir).glob("**/*.pdf")):
        if not file_path.is_file():
            continue

        documents.extend(
            _load_single_file(
                file_path,
                PyMuPDFLoader,
                {
                    "mode": "page",
                    "extract_tables": "markdown",
                },
            )
        )

    logger.info("Successfully loaded %s pages using PyMuPDFLoader.", len(documents))
    return documents


def load_with_pdfplumber(pdf_dir: str | Path) -> list[Document]:
    """Optional loader: layout/table-aware extraction for explicit comparison."""
    logger.info("Starting PDF loading using PDFPlumberLoader.")
    documents: list[Document] = []
    for file_path in sorted(Path(pdf_dir).glob("**/*.pdf")):
        if not file_path.is_file():
            continue

        documents.extend(
            _load_single_file(
                file_path,
                PDFPlumberLoader,
                {
                    "dedupe": True,
                },
            )
        )

    logger.info("Successfully loaded %s pages using PDFPlumberLoader.", len(documents))
    return documents


def load_non_pdf_documents(
    data_dir: str | Path,
    glob_pattern: str,
    loader_cls: type,
    loader_kwargs: dict[str, Any] | None = None,
) -> list[Document]:
    """Load non-PDF documents using a selected LangChain loader."""
    documents: list[Document] = []
    data_path = Path(data_dir)

    for file_path in sorted(data_path.glob(glob_pattern)):
        if not file_path.is_file():
            continue

        documents.extend(
            _load_single_file(file_path, loader_cls, loader_kwargs)
        )

    return documents


def _load_single_file(
    file_path: Path,
    loader_cls: type,
    loader_kwargs: dict[str, Any] | None = None,
) -> list[Document]:
    """Load one source file so one bad document cannot stop the full batch."""
    try:
        loader = DirectoryLoader(
            str(file_path.parent),
            glob=file_path.name,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            show_progress=True,
        )
        return loader.load()
    except Exception as exc:
        logger.error(
            "Loading failed for document source=%r: %s",
            file_path.as_posix(),
            exc,
        )
        return []


def _file_checksum(file_path: Path) -> str:
    """Generate SHA-256 from the original PDF file for version tracking."""
    sha256 = hashlib.sha256() # SHA256 IS A UNIQUE FINGERPRINT OF THE FILE, IT IS A CRYPTOGRAPHIC HASH FUNCTION THAT TAKES AN INPUT (OR 'MESSAGE') AND RETURNS A FIXED-LENGTH STRING, WHICH IS UNIQUE TO THAT INPUT. EVEN A SMALL CHANGE IN THE INPUT WILL PRODUCE A SIGNIFICANTLY DIFFERENT OUTPUT.
    with file_path.open("rb") as file: # READ AS BINARY BEACUSE IT MAY CONTAIN NON-TEXT DATA LIKE IMAGES OR TABLES
        for chunk in iter(lambda: file.read(1024 * 1024), b""): # DONT READ WHOLW FILE INTO MEMORY, READ IN CHUNKS OF 1MB AND WHEN B"" COMES IT MEANS END OF FILE"
            sha256.update(chunk)
    return sha256.hexdigest() # TO RETURN CRYPTOPGRAPHICALLY HASH VALUE AS HUMAN READABLE STRING


def _relative_path(file_path: Path) -> str:
    """Return a readable relative path when the file is inside the project."""
    try:
        return file_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return file_path.as_posix()


def load_document_metadata(csv_path: str | Path) -> dict[str, dict[str, str]]:
    """Load business metadata from CSV once for O(1) filename lookup."""
    metadata_path = Path(csv_path)
    required_columns = {"filename", "domain", "document_type"}
    logger.info("Loading business metadata CSV: %s", metadata_path)

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV is empty: {metadata_path}")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Metadata CSV missing columns: {sorted(missing_columns)}"
            )

        metadata_lookup = {
            row["filename"]: {
                "domain": row["domain"],
                "document_type": row["document_type"],
            }
            for row in reader
            if row.get("filename")
        }
        logger.info("Successfully loaded metadata for %s documents.", len(metadata_lookup))
        return metadata_lookup


def enrich_metadata(
    documents: list[Document],
    loader_name: str,
    metadata_lookup: dict[str, dict[str, str]],
) -> list[Document]:
    """Add citation and extraction metadata to loaded documents."""
    logger.info("Starting metadata enrichment.")
    checksum_cache: dict[Path, str] = {}
    missing_metadata_files: set[str] = set()
    ingested_at = datetime.now(UTC).isoformat()
    

    for doc in documents:
        source_path = Path(doc.metadata["source"])
        page_number = int(doc.metadata.get("page", 0)) + 1
        text = doc.page_content or ""
        source_file = source_path.name
        business_metadata = metadata_lookup.get(source_file)


        if business_metadata is None:
            if source_file not in missing_metadata_files:
            
                missing_metadata_files.add(source_file)

                logger.warning(
                    "No CSV metadata found for document: %r",source_file,
                )

                logger.info("Available CSV filenames:")

                for filename in metadata_lookup:
                    logger.info("  %r", filename)


            
            domain = None
            document_type = None
        else:
            domain = business_metadata["domain"]
            document_type = business_metadata["document_type"]

        if source_path not in checksum_cache:
            checksum_cache[source_path] = _file_checksum(source_path)
        checksum = checksum_cache[source_path]
        
        relative_path = _relative_path(source_path)
        document_id=hashlib.sha256(relative_path.encode("utf-8")).hexdigest()  # Create a unique document ID based on the relative path

        doc.metadata.update(
            {
                "document_id": document_id,
                "source_file": source_file,
                "source_path": relative_path,
                "page_number": page_number,
                "character_count": len(text),
                "is_empty_page": len(text.strip()) == 0,
                "loader_used": loader_name,
                "document_type": document_type,
                "domain": domain,
                "document_checksum": checksum,
                "ingested_at": ingested_at,
            }
        )

    logger.info("Metadata enrichment completed.")
    return documents


def enrich_metadata_with_fault_isolation(
    documents: list[Document],
    loader_name: str,
    metadata_lookup: dict[str, dict[str, str]],
) -> list[Document]:
    """Enrich one source at a time so one bad file does not stop ingestion."""
    documents_by_source: dict[str, list[Document]] = {}

    for document in documents:
        source = document.metadata.get("source", "unknown")
        documents_by_source.setdefault(source, []).append(document)

    enriched_documents: list[Document] = []
    for source, source_documents in documents_by_source.items():
        try:
            enriched_documents.extend(
                enrich_metadata(source_documents, loader_name, metadata_lookup)
            )
        except Exception as exc:
            document_id = source_documents[0].metadata.get("document_id")
            if document_id:
                logger.error(
                    "Metadata enrichment failed for document source=%r document_id=%r: %s",
                    source,
                    document_id,
                    exc,
                )
            else:
                logger.error(
                    "Metadata enrichment failed for document source=%r: %s",
                    source,
                    exc,
                )

    return enriched_documents


def summarize_loaded_documents(
    documents: list[Document],
    discovered_source_files: int = 0,
    duration_seconds: float = 0.0,
) -> dict[str, int | float]:
    """Return a small loading summary for monitoring and debugging."""
    source_files = {
        doc.metadata.get("source_file", "unknown")
        for doc in documents
    }
    empty_pages = sum(
        1 for doc in documents if doc.metadata.get("is_empty_page", False)
    )

    successful_source_files = len(source_files)
    failed_source_files = max(discovered_source_files - successful_source_files, 0)

    return {
        "documents_loaded": len(source_files),
        "discovered_source_files": discovered_source_files,
        "successful_source_files": successful_source_files,
        "failed_source_files": failed_source_files,
        "pages_loaded": len(documents),
        "empty_pages": empty_pages,
        "duration_seconds": round(duration_seconds, 3),
    }


def load_documents(
    pdf_dir: str | Path = "data/pdf",
    pdf_loader: str = "pymupdf",
    metadata_csv_path: str | Path = "configs/document_metadata.csv",
) -> tuple[list[Document], dict[str, int | float]]:
    """Main loading function used by the ingestion pipeline."""
    start_time = perf_counter()
    logger.info("Starting document ingestion.")
    selected_pdf_loader = pdf_loader.lower().strip()
    if selected_pdf_loader not in SUPPORTED_PDF_LOADERS:
        raise ValueError(
            f"Unsupported loader '{pdf_loader}'. Use one of: {sorted(SUPPORTED_PDF_LOADERS)}"
        )
    logger.info("Selected PDF_loader: %s", selected_pdf_loader)
    logger.info(
        "CONFIG Loading: document_dir=%s metadata_csv_path=%s selected_pdf_loader=%s",
        pdf_dir,
        metadata_csv_path,
        selected_pdf_loader,
    )

    # Validate the folder before running the selected LangChain loaders.
    supported_files = get_supported_files(pdf_dir)
    metadata_lookup = load_document_metadata(metadata_csv_path)

    documents: list[Document] = []
    has_pdf_files = any(file.suffix.lower() == ".pdf" for file in supported_files)

    if has_pdf_files:
        if selected_pdf_loader == "pymupdf":
            pdf_docs = load_with_pymupdf(pdf_dir)
            documents.extend(
                enrich_metadata_with_fault_isolation(
                    pdf_docs,
                    "DirectoryLoader+PyMuPDFLoader",
                    metadata_lookup,
                )
            )
        else:
            pdf_docs = load_with_pdfplumber(pdf_dir)
            documents.extend(
                enrich_metadata_with_fault_isolation(
                    pdf_docs,
                    "DirectoryLoader+PDFPlumberLoader",
                    metadata_lookup,
                )
            )

    if any(file.suffix.lower() == ".docx" for file in supported_files):
        docx_docs = load_non_pdf_documents(pdf_dir, "**/*.docx", Docx2txtLoader)
        documents.extend(
            enrich_metadata_with_fault_isolation(
                docx_docs,
                "DirectoryLoader+Docx2txtLoader",
                metadata_lookup,
            )
        )

    if any(file.suffix.lower() == ".pptx" for file in supported_files):
        pptx_docs = load_non_pdf_documents(
            pdf_dir,
            "**/*.pptx",
            UnstructuredPowerPointLoader,
        )
        documents.extend(
            enrich_metadata_with_fault_isolation(
                pptx_docs,
                "DirectoryLoader+UnstructuredPowerPointLoader",
                metadata_lookup,
            )
        )

    if any(file.suffix.lower() == ".txt" for file in supported_files):
        text_docs = load_non_pdf_documents(
            pdf_dir,
            "**/*.txt",
            TextLoader,
            {"encoding": "utf-8"},
        )
        documents.extend(
            enrich_metadata_with_fault_isolation(
                text_docs,
                "DirectoryLoader+TextLoader",
                metadata_lookup,
            )
        )

    if any(file.suffix.lower() == ".html" for file in supported_files):
        html_docs = load_non_pdf_documents(
            pdf_dir,
            "**/*.html",
            BSHTMLLoader,
            {"bs_kwargs": {"features": "html.parser"}},
        )
        documents.extend(
            enrich_metadata_with_fault_isolation(
                html_docs,
                "DirectoryLoader+BSHTMLLoader",
                metadata_lookup,
            )
        )

    if any(file.suffix.lower() == ".md" for file in supported_files):
        markdown_docs = load_non_pdf_documents(
            pdf_dir,
            "**/*.md",
            TextLoader,
            {"encoding": "utf-8"},
        )
        documents.extend(
            enrich_metadata_with_fault_isolation(
                markdown_docs,
                "DirectoryLoader+TextLoader",
                metadata_lookup,
            )
        )

    duration_seconds = perf_counter() - start_time
    summary = summarize_loaded_documents(
        documents,
        discovered_source_files=len(supported_files),
        duration_seconds=duration_seconds,
    )
    logger.info(
        "Loading completed successfully. Documents=%s, pages=%s, empty_pages=%s",
        summary["documents_loaded"],
        summary["pages_loaded"],
        summary["empty_pages"],
    )
    logger.info(
        "STATISTICS Loading: discovered=%s succeeded=%s failed=%s pages=%s empty_pages=%s duration=%.3fs",
        summary["discovered_source_files"],
        summary["successful_source_files"],
        summary["failed_source_files"],
        summary["pages_loaded"],
        summary["empty_pages"],
        summary["duration_seconds"],
    )

    return documents, summary
