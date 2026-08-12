import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from rag_app.chunking import chunker as chunker_module
from rag_app.chunking.chunker import create_parent_child_chunks
from rag_app.ingestion.loader import load_documents, load_non_pdf_documents
from langchain_community.document_loaders import TextLoader


def _required_metadata(source_file: str) -> dict[str, object]:
    return {
        "document_id": f"doc-{source_file}",
        "document_checksum": f"checksum-{source_file}",
        "source_file": source_file,
        "source_path": source_file,
        "domain": "test",
        "document_type": "test_doc",
        "loader_used": "test_loader",
        "page_number": 1,
        "character_count": 120,
        "is_empty_page": False,
        "ingested_at": "2026-08-11T00:00:00+00:00",
    }


class FaultIsolationTests(unittest.TestCase):
    def test_loading_discovers_pdf_html_md_and_skips_failed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            config_dir = Path(temp_dir) / "configs"
            data_dir.mkdir()
            config_dir.mkdir()

            for filename in ("a.pdf", "b.html", "bad.md", "c.md"):
                (data_dir / filename).write_text("sample", encoding="utf-8")

            metadata_csv = config_dir / "document_metadata.csv"
            metadata_csv.write_text(
                "filename,domain,document_type\n"
                "a.pdf,test,pdf\n"
                "b.html,test,html\n"
                "c.md,test,markdown\n",
                encoding="utf-8",
            )

            def fake_single_file_loader(file_path, loader_cls, loader_kwargs=None):
                if file_path.name == "bad.md":
                    return []
                return [
                    Document(
                        page_content=f"loaded {file_path.suffix}",
                        metadata={"source": str(file_path), "page": 0},
                    )
                ]

            with patch(
                "rag_app.ingestion.loader._load_single_file",
                side_effect=fake_single_file_loader,
            ), self.assertLogs("rag_app", level="INFO") as logs:
                documents, summary = load_documents(data_dir, metadata_csv_path=metadata_csv)

            loaded_files = {document.metadata["source_file"] for document in documents}
            self.assertEqual(loaded_files, {"a.pdf", "b.html", "c.md"})
            self.assertEqual(summary["documents_loaded"], 3)
            self.assertEqual(summary["discovered_source_files"], 4)
            self.assertEqual(summary["successful_source_files"], 3)
            self.assertEqual(summary["failed_source_files"], 1)
            self.assertIn("duration_seconds", summary)
            log_output = "\n".join(logs.output)
            self.assertIn("CONFIG Loading:", log_output)
            self.assertIn("STATISTICS Loading:", log_output)
            self.assertIn("selected_pdf_loader=pymupdf", log_output)
            self.assertNotIn("loaded .pdf", log_output)

    def test_markdown_loader_continues_after_utf8_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "a.md").write_text("good markdown", encoding="utf-8")
            (data_dir / "bad.md").write_bytes(b"\xff\xfe\xfa")
            (data_dir / "c.md").write_text("another good markdown", encoding="utf-8")

            with self.assertLogs("rag_app", level="ERROR") as logs:
                documents = load_non_pdf_documents(
                    data_dir,
                    "**/*.md",
                    TextLoader,
                    {"encoding": "utf-8"},
                )

            loaded_files = {Path(document.metadata["source"]).name for document in documents}
            self.assertEqual(loaded_files, {"a.md", "c.md"})
            log_output = "\n".join(logs.output)
            self.assertIn("bad.md", log_output)
            self.assertIn("Loading failed for document source=", log_output)
            self.assertNotIn("document_id=None", log_output)
            self.assertNotIn("good markdown", log_output)

    def test_metadata_enrichment_failure_does_not_stop_other_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            config_dir = Path(temp_dir) / "configs"
            data_dir.mkdir()
            config_dir.mkdir()

            for filename in ("a.md", "bad.md", "c.md"):
                (data_dir / filename).write_text(f"{filename} body", encoding="utf-8")

            metadata_csv = config_dir / "document_metadata.csv"
            metadata_csv.write_text(
                "\ufefffilename,domain,document_type\n"
                "a.md,test,markdown\n"
                "bad.md,test,markdown\n"
                "c.md,test,markdown\n",
                encoding="utf-8",
            )

            def fake_checksum(file_path):
                if file_path.name == "bad.md":
                    raise OSError("checksum failed")
                return f"checksum-{file_path.name}"

            with patch(
                "rag_app.ingestion.loader._file_checksum",
                side_effect=fake_checksum,
            ), self.assertLogs("rag_app", level="INFO") as logs:
                documents, summary = load_documents(data_dir, metadata_csv_path=metadata_csv)

            loaded_files = {document.metadata["source_file"] for document in documents}
            self.assertEqual(loaded_files, {"a.md", "c.md"})
            self.assertEqual(summary["discovered_source_files"], 3)
            self.assertEqual(summary["successful_source_files"], 2)
            self.assertEqual(summary["failed_source_files"], 1)
            self.assertTrue(all("document_id" in doc.metadata for doc in documents))
            self.assertTrue(all("document_checksum" in doc.metadata for doc in documents))
            log_output = "\n".join(logs.output)
            self.assertIn("Metadata enrichment failed", log_output)
            self.assertIn("bad.md", log_output)
            self.assertIn("duration=", log_output)

    def test_chunking_skips_failed_document_without_partial_results(self) -> None:
        documents = [
            Document(
                page_content="alpha " * 80,
                metadata=_required_metadata("a.pdf"),
            ),
            Document(
                page_content="broken " * 80,
                metadata={"source_file": "b.pdf", "document_id": "doc-b.pdf"},
            ),
            Document(
                page_content="charlie " * 80,
                metadata=_required_metadata("c.pdf"),
            ),
        ]

        with self.assertLogs("rag_app", level="INFO") as logs:
            groups = create_parent_child_chunks(
                documents,
                parent_chunk_size=200,
                parent_chunk_overlap=20,
                child_chunk_size=80,
                child_chunk_overlap=10,
            )

        parent_files = {group["parent"].metadata["source_file"] for group in groups}
        child_files = {
            child.metadata["source_file"]
            for group in groups
            for child in group["children"]
        }

        self.assertEqual(parent_files, {"a.pdf", "c.pdf"})
        self.assertEqual(child_files, {"a.pdf", "c.pdf"})
        self.assertNotIn("parent_id", documents[0].metadata)
        self.assertNotIn("parent_id", documents[2].metadata)
        for group in groups:
            parent = group["parent"]
            self.assertIn("parent_id", parent.metadata)
            self.assertIn("parent_checksum", parent.metadata)
            self.assertIn("chunking_config_hash", parent.metadata)
            self.assertEqual(parent.metadata["status"], "ACTIVE")
            self.assertIsNone(parent.metadata["version"])
            self.assertIsNone(parent.metadata["updated_at"])
            self.assertIsNone(parent.metadata["indexed_at"])
            for child in group["children"]:
                self.assertEqual(child.metadata["parent_id"], parent.metadata["parent_id"])
                self.assertIn("child_id", child.metadata)
                self.assertIn("child_checksum", child.metadata)
                self.assertEqual(
                    child.metadata["chunking_config_hash"],
                    parent.metadata["chunking_config_hash"],
                )

        log_output = "\n".join(logs.output)
        self.assertIn("Starting chunking.", log_output)
        self.assertIn("CONFIG Chunking:", log_output)
        self.assertIn("parent_chunk_size=200", log_output)
        self.assertIn("child_chunk_overlap=10", log_output)
        self.assertIn("splitter=RecursiveCharacterTextSplitter", log_output)
        self.assertIn("Chunking completed.", log_output)
        self.assertIn("STATISTICS Chunking:", log_output)
        self.assertIn("documents_received=3", log_output)
        self.assertIn("parent_chunks_created=", log_output)
        self.assertIn("child_chunks_created=", log_output)
        self.assertIn("duration=", log_output)
        self.assertIn("status=FULL_SUCCESS", log_output)
        self.assertIn("Document chunking failed source_file='b.pdf'", log_output)
        self.assertIn("status=FAILED", log_output)
        self.assertNotIn("alpha alpha", log_output)
        self.assertNotIn("charlie charlie", log_output)

    def test_chunking_parent_level_partial_success_keeps_later_parents(self) -> None:
        document = Document(
            page_content=("first parent text. " * 20)
            + ("second parent text. " * 20)
            + ("third parent text. " * 20),
            metadata=_required_metadata("partial.pdf"),
        )
        original_apply = chunker_module._apply_common_chunk_metadata

        def fail_middle_parent(chunk: Document, chunking_config_hash: str) -> None:
            if (
                chunk.metadata.get("source_file") == "partial.pdf"
                and chunk.metadata.get("parent_index") == 1
            ):
                raise ValueError("planned parent failure")
            original_apply(chunk, chunking_config_hash)

        with patch(
            "rag_app.chunking.chunker._apply_common_chunk_metadata",
            side_effect=fail_middle_parent,
        ), self.assertLogs("rag_app", level="INFO") as logs:
            groups = create_parent_child_chunks(
                [document],
                parent_chunk_size=180,
                parent_chunk_overlap=0,
                child_chunk_size=70,
                child_chunk_overlap=0,
            )

        parent_indexes = {group["parent"].metadata["parent_index"] for group in groups}
        self.assertIn(0, parent_indexes)
        self.assertIn(2, parent_indexes)
        self.assertNotIn(1, parent_indexes)
        self.assertNotIn("parent_id", document.metadata)

        for group in groups:
            parent = group["parent"]
            self.assertIn("parent_id", parent.metadata)
            self.assertIn("parent_checksum", parent.metadata)
            self.assertIn("chunking_config_hash", parent.metadata)
            for child in group["children"]:
                self.assertEqual(child.metadata["parent_id"], parent.metadata["parent_id"])
                self.assertIn("child_id", child.metadata)
                self.assertIn("child_checksum", child.metadata)
                self.assertEqual(
                    child.metadata["chunking_config_hash"],
                    parent.metadata["chunking_config_hash"],
                )

        log_output = "\n".join(logs.output)
        self.assertIn("Parent chunking failed", log_output)
        self.assertIn("parent_index=1", log_output)
        self.assertIn("status=PARTIAL_SUCCESS", log_output)
        self.assertIn("Chunking completed.", log_output)
        self.assertIn("STATISTICS Chunking:", log_output)
        self.assertNotIn("second parent text", log_output)

    def test_chunking_config_hash_is_deterministic_and_config_sensitive(self) -> None:
        document = Document(
            page_content="stable chunk config hash text " * 20,
            metadata=_required_metadata("hash.pdf"),
        )

        first_groups = create_parent_child_chunks(
            [document],
            parent_chunk_size=200,
            parent_chunk_overlap=20,
            child_chunk_size=80,
            child_chunk_overlap=10,
        )
        second_groups = create_parent_child_chunks(
            [document],
            parent_chunk_size=200,
            parent_chunk_overlap=20,
            child_chunk_size=80,
            child_chunk_overlap=10,
        )
        changed_groups = create_parent_child_chunks(
            [document],
            parent_chunk_size=201,
            parent_chunk_overlap=20,
            child_chunk_size=80,
            child_chunk_overlap=10,
        )

        first_hash = first_groups[0]["parent"].metadata["chunking_config_hash"]
        second_hash = second_groups[0]["parent"].metadata["chunking_config_hash"]
        changed_hash = changed_groups[0]["parent"].metadata["chunking_config_hash"]

        self.assertEqual(first_hash, second_hash)
        self.assertNotEqual(first_hash, changed_hash)
        self.assertEqual(len(first_hash), 64)

        for group in first_groups:
            parent_hash = group["parent"].metadata["chunking_config_hash"]
            self.assertEqual(parent_hash, first_hash)
            for child in group["children"]:
                self.assertEqual(child.metadata["chunking_config_hash"], first_hash)


if __name__ == "__main__":
    unittest.main()
