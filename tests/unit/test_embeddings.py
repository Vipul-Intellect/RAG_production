import re
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from rag_app.embeddings.embedder import (
    BGEEmbeddingModel,
    EMBEDDING_MODEL_NAME,
    EXPECTED_EMBEDDING_DIMENSION,
    embed_chunks,
)


class FakeSentenceTransformer:
    init_count = 0

    def __init__(self, model_name: str) -> None:
        FakeSentenceTransformer.init_count += 1
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return EXPECTED_EMBEDDING_DIMENSION

    def encode(self, text: str, normalize_embeddings: bool = True) -> list[float]:
        return [0.1] * EXPECTED_EMBEDDING_DIMENSION


class WrongDimensionSentenceTransformer(FakeSentenceTransformer):
    def get_sentence_embedding_dimension(self) -> int:
        return 768


class RecordingEmbeddingModel:
    def __init__(self, failing_text: str | None = None) -> None:
        self.model_name = EMBEDDING_MODEL_NAME
        self.expected_dimension = EXPECTED_EMBEDDING_DIMENSION
        self.calls: list[str] = []
        self.failing_text = failing_text

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        if text == self.failing_text:
            raise ValueError("planned embedding failure")
        return [0.2] * EXPECTED_EMBEDDING_DIMENSION


class EmbeddingSubPhaseOneTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSentenceTransformer.init_count = 0

    def test_model_initialization_name_and_dimension(self) -> None:
        with patch(
            "rag_app.embeddings.embedder._load_sentence_transformer",
            side_effect=FakeSentenceTransformer,
        ):
            model = BGEEmbeddingModel()

        self.assertEqual(model.model_name, EMBEDDING_MODEL_NAME)
        self.assertEqual(model.expected_dimension, EXPECTED_EMBEDDING_DIMENSION)
        self.assertEqual(model.actual_dimension, EXPECTED_EMBEDDING_DIMENSION)
        self.assertEqual(FakeSentenceTransformer.init_count, 1)

    def test_model_initialization_fails_for_wrong_actual_dimension(self) -> None:
        with patch(
            "rag_app.embeddings.embedder._load_sentence_transformer",
            side_effect=WrongDimensionSentenceTransformer,
        ):
            with self.assertRaisesRegex(ValueError, "Embedding model dimension mismatch"):
                BGEEmbeddingModel()

    def test_embedding_vector_has_expected_numeric_values(self) -> None:
        with patch(
            "rag_app.embeddings.embedder._load_sentence_transformer",
            side_effect=FakeSentenceTransformer,
        ):
            model = BGEEmbeddingModel()
            embedding = model.embed_text("short test text")

        self.assertEqual(len(embedding), EXPECTED_EMBEDDING_DIMENSION)
        self.assertTrue(all(isinstance(value, float) for value in embedding))

    def test_repeated_embedding_calls_reuse_loaded_model(self) -> None:
        with patch(
            "rag_app.embeddings.embedder._load_sentence_transformer",
            side_effect=FakeSentenceTransformer,
        ):
            model = BGEEmbeddingModel()
            model.embed_text("first text")
            model.embed_text("second text")

        self.assertEqual(FakeSentenceTransformer.init_count, 1)

    def test_public_embed_chunks_entry_point_embeds_parent_and_children(self) -> None:
        model = RecordingEmbeddingModel()
        chunk_groups = [
            {
                "parent": Document(
                    page_content="parent text",
                    metadata={
                        "document_id": "doc-1",
                        "document_checksum": "doc-checksum",
                        "parent_id": "parent-1",
                        "parent_checksum": "parent-checksum",
                        "chunking_config_hash": "chunk-config",
                    },
                ),
                "children": [
                    Document(
                        page_content="child one text",
                        metadata={
                            "document_id": "doc-1",
                            "document_checksum": "doc-checksum",
                            "parent_id": "parent-1",
                            "child_id": "child-1",
                            "child_checksum": "child-checksum-1",
                            "chunking_config_hash": "chunk-config",
                        },
                    ),
                    Document(
                        page_content="child two text",
                        metadata={
                            "document_id": "doc-1",
                            "document_checksum": "doc-checksum",
                            "parent_id": "parent-1",
                            "child_id": "child-2",
                            "child_checksum": "child-checksum-2",
                            "chunking_config_hash": "chunk-config",
                        },
                    )
                ],
            }
        ]
        parent_metadata_before = chunk_groups[0]["parent"].metadata.copy()
        child_metadata_before = [
            child.metadata.copy() for child in chunk_groups[0]["children"]
        ]

        with self.assertLogs("rag_app", level="INFO") as logs:
            returned_groups = embed_chunks(chunk_groups, embedding_model=model)

        self.assertEqual(len(returned_groups), 1)
        self.assertIs(returned_groups[0]["parent"]["document"], chunk_groups[0]["parent"])
        self.assertEqual(
            len(returned_groups[0]["parent"]["embedding"]),
            EXPECTED_EMBEDDING_DIMENSION,
        )
        self.assertEqual(len(returned_groups[0]["children"]), 2)
        for child in returned_groups[0]["children"]:
            self.assertEqual(len(child["embedding"]), EXPECTED_EMBEDDING_DIMENSION)
            self.assertEqual(
                child["document"].metadata["parent_id"],
                returned_groups[0]["parent"]["document"].metadata["parent_id"],
            )

        self.assertEqual(chunk_groups[0]["parent"].metadata, parent_metadata_before)
        self.assertEqual(
            [child.metadata for child in chunk_groups[0]["children"]],
            child_metadata_before,
        )
        self.assertEqual(
            returned_groups[0]["parent"]["document"].metadata["document_checksum"],
            "doc-checksum",
        )
        self.assertEqual(
            returned_groups[0]["parent"]["document"].metadata["parent_checksum"],
            "parent-checksum",
        )
        self.assertEqual(
            returned_groups[0]["children"][0]["document"].metadata["child_checksum"],
            "child-checksum-1",
        )
        self.assertEqual(
            returned_groups[0]["children"][1]["document"].metadata["child_id"],
            "child-2",
        )
        self.assertEqual(
            returned_groups[0]["parent"]["document"].metadata["chunking_config_hash"],
            "chunk-config",
        )
        self.assertEqual(
            model.calls,
            ["parent text", "child one text", "child two text"],
        )
        log_output = "\n".join(logs.output)
        self.assertIn("CONFIG Embedding:", log_output)
        self.assertIn(f"model_name={EMBEDDING_MODEL_NAME}", log_output)
        self.assertIn(f"expected_dimension={EXPECTED_EMBEDDING_DIMENSION}", log_output)
        self.assertIn("STATISTICS Embedding:", log_output)
        self.assertIn("parents_received=1", log_output)
        self.assertIn("parents_embedded=1", log_output)
        self.assertIn("children_received=2", log_output)
        self.assertIn("children_embedded=2", log_output)
        self.assertIn("failed_groups=0", log_output)
        self.assertIn(f"dimension={EXPECTED_EMBEDDING_DIMENSION}", log_output)
        duration_match = re.search(r"duration=([0-9.]+)s", log_output)
        self.assertIsNotNone(duration_match)
        self.assertGreaterEqual(float(duration_match.group(1)), 0.0)
        self.assertNotIn("0.2", log_output)

    def test_failed_child_discards_group_and_next_group_continues(self) -> None:
        model = RecordingEmbeddingModel(failing_text="bad child")
        chunk_groups = [
            {
                "parent": Document(
                    page_content="bad parent",
                    metadata={"source_file": "bad.pdf", "document_id": "doc-bad"},
                ),
                "children": [
                    Document(
                        page_content="bad child",
                        metadata={"parent_id": "bad-parent", "child_id": "bad-child"},
                    )
                ],
            },
            {
                "parent": Document(
                    page_content="good parent",
                    metadata={"source_file": "good.pdf", "document_id": "doc-good"},
                ),
                "children": [
                    Document(
                        page_content="good child",
                        metadata={"parent_id": "good-parent", "child_id": "good-child"},
                    )
                ],
            },
        ]

        with self.assertLogs("rag_app", level="INFO") as logs:
            returned_groups = embed_chunks(chunk_groups, embedding_model=model)

        self.assertEqual(len(returned_groups), 1)
        self.assertEqual(
            returned_groups[0]["parent"]["document"].metadata["document_id"],
            "doc-good",
        )
        returned_document_ids = {
            group["parent"]["document"].metadata["document_id"]
            for group in returned_groups
        }
        self.assertNotIn("doc-bad", returned_document_ids)
        self.assertEqual(len(returned_groups[0]["children"]), 1)
        log_output = "\n".join(logs.output)
        self.assertIn("Embedding failed for chunk group", log_output)
        self.assertIn("bad.pdf", log_output)
        self.assertIn("parents_received=2", log_output)
        self.assertIn("parents_embedded=1", log_output)
        self.assertIn("children_received=2", log_output)
        self.assertIn("children_embedded=1", log_output)
        self.assertIn("failed_groups=1", log_output)
        self.assertIn(f"model_name={EMBEDDING_MODEL_NAME}", log_output)
        self.assertIn(f"dimension={EXPECTED_EMBEDDING_DIMENSION}", log_output)
        self.assertNotIn("bad child", log_output)
        self.assertNotIn("good child", log_output)
        self.assertNotIn("0.2", log_output)

    def test_failed_parent_discards_group_and_next_group_continues(self) -> None:
        model = RecordingEmbeddingModel(failing_text="bad parent")
        chunk_groups = [
            {
                "parent": Document(
                    page_content="bad parent",
                    metadata={"source_file": "bad.pdf", "document_id": "doc-bad"},
                ),
                "children": [
                    Document(
                        page_content="bad child should not run",
                        metadata={"parent_id": "bad-parent", "child_id": "bad-child"},
                    )
                ],
            },
            {
                "parent": Document(
                    page_content="good parent",
                    metadata={"source_file": "good.pdf", "document_id": "doc-good"},
                ),
                "children": [
                    Document(
                        page_content="good child",
                        metadata={"parent_id": "good-parent", "child_id": "good-child"},
                    )
                ],
            },
        ]

        with self.assertLogs("rag_app", level="INFO") as logs:
            returned_groups = embed_chunks(chunk_groups, embedding_model=model)

        self.assertEqual(len(returned_groups), 1)
        self.assertEqual(
            returned_groups[0]["parent"]["document"].metadata["document_id"],
            "doc-good",
        )
        self.assertEqual(len(returned_groups[0]["children"]), 1)
        self.assertEqual(
            model.calls,
            ["bad parent", "good parent", "good child"],
        )
        log_output = "\n".join(logs.output)
        self.assertIn("Embedding failed for chunk group", log_output)
        self.assertIn("bad.pdf", log_output)
        self.assertIn("parents_received=2", log_output)
        self.assertIn("parents_embedded=1", log_output)
        self.assertIn("children_received=2", log_output)
        self.assertIn("children_embedded=1", log_output)
        self.assertIn("failed_groups=1", log_output)
        self.assertNotIn("bad parent", log_output)
        self.assertNotIn("bad child should not run", log_output)
        self.assertNotIn("good child", log_output)
        self.assertNotIn("0.2", log_output)


if __name__ == "__main__":
    unittest.main()
