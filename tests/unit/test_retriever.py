import unittest
from types import SimpleNamespace

from langchain_core.documents import Document

from rag_app.embeddings.embedder import EXPECTED_EMBEDDING_DIMENSION
from rag_app.retrieval.retriever import retrieve_context
from rag_app.storage.qdrant_collections import (
    CHILD_CHUNKS_COLLECTION,
    PARENT_CHUNKS_COLLECTION,
)
from rag_app.storage.qdrant_storage import (
    ACTIVE_STATUS,
    INACTIVE_STATUS,
    PAYLOAD_CONTENT_FIELD,
    build_parent_point_id,
)


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.expected_dimension = EXPECTED_EMBEDDING_DIMENSION
        self.calls: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.3] * EXPECTED_EMBEDDING_DIMENSION


def _child_payload(status: str = ACTIVE_STATUS) -> dict[str, object]:
    return {
        PAYLOAD_CONTENT_FIELD: "child content",
        "child_id": "child-1",
        "parent_id": "parent-1",
        "document_id": "doc-1",
        "document_checksum": "checksum-1",
        "chunking_config_hash": "chunk-config",
        "parent_index": 0,
        "child_index": 0,
        "status": status,
        "domain": "policy",
    }


def _parent_payload() -> dict[str, object]:
    return {
        PAYLOAD_CONTENT_FIELD: "parent content",
        "parent_id": "parent-1",
        "document_id": "doc-1",
        "document_checksum": "checksum-1",
        "chunking_config_hash": "chunk-config",
        "parent_index": 0,
        "status": ACTIVE_STATUS,
        "domain": "policy",
    }


def _parent_point_id_from_payload(payload: dict[str, object]) -> str:
    return build_parent_point_id(
        Document(
            page_content="",
            metadata={
                "document_id": payload["document_id"],
                "document_checksum": payload["document_checksum"],
                "chunking_config_hash": payload["chunking_config_hash"],
                "parent_index": payload["parent_index"],
            },
        )
    )


class FakeQdrantClient:
    def __init__(
        self,
        child_records: list[SimpleNamespace],
        parent_records: list[SimpleNamespace],
    ) -> None:
        self.child_records = child_records
        self.parent_records = parent_records
        self.query_points_calls: list[dict[str, object]] = []
        self.retrieve_calls: list[dict[str, object]] = []

    def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return SimpleNamespace(points=self.child_records)

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        requested_ids = set(kwargs["ids"])
        return [
            record
            for record in self.parent_records
            if str(record.id) in requested_ids
        ]


class RetrievalTests(unittest.TestCase):
    def test_retrieval_searches_active_children_and_recovers_parent(self) -> None:
        child_payload = _child_payload()
        parent_payload = _parent_payload()
        parent_point_id = _parent_point_id_from_payload(child_payload)
        client = FakeQdrantClient(
            child_records=[
                SimpleNamespace(id="child-point-1", score=0.87, payload=child_payload)
            ],
            parent_records=[
                SimpleNamespace(id=parent_point_id, payload=parent_payload)
            ],
        )
        model = FakeEmbeddingModel()

        result = retrieve_context(
            "security policy",
            k=3,
            embedding_model=model,
            client=client,
        )

        self.assertEqual(model.calls, ["security policy"])
        query_call = client.query_points_calls[0]
        self.assertEqual(query_call["collection_name"], CHILD_CHUNKS_COLLECTION)
        self.assertEqual(len(query_call["query"]), EXPECTED_EMBEDDING_DIMENSION)
        self.assertEqual(query_call["limit"], 3)
        self.assertTrue(query_call["with_payload"])
        self.assertFalse(query_call["with_vectors"])
        active_condition = query_call["query_filter"].must[0]
        self.assertEqual(active_condition.key, "status")
        self.assertEqual(active_condition.match.value, ACTIVE_STATUS)

        retrieve_call = client.retrieve_calls[0]
        self.assertEqual(retrieve_call["collection_name"], PARENT_CHUNKS_COLLECTION)
        self.assertEqual(retrieve_call["ids"], [parent_point_id])
        self.assertTrue(retrieve_call["with_payload"])
        self.assertFalse(retrieve_call["with_vectors"])

        self.assertEqual(result["requested_k"], 3)
        self.assertEqual(result["retrieved_child_count"], 1)
        self.assertEqual(result["retrieved_parent_count"], 1)
        self.assertGreaterEqual(result["duration_seconds"], 0.0)
        child = result["retrieved_children"][0]
        parent = result["corresponding_parents"][0]
        self.assertEqual(child["score"], 0.87)
        self.assertEqual(child["content"], "child content")
        self.assertEqual(child["child_id"], "child-1")
        self.assertEqual(child["parent_id"], "parent-1")
        self.assertEqual(child["document_id"], "doc-1")
        self.assertEqual(child["document_checksum"], "checksum-1")
        self.assertEqual(child["chunking_config_hash"], "chunk-config")
        self.assertEqual(child["parent_index"], 0)
        self.assertEqual(child["child_index"], 0)
        self.assertEqual(child["status"], ACTIVE_STATUS)
        self.assertEqual(child["metadata"]["domain"], "policy")
        self.assertEqual(parent["content"], "parent content")
        self.assertEqual(parent["parent_id"], "parent-1")

    def test_query_and_k_validation(self) -> None:
        client = FakeQdrantClient(child_records=[], parent_records=[])
        model = FakeEmbeddingModel()

        with self.assertRaisesRegex(ValueError, "non-empty string"):
            retrieve_context("", k=1, embedding_model=model, client=client)
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            retrieve_context("   ", k=1, embedding_model=model, client=client)
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            retrieve_context("query", k=0, embedding_model=model, client=client)
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            retrieve_context("query", k=-1, embedding_model=model, client=client)

        self.assertEqual(model.calls, [])
        self.assertEqual(client.query_points_calls, [])

    def test_duplicate_parent_ids_are_retrieved_once(self) -> None:
        child_payload_one = _child_payload()
        child_payload_two = {**_child_payload(), "child_id": "child-2", "child_index": 1}
        parent_payload = _parent_payload()
        parent_point_id = _parent_point_id_from_payload(child_payload_one)
        client = FakeQdrantClient(
            child_records=[
                SimpleNamespace(id="child-point-1", score=0.8, payload=child_payload_one),
                SimpleNamespace(id="child-point-2", score=0.7, payload=child_payload_two),
            ],
            parent_records=[
                SimpleNamespace(id=parent_point_id, payload=parent_payload)
            ],
        )

        result = retrieve_context(
            "query",
            k=2,
            embedding_model=FakeEmbeddingModel(),
            client=client,
        )

        self.assertEqual(client.retrieve_calls[0]["ids"], [parent_point_id])
        self.assertEqual(result["retrieved_child_count"], 2)
        self.assertEqual(result["retrieved_parent_count"], 1)

    def test_parent_id_mismatch_is_excluded_safely(self) -> None:
        child_payload = _child_payload()
        mismatched_parent_payload = {**_parent_payload(), "parent_id": "wrong-parent"}
        parent_point_id = _parent_point_id_from_payload(child_payload)
        client = FakeQdrantClient(
            child_records=[
                SimpleNamespace(id="child-point-1", score=0.8, payload=child_payload),
            ],
            parent_records=[
                SimpleNamespace(id=parent_point_id, payload=mismatched_parent_payload)
            ],
        )

        with self.assertLogs("rag_app", level="WARNING") as logs:
            result = retrieve_context(
                "query",
                k=1,
                embedding_model=FakeEmbeddingModel(),
                client=client,
            )

        self.assertEqual(result["retrieved_child_count"], 1)
        self.assertEqual(result["retrieved_parent_count"], 0)
        log_output = "\n".join(logs.output)
        self.assertIn("parent_id mismatch", log_output)
        self.assertNotIn("child content", log_output)
        self.assertNotIn("parent content", log_output)
        self.assertNotIn("0.3", log_output)

    def test_inactive_child_records_are_not_returned_as_context(self) -> None:
        client = FakeQdrantClient(
            child_records=[
                SimpleNamespace(
                    id="inactive-child",
                    score=0.9,
                    payload=_child_payload(status=INACTIVE_STATUS),
                )
            ],
            parent_records=[],
        )

        result = retrieve_context(
            "query",
            k=5,
            embedding_model=FakeEmbeddingModel(),
            client=client,
        )

        self.assertEqual(result["retrieved_child_count"], 0)
        self.assertEqual(result["retrieved_parent_count"], 0)
        self.assertEqual(client.retrieve_calls, [])

    def test_empty_search_returns_valid_empty_result(self) -> None:
        client = FakeQdrantClient(child_records=[], parent_records=[])

        with self.assertLogs("rag_app", level="INFO") as logs:
            result = retrieve_context(
                "query",
                k=2,
                embedding_model=FakeEmbeddingModel(),
                client=client,
            )

        self.assertEqual(result["query"], "query")
        self.assertEqual(result["requested_k"], 2)
        self.assertEqual(result["retrieved_children"], [])
        self.assertEqual(result["corresponding_parents"], [])
        self.assertIn("no ACTIVE child results", "\n".join(logs.output))

    def test_missing_parent_is_handled_safely(self) -> None:
        client = FakeQdrantClient(
            child_records=[
                SimpleNamespace(id="child-point-1", score=0.5, payload=_child_payload())
            ],
            parent_records=[],
        )

        with self.assertLogs("rag_app", level="WARNING") as logs:
            result = retrieve_context(
                "query",
                k=1,
                embedding_model=FakeEmbeddingModel(),
                client=client,
            )

        self.assertEqual(result["retrieved_child_count"], 1)
        self.assertEqual(result["retrieved_parent_count"], 0)
        log_output = "\n".join(logs.output)
        self.assertIn("parent chunks were missing", log_output)
        self.assertNotIn("child content", log_output)
        self.assertNotIn("parent content", log_output)
        self.assertNotIn("0.3", log_output)
        self.assertNotIn("secret-api-key", log_output)

    def test_child_missing_parent_lookup_metadata_warns_without_content(self) -> None:
        payload = _child_payload()
        del payload["parent_index"]
        client = FakeQdrantClient(
            child_records=[
                SimpleNamespace(id="child-point-1", score=0.5, payload=payload)
            ],
            parent_records=[],
        )

        with self.assertLogs("rag_app", level="WARNING") as logs:
            result = retrieve_context(
                "query",
                k=1,
                embedding_model=FakeEmbeddingModel(),
                client=client,
            )

        self.assertEqual(result["retrieved_child_count"], 1)
        self.assertEqual(result["retrieved_parent_count"], 0)
        log_output = "\n".join(logs.output)
        self.assertIn("missing parent lookup metadata", log_output)
        self.assertNotIn("child content", log_output)
        self.assertNotIn("0.3", log_output)


if __name__ == "__main__":
    unittest.main()
