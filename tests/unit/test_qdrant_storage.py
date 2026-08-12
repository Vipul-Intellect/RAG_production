import unittest
from copy import deepcopy
from types import SimpleNamespace

from langchain_core.documents import Document

from rag_app.embeddings.embedder import EXPECTED_EMBEDDING_DIMENSION
from rag_app.storage.qdrant_collections import (
    CHILD_CHUNKS_COLLECTION,
    PARENT_CHUNKS_COLLECTION,
)
from rag_app.storage.qdrant_storage import (
    PAYLOAD_CONTENT_FIELD,
    ACTIVE_STATUS,
    INACTIVE_STATUS,
    build_child_point_id,
    build_parent_point_id,
    store_embedded_chunks,
)


def _vector(value: float = 0.1) -> list[float]:
    return [value] * EXPECTED_EMBEDDING_DIMENSION


def _parent_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "document_id": "doc-1",
        "document_checksum": "doc-checksum",
        "source_file": "source.pdf",
        "source_path": "data/source.pdf",
        "domain": "policy",
        "document_type": "manual",
        "loader_used": "loader",
        "page_number": 1,
        "character_count": 100,
        "is_empty_page": False,
        "ingested_at": "2026-08-12T00:00:00+00:00",
        "parent_id": "parent-runtime-id",
        "parent_index": 0,
        "parent_checksum": "parent-checksum",
        "chunking_config_hash": "chunk-config",
        "page_start": 1,
        "page_end": 1,
        "status": "ACTIVE",
        "version": None,
        "updated_at": None,
        "indexed_at": None,
    }
    metadata.update(overrides)
    return metadata


def _child_metadata(**overrides: object) -> dict[str, object]:
    metadata = _parent_metadata(
        child_id="child-runtime-id",
        child_index=0,
        child_checksum="child-checksum",
    )
    metadata.update(overrides)
    return metadata


def _embedded_group(
    parent_metadata: dict[str, object] | None = None,
    child_metadata: dict[str, object] | None = None,
):
    return {
        "parent": {
            "document": Document(
                page_content="parent content",
                metadata=parent_metadata or _parent_metadata(),
            ),
            "embedding": _vector(0.1),
        },
        "children": [
            {
                "document": Document(
                    page_content="child content",
                    metadata=child_metadata or _child_metadata(),
                ),
                "embedding": _vector(0.2),
            }
        ],
    }


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections = {
            PARENT_CHUNKS_COLLECTION: {},
            CHILD_CHUNKS_COLLECTION: {},
        }
        self.upsert_calls: list[dict[str, object]] = []
        self.set_payload_calls: list[dict[str, object]] = []
        self.fail_parent_upsert = False
        self.fail_child_upsert = False
        self.fail_retrieve = False
        self.fail_next_active_child_status = False
        self.fail_next_inactive_parent_status = False

    def count(self, collection_name: str, exact: bool = True):
        return SimpleNamespace(count=len(self.collections[collection_name]))

    def scroll(
        self,
        collection_name: str,
        scroll_filter,
        with_payload: bool,
        with_vectors: bool,
        limit: int,
        offset=None,
    ):
        conditions = {
            condition.key: condition.match.value
            for condition in scroll_filter.must
        }
        records = []
        for point_id, point in self.collections[collection_name].items():
            payload = point.payload or {}
            if all(payload.get(key) == value for key, value in conditions.items()):
                records.append(SimpleNamespace(id=point_id, payload=payload))
        return records, None

    def upsert(self, collection_name: str, points):
        if collection_name == PARENT_CHUNKS_COLLECTION and self.fail_parent_upsert:
            raise RuntimeError("secret-api-key")
        if collection_name == CHILD_CHUNKS_COLLECTION and self.fail_child_upsert:
            raise RuntimeError("secret-api-key")
        self.upsert_calls.append(
            {"collection_name": collection_name, "points": points}
        )
        for point in points:
            self.collections[collection_name][str(point.id)] = point

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool,
        with_vectors: bool,
    ):
        if self.fail_retrieve:
            return []
        return [
            self.collections[collection_name][point_id]
            for point_id in ids
            if point_id in self.collections[collection_name]
        ]

    def set_payload(self, collection_name: str, payload: dict[str, object], points: list[str]):
        if (
            collection_name == CHILD_CHUNKS_COLLECTION
            and payload.get("status") == ACTIVE_STATUS
            and self.fail_next_active_child_status
        ):
            self.fail_next_active_child_status = False
            raise RuntimeError("secret-api-key")
        if (
            collection_name == PARENT_CHUNKS_COLLECTION
            and payload.get("status") == INACTIVE_STATUS
            and self.fail_next_inactive_parent_status
        ):
            self.fail_next_inactive_parent_status = False
            raise RuntimeError("secret-api-key")
        self.set_payload_calls.append(
            {"collection_name": collection_name, "payload": payload, "points": points}
        )
        for point_id in points:
            self.collections[collection_name][point_id].payload.update(payload)


class QdrantStorageTests(unittest.TestCase):
    def test_parent_and_child_points_are_upserted_to_correct_collections(self) -> None:
        client = FakeQdrantClient()
        group = _embedded_group()

        result = store_embedded_chunks([group], client)

        self.assertEqual(result["parents_stored"], 1)
        self.assertEqual(result["children_stored"], 1)
        self.assertEqual(len(client.upsert_calls), 2)
        parent_call = client.upsert_calls[0]
        child_call = client.upsert_calls[1]
        self.assertEqual(parent_call["collection_name"], PARENT_CHUNKS_COLLECTION)
        self.assertEqual(child_call["collection_name"], CHILD_CHUNKS_COLLECTION)

        parent_point = parent_call["points"][0]
        child_point = child_call["points"][0]
        self.assertEqual(parent_point.vector, _vector(0.1))
        self.assertEqual(child_point.vector, _vector(0.2))
        self.assertEqual(parent_point.payload["document_id"], "doc-1")
        self.assertEqual(child_point.payload["child_id"], "child-runtime-id")
        self.assertEqual(child_point.payload["parent_id"], "parent-runtime-id")
        self.assertEqual(parent_point.payload[PAYLOAD_CONTENT_FIELD], "parent content")
        self.assertEqual(child_point.payload[PAYLOAD_CONTENT_FIELD], "child content")
        self.assertNotEqual(child_point.payload[PAYLOAD_CONTENT_FIELD], "parent content")
        self.assertEqual(parent_point.payload["status"], ACTIVE_STATUS)
        self.assertEqual(child_point.payload["status"], ACTIVE_STATUS)

    def test_point_ids_are_deterministic_and_config_sensitive(self) -> None:
        parent_doc = Document(page_content="parent content", metadata=_parent_metadata())
        child_doc = Document(page_content="child content", metadata=_child_metadata())

        self.assertEqual(build_parent_point_id(parent_doc), build_parent_point_id(parent_doc))
        self.assertEqual(build_child_point_id(child_doc), build_child_point_id(child_doc))
        self.assertNotEqual(build_parent_point_id(parent_doc), build_child_point_id(child_doc))

        changed_parent = Document(
            page_content="parent content",
            metadata={**_parent_metadata(), "document_checksum": "changed"},
        )
        changed_child = Document(
            page_content="child content",
            metadata={**_child_metadata(), "document_checksum": "changed"},
        )
        self.assertNotEqual(build_parent_point_id(parent_doc), build_parent_point_id(changed_parent))
        self.assertNotEqual(build_child_point_id(child_doc), build_child_point_id(changed_child))

        changed_config_parent = Document(
            page_content="parent content",
            metadata={**_parent_metadata(), "chunking_config_hash": "changed"},
        )
        changed_config_child = Document(
            page_content="child content",
            metadata={**_child_metadata(), "chunking_config_hash": "changed"},
        )
        self.assertNotEqual(
            build_parent_point_id(parent_doc),
            build_parent_point_id(changed_config_parent),
        )
        self.assertNotEqual(
            build_child_point_id(child_doc),
            build_child_point_id(changed_config_child),
        )

        changed_parent_index = Document(
            page_content="parent content",
            metadata={**_parent_metadata(), "parent_index": 1},
        )
        changed_child_index = Document(
            page_content="child content",
            metadata={**_child_metadata(), "child_index": 1},
        )
        self.assertNotEqual(
            build_parent_point_id(parent_doc),
            build_parent_point_id(changed_parent_index),
        )
        self.assertNotEqual(
            build_child_point_id(child_doc),
            build_child_point_id(changed_child_index),
        )

    def test_missing_required_identity_metadata_fails_clearly(self) -> None:
        group = _embedded_group(parent_metadata=_parent_metadata())
        del group["parent"]["document"].metadata["document_id"]

        with self.assertRaisesRegex(ValueError, "document_id"):
            store_embedded_chunks([group], FakeQdrantClient())

    def test_invalid_vector_fails_before_upsert(self) -> None:
        client = FakeQdrantClient()
        group = _embedded_group()
        group["parent"]["embedding"] = [0.1]

        with self.assertRaisesRegex(ValueError, "Parent vector dimension mismatch"):
            store_embedded_chunks([group], client)

        self.assertEqual(client.upsert_calls, [])

    def test_metadata_is_not_mutated_and_logs_do_not_expose_content_or_vectors(self) -> None:
        client = FakeQdrantClient()
        group = _embedded_group()
        parent_metadata_before = deepcopy(group["parent"]["document"].metadata)
        child_metadata_before = deepcopy(group["children"][0]["document"].metadata)

        with self.assertLogs("rag_app", level="INFO") as logs:
            store_embedded_chunks([group], client)

        self.assertEqual(group["parent"]["document"].metadata, parent_metadata_before)
        self.assertEqual(group["children"][0]["document"].metadata, child_metadata_before)
        log_output = "\n".join(logs.output)
        self.assertNotIn("secret-api-key", log_output)
        self.assertNotIn("parent content", log_output)
        self.assertNotIn("child content", log_output)
        self.assertNotIn("0.1", log_output)
        self.assertNotIn("0.2", log_output)

    def test_same_generation_is_idempotent_and_counts_do_not_increase(self) -> None:
        client = FakeQdrantClient()
        group = _embedded_group()
        parent_id = build_parent_point_id(group["parent"]["document"])
        child_id = build_child_point_id(group["children"][0]["document"])

        first = store_embedded_chunks([group], client)
        second = store_embedded_chunks([group], client)

        self.assertEqual(first["after_parent_count"], 1)
        self.assertEqual(first["after_child_count"], 1)
        self.assertEqual(second["before_parent_count"], 1)
        self.assertEqual(second["before_child_count"], 1)
        self.assertEqual(second["after_parent_count"], 1)
        self.assertEqual(second["after_child_count"], 1)
        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][child_id].payload["status"],
            ACTIVE_STATUS,
        )

    def test_same_generation_parent_upsert_failure_keeps_existing_active(self) -> None:
        client = FakeQdrantClient()
        group = _embedded_group()
        parent_id = build_parent_point_id(group["parent"]["document"])
        child_id = build_child_point_id(group["children"][0]["document"])

        store_embedded_chunks([group], client)
        client.fail_parent_upsert = True

        with self.assertRaises(RuntimeError):
            store_embedded_chunks([group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][child_id].payload["status"],
            ACTIVE_STATUS,
        )

    def test_same_generation_child_upsert_failure_keeps_existing_active(self) -> None:
        client = FakeQdrantClient()
        group = _embedded_group()
        parent_id = build_parent_point_id(group["parent"]["document"])
        child_id = build_child_point_id(group["children"][0]["document"])

        store_embedded_chunks([group], client)
        client.fail_child_upsert = True

        with self.assertRaises(RuntimeError):
            store_embedded_chunks([group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][child_id].payload["status"],
            ACTIVE_STATUS,
        )

    def test_same_generation_verification_failure_keeps_existing_active(self) -> None:
        client = FakeQdrantClient()
        group = _embedded_group()
        parent_id = build_parent_point_id(group["parent"]["document"])
        child_id = build_child_point_id(group["children"][0]["document"])

        store_embedded_chunks([group], client)
        client.fail_retrieve = True

        with self.assertRaises(RuntimeError):
            store_embedded_chunks([group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][child_id].payload["status"],
            ACTIVE_STATUS,
        )

    def test_new_generation_activates_new_points_and_inactivates_old_points(self) -> None:
        client = FakeQdrantClient()
        old_group = _embedded_group()
        new_group = _embedded_group(
            parent_metadata=_parent_metadata(document_checksum="new-checksum"),
            child_metadata=_child_metadata(document_checksum="new-checksum"),
        )
        old_parent_id = build_parent_point_id(old_group["parent"]["document"])
        old_child_id = build_child_point_id(old_group["children"][0]["document"])
        new_parent_id = build_parent_point_id(new_group["parent"]["document"])
        new_child_id = build_child_point_id(new_group["children"][0]["document"])

        store_embedded_chunks([old_group], client)
        store_embedded_chunks([new_group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][old_parent_id].payload["status"],
            INACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][old_child_id].payload["status"],
            INACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][new_parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][new_child_id].payload["status"],
            ACTIVE_STATUS,
        )

    def test_changed_chunking_config_hash_creates_new_generation(self) -> None:
        old_group = _embedded_group()
        new_group = _embedded_group(
            parent_metadata=_parent_metadata(chunking_config_hash="new-config"),
            child_metadata=_child_metadata(chunking_config_hash="new-config"),
        )

        self.assertNotEqual(
            build_parent_point_id(old_group["parent"]["document"]),
            build_parent_point_id(new_group["parent"]["document"]),
        )
        self.assertNotEqual(
            build_child_point_id(old_group["children"][0]["document"]),
            build_child_point_id(new_group["children"][0]["document"]),
        )

    def test_failed_new_generation_keeps_old_generation_active(self) -> None:
        client = FakeQdrantClient()
        old_group = _embedded_group()
        new_group = _embedded_group(
            parent_metadata=_parent_metadata(document_checksum="new-checksum"),
            child_metadata=_child_metadata(document_checksum="new-checksum"),
        )
        old_parent_id = build_parent_point_id(old_group["parent"]["document"])
        old_child_id = build_child_point_id(old_group["children"][0]["document"])

        store_embedded_chunks([old_group], client)
        client.fail_child_upsert = True

        with self.assertLogs("rag_app", level="ERROR") as logs:
            with self.assertRaises(RuntimeError):
                store_embedded_chunks([new_group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][old_parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][old_child_id].payload["status"],
            ACTIVE_STATUS,
        )
        log_output = "\n".join(logs.output)
        self.assertIn("Qdrant generation storage failed", log_output)
        self.assertNotIn("secret-api-key", log_output)
        self.assertNotIn("parent content", log_output)
        self.assertNotIn("child content", log_output)

    def test_parent_upsert_failure_keeps_old_generation_active(self) -> None:
        client = FakeQdrantClient()
        old_group = _embedded_group()
        new_group = _embedded_group(
            parent_metadata=_parent_metadata(document_checksum="new-checksum"),
            child_metadata=_child_metadata(document_checksum="new-checksum"),
        )
        old_parent_id = build_parent_point_id(old_group["parent"]["document"])
        old_child_id = build_child_point_id(old_group["children"][0]["document"])

        store_embedded_chunks([old_group], client)
        client.fail_parent_upsert = True

        with self.assertRaises(RuntimeError):
            store_embedded_chunks([new_group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][old_parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][old_child_id].payload["status"],
            ACTIVE_STATUS,
        )

    def test_verification_failure_keeps_old_generation_active_and_new_inactive(self) -> None:
        client = FakeQdrantClient()
        old_group = _embedded_group()
        new_group = _embedded_group(
            parent_metadata=_parent_metadata(document_checksum="new-checksum"),
            child_metadata=_child_metadata(document_checksum="new-checksum"),
        )
        old_parent_id = build_parent_point_id(old_group["parent"]["document"])
        old_child_id = build_child_point_id(old_group["children"][0]["document"])
        new_parent_id = build_parent_point_id(new_group["parent"]["document"])
        new_child_id = build_child_point_id(new_group["children"][0]["document"])

        store_embedded_chunks([old_group], client)
        client.fail_retrieve = True

        with self.assertRaises(RuntimeError):
            store_embedded_chunks([new_group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][old_parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][old_child_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][new_parent_id].payload["status"],
            INACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][new_child_id].payload["status"],
            INACTIVE_STATUS,
        )

    def test_activation_failure_restores_new_inactive_and_old_active(self) -> None:
        client = FakeQdrantClient()
        old_group = _embedded_group()
        new_group = _embedded_group(
            parent_metadata=_parent_metadata(document_checksum="new-checksum"),
            child_metadata=_child_metadata(document_checksum="new-checksum"),
        )
        old_parent_id = build_parent_point_id(old_group["parent"]["document"])
        old_child_id = build_child_point_id(old_group["children"][0]["document"])
        new_parent_id = build_parent_point_id(new_group["parent"]["document"])
        new_child_id = build_child_point_id(new_group["children"][0]["document"])

        store_embedded_chunks([old_group], client)
        client.fail_next_active_child_status = True

        with self.assertLogs("rag_app", level="INFO") as logs:
            with self.assertRaises(RuntimeError):
                store_embedded_chunks([new_group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][old_parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][old_child_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][new_parent_id].payload["status"],
            INACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][new_child_id].payload["status"],
            INACTIVE_STATUS,
        )
        log_output = "\n".join(logs.output)
        self.assertIn("Qdrant lifecycle recovery attempted", log_output)
        self.assertIn("Qdrant lifecycle recovery succeeded", log_output)
        self.assertNotIn("parent content", log_output)
        self.assertNotIn("child content", log_output)
        self.assertNotIn("secret-api-key", log_output)

    def test_old_deactivation_failure_recovers_new_inactive_and_old_active(self) -> None:
        client = FakeQdrantClient()
        old_group = _embedded_group()
        new_group = _embedded_group(
            parent_metadata=_parent_metadata(document_checksum="new-checksum"),
            child_metadata=_child_metadata(document_checksum="new-checksum"),
        )
        old_parent_id = build_parent_point_id(old_group["parent"]["document"])
        old_child_id = build_child_point_id(old_group["children"][0]["document"])
        new_parent_id = build_parent_point_id(new_group["parent"]["document"])
        new_child_id = build_child_point_id(new_group["children"][0]["document"])

        store_embedded_chunks([old_group], client)
        client.fail_next_inactive_parent_status = True

        with self.assertLogs("rag_app", level="INFO") as logs:
            with self.assertRaises(RuntimeError):
                store_embedded_chunks([new_group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][old_parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][old_child_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][new_parent_id].payload["status"],
            INACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][new_child_id].payload["status"],
            INACTIVE_STATUS,
        )
        log_output = "\n".join(logs.output)
        self.assertIn("Qdrant lifecycle recovery attempted", log_output)
        self.assertIn("Qdrant lifecycle recovery succeeded", log_output)
        self.assertNotIn("parent content", log_output)
        self.assertNotIn("child content", log_output)
        self.assertNotIn("secret-api-key", log_output)

    def test_unrelated_document_generation_is_not_modified(self) -> None:
        client = FakeQdrantClient()
        original_group = _embedded_group()
        unrelated_group = _embedded_group(
            parent_metadata=_parent_metadata(document_id="doc-2", parent_id="parent-doc-2"),
            child_metadata=_child_metadata(
                document_id="doc-2",
                parent_id="parent-doc-2",
                child_id="child-doc-2",
            ),
        )
        new_group = _embedded_group(
            parent_metadata=_parent_metadata(document_checksum="new-checksum"),
            child_metadata=_child_metadata(document_checksum="new-checksum"),
        )
        unrelated_parent_id = build_parent_point_id(unrelated_group["parent"]["document"])
        unrelated_child_id = build_child_point_id(unrelated_group["children"][0]["document"])

        store_embedded_chunks([original_group, unrelated_group], client)
        store_embedded_chunks([new_group], client)

        self.assertEqual(
            client.collections[PARENT_CHUNKS_COLLECTION][unrelated_parent_id].payload["status"],
            ACTIVE_STATUS,
        )
        self.assertEqual(
            client.collections[CHILD_CHUNKS_COLLECTION][unrelated_child_id].payload["status"],
            ACTIVE_STATUS,
        )


if __name__ == "__main__":
    unittest.main()
