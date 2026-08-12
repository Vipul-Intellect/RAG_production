import unittest
from unittest.mock import Mock

from qdrant_client.models import Distance

from rag_app.embeddings.embedder import EXPECTED_EMBEDDING_DIMENSION
from rag_app.storage.qdrant_collections import (
    CHILD_CHUNKS_COLLECTION,
    PARENT_CHUNKS_COLLECTION,
    REQUIRED_COLLECTIONS,
    ensure_qdrant_collections,
)


class QdrantCollectionTests(unittest.TestCase):
    def test_missing_parent_and_child_collections_are_created(self) -> None:
        client = Mock()
        client.collection_exists.return_value = False

        with self.assertLogs("rag_app", level="INFO") as logs:
            ensure_qdrant_collections(client)

        self.assertEqual(client.create_collection.call_count, 2)
        created_names = [
            call.kwargs["collection_name"]
            for call in client.create_collection.call_args_list
        ]
        self.assertEqual(created_names, list(REQUIRED_COLLECTIONS))

        for call in client.create_collection.call_args_list:
            vectors_config = call.kwargs["vectors_config"]
            self.assertEqual(vectors_config.size, EXPECTED_EMBEDDING_DIMENSION)
            self.assertEqual(vectors_config.distance, Distance.COSINE)

        log_output = "\n".join(logs.output)
        self.assertIn("Qdrant collection initialization started.", log_output)
        self.assertIn(PARENT_CHUNKS_COLLECTION, log_output)
        self.assertIn(CHILD_CHUNKS_COLLECTION, log_output)

    def test_existing_collections_are_not_recreated_or_deleted(self) -> None:
        client = Mock()
        client.collection_exists.return_value = True

        ensure_qdrant_collections(client)
        ensure_qdrant_collections(client)

        client.create_collection.assert_not_called()
        self.assertFalse(client.delete_collection.called)
        self.assertEqual(client.collection_exists.call_count, 4)

    def test_mixed_existing_and_missing_collection_creates_only_missing(self) -> None:
        client = Mock()
        client.collection_exists.side_effect = [True, False]

        ensure_qdrant_collections(client)

        client.create_collection.assert_called_once()
        self.assertEqual(
            client.create_collection.call_args.kwargs["collection_name"],
            CHILD_CHUNKS_COLLECTION,
        )

    def test_creation_failure_is_clear_and_secret_safe(self) -> None:
        client = Mock()
        client.collection_exists.return_value = False
        client.create_collection.side_effect = RuntimeError("secret-api-key")

        with self.assertLogs("rag_app", level="ERROR") as logs:
            with self.assertRaisesRegex(
                RuntimeError,
                f"Qdrant collection initialization failed for {PARENT_CHUNKS_COLLECTION}",
            ) as error:
                ensure_qdrant_collections(client)

        self.assertNotIn("secret-api-key", str(error.exception))
        log_output = "\n".join(logs.output)
        self.assertIn("Qdrant collection initialization failed", log_output)
        self.assertNotIn("secret-api-key", log_output)
        self.assertNotIn("[0.1", log_output)


if __name__ == "__main__":
    unittest.main()
