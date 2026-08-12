import os
import unittest
from unittest.mock import Mock, patch

from rag_app.storage.qdrant_connection import (
    QDRANT_API_KEY_ENV,
    QDRANT_URL_ENV,
    QdrantConfig,
    get_qdrant_client,
    load_qdrant_config,
    verify_qdrant_connection,
)


class QdrantConnectionTests(unittest.TestCase):
    def test_missing_qdrant_url_fails_clearly(self) -> None:
        with patch.dict(os.environ, {QDRANT_API_KEY_ENV: "secret-key"}, clear=True):
            with patch("rag_app.storage.qdrant_connection.load_dotenv"):
                with self.assertRaisesRegex(ValueError, QDRANT_URL_ENV) as error:
                    load_qdrant_config()

        self.assertNotIn("secret-key", str(error.exception))

    def test_missing_qdrant_api_key_fails_clearly(self) -> None:
        with patch.dict(os.environ, {QDRANT_URL_ENV: "https://qdrant.example"}, clear=True):
            with patch("rag_app.storage.qdrant_connection.load_dotenv"):
                with self.assertRaisesRegex(ValueError, QDRANT_API_KEY_ENV) as error:
                    load_qdrant_config()

        self.assertNotIn("https://qdrant.example", str(error.exception))

    def test_valid_configuration_loads_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                QDRANT_URL_ENV: "https://qdrant.example",
                QDRANT_API_KEY_ENV: "secret-key",
            },
            clear=True,
        ):
            with patch("rag_app.storage.qdrant_connection.load_dotenv"):
                with self.assertLogs("rag_app", level="INFO") as logs:
                    config = load_qdrant_config()

        self.assertEqual(config.url, "https://qdrant.example")
        self.assertEqual(config.api_key, "secret-key")
        self.assertNotIn("secret-key", "\n".join(logs.output))
        self.assertNotIn("https://qdrant.example", "\n".join(logs.output))

    def test_client_creation_uses_configured_url_and_api_key(self) -> None:
        config = QdrantConfig(url="https://qdrant.example", api_key="secret-key")

        with patch("rag_app.storage.qdrant_connection.QdrantClient") as client_cls:
            client = get_qdrant_client(config)

        client_cls.assert_called_once_with(
            url="https://qdrant.example",
            api_key="secret-key",
        )
        self.assertIs(client, client_cls.return_value)

    def test_connection_verification_success(self) -> None:
        client = Mock()
        client.get_collections.return_value = []

        with self.assertLogs("rag_app", level="INFO") as logs:
            result = verify_qdrant_connection(client)

        self.assertTrue(result)
        self.assertIn("Qdrant connection verified successfully.", "\n".join(logs.output))

    def test_connection_verification_failure_does_not_expose_secret(self) -> None:
        client = Mock()
        client.get_collections.side_effect = RuntimeError("secret-key leaked")

        with self.assertLogs("rag_app", level="ERROR") as logs:
            result = verify_qdrant_connection(client)

        self.assertFalse(result)
        log_output = "\n".join(logs.output)
        self.assertIn("Qdrant connection verification failed", log_output)
        self.assertNotIn("secret-key", log_output)


if __name__ == "__main__":
    unittest.main()
