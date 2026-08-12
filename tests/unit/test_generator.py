import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_app.generation.generator import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_KEY_ENV_VAR,
    GROUNDING_SYSTEM_PROMPT,
    _load_gemini_client,
    generate_answer,
)


class FakeGeminiClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response or SimpleNamespace(
            text="Grounded answer [C1]",
            usage_metadata={"promptTokenCount": 12, "candidatesTokenCount": 4},
        )
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _retrieval_result(parent_content: str | None = "parent policy text") -> dict[str, object]:
    return {
        "query": "What is the policy?",
        "requested_k": 2,
        "retrieved_children": [
            {
                "point_id": "child-point-1",
                "score": 0.9,
                "content": "child policy text",
                "child_id": "child-1",
                "parent_id": "parent-1",
                "document_id": "doc-1",
                "document_checksum": "checksum-1",
                "chunking_config_hash": "chunk-hash",
                "parent_index": 0,
                "child_index": 0,
                "status": "ACTIVE",
                "metadata": {
                    "source_file": "policy.pdf",
                    "source_path": "data/pdf/policy.pdf",
                    "page_start": 3,
                    "page_end": 4,
                    "document_id": "doc-1",
                    "parent_index": 0,
                },
            }
        ],
        "corresponding_parents": [
            {
                "point_id": "parent-point-1",
                "content": parent_content,
                "parent_id": "parent-1",
                "document_id": "doc-1",
                "document_checksum": "checksum-1",
                "chunking_config_hash": "chunk-hash",
                "parent_index": 0,
                "status": "ACTIVE",
                "metadata": {
                    "source_file": "policy.pdf",
                    "source_path": "data/pdf/policy.pdf",
                    "page_start": 3,
                    "page_end": 4,
                    "document_id": "doc-1",
                    "parent_index": 0,
                },
            }
        ],
        "retrieved_child_count": 1,
        "retrieved_parent_count": 1,
        "duration_seconds": 0.01,
    }


class GenerationTests(unittest.TestCase):
    def test_retrieved_context_and_query_are_passed_to_gemini(self) -> None:
        client = FakeGeminiClient()

        result = generate_answer(_retrieval_result(), gemini_client=client)

        call = client.calls[0]
        self.assertIn("parent policy text", call["contents"])
        self.assertIn("What is the policy?", call["contents"])
        self.assertEqual(result["answer"], "Grounded answer [C1]")
        self.assertFalse(result["no_context"])

    def test_grounding_instructions_are_included(self) -> None:
        client = FakeGeminiClient()

        generate_answer(_retrieval_result(), gemini_client=client)

        instruction = client.calls[0]["system_instruction"]
        self.assertEqual(instruction, GROUNDING_SYSTEM_PROMPT)
        self.assertIn("Answer only from the supplied context", instruction)
        self.assertIn("Do not invent facts", instruction)
        self.assertIn("Do not use outside knowledge", instruction)
        self.assertIn("Cite the supplied source labels", instruction)

    def test_no_context_result_does_not_call_gemini(self) -> None:
        client = FakeGeminiClient()
        retrieval_result = _retrieval_result(parent_content="")
        retrieval_result["retrieved_children"] = []
        retrieval_result["corresponding_parents"] = []

        result = generate_answer(retrieval_result, gemini_client=client)

        self.assertTrue(result["no_context"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["used_context"], [])
        self.assertEqual(client.calls, [])
        self.assertIn("not have enough retrieved context", result["answer"])

    def test_no_context_result_contains_no_fabricated_citation(self) -> None:
        client = FakeGeminiClient()
        retrieval_result = _retrieval_result(parent_content="")
        retrieval_result["retrieved_children"] = []
        retrieval_result["corresponding_parents"] = []

        result = generate_answer(retrieval_result, gemini_client=client)

        self.assertEqual(result["citations"], [])
        self.assertNotIn("policy.pdf", result["answer"])
        self.assertEqual(client.calls, [])

    def test_citations_are_generated_only_from_retrieved_metadata(self) -> None:
        result = generate_answer(_retrieval_result(), gemini_client=FakeGeminiClient())

        self.assertEqual(
            result["citations"],
            [
                {
                    "label": "C1",
                    "source_file": "policy.pdf",
                    "source_path": "data/pdf/policy.pdf",
                    "page_start": 3,
                    "page_end": 4,
                    "document_id": "doc-1",
                    "parent_index": 0,
                }
            ],
        )

    def test_api_key_is_loaded_from_environment(self) -> None:
        with patch.dict(os.environ, {GEMINI_API_KEY_ENV_VAR: "secret-api-key"}, clear=False):
            client = _load_gemini_client()

        self.assertEqual(client.api_key, "secret-api-key")

    def test_api_key_is_never_logged(self) -> None:
        client = FakeGeminiClient()

        with patch.dict(os.environ, {GEMINI_API_KEY_ENV_VAR: "secret-api-key"}, clear=False):
            with self.assertLogs("rag_app", level="INFO") as logs:
                generate_answer(_retrieval_result(), gemini_client=client)

        self.assertNotIn("secret-api-key", "\n".join(logs.output))

    def test_gemini_failure_is_handled_cleanly_without_sensitive_logs(self) -> None:
        client = FakeGeminiClient(error=RuntimeError("provider failed with secret-api-key"))

        with self.assertLogs("rag_app", level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "Gemini generation failed"):
                generate_answer(_retrieval_result(), gemini_client=client)

        log_output = "\n".join(logs.output)
        self.assertIn("Gemini generation failed", log_output)
        self.assertIn("RuntimeError", log_output)
        self.assertNotIn("secret-api-key", log_output)
        self.assertNotIn("parent policy text", log_output)
        self.assertNotIn("child policy text", log_output)

    def test_generation_duration_is_recorded(self) -> None:
        result = generate_answer(_retrieval_result(), gemini_client=FakeGeminiClient())

        self.assertGreaterEqual(result["duration_seconds"], 0.0)

    def test_token_usage_is_recorded_when_available(self) -> None:
        result = generate_answer(_retrieval_result(), gemini_client=FakeGeminiClient())

        self.assertEqual(
            result["token_usage"],
            {"promptTokenCount": 12, "candidatesTokenCount": 4},
        )

    def test_model_name_defaults_and_can_be_overridden(self) -> None:
        client = FakeGeminiClient()

        result = generate_answer(_retrieval_result(), gemini_client=client)
        custom_result = generate_answer(
            _retrieval_result(),
            gemini_client=client,
            model_name="gemini-custom",
        )

        self.assertEqual(result["model_name"], DEFAULT_GEMINI_MODEL)
        self.assertEqual(custom_result["model_name"], "gemini-custom")


if __name__ == "__main__":
    unittest.main()
