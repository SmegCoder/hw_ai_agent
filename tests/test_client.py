import io
import json
import os
import socket
import ssl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from hw_ai_agent.config import ConfigurationError, Settings, load_env
from hw_ai_agent.client import ModelClientError, OpenAICompatibleClient

class ClientTests(unittest.TestCase):
    def client(self, provider="openrouter"):
        return OpenAICompatibleClient(Settings(provider, "model", "http://127.0.0.1:1/v1", "secret-example"))

    def response(self, client, body):
        return patch.object(client.opener, "open", return_value=io.BytesIO(json.dumps(body).encode()))

    def test_three_provider_request_formats(self):
        for provider in ("openrouter", "openai", "ollama"):
            with self.subTest(provider=provider):
                client = self.client(provider)
                with self.response(client, {"choices": [{"message": {"content": "OK"}}], "usage": {"total_tokens": 7}}) as transport:
                    self.assertEqual(client.complete([], []), {"content": "OK"})
                request = transport.call_args.args[0]
                body = json.loads(request.data)
                self.assertEqual(body["model"], "model")
                self.assertFalse(body["stream"])
                self.assertIn("tools", body)
                self.assertIn("max_completion_tokens" if provider == "openai" else "max_tokens", body)
                if provider == "ollama":
                    self.assertEqual(body["reasoning_effort"], "none")
                self.assertEqual(client.usage[0]["total_tokens"], 7)

    def test_transport_errors_do_not_leak_key(self):
        client = self.client()
        errors = [HTTPError("http://example", 401, "bad", {}, io.BytesIO(b"secret-example")), URLError("secret-example")]
        for error in errors:
            with patch.object(client.opener, "open", side_effect=error):
                with self.assertRaises(ModelClientError) as caught:
                    client.complete([], [])
                self.assertNotIn("secret-example", str(caught.exception))

    def test_model_history_escapes_surrogates_for_the_provider(self):
        client = self.client()
        messages = [{"role": "assistant", "content": "Целый текст ✓",
                     "reasoning_details": [{"text": "Данные\udcd0"}]}]
        with self.response(client, {"choices": [{"message": {"content": "OK"}}]}) as transport:
            client.complete(messages, [])
        sent = json.loads(transport.call_args.args[0].data)["messages"][0]
        self.assertEqual(sent["reasoning_details"][0]["text"], r"Данные\udcd0")
        self.assertEqual(sent["content"], "Целый текст ✓")
        self.assertEqual(messages[0]["reasoning_details"][0]["text"], "Данные\udcd0")

    def test_openrouter_distinguishes_retryable_errors_and_redacts_provider_messages(self):
        client = self.client()
        cases = [
            (HTTPError("http://example", 429, "secret", {"Retry-After": "12"}, None), "HTTP 429", 12),
            (HTTPError("http://example", 402, "secret", {}, None), "HTTP 402", None),
            (URLError(TimeoutError("secret")), "время ожидания", 0),
            (URLError(socket.gaierror(socket.EAI_AGAIN, "secret")), "DNS", 0),
            (URLError(ssl.SSLCertVerificationError("secret")), "TLS", None),
        ]
        for error, message, wait in cases:
            with self.subTest(message=message), patch.object(client.opener, "open", side_effect=error):
                with self.assertRaisesRegex(ModelClientError, message) as caught:
                    client.complete([], [])
                self.assertNotIn("secret", str(caught.exception))
                self.assertEqual(caught.exception.retry_after, wait)
        body = {"choices": [{"finish_reason": "error", "message": {"content": "partial"},
                             "error": {"code": 502, "message": "secret"}}]}
        with self.response(client, body), self.assertRaisesRegex(ModelClientError, "код 502"):
            client.complete([], [])

    def test_malformed_and_truncated_responses(self):
        for body in ({}, {"choices": []}, {"choices": [{"message": {}, "finish_reason": "length"}]}, []):
            client = self.client()
            with self.response(client, body), self.assertRaises(ModelClientError):
                client.complete([], [])

    def test_invalid_json_and_oversized_response(self):
        client = self.client()
        for raw in (b"not json", b"x" * (client.max_response_bytes + 1)):
            with patch.object(client.opener, "open", return_value=io.BytesIO(raw)), self.assertRaises(ModelClientError):
                client.complete([], [])

    def test_provider_configuration_and_key_isolation(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}, clear=True):
            self.assertEqual(Settings.from_env().provider, "openrouter")
            self.assertEqual(Settings.from_env("ollama").api_key, "ollama")
            with self.assertRaises(ConfigurationError):
                Settings.from_env("openai", "any-model")
            with patch.dict(os.environ, {"OPENROUTER_BASE_URL": "http://example.com/v1"}):
                with self.assertRaises(ConfigurationError):
                    Settings.from_env()

    def test_dotenv_is_not_executed_and_does_not_overwrite_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"AGENT_PROVIDER": "ollama"}, clear=True):
            path = Path(directory) / ".env"
            path.write_text("AGENT_PROVIDER=openrouter\nOPENROUTER_API_KEY='$(touch secret)'\n")
            load_env(path)
            self.assertEqual(os.environ["AGENT_PROVIDER"], "ollama")
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "$(touch secret)")

if __name__ == "__main__":
    unittest.main()
