import io
import json
import os
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
