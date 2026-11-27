import http.client
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_fixtures import tool_call
from hw_ai_agent.web_server import Handler
from hw_ai_agent.web_service import WebService


class WebTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.project = Path(__file__).resolve().parents[1]
        self.app = WebService(self.root / "state/db", self.workspace, self.project / "policies/tracker.json")
        self.profile = dict(provider="ollama", model="test-model", source="", check_on_start=True, auto_update=True,
                            disabled=["review-python-tests"], demo=False, mcp=str(self.project / "mcp/tracker.json"))

    def http(self, path, body=None, *, origin=None):
        client, server = socket.socketpair()
        with client, server:
            thread = threading.Thread(target=Handler, args=(server, ("127.0.0.1", 1),
                SimpleNamespace(app=self.app, server_port=8765)))
            thread.start()
            request = http.client.HTTPConnection("127.0.0.1", 8765)
            request.sock = client
            headers = {"X-Agi-UI": "1", "Content-Type": "application/json"}
            if origin:
                headers["Origin"] = origin
            request.request("GET" if body is None else "POST", path, None if body is None else json.dumps(body), headers)
            reply = request.getresponse()
            content = reply.read()
            thread.join(5)
            return reply.status, content

    def api(self, path, body=None):
        status, content = self.http("/api/" + path, body)
        self.assertEqual(status, 200, content)
        return json.loads(content)["data"]

    def wait(self, identifier, condition):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            data = self.api("chat?id=" + identifier)
            if condition(data):
                return data
            if data["error"]:
                self.fail(data["error"])
            time.sleep(.03)
        self.fail("Timed out waiting for the run")

    def test_http_origin_assets_and_saved_profiles(self):
        self.assertEqual(self.http("/api/new", {}, origin="http://elsewhere.test")[0], 403)
        for page in ("/", "/style.css", "/app.js"):
            self.assertEqual(self.http(page)[0], 200)
        self.api("settings", self.profile)
        restored = WebService(self.app.db, self.workspace)
        self.assertEqual(restored.prefs["disabled"], ["review-python-tests"])
        self.assertEqual(self.api("mcp")["tools"][0]["function"]["name"], "mcp__tracker__list_tasks")

    def test_model_choice_is_bound_only_to_new_chat(self):
        self.api("settings", self.profile)
        default_id = self.api("new", {})
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-web-key"}), \
                patch("hw_ai_agent.cli.configure_client", side_effect=lambda client, *args: client):
            chosen = self.api("new", {"provider":"openrouter", "model":"deepseek/deepseek-v4.1-flash-1017"})
        restarted = WebService(self.app.db, self.workspace)
        self.assertEqual(restarted.chat(chosen)["model"], "deepseek/deepseek-v4.1-flash-1017")
        self.assertEqual(restarted.chat(default_id)["model"], self.profile["model"])
        self.assertEqual(restarted.prefs["provider"], self.profile["provider"])
        self.assertEqual(self.api("chat?id="+self.api("new", {}))["model"], self.profile["model"])
        before = self.api("sessions")
        self.assertEqual(self.http("/api/new", {"model":"invalid"})[0], 400)
        self.assertEqual(self.api("sessions"), before)

    def test_workspace_creation_reuse_and_recovery(self):
        self.api("settings", self.profile)
        project = self.workspace / "курс" / "дз"
        project.mkdir(parents=True)
        (self.workspace / "external").symlink_to(self.root)
        self.assertEqual(self.api("folders")["folders"], ["курс"])
        self.assertEqual(self.app.folders("курс")["folders"], ["дз"])
        ids = [self.api("new", {}) for _ in range(2)]
        folders = [self.workspace / i[:8] for i in ids]
        self.assertNotEqual(*folders)
        for identifier, folder in zip(ids, folders):
            self.assertTrue(folder.is_dir())
            self.assertEqual(self.api("chat?id=" + identifier)["workspace"], str(folder.resolve()))
        for relative, expected in (("курс/дз", project), (".", self.workspace)):
            identifier = self.api("new", {"folder": relative})
            restored = WebService(self.app.db, self.workspace)
            self.assertEqual(restored.chat(identifier)["workspace"], str(expected.resolve()))
        for relative in ("../state", str(self.root), "external", "unknown"):
            self.assertEqual(self.http("/api/new", {"folder": relative})[0], 400)
        before = set(self.workspace.iterdir())
        with patch("hw_ai_agent.cli.new_agent", side_effect=ValueError("invalid config")):
            self.assertEqual(self.http("/api/new", {})[0], 400)
            self.assertEqual(self.http("/api/new", {"folder": "курс/дз"})[0], 400)
        self.assertEqual(set(self.workspace.iterdir()), before)
        self.assertTrue(project.is_dir())

    def test_available_models_and_saved_defaults(self):
        self.api("settings", self.profile)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-web-key"}):
            options = self.api("chat-options")
            self.assertIn("openai", options["models"])
            self.api("model", {"provider": "openai", "model": "gpt-4.1-mini"})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENROUTER_API_KEY": ""}):
            options = self.api("chat-options")
            self.assertNotIn("openai", options["models"])
            self.assertEqual(options["provider"], "ollama")
            router = options["models"]["openrouter"]
            self.assertEqual(len(router), 10)
            self.assertEqual(router[0][0], "nvidia/nemotron-3-ultra-550b-a55b:free")
            self.assertIn(["openai/gpt-6.2-sol", "GPT-6.2 Sol"], router)
            direct = {"provider": "openai", "model": "gpt-4.1-mini"}
            for endpoint in ("model", "new"):
                self.assertEqual(self.http("/api/" + endpoint, direct)[0], 400)
            self.assertEqual(self.http("/api/settings", {**self.profile, **direct})[0], 400)
            self.api("model", {"provider": "openrouter", "model": router[-1][0]})
            integrations = {k: v for k, v in self.profile.items() if k not in {"provider", "model"}}
            self.api("settings", integrations)
            self.assertEqual(WebService(self.app.db, self.workspace).chat_options()["model"], router[-1][0])

    def test_dialogue_tool_policy_history_and_doctor(self):
        self.api("settings", self.profile)
        responses = []
        history = []
        class Client:
            def __init__(self, settings, **options):
                self.settings = settings
                self.usage = []
            def complete(self, messages, tools):
                history.append(list(messages))
                self.usage.append({"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16})
                return responses.pop(0)

        with patch("hw_ai_agent.cli.OpenAICompatibleClient", Client):
            identifier = self.api("new", {})
            folder = self.workspace / identifier[:8]
            for number, allow in enumerate((False, True)):
                responses.extend([
                    {"role": "assistant", "tool_calls": [tool_call("list", "mcp__tracker__list_tasks", {})]},
                    {"role": "assistant", "tool_calls": [tool_call("write", "write_file", {"path":"report.md","content":"report"})]},
                    {"role": "assistant", "content": "Готово" if allow else "Запись отклонена"},
                ])
                self.api("send", {"id":identifier, "prompt":["Собери отчёт", "Теперь сохрани"][number]})
                state = self.wait(identifier, lambda d: d["approval"])
                self.assertFalse((self.workspace / "report.md").exists())
                self.assertEqual(state["approval"]["target"], "report.md")
                state_key = {"id":identifier, "request":state["approval"]["id"], "allow":allow}
                self.api("decision", state_key)
                self.assertEqual(self.http("/api/decision", state_key)[0], 400)
                state = self.wait(identifier, lambda d: not d["running"])
                self.assertFalse(state["error"], state["error"])
                self.assertEqual((folder / "report.md").exists(), allow)
                self.assertFalse((self.workspace / "report.md").exists())
            self.assertIn("Запись отклонена", [m.get("content") for m in history[-1]])
            self.assertEqual(state["stats"]["usage"]["prompt_tokens"]["tokens"], 60)
            self.assertEqual(len(self.api("doctor?id="+identifier)["runs"]), 2)
            self.assertEqual(state["stats"]["mcp_calls"]["tracker/list_tasks"], 2)
            self.assertEqual(self.api("sessions")[0]["id"], identifier)
            self.assertNotIn("review-python-tests", history[0][0]["content"])

    def test_evaluation_actions_and_update_rollback(self):
        self.api("settings", {**self.profile, "source":"demo"})
        self.api("check", {})
        name = "tracker-json-deadlines"
        self.api("install", {"name":name})
        self.api("publish-demo", {"version":"1.1.0"})
        self.api("check", {})
        def reuse_evidence(output, *args, **kwargs):
            saved = "skill-update-eval" if kwargs.get("comparison") else "study-tracker/eval"
            shutil.copytree(self.project / "docs" / saved, output, dirs_exist_ok=True)
            return 0
        def finished():
            for _ in range(300):
                job = self.api("reports")
                if not job["running"]:
                    self.assertFalse(job["error"], job["error"])
                    return
                time.sleep(.02)
            self.fail("Evaluation did not finish")
        with patch("hw_ai_agent.web_service.run_evaluation", side_effect=reuse_evidence):
            series = self.api("eval", {})
            finished()
            self.assertEqual(self.api("report?key="+series)["summary"]["completed"], 30)
            self.assertEqual(self.api("eval", {"series":series}), series)
            finished()
        with patch("hw_ai_agent.skill_updates.run_evaluation", side_effect=reuse_evidence):
            self.api("eval", {"candidate":name, "focus":["deadlines","missing-deadlines"]})
            finished()
        candidate = self.api("settings")["candidates"][0]
        self.assertEqual(candidate["status"], "accepted")
        self.assertEqual(self.api("rollback", {"name":name}), "1.0.0")

    def test_configured_secret_removed_before_persistence(self):
        self.api("settings", self.profile)
        with patch.dict(os.environ, {"OPENAI_API_KEY":"example-secret-value"}):
            identifier = self.api("new", {})
            with patch("hw_ai_agent.client.OpenAICompatibleClient.complete",
                       return_value={"role":"assistant","content":"echo example-secret-value"}):
                self.api("send", {"id":identifier, "prompt":"Случайно вставлен example-secret-value"})
                self.wait(identifier, lambda d: not d["running"])
            self.assertNotIn(b"example-secret-value", self.app.db.read_bytes())
            self.assertIn("[REDACTED]", self.api("sessions")[0]["title"])

    def test_market_update_and_evidence(self):
        source = self.root / "market"
        shutil.copytree(self.project / "examples/skill-update", source)
        live = source / "catalog.json"
        shutil.copyfile(source / "catalog-1.0.0.json", live)
        self.api("settings", {**self.profile, "source":str(live)})
        self.api("check", {})
        self.api("install", {"name":"tracker-json-deadlines"})
        shutil.copyfile(source / "catalog-1.1.0.json", live)
        self.api("check", {})
        self.assertEqual(self.api("settings")["candidates"][0]["status"], "pending")
        report = self.api("report?key=archive-update")
        self.assertEqual(report["summary"]["completed"], 30)
        run = report["rows"][0]["run_id"]
        self.assertTrue(self.api("report?key=archive-update&run="+run))
