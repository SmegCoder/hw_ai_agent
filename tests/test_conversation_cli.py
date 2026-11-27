from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import chdir, redirect_stderr, redirect_stdout
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_fixtures import PLAN, tool_call
from hw_ai_agent import cli
from hw_ai_agent.agent import DEFAULT_SYSTEM_PROMPT
from hw_ai_agent.conversation import Conversation
from hw_ai_agent.journal import ChangeJournal
from hw_ai_agent.mcp import TRACKER
from hw_ai_agent.model_catalog import Model
from hw_ai_agent.pricing import Price, estimate_tokens
from hw_ai_agent.session import SessionStore, encoded
from hw_ai_agent.skills import SkillCatalog
from hw_ai_agent.tools import ToolRuntime


class ConversationCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.file = self.workspace / "value.py"
        self.file.write_text("value = 0\n")
        self.db = self.root / "state.db"
        self.env_file = self.root / ".env"
        self.responses = []
        self.requests = []
        self.schemas = []
        self.output_limits = []
        self.window = 20_000

        environment = patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        test = self

        class Client:
            billing_enabled = True

            def __init__(self, settings, *, max_tokens, **kwargs):
                self.settings, self.max_tokens = settings, max_tokens
                self.usage = []

            def complete(self, messages, tools):
                test.requests.append([(item["role"], item.get("content")) for item in messages])
                test.schemas.append({schema["function"]["name"] for schema in tools})
                test.output_limits.append(self.max_tokens)
                response = test.responses.pop(0)
                if isinstance(response, BaseException):
                    raise response
                if isinstance(response, tuple):
                    response, usage = response
                else:
                    usage = {"total_tokens": 10}
                self.usage.append(usage)
                return response

        for temporary_patch in (
            patch.object(cli, "OpenAICompatibleClient", Client),
            patch.object(cli, "catalog_for", side_effect=self.catalog),
            patch("sys.stdin.isatty", return_value=True),
        ):
            temporary_patch.start()
            self.addCleanup(temporary_patch.stop)

    def catalog(self, args, provider):
        model = Model("test/model", Price("0", "0"), context=self.window, tools=True)
        return SimpleNamespace(find=lambda name: model if name == model.id else None,
                               source="test", updated_at="test", status="test", warning=None)

    def invoke(self, action, *arguments, inputs=(), tty_output=False):
        out, err = io.StringIO(), io.StringIO()
        argv = ["hw-ai-agent", action, "--env-file", str(self.env_file), "--state-db", str(self.db), *map(str, arguments)]
        with patch.object(sys, "argv", argv), patch("builtins.input", side_effect=inputs), \
             patch.object(out, "isatty", return_value=tty_output), \
             redirect_stdout(out), redirect_stderr(err):
            status = cli.main()
        return status, out.getvalue(), err.getvalue()

    def start(self, *inputs):
        return self.invoke("chat", "--provider", "openrouter", "--model", "test/model", "--mode", "auto",
                           "--max-tokens", 20, "--workspace", self.workspace, inputs=inputs)

    def conversation_id(self):
        with SessionStore(self.db) as store:
            return store.db.execute("SELECT id FROM conversations").fetchone()["id"]

    @staticmethod
    def write(value, path="value.py"):
        return {"tool_calls": [tool_call("write", "write_file", {"path": path, "content": f"value = {value}\n"})]}

    def test_rollback_latest_turn_then_continue_with_history_and_totals(self):
        self.responses = [self.write(1), {"content": "Первое изменение."},
                          self.write(2), {"content": "Второе изменение."},
                          ({"content": "После отката."}, {})]
        code, out, err = self.start("Первая правка", "Вторая правка", "/changes", "/rollback",
                                    "Что осталось?", "/tokens", "/exit")
        self.assertEqual(code, 0, err)
        self.assertIn("+value = 2", out)
        self.assertIn("Запрос 2 отменён", out)
        self.assertIn("Токены за диалог · ≈", out)
        self.assertNotIn("Без отчёта провайдера", out)
        self.assertNotIn("расход известен частично", out)
        self.assertEqual(out.count("Окно модели · 20 000 токенов"), 1)
        self.assertLess(out.index("Окно модели"), out.index("Первое изменение."))
        self.assertIn("▶ Изменения · /changes", out)
        self.assertEqual(self.file.read_text(), "value = 1\n")
        self.assertIn(("assistant", "Первое изменение."), self.requests[-1])
        self.assertTrue(any(content and "отменил правки запросов 2" in content for _, content in self.requests[-1]))
        identifier = self.conversation_id()
        self.responses = [{"content": "Память сохранилась."}]
        code, out, err = self.invoke("chat", "--resume", identifier, inputs=("Напомни", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn(("assistant", "После отката."), self.requests[-1])
        self.assertIn("Диалог завершён", out)
        with SessionStore(self.db) as store:
            turns = Conversation(store, identifier).turns()
            self.assertEqual([turn.state.status for turn in turns], ["completed", "rolled_back", "completed", "completed"])
            self.assertEqual(Conversation.tokens(turns)[0], 50)
            self.assertEqual(len({turn.state.id for turn in turns}), 4)
            self.assertNotIn("test-key", store.db.execute("SELECT configuration FROM conversations").fetchone()[0])
        code, _, err = self.invoke("chat", "--resume", identifier)
        self.assertEqual(code, 1)
        self.assertIn("окончательно завершён", err)

    def test_ctrl_c_saves_request_and_resume_keeps_its_journal(self):
        self.responses = [self.write(1), KeyboardInterrupt()]
        code, out, err = self.start("Исправь", "/end")
        self.assertEqual(code, 130, err)
        self.assertIn("Диалог на паузе", out)
        identifier = self.conversation_id()
        self.responses = [{"content": "Готово."}, {"content": "После отката."}]
        code, out, err = self.invoke("chat", "--resume", identifier,
                                     inputs=("Новый запрос слишком рано", "/resume", "/rollback", "Продолжим", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn("Есть незавершённый запрос", out)
        self.assertIn("Токены за диалог · ≈", out)
        self.assertNotIn("расход известен частично", out)
        self.assertEqual(self.file.read_text(), "value = 0\n")
        self.assertNotIn(("user", "Новый запрос слишком рано"), self.requests[-1])

    def test_stats_and_jsonl_cover_approved_and_refused_followup(self):
        self.responses = [
            (self.write(1), {"prompt_tokens": 12, "completion_tokens": 4,
                             "prompt_tokens_details": {"cached_tokens": 8}}),
            ({"content": "Сохранено в value.py."}, {"prompt_tokens": 3}),
            (self.write(2), {}),
            ({"content": "Изменение отклонено."}, {"total_tokens": 7}),
        ]
        code, out, err = self.start("/stats", "/set mode safe", "Запиши результат.", "y",
                                    "Теперь замени его.", "n", "/stats", "/exit")
        self.assertEqual(code, 0, err)
        self.assertIn("Обращения к модели · 0", out)
        self.assertIn("вход 15 (частично, 2/4) · выход 4 (частично, 1/4) · всего 23 (частично, 2/4)", out)
        self.assertIn("вход unknown · выход unknown · всего 7 (частично, 1/2)", out)
        self.assertIn("Вызовы инструментов · 2 · уникальных 1 · write_file × 2", out)
        self.assertEqual(self.file.read_text(), "value = 1\n")
        self.assertIn(("assistant", "Сохранено в value.py."), self.requests[2])
        identifier = self.conversation_id()
        code, log, err = self.invoke("events", identifier)
        self.assertEqual(code, 0, err)
        events = [json.loads(line) for line in log.splitlines()]
        self.assertEqual({e["session_id"] for e in events}, {identifier})
        self.assertEqual(len(events), len({e["id"] for e in events}))
        self.assertTrue(all(datetime.fromisoformat(e["time"]).tzinfo for e in events))
        self.assertEqual([e["data"]["content"] for e in events if e["type"] == "user_request"],
                         ["Запиши результат.", "Теперь замени его."])
        runs = [e["run_id"] for e in events if e["type"] == "user_request"]
        self.assertNotEqual(*runs)
        self.assertIn("refused", [e["data"]["outcome"] for e in events if e["type"] == "policy_decision"])
        self.assertEqual([e["data"]["failed"] for e in events if e["type"] == "tool_result"], [False, True])
        self.assertEqual([json.loads(e["data"]["arguments"])["path"] for e in events if e["type"] == "tool_call"],
                         ["value.py", "value.py"])
        code, single, err = self.invoke("events", runs[0])
        self.assertEqual(code, 0, err)
        self.assertEqual([json.loads(line) for line in single.splitlines()],
                         [e for e in events if e["run_id"] == runs[0]])
        self.assertEqual(self.invoke("chat", "--resume", identifier, inputs=("/end",))[0], 0)
        self.assertEqual(self.invoke("events", identifier)[1], log)

    def test_skills_load_on_demand_and_are_counted_after_reopening_the_chat(self):
        self.responses = [
            {"tool_calls": [tool_call("open-skill", "load_skill", {"name": "open-task-export"})]},
            {"content": "Открытые задачи подготовлены."},
            {"tool_calls": [tool_call("deadline-skill", "load_skill", {"name": "deadline-report"})]},
            {"content": "Отчёт о дедлайнах подготовлен."},
            self.write(1), {"content": "Заметка записана."},
        ]
        code, _, err = self.start("Выгрузи открытые учебные задачи", "Отчёт на 2026-11-25",
                                  "Сохрани заметку", "/exit")
        self.assertEqual(code, 0, err)
        system_prompt = self.requests[0][0][1]
        for name, heading in (("open-task-export", "# Export open tasks"),
                              ("deadline-report", "# Report overdue tasks"),
                              ("review-python-tests", "# Review Python tests")):
            self.assertIn(name, system_prompt)
            self.assertNotIn(heading, system_prompt)
        self.assertIn("load_skill", self.schemas[0])
        self.assertTrue(self.requests[1][-1][1].startswith("---\nname: open-task-export\n"))
        self.assertTrue(self.requests[3][-1][1].startswith("---\nname: deadline-report\n"))
        self.assertFalse(any(role == "tool" for role, _ in self.requests[4]))

        identifier = self.conversation_id()
        code, out, err = self.invoke("chat", "--resume", identifier, inputs=("/stats", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn("обнаружено 3 · загрузок 0", out)
        self.assertIn("обнаружено 3 · загрузок 2 · уникальных скиллов 2 · deadline-report × 1, open-task-export × 1", out)
        code, log, err = self.invoke("events", identifier)
        self.assertEqual(code, 0, err)
        events = [json.loads(line) for line in log.splitlines()]
        self.assertEqual(sum(event["type"] == "skills_discovered" for event in events), 3)
        self.assertEqual([(event["data"]["name"], event["data"]["tool_call_id"])
                          for event in events if event["type"] == "skill_loaded"],
                         [("open-task-export", "open-skill"), ("deadline-report", "deadline-skill")])

    def test_no_skills_removes_prompt_catalog_and_tool_across_chat_resume(self):
        code, _, err = self.invoke("chat", "--provider", "openrouter", "--model", "test/model",
                                   "--workspace", self.workspace, "--no-skills", inputs=("/exit",))
        self.assertEqual(code, 0, err)
        identifier = self.conversation_id()
        self.responses = [{"content": "Готово."}]
        code, out, err = self.invoke("chat", "--resume", identifier, inputs=("Запрос", "/stats", "/end"))
        self.assertEqual(code, 0, err)
        self.assertEqual(self.requests[0][0], ("system", DEFAULT_SYSTEM_PROMPT))
        self.assertNotIn("load_skill", self.schemas[0])
        self.assertIn("обнаружено 0 · загрузок 0", out)
        code, log, err = self.invoke("events", identifier)
        self.assertEqual(code, 0, err)
        self.assertNotIn("skills_discovered", log)

    def test_resume_after_skill_load_reuses_saved_instructions_and_event(self):
        self.responses = [{"tool_calls": [tool_call("review", "load_skill", {"name": "review-python-tests"})]}]
        code, _, err = self.invoke("run", "--provider", "openrouter", "--model", "test/model", "--max-tokens", "20",
                                   "--workspace", self.workspace, "--stop-after-tools", "1", "Проверь Python-тесты")
        self.assertEqual(code, 130, err)
        with SessionStore(self.db) as store:
            identifier = store.list_sessions()[0]["id"]
        self.responses = [{"content": "Проверка завершена."}]
        code, _, err = self.invoke("resume", identifier)
        self.assertEqual(code, 0, err)
        self.assertTrue(self.requests[-1][-1][1].startswith("---\nname: review-python-tests\n"))
        with SessionStore(self.db) as store:
            events = [event["event"] for event in store.get(identifier).events]
        self.assertEqual(events.count("skills_discovered"), 1)
        self.assertEqual(events.count("skill_loaded"), 1)

    def mcp_options(self, *, action="allow", timeout=2):
        snapshot = self.root / "tasks.json"
        snapshot.write_text(json.dumps({"tasks": [{"id": "T1", "status": "open"}]}), encoding="utf-8")
        config = self.root / "mcp.json"
        config.write_text(json.dumps({"servers": {"tracker": {"snapshot": "tasks.json", "timeout": timeout,
                                                               "allow_tools": ["list_tasks"]}}}), encoding="utf-8")
        policy = self.root / "policy.json"
        policy.write_text(json.dumps({"rules": [{"tool": "mcp__tracker__list_tasks", "action": action}]}), encoding="utf-8")
        return ["--workspace", self.workspace, "--provider", "openrouter", "--model", "test/model",
                "--mcp-config", config, "--policy", policy]

    def test_chat_mcp_tools_reach_model_and_are_exported_and_counted(self):
        alias = "mcp__tracker__list_tasks"
        self.responses = [{"tool_calls": [tool_call("tracker-read", alias, {})]}, {"content": "Открыта T1."}]
        code, out, err = self.invoke("chat", *self.mcp_options(), inputs=("Что открыто в трекере?", "/stats", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn(alias, self.schemas[0])
        self.assertEqual(json.loads(self.requests[1][-1][1])["tasks"], [{"id": "T1", "status": "open"}])
        self.assertIn("MCP-вызовы · 1 · tracker/list_tasks × 1", out)
        code, log, err = self.invoke("events", self.conversation_id())
        self.assertEqual(code, 0, err)
        events = {event["type"]: event["data"] for event in map(json.loads, log.splitlines())}
        self.assertIn("list_tasks", events["mcp_discovered"]["names"])
        for key in ("tool_call", "tool_result", "policy_decision"):
            self.assertEqual((events[key]["server"], events[key]["mcp_tool"]), ("tracker", "list_tasks"))
            self.assertEqual(events[key]["tool_call_id"], "tracker-read")
        self.assertEqual(json.loads(events["tool_call"]["arguments"]), {})
        self.assertFalse(events["tool_result"]["failed"])

    def test_mcp_policy_denial_and_timeout_are_saved_as_failed_results(self):
        alias = "mcp__tracker__list_tasks"
        slow = self.root / "slow_tracker.py"
        source = TRACKER.read_text(encoding="utf-8").replace(
            "    try:\n        with os.fdopen", "    import time\n    time.sleep(2)\n    try:\n        with os.fdopen", 1)
        slow.write_text(source, encoding="utf-8")
        for tool_id, action, error in (("denied", "deny", "POLICY_DENIED"), ("timeout", "allow", "MCP_TIMEOUT")):
            with self.subTest(action=action):
                self.responses = [{"tool_calls": [tool_call(tool_id, alias, {})]}, {"content": "Данные недоступны."}]
                options = self.mcp_options(action=action, timeout=0.5)
                with patch("hw_ai_agent.mcp.TRACKER", slow):
                    code, _, err = self.invoke("run", *options, "Прочитай трекер")
                self.assertEqual(code, 0, err)
                self.assertIn(error, self.requests[-1][-1][1])
                with SessionStore(self.db) as store:
                    session = next(session for session in map(lambda row: store.get(row["id"]), store.list_sessions())
                                   if any(event.get("tool_call_id") == tool_id for event in session.events))
                    result = next(event for event in session.events if event["event"] == "tool_result")
                    decision = next(event for event in session.events if event["event"] == "policy_decision")
                self.assertTrue(result["failed"])
                self.assertIn(error, result["content"])
                self.assertEqual(result["server"], "tracker")
                self.assertEqual(decision["outcome"], "blocked" if action == "deny" else "permitted")
                self.assertEqual(alias in self.schemas[-2], action == "allow")

    def test_mcp_servers_are_namespaced_and_a_missing_snapshot_does_not_block_another_server(self):
        options = self.mcp_options()
        config, policy = self.root / "mcp.json", self.root / "policy.json"
        settings = json.loads(config.read_text())
        settings["servers"]["backup"] = {"snapshot": "missing.json", "allow_tools": ["list_tasks"]}
        config.write_text(json.dumps(settings))
        rules = json.loads(policy.read_text())
        rules["rules"].append({"tool": "mcp__backup__list_tasks", "action": "allow"})
        policy.write_text(json.dumps(rules))
        calls = [tool_call("backup", "mcp__backup__list_tasks", {}), tool_call("tracker", "mcp__tracker__list_tasks", {})]
        self.responses = [{"tool_calls": calls}, {"content": "T1 получена из основного трекера."}]
        code, _, err = self.invoke("run", *options, "--no-skills", "Сопоставь задачи двух трекеров")
        self.assertEqual(code, 0, err)
        self.assertTrue({"mcp__backup__list_tasks", "mcp__tracker__list_tasks"} <= self.schemas[0])
        self.assertNotIn("load_skill", self.schemas[0])
        self.assertIn("MCP_TOOL_ERROR", self.requests[1][-2][1])
        self.assertEqual(json.loads(self.requests[1][-1][1])["tasks"][0]["id"], "T1")

    def test_failed_model_request_can_resume_with_same_run_and_missing_usage(self):
        self.responses = [KeyboardInterrupt()]
        self.assertEqual(self.start("Запрос с test-key")[0], 130)
        identifier = self.conversation_id()
        _, before, _ = self.invoke("events", identifier)
        first = [json.loads(line) for line in before.splitlines()]
        run_id = first[0]["run_id"]
        self.assertNotIn("test-key", before)
        self.assertIn("run_error", [e["type"] for e in first])
        self.responses = [({"content": "Готово."}, {})]
        code, out, err = self.invoke("chat", "--resume", identifier, inputs=("/resume", "/stats", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("Обращения к модели · 2", out)
        self.assertIn("вход unknown · выход unknown · всего unknown", out)
        _, after, _ = self.invoke("events", identifier)
        self.assertTrue(after.startswith(before))
        self.assertEqual({json.loads(line)["run_id"] for line in after.splitlines()}, {run_id})

    def test_export_old_event_without_timestamp(self):
        self.responses = [{"content": "Готово."}]
        self.assertEqual(self.start("Запрос", "/exit")[0], 0)
        identifier = self.conversation_id()
        with SessionStore(self.db) as store:
            run_id = Conversation(store, identifier).turns()[0].state.id
            store.db.execute("UPDATE sessions SET events=? WHERE id=?",
                             (encoded([{"event": "final", "content": "старый ответ"}]), run_id))
        code, out, err = self.invoke("events", identifier)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), {"id": run_id + ":1", "time": None, "type": "final",
                                         "session_id": identifier, "run_id": run_id,
                                         "data": {"content": "старый ответ"}})

    def test_resume_chat_picker_filters_closed_chats_and_opens_selected_history(self):
        self.responses = [{"content": "Закрытый ответ"}, {"content": "Отчёт сохранён"},
                          {"content": "Список готов"}, {"content": "Продолжаю отчёт"}]
        self.assertEqual(self.start("Архивный запрос", "/end")[0], 0)
        self.assertEqual(self.start("Первый отчёт", "/exit")[0], 0)
        self.assertEqual(self.start("Другой список", "/exit")[0], 0)
        self.assertEqual(self.invoke("resume", "chat", inputs=("",))[0], 0)
        code, out, err = self.invoke("resume", "chat", inputs=("5", "2", "Уточнение", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("Введите номер из списка", out)
        self.assertLess(out.index("Другой список"), out.index("Первый отчёт"))
        self.assertNotIn("Архивный запрос", out)
        self.assertIn(("assistant", "Отчёт сохранён"), self.requests[-1])
        self.assertNotIn(("assistant", "Список готов"), self.requests[-1])

    def test_empty_picker_and_removed_stop_command(self):
        code, out, err = self.invoke("resume", "chat")
        self.assertEqual(code, 0, err)
        self.assertIn("Нет открытых диалогов", out)
        code, out, err = self.start("/help", "/stop", "/exit")
        self.assertEqual(code, 0, err)
        self.assertIn("Команда не распознана", out)
        self.assertNotIn("/stop", out)
        self.assertIn("/exit", out)
        self.assertEqual(self.requests, [])

    def test_progress_controls_only_display_and_journal_keeps_tool_details(self):
        self.file.write_text("payload-line test-key\n")
        call = {"tool_calls": [tool_call("r1", "read_file", {"path": "value.py"})]}
        for flags, full in [((), False), (("--progress", "full"), True), (("--no-progress",), False),
                            (("--verbose",), False)]:
            with self.subTest(flags=flags):
                self.responses = [(call, {"prompt_tokens": 20, "completion_tokens": 4}),
                                  ({"content": "Готово"}, {})]
                code, out, err = self.invoke("chat", "--model", "test/model", "--workspace", self.workspace,
                                             *flags, inputs=("Прочитай файл", "/exit"))
                self.assertEqual(code, 0, err)
                self.assertEqual("payload-line" in out, full)
                self.assertEqual('"path": "value.py"' in out, full)
                self.assertNotIn("test-key", out)
                if not flags:
                    self.assertRegex(out, r"\d{2}:\d{2}  Чтение · value.py")
                    self.assertNotIn("Инструмент · read_file", out)
                if full:
                    self.assertIn("Токены запроса 1 · вход 20 · выход 4", out)
                with SessionStore(self.db) as store:
                    identifier = store.db.execute("SELECT id FROM conversations ORDER BY rowid DESC LIMIT 1").fetchone()[0]
                    records = list(store.event_records(identifier))
                self.assertEqual({r["session_id"] for r in records}, {identifier})
                self.assertEqual(len({r["run_id"] for r in records}), 1)
                self.assertTrue(all(r["time"] and r["id"] for r in records))
                tools = [r["data"] for r in records if r["type"] in {"tool_call", "tool_result", "policy_decision"}]
                self.assertEqual({item["tool_call_id"] for item in tools}, {"r1"})
                self.assertEqual({item["call"] for item in tools}, {"1:0"})
                self.assertEqual(json.loads(tools[0]["arguments"]), {"path": "value.py"})
                self.assertIn("payload-line [REDACTED]", tools[-1]["content"])
                self.assertEqual([r["data"]["usage_status"] for r in records if r["type"] == "model_usage"],
                                 ["complete", "unknown"])

    def test_full_progress_saved_and_pending_request_resumed_via_picker(self):
        self.responses = [KeyboardInterrupt()]
        self.assertEqual(self.start("/set progress full", "Первый запрос")[0], 130)
        identifier = self.conversation_id()
        with SessionStore(self.db) as store:
            run_id = Conversation(store, identifier).turns()[0].state.id
        self.responses = [self.write(1), {"content": "Завершено"}]
        code, out, err = self.invoke("resume", "chat", inputs=("1", "/resume", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("Вызов write_file", out)
        self.assertIn("Результат write_file", out)
        with SessionStore(self.db) as store:
            self.assertEqual([t.state.id for t in Conversation(store, identifier).turns()], [run_id])

    def test_brief_groups_actual_file_calls_across_rounds_and_keeps_every_event(self):
        paths = ["a.py", "b.py", "a.py", "c.py", "d.py"]
        for path in paths:
            (self.workspace / path).write_text("original")
        reads = [tool_call(f"r{i}", "read_file", {"path": p}) for i, p in enumerate(paths)]
        writes = [tool_call(f"w{i}", "write_file", {"path": p, "content": "changed"}) for i, p in enumerate(paths[:2])]
        check = tool_call("tests", "Bash", {"command": "python -m unittest"})
        self.responses = [{"tool_calls": reads[:2]}, {"tool_calls": [*reads[2:], *writes, check]},
                          {"content": "Готово."}]
        with patch.object(ToolRuntime, "bash", return_value="exit_code=0\nRan 1 test in 0.001s\n\nOK"):
            code, out, err = self.invoke("chat", "--model", "test/model", "--workspace", self.workspace,
                                         "--mode", "auto", "--progress", "brief", inputs=("Проверь", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("Читаю a.py, b.py, c.py и ещё 1\n", out)
        self.assertIn("Изменяю a.py, b.py\n", out)
        self.assertIn("Запускаю тесты\n", out)
        self.assertNotIn("Запрос к модели", out)
        self.assertNotIn("changed", out)
        identifier = self.conversation_id()
        with SessionStore(self.db) as store:
            calls = [r for r in store.event_records(identifier) if r["type"] == "tool_call"]
            self.assertEqual(len(calls), len(reads) + len(writes) + 1)
        self.responses = [{"tool_calls": [self.write(2)["tool_calls"][0]]}, {"content": "Отклонено."}]
        code, out, err = self.invoke("resume", "chat", inputs=("1", "/set mode safe", "Уточнение", "n", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("Изменяю value.py\n", out)
        self.assertIn("Ошибка write_file", out)
        self.assertEqual(self.file.read_text(), "value = 0\n")

    def test_brief_finishes_partial_line_before_interruption_and_next_prompt(self):
        read = {"tool_calls": [tool_call("r", "read_file", {"path": "value.py"})]}
        self.responses = [read, KeyboardInterrupt()]
        code, out, err = self.start("/set progress brief", "Прочитай")
        self.assertEqual(code, 130, err)
        self.assertIn("Читаю value.py\n\nДиалог на паузе", out)
        self.responses = [{"content": "Готово."}, read, {"content": "Готово."}]
        code, out, err = self.invoke("resume", "chat", inputs=("1", "/resume", "Прочитай опять", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("Читаю value.py\n", out)

    def test_chat_settings_persist_and_resumed_turn_uses_its_checkpoint(self):
        self.responses = [self.write(1), KeyboardInterrupt()]
        self.assertEqual(self.start("Первая правка")[0], 130)
        identifier = self.conversation_id()
        self.responses = [{"content": "Продолжено."}, self.write(2), {"content": "Запись отклонена."}]
        code, out, err = self.invoke("chat", "--resume", identifier,
            inputs=("/set", "/set approvals invalid", "/set mode safe", "/set approvals deny",
                    "/set max-rounds 4", "/set max-tool-calls 1", "/set max-side-effects 1",
                    "/set max-tokens 50", "/settings", "/resume", "Вторая правка", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("Формат · /set", out)
        self.assertIn("допустимы ask, deny", out)
        self.assertIn("approvals · deny", out)
        self.assertIn("max-tool-calls · 1", out)
        self.assertIn("APPROVAL_REQUIRED", out)
        self.assertEqual(self.output_limits, [20, 20, 20, 50, 50])
        self.assertEqual(self.file.read_text(), "value = 1\n")
        self.responses = [self.write(3), {"content": "Записано с подтверждением."}]
        code, out, err = self.invoke("chat", "--resume", identifier,
                                     inputs=("/settings", "/set approvals ask", "Третья правка", "y", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn("mode · safe", out)
        self.assertIn("max-tokens · 50", out)
        self.assertEqual(self.file.read_text(), "value = 3\n")
        with SessionStore(self.db) as store:
            first, second, third = Conversation(store, identifier).turns()
            self.assertEqual(first.state.metadata["execution_mode"], "auto")
            self.assertEqual(second.state.metadata["approvals"], "deny")
            self.assertEqual(third.state.metadata["approvals"], "ask")
            self.assertEqual(store.budget(second.state.id)["limits"],
                             {"model_requests": 4, "tool_calls": 1, "side_effects": 1, "result_bytes": 1_000_000})

    def test_chat_progress_persists_and_applies_to_the_pending_request(self):
        self.responses = [self.write(1), KeyboardInterrupt()]
        code, out, err = self.start("/set progress true", "Первая правка")
        self.assertEqual(code, 130, err)
        self.assertIn("Файл записан · value.py", out)
        identifier = self.conversation_id()
        self.responses = [{"content": "Продолжено."}]
        code, out, err = self.invoke("chat", "--resume", identifier, inputs=("/settings", "/resume", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("progress · compact", out)
        self.assertRegex(out, r"\d{2}:\d{2}  Запрос к модели · 3")

        self.responses = [{"content": "Готово."}]
        code, out, err = self.invoke("chat", "--resume", identifier, "--no-progress",
                                     inputs=("/settings", "Ещё запрос", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("progress · off", out)
        self.assertIn("Агент работает…", out)
        self.assertNotIn("Запрос к модели", out)
        with SessionStore(self.db) as store:
            self.assertEqual(Conversation(store, identifier).configuration()["progress"], "compact")

    def test_chat_handles_surrogate_text_without_rewriting_invalid_tool_arguments(self):
        call = tool_call("broken-write", "write_file", {"path": "value.py", "content": "value = 1"})
        call["function"]["arguments"] = call["function"]["arguments"].replace("value = 1", "value = 1\udcd0")
        self.responses = [
            {"tool_calls": [call], "reasoning_details": [{"text": "Проверка\udcd0"}]},
            {"content": "Ответ\udcd0"},
            {"content": "Продолжение ✓"},
        ]
        code, out, err = self.start("Запрос\udcd0", "Продолжи", "/end")
        self.assertEqual(code, 0, err)
        self.assertIn(r"Ответ\udcd0", out)
        self.assertIn("Продолжение ✓", out)
        self.assertEqual(self.file.read_text(), "value = 0\n")
        self.assertIn(("user", r"Запрос\udcd0"), self.requests[0])
        self.assertIn(("assistant", r"Ответ\udcd0"), self.requests[-1])
        with SessionStore(self.db) as store:
            first, second = Conversation(store, self.conversation_id()).turns()
            self.assertEqual([first.state.status, second.state.status], ["completed", "completed"])
            rejected = store.calls(first.state.id)[0]
            self.assertEqual(rejected["state"], "failed")
            self.assertIn("valid UTF-8 text", rejected["result"]["content"])

    def test_old_chat_without_progress_uses_the_env_default(self):
        self.assertEqual(self.start("/exit")[0], 0)
        identifier = self.conversation_id()
        with SessionStore(self.db) as store:
            configuration = Conversation(store, identifier).configuration()
            del configuration["progress"]
            store.db.execute("UPDATE conversations SET configuration=? WHERE id=?", (encoded(configuration), identifier))
        self.env_file.write_text("AGENT_PROGRESS=true\n")
        self.responses = [{"content": "Готово."}]
        code, out, err = self.invoke("chat", "--resume", identifier, inputs=("/settings", "Запрос", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn("progress · compact", out)
        self.assertIn("Запрос к модели", out)

    def test_interrupted_rollback_must_finish_before_next_prompt(self):
        writes = self.write(1)["tool_calls"] + [
            tool_call("write_other", "write_file", {"path": "z.py", "content": "value = 1\n"})
        ]
        (self.workspace / "z.py").write_text("value = 0\n")
        self.responses = [{"tool_calls": writes}, {"content": "Два файла."}]
        original = ChangeJournal._rollback

        def interrupted(journal):
            def pause(_path):
                raise KeyboardInterrupt
            return original(journal, after_file=pause)

        with patch.object(ChangeJournal, "_rollback", interrupted):
            code, _, err = self.start("Измени оба файла", "/rollback")
        self.assertEqual(code, 130, err)
        self.assertEqual(self.file.read_text(), "value = 0\n")
        self.assertEqual((self.workspace / "z.py").read_text(), "value = 1\n")
        code, out, err = self.invoke("chat", "--resume", self.conversation_id(),
                                     inputs=("Новая задача", "/resume", "/end", "/rollback", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn("Сначала завершите прерванный откат", out)
        self.assertEqual((self.workspace / "z.py").read_text(), "value = 0\n")
        self.assertEqual(len(self.requests), 2)

    def test_context_warning_and_refusal_happen_before_model_request(self):
        prompt = "Первый вопрос"
        catalog = SkillCatalog(self.workspace)
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT + catalog.prompt()},
                    {"role": "user", "content": prompt}]
        self.window = estimate_tokens(messages, ToolRuntime(self.workspace, skills=catalog).tool_schemas) + 240
        self.responses = [{"content": "Ответ."}]
        oversized = "Очень большой запрос " * 1_000
        code, out, err = self.start(prompt, oversized, "/tokens", "/end")
        self.assertEqual(code, 0, err)
        self.assertIn("Контекст близок к пределу", err)
        self.assertIn("Запрос не отправлен", out)
        self.assertEqual(len(self.requests), 1)
        with SessionStore(self.db) as store:
            turns = Conversation(store, self.conversation_id()).turns()
            self.assertEqual(turns[-1].state.status, "context_limit")
            self.assertEqual(turns[-1].state.rounds, 0)

    def test_run_without_workspace_selects_a_numbered_folder(self):
        for name in ("wok", "fold"):
            (self.workspace / name).mkdir()
        (self.workspace / "outside").symlink_to(self.root, target_is_directory=True)
        self.responses = [{"content": "Готово."}]
        with chdir(self.root), patch.dict(os.environ, {"AGENT_WORKSPACE": str(self.root / "other")}):
            code, out, err = self.invoke("run", "--model", "test/model", "Проверь проект", inputs=("2",))
        self.assertEqual(code, 0, err)
        self.assertIn("1. fold", out)
        self.assertIn("2. wok", out)
        self.assertNotIn("outside", out)
        with SessionStore(self.db) as store:
            session = store.get(store.list_sessions()[0]["id"])
            self.assertEqual(Path(session.metadata["workspace"]), (self.workspace / "wok").resolve())

    def test_enter_creates_dated_folders_for_each_new_plan(self):
        with chdir(self.root), patch("hw_ai_agent.terminal.date") as clock:
            clock.today.return_value = date(2026, 11, 24)
            for name in ("2026-11-24", "2026-11-24-2"):
                self.responses = [{"tool_calls": [tool_call("plan", "submit_plan", PLAN)]}, {"content": "Готово."}]
                code, out, err = self.invoke("plan", "--model", "test/model", "Проверь проект", inputs=("", "y"))
                self.assertEqual(code, 0, err)
                self.assertIn(f"Создана папка · workspace/{name}", out)
                self.assertTrue((self.workspace / name).is_dir())
        with SessionStore(self.db) as store:
            self.assertEqual({Path(session["workspace"]).name for session in store.list_sessions()},
                             {"2026-11-24", "2026-11-24-2"})

    def test_enter_creates_short_workspace_for_new_chat(self):
        with chdir(self.root):
            code, out, err = self.invoke("chat", "--model", "test/model", inputs=("", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertRegex(out, r"Создана папка · workspace/[0-9a-f]{8}\b")
        name = out.split("Создана папка · workspace/", 1)[1].splitlines()[0]
        self.assertTrue((self.workspace / name).is_dir())

    def test_chat_can_create_a_named_workspace_and_reopen_it_without_menu(self):
        (self.workspace / "wok").mkdir()
        with chdir(self.root):
            code, out, err = self.invoke("chat", "--model", "test/model",
                                         inputs=("n", "../outside", "n", "wok", "n", "Новый проект", "/exit"))
            self.assertEqual(code, 0, err)
            self.assertIn("одно имя папки без пути", out)
            self.assertIn("Это имя уже занято", out)
            self.assertIn("Создана папка · workspace/Новый проект", out)
            code, out, err = self.invoke("chat", "--resume", self.conversation_id(), inputs=("/end",))
        self.assertEqual(code, 0, err)
        self.assertNotIn("Рабочая папка · workspace", out)
        self.assertIn("Новый проект", out)
        self.assertFalse((self.root / "outside").exists())

    def test_workspace_selection_can_be_cancelled_before_starting(self):
        with chdir(self.root):
            code, _, err = self.invoke("run", "--model", "test/model", "Проверь проект",
                                       inputs=(KeyboardInterrupt(),))
        self.assertEqual(code, 130)
        self.assertIn("Выбор рабочей папки отменён", err)
        self.assertFalse(self.db.exists())
        self.assertEqual(self.requests, [])

    def test_ollama_prints_only_final_text_in_run_and_cached_resume(self):
        self.responses = [{"content": "Черновик модели.\n\n</think>\n\nГотовый ответ."}]
        code, out, err = self.invoke("run", "--provider", "ollama", "--workspace", self.workspace, "Проверь")
        self.assertEqual(code, 0, err)
        self.assertIn("Готовый ответ.", out)
        self.assertNotIn("Черновик модели.", out)
        self.assertNotIn("</think>", out)
        with SessionStore(self.db) as store:
            identifier = store.list_sessions()[0]["id"]
        code, out, err = self.invoke("resume", identifier, "--verbose")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip().splitlines()[-1], "Готовый ответ.")
        self.assertNotIn("Черновик модели.", out)

    def test_ollama_verbose_chat_hides_tagged_blocks_and_preserves_code_examples(self):
        example = "Пример разметки.\n\n```xml\n<think>\n</think>\n```"
        self.responses = [{"content": "\n<think>Черновик.</think>\nИтог."},
                          {"content": example}, {"content": "<think>Незавершённый черновик."}]
        code, out, err = self.invoke("chat", "--provider", "ollama", "--workspace", self.workspace, "--verbose",
                                    inputs=("Задача", "Пример", "Ещё задача", "/exit"))
        self.assertEqual(code, 0, err)
        self.assertIn("Итог.", out)
        self.assertIn(example, out)
        self.assertIn("Модель не вернула итоговый текст.", out)
        self.assertNotIn("Черновик.", out)
        self.assertNotIn("Незавершённый черновик.", out)

    def test_single_prompt_output_is_compact_and_verbose_is_opt_in(self):
        options = ("--provider", "openrouter", "--model", "test/model", "--mode", "auto",
                   "--workspace", self.workspace, "--max-tokens", 20)
        self.responses = [self.write(1), {"content": "Готово."}, self.write(2), {"content": "Ещё готово."}]
        code, out, err = self.invoke("run", *options, "--progress", "Исправь")
        self.assertEqual(code, 0, err)
        self.assertIn(("user", "Исправь"), self.requests[0])
        self.assertIn("✓  value.py записан", out)
        self.assertIn("Тесты не запускались", out)
        self.assertNotIn("Вызов write_file", out)
        self.assertNotIn("Бюджет:", out)
        self.assertRegex(out, r"\d{2}:\d{2}  Запрос к модели")
        code, out, err = self.invoke("run", *options, "--verbose", "--progress", "full", "Ещё исправь")
        self.assertEqual(code, 0, err)
        self.assertIn("Вызов write_file", out)
        self.assertIn("Оценка запроса:", out)
        with SessionStore(self.db) as store:
            self.assertEqual(len(store.list_sessions()), 2)
            self.assertEqual(store.db.execute("SELECT count(*) FROM conversations").fetchone()[0], 0)


    def test_result_shows_verified_files_test_count_and_saved_diff(self):
        options = ("--provider", "openrouter", "--model", "test/model", "--mode", "auto", "--journal",
                   "--workspace", self.workspace, "--max-tokens", 20)
        self.env_file.write_text("AGENT_PROGRESS=true\n")
        check = {"tool_calls": [tool_call("tests", "Bash", {"command": "python3 -m unittest discover -v"})]}
        self.responses = [self.write(1), check, {"content": "Готово."}]
        output = "exit_code=0\nstdout:\n\nstderr:\nRan 1 test in 0.001s\n\nOK"
        with patch.object(ToolRuntime, "bash", return_value=output):
            code, out, err = self.invoke("run", *options, "Исправь и проверь")

        self.assertEqual(code, 0, err)
        self.assertIn("agi run", out)
        self.assertIn("OpenRouter · test/model", out)
        self.assertIn("Задача выполнена", out)
        self.assertIn("✓  value.py обновлён", out)
        self.assertIn("✓  Тесты прошли · 1 из 1", out)
        for line in ("Запрос к модели · 1", "Файл записан · value.py", "Проверка · python3", "Тесты прошли\n"):
            self.assertIn(line, out)
        self.assertIn("▶ Изменения · agi changes", out)
        self.assertIn("--state-db", out)
        with SessionStore(self.db) as store:
            session_id = store.list_sessions()[0]["id"]
        code, out, err = self.invoke("changes", session_id)
        self.assertEqual(code, 0, err)
        self.assertIn("+value = 1", out)

    def test_safe_approval_shows_color_before_asking_and_sanitizes_file_content(self):
        self.file.write_text("-- old\x1b[2J")
        self.responses = [self.write(1), {"content": "Готово."}]
        options = ("--model", "test/model", "--mode", "safe", "--journal", "--workspace", self.workspace,
                   "--max-tokens", 20, "Исправь")

        def approve(prompt):
            self.assertIn("[y/N]", prompt)
            output = sys.stdout.getvalue()
            self.assertIn("\x1b[31m--- old\\x1b[2J\x1b[0m", output)
            self.assertIn("\x1b[32m+value = 1\x1b[0m", output)
            self.assertIn("\\ No newline at end of file", output)
            self.assertNotIn("\x1b[2J", output)
            return "y"

        with patch.dict(os.environ, {"TERM": "xterm"}):
            code, _, err = self.invoke("run", *options, inputs=approve, tty_output=True)
            self.assertEqual(code, 0, err)
            self.assertEqual(self.file.read_text(), "value = 1\n")
            with SessionStore(self.db) as store:
                identifier = store.list_sessions()[0]["id"]
            code, out, err = self.invoke("changes", identifier, tty_output=True)
            self.assertEqual(code, 0, err)
            self.assertIn("\x1b[32m+value = 1\x1b[0m", out)
            with patch.dict(os.environ, {"NO_COLOR": "1"}):
                code, out, err = self.invoke("changes", identifier, tty_output=True)
            self.assertEqual(code, 0, err)
            self.assertNotIn("\x1b[", out)
            self.assertIn("+value = 1", out)

    def test_failed_tests_do_not_get_a_success_label(self):
        options = ("--provider", "openrouter", "--model", "test/model", "--mode", "auto",
                   "--workspace", self.workspace, "--max-tokens", 20)
        check = {"tool_calls": [tool_call("tests", "Bash", {"command": "python -m unittest discover"})]}
        self.responses = [self.write(1), check, {"content": "Прочтите результат проверки."}]
        output = "exit_code=1\nstdout:\n\nstderr:\nRan 1 test in 0.001s\n\nFAILED (failures=1)"
        with patch.object(ToolRuntime, "bash", return_value=output):
            code, out, err = self.invoke("run", *options, "Исправь и проверь")
        self.assertEqual(code, 0, err)
        self.assertIn("Успех тестов не подтверждён", out)
        self.assertNotIn("Задача выполнена", out)
        self.assertNotIn("▶ Изменения", out)


if __name__ == "__main__":
    unittest.main()
