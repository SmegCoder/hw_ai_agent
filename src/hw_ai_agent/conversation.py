from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass

from .agent import AgentError
from .budget import BudgetExceeded, BudgetLimits
from .client import ModelClientError
from .config import option_value
from .journal import ChangeJournal, JournalError
from .policy import ApprovalDeferred, PolicyError
from .session import SessionError, SessionState, encoded
from .terminal import color_diff, report_header, visible, visible_line
from .tools import ToolError

FINISHED = {"completed", "aborted", "rolled_back", "budget_exhausted", "context_limit"}

CHAT_OPTIONS = {
    "mode": ("binding", "execution_mode", "safe|auto"),
    "approvals": ("binding", "approvals", "ask|deny"),
    "max-rounds": ("budget", "model_requests", "N"),
    "max-tool-calls": ("budget", "tool_calls", "N"),
    "max-side-effects": ("budget", "side_effects", "N"),
    "max-tokens": ("binding", "max_tokens", "N"),
    "progress": (None, "progress", "brief|compact|full|off"),
}

COMMANDS = """Введите запрос или команду.
  /plan ТЕКСТ    Запрос с утверждением плана перед правками
  /changes       Показать правки последнего запроса с изменениями
  /rollback      Отменить правки последнего запроса и остаться в диалоге
  /tokens        Токены всех обращений к модели в диалоге
  /stats         Статистика каждого запроса и всего диалога
  /settings      Показать настройки диалога
  /set ИМЯ ЗНАЧЕНИЕ   Сохранить настройку диалога
  /resume        Продолжить приостановленный запрос
  /exit          Выйти с возможностью продолжить диалог
  /end           Окончательно завершить диалог
  /help          Показать команды"""


@dataclass(frozen=True)
class Turn:
    number: int
    prompt: str
    state: SessionState
    journal_status: str | None


class RunStats:
    def __init__(self):
        self.requests = 0
        self.tools = Counter()
        self.mcp_calls = Counter()
        self.discovered_skills = set()
        self.loaded_skills = Counter()
        self.known = Counter()
        self.tokens = Counter()

    def include(self, state: SessionState):
        self.requests += state.rounds
        usage_by_round = {}
        legacy_usage = []
        for event in state.events:
            if event["event"] == "tool_call":
                self.tools[event["name"]] += 1
                if "server" in event and "mcp_tool" in event:
                    self.mcp_calls[event["server"], event["mcp_tool"]] += 1
            elif event["event"] == "skills_discovered":
                self.discovered_skills.update(event["names"])
            elif event["event"] == "skill_loaded":
                self.loaded_skills[event["name"]] += 1
            elif event["event"] in {"model_cost", "model_usage"} and event.get("usage"):
                if "round" in event:
                    # Both events contain the API counters. A crash can leave only model_cost.
                    usage_by_round[event["round"]] = event["usage"]
                elif event["event"] == "model_usage":
                    legacy_usage.append(event["usage"])
        for usage in [*usage_by_round.values(), *legacy_usage]:
            values = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                      if type(usage.get(key)) is int and usage[key] >= 0}
            if "total_tokens" not in values and {"prompt_tokens", "completion_tokens"} <= values.keys():
                values["total_tokens"] = values["prompt_tokens"] + values["completion_tokens"]
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                values.update({key: details[key] for key in ("cached_tokens", "cache_write_tokens")
                               if type(details.get(key)) is int and details[key] >= 0})
            self.known.update(values.keys())
            self.tokens.update(values)

    def token_count(self, key):
        if self.requests == 0:
            return "0"
        count = self.known[key]
        if count == 0:
            return "unknown"
        if count < self.requests:
            return f"{self.tokens[key]} (частично, {count}/{self.requests})"
        return str(self.tokens[key])

    def display(self):
        print(f"  Обращения к модели · {self.requests}")
        print(f"  Токены API · вход {self.token_count('prompt_tokens')} · выход {self.token_count('completion_tokens')}"
              f" · всего {self.token_count('total_tokens')}")
        tools = ", ".join(f"{visible(name)} × {count}" for name, count in sorted(self.tools.items()))
        print(f"  Вызовы инструментов · {self.tools.total()} · уникальных {len(self.tools)}"
              + (f" · {tools}" if tools else ""))
        skills = ", ".join(f"{visible(name)} × {count}" for name, count in sorted(self.loaded_skills.items()))
        print(f"  Скиллы · обнаружено {len(self.discovered_skills)} · загрузок {self.loaded_skills.total()}"
              + f" · уникальных скиллов {len(self.loaded_skills)}" + (f" · {skills}" if skills else ""))
        if self.mcp_calls:
            calls = ", ".join(f"{visible(server)}/{visible(tool)} × {count}"
                              for (server, tool), count in sorted(self.mcp_calls.items()))
            print(f"  MCP-вызовы · {self.mcp_calls.total()} · {calls}")


class Conversation:
    def __init__(self, store, identifier: str):
        self.store = store
        self.id = identifier
        store.validate_id(identifier)

    @staticmethod
    def choose_open(store):
        rows = store.db.execute("""
            SELECT c.id, json_extract(c.configuration, '$.binding.workspace') AS workspace,
                   t.prompt, s.updated_at
            FROM conversations AS c
            LEFT JOIN conversation_turns AS t ON t.session_id=(
                SELECT session_id FROM conversation_turns
                WHERE conversation_id=c.id ORDER BY number DESC LIMIT 1)
            LEFT JOIN sessions AS s ON s.id=t.session_id
            WHERE c.ended=0 ORDER BY s.updated_at DESC, c.rowid DESC
        """).fetchall()
        if not rows:
            print("Нет открытых диалогов. Начать новый · agi chat")
            return None
        print("Выберите диалог · время UTC, рабочая папка и последний запрос")
        for number, row in enumerate(rows, 1):
            title = visible_line(row["prompt"] or "Пока без запросов")
            updated = (row["updated_at"] or "—")[:16].replace("T", " ")
            print(f"{number}. {updated} · {visible_line(row['workspace'])} · {title[:90]}")
        while True:
            try:
                value = input("Номер диалога (Enter — отмена): ").strip()
            except (KeyboardInterrupt, EOFError):
                return None
            if not value:
                return None
            if value.isascii() and value.isdecimal() and len(value) <= len(str(len(rows))):
                index = int(value)
                if 1 <= index <= len(rows):
                    return rows[index - 1]["id"]
            print("Введите номер из списка.")

    @classmethod
    def create(cls, store, configuration: dict) -> Conversation:
        identifier = uuid.uuid4().hex
        with store.transaction():
            store.db.execute("INSERT INTO conversations (id, configuration) VALUES (?, ?)",
                             (identifier, encoded(configuration)))
        return cls(store, identifier)

    def configuration(self) -> dict:
        row = self.store.db.execute("SELECT configuration, ended FROM conversations WHERE id=?", (self.id,)).fetchone()
        if row is None:
            raise SessionError("Диалог не найден")
        if row["ended"]:
            raise SessionError("Диалог окончательно завершён")
        try:
            configuration = json.loads(row["configuration"])
            if not isinstance(configuration["binding"], dict) or not isinstance(configuration["budget"], dict):
                raise ValueError
            return configuration
        except (KeyError, TypeError, ValueError):
            raise SessionError("Не удалось прочитать настройки диалога") from None

    def end(self) -> None:
        with self.store.transaction():
            self.store.db.execute("UPDATE conversations SET ended=1 WHERE id=?", (self.id,))

    def set_option(self, name: str, value: str):
        group, key, _ = CHAT_OPTIONS[name]
        value = option_value(name.replace("-", "_"), value)
        with self.store.transaction():
            configuration = self.configuration()
            options = configuration if group is None else configuration[group]
            options[key] = value
            self.store.db.execute("UPDATE conversations SET configuration=? WHERE id=?",
                                  (encoded(configuration), self.id))
        return value

    def turns(self) -> list[Turn]:
        rows = self.store.db.execute(
            """SELECT number, prompt, conversation_turns.session_id, journals.status AS journal_status
               FROM conversation_turns LEFT JOIN journals USING (session_id)
               WHERE conversation_id=? ORDER BY number""",
            (self.id,),
        )
        return [
            Turn(row["number"], row["prompt"], self.store.get(row["session_id"]), row["journal_status"])
            for row in rows
        ]

    @staticmethod
    def pending(turns: list[Turn]) -> Turn | None:
        return next((turn for turn in reversed(turns)
                     if turn.journal_status == "rollback_pending" or turn.state.status not in FINISHED), None)

    @staticmethod
    def history(turns: list[Turn]) -> list[dict]:
        messages = []
        reverted = []
        for turn in turns:
            state = turn.state
            if state.final_answer:
                messages.extend(({"role": "user", "content": turn.prompt},
                                 {"role": "assistant", "content": state.final_answer}))
            elif state.status != "rolled_back" and any(
                e["event"] == "journal_action_completed" and e["paths"] for e in state.events
            ):
                messages.extend(({"role": "user", "content": turn.prompt},
                                 {"role": "assistant", "content": "Запрос прерван после правок. Проверю текущее состояние файлов."}))
            if state.status == "rolled_back":
                reverted.append(str(turn.number))
        if reverted:
            messages.append({
                "role": "user",
                "content": f"Я отменил правки запросов {', '.join(reverted)}. Учитывай, что эти правки больше не действуют, "
                           "и проверь текущее состояние файлов перед изменениями.",
            })
        return messages

    @staticmethod
    def tokens(turns: list[Turn]) -> tuple[int, int, int]:
        reported = estimated = incomplete = 0
        for turn in turns:
            events = turn.state.events
            requests = {e["round"]: e for e in events if e["event"] == "model_request"}
            responses = {e["round"]: e for e in events if e["event"] == "model_usage" and "round" in e}
            for number, request in requests.items():
                response = responses.get(number, {})
                usage = response.get("usage", {})
                if "total_tokens" in usage:
                    reported += usage["total_tokens"]
                    continue
                for field, fallback in (("prompt_tokens", request.get("input_tokens_estimate")),
                                        ("completion_tokens", response.get("output_tokens_estimate"))):
                    if field in usage:
                        reported += usage[field]
                    elif fallback is not None:
                        estimated += fallback
                    else:
                        incomplete += 1
        return reported, estimated, incomplete


class ConversationShell:
    def __init__(self, conversation, agent, *, restore_runtime, report_answer, resume_command, approve_action, progress):
        self.conversation = conversation
        self.agent = agent
        self.store = conversation.store
        self.restore_runtime = restore_runtime
        self.report_answer = report_answer
        self.resume_command = resume_command
        self.approve_action = approve_action
        self.progress = progress
        self.default_plan = agent.plan_mode

    def run(self, first_prompt: str = "") -> int:
        report_header("chat", self.agent.runtime.root, self.agent.client.settings)
        window = getattr(self.agent.client, "context_window", None)
        if window:
            print(f"Окно модели · {window:,} токенов".replace(",", " "))
        print(f"Диалог · {self.conversation.id}")
        print("/help — команды · /settings — настройки · Ctrl+C — пауза")
        try:
            if first_prompt:
                self.execute(lambda: self.prompt(first_prompt))
            while True:
                line = input("\nВы › ").strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                command, argument = parts[0], parts[1] if len(parts) == 2 else ""
                if command in {"/pause", "/exit"} and not argument:
                    self.pause()
                    return 0
                if command == "/end" and not argument:
                    if any(turn.journal_status == "rollback_pending" for turn in self.conversation.turns()):
                        print("Сначала завершите прерванный откат через /rollback.")
                        continue
                    self.conversation.end()
                    self.show_tokens()
                    print("Диалог завершён.")
                    return 0
                self.execute(lambda: self.dispatch(command, argument, line))
        except (EOFError, KeyboardInterrupt) as exc:
            self.pause()
            return 130 if isinstance(exc, KeyboardInterrupt) else 0

    def execute(self, action):
        try:
            with self.progress:
                action()
        except ApprovalDeferred:
            print("Запрос на паузе · продолжить через /resume")
        except ModelClientError as exc:
            print(f"Ошибка · {visible(exc)}")
            if self.agent.session_id:
                state = self.store.get(self.agent.session_id)
                if state.rounds < state.max_rounds:
                    print("Продолжить запрос · /resume")
                else:
                    print("Лимит обращений исчерпан · /rollback отменит запрос")
        except (AgentError, BudgetExceeded, JournalError, PolicyError,
                SessionError, ToolError, OSError, ValueError) as exc:
            print(f"Ошибка · {visible(exc)}")

    def dispatch(self, command, argument, line):
        if not command.startswith("/"):
            self.prompt(line)
        elif command == "/plan" and argument.strip():
            self.prompt(argument.strip(), plan=True)
        elif command == "/resume" and not argument:
            self.resume_turn()
        elif command in {"/changes", "/rollback"} and not argument:
            self.changes(rollback=command == "/rollback")
        elif command == "/tokens" and not argument:
            self.show_tokens()
        elif command == "/stats" and not argument:
            self.show_stats()
        elif command == "/settings" and not argument:
            self.show_settings()
        elif command == "/set":
            parts = argument.split()
            if len(parts) != 2 or parts[0] not in CHAT_OPTIONS:
                print("Формат · /set ИМЯ ЗНАЧЕНИЕ · /settings — список")
                return
            name, value = parts
            saved = self.conversation.set_option(name, value)
            if name == "progress":
                self.progress.mode = saved
                print(f"Ход работы · {saved}")
            else:
                print(f"Для новых запросов · {name} = {visible(value)}")
        elif command == "/help" and not argument:
            print(COMMANDS)
        else:
            print("Команда не распознана · /help")

    def show_settings(self):
        configuration = self.conversation.configuration()
        print("Настройки новых запросов")
        for name, (group, key, allowed) in CHAT_OPTIONS.items():
            value = self.progress.mode if name == "progress" else configuration[group][key]
            print(f"  {name} · {visible(value)}   /set {name} {allowed}")

    def apply_turn_binding(self, binding):
        approvals = binding.get("approvals", "ask")
        self.agent.runtime.mode = binding.get("execution_mode", "safe")
        self.agent.runtime.approve = self.approve_action if approvals == "ask" else None
        self.agent.client.max_tokens = binding["max_tokens"]
        self.agent.session_metadata.update(approvals=approvals, max_tokens=binding["max_tokens"])

    def prompt(self, prompt, *, plan=False):
        turns = self.conversation.turns()
        if self.conversation.pending(turns):
            raise SessionError("Есть незавершённый запрос. Введите /resume или /rollback")
        configuration = self.conversation.configuration()
        self.apply_turn_binding(configuration["binding"])
        self.agent.budget_limits = BudgetLimits.restore(configuration["budget"])
        self.agent.max_rounds = self.agent.budget_limits.model_requests
        self.agent.plan_mode = plan or self.agent.task_mode or self.default_plan
        self.finish_turn(self.agent.run(prompt, history=self.conversation.history(turns), conversation_id=self.conversation.id))

    def resume_turn(self):
        turn = self.conversation.pending(self.conversation.turns())
        if turn is None:
            print("Нет приостановленного запроса.")
            return
        if turn.journal_status == "rollback_pending":
            raise JournalError("Откат прерван. Введите /rollback для продолжения")
        try:
            self.apply_turn_binding(turn.state.metadata)
            self.finish_turn(self.agent.resume(turn.state.id))
        except SessionError:
            print(f"Запрос {turn.number} · {turn.state.id}")
            raise

    def finish_turn(self, answer):
        self.report_answer(self.agent, answer)
        self.show_tokens()

    def show_tokens(self):
        turns = self.conversation.turns()
        reported, estimated, incomplete = self.conversation.tokens(turns)
        prefix = "≈" if estimated or incomplete else ""
        print(f"Токены за диалог · {prefix}{reported + estimated:,}".replace(",", " "))

    def show_stats(self):
        total = RunStats()
        for turn in self.conversation.turns():
            run = RunStats()
            run.include(turn.state)
            total.include(turn.state)
            print(f"Запрос {turn.number} · {turn.state.id} · {turn.state.status}")
            run.display()
        print(f"Итого за диалог · {self.conversation.id}")
        total.display()

    def pause(self):
        print(f"\nДиалог на паузе · {self.resume_command}")

    def changes(self, *, rollback=False):
        turns = self.conversation.turns()
        pending = self.conversation.pending(turns)
        rollable = [pending] if pending and rollback else [
            turn for turn in reversed(turns) if turn.state.status != "rolled_back"
        ]
        for turn in rollable:
            identifier = turn.state.id
            with self.store.lock(identifier):
                journal_row = self.store.db.execute("SELECT status FROM journals WHERE session_id=?", (identifier,)).fetchone()
                if journal_row is None:
                    if rollback and turn == pending and not self.store.calls(identifier):
                        self.store.set_status(identifier, "rolled_back")
                        print(f"Запрос {turn.number} отменён.")
                        return
                    continue
                runtime = self.restore_runtime(turn.state, allow_policy_change=rollback)
                journal = ChangeJournal(self.store, identifier, runtime)
                diff = journal.diff()
                if not diff and turn != pending and journal_row["status"] != "rollback_pending":
                    continue
                if not rollback:
                    print(f"Запрос {turn.number} · {color_diff(diff) if diff else 'сохранённых правок пока нет'}")
                    if journal_row["status"] == "rollback_pending":
                        print("Откат прерван. Введите /rollback для продолжения.")
                    return
                result = journal._rollback()
                print(f"Запрос {turn.number} отменён · файлов в откате: {len(result['paths'])}")
                return
        print("Нет сохранённых изменений.")
