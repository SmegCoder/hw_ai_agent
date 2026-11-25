from __future__ import annotations

import difflib
import json
import os
import re
import shlex
import shutil
import sys
from time import monotonic
from datetime import date
from itertools import count
from pathlib import Path

from .planning import PlanApprovalDeferred, PlanError
from .pricing import amount, usd
from .tasks import passing_tests, test_command
from .session import SessionError
from .tools import ToolError


def visible(value: str) -> str:
    return ''.join(c if c.isprintable() or c in '\n\t' else repr(c)[1:-1] for c in str(value))


def emphasis(text: str, style: str = "1") -> str:
    if sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ:
        return f"\x1b[{style}m{text}\x1b[0m"
    return text


def visible_line(value: str) -> str:
    return visible(value).replace("\n", "\\n").replace("\t", "\\t")


DIFF_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


def color_diff(diff: str) -> str:
    lines = []
    remaining = 0
    for line in visible(diff).splitlines(keepends=True):
        hunk = DIFF_HUNK.match(line)
        if hunk:
            remaining = sum(int(count or "1") for count in hunk.groups())
        elif remaining and line.startswith(" "):
            remaining -= 2
        elif remaining and line.startswith(("+", "-")):
            remaining -= 1
            ending = "\n" if line.endswith("\n") else ""
            line = emphasis(line.removesuffix("\n"), "32" if line.startswith("+") else "31") + ending
        lines.append(line)
    return "".join(lines)


class ProgressDisplay:
    FILE_ACTIONS = {"read_file": "Читаю", "write_file": "Изменяю", "delete_file": "Удаляю", "list_files": "Просматриваю"}
    TASK_ACTIONS = {"submit_plan": "Предлагаю план", "complete_task": "Завершаю задачу", "fail_task": "Отмечаю ошибку задачи"}

    def __init__(self, mode: str = "compact"):
        self.mode = mode
        self.started_at = monotonic()
        self.target = ""
        self.is_test = False
        self.brief_action = ""
        self.brief_paths: set[str] = set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish_line()

    def finish_line(self):
        if self.brief_action:
            extra = len(self.brief_paths) - 3
            print(f" и ещё {extra}" if extra > 0 else "", flush=True)
            self.brief_paths.clear()
            self.brief_action = ""

    def brief_file(self, action: str, path: str):
        if self.brief_action != action:
            self.finish_line()
            self.brief_action = action
        if path in self.brief_paths:
            return
        self.brief_paths.add(path)
        label = path if len(path) <= 40 else "…" + path[-39:]
        if len(self.brief_paths) == 1:
            self.line(f"{action} {label}", end="")
        elif len(self.brief_paths) <= 3:
            print(f", {label}", end="", flush=True)

    def render(self, event: dict) -> bool:
        """Render a progress line when this event has useful information."""
        kind = event["event"]
        brief = self.mode == "brief"
        if brief and (kind in {"final", "context_warning", "model_retry", "task_completed",
                               "session_created", "session_resumed", "session_cached"}
                      or (kind == "policy_decision" and event["outcome"] == "waiting")):
            self.finish_line()
        if kind in {"session_created", "session_resumed"}:
            self.started_at = monotonic()
            self.target, self.is_test = "", False
        elif kind == "model_request":
            if not brief:
                self.line(f"Запрос к модели · {event['round']}")
            elif event["round"] == 1:
                self.line("Думаю…")
            return True
        elif kind == "model_usage" and self.mode == "full":
            usage = event["usage"]
            self.line(f"Токены запроса {event['round']} · вход {usage.get('prompt_tokens', 'unknown')}"
                      f" · выход {usage.get('completion_tokens', 'unknown')}")
            return True
        elif kind == "tool_call":
            if self.mode == "full":
                self.line(f"Вызов {visible_line(event['name'])} · {visible(event['arguments'])}")
            else:
                self.tool_start(event)
            return True
        elif kind == "tool_result":
            if self.mode == "full":
                self.line(f"Результат {visible_line(event['name'])} · {visible(event['content'])}")
            elif event.get("failed", event["content"].startswith("ERROR:")):
                self.finish_line()
                self.line(f"Ошибка {visible_line(event['name'])} · {visible_line(event['content'][:240])}")
            elif not brief or event["name"] in {"Bash", "bash"}:
                self.tool_result(event)
            return True
        return False

    def tool_start(self, event: dict) -> None:
        self.target, self.is_test = "", False
        name = event["name"]
        try:
            arguments = json.loads(event["arguments"])
        except (ValueError, TypeError):
            return
        if not isinstance(arguments, dict):
            return
        target_key = {"Bash": "command", "bash": "command", "load_skill": "name"}.get(name, "path")
        target = arguments.get(target_key, ".")
        if not isinstance(target, str):
            return
        self.target = visible_line(target)
        brief = self.mode == "brief"
        if brief:
            if name in self.FILE_ACTIONS:
                self.brief_file(self.FILE_ACTIONS[name], self.target)
                return
            self.finish_line()
        if len(self.target) > 120:
            self.target = self.target[:119] + "…"
        if name in {"Bash", "bash"}:
            try:
                test_command(target)
                self.is_test = True
            except PlanError:
                pass
            if brief:
                self.line("Запускаю тесты" if self.is_test else "Выполняю команду")
            else:
                self.line(f"{'Проверка' if self.is_test else 'Команда'} · {self.target}")
        elif name in {"read_file", "list_files"}:
            self.line(f"{'Чтение' if name == 'read_file' else 'Список файлов'} · {self.target}")
        elif name == "load_skill":
            self.line(f"Загрузка скилла · {self.target}")
        elif name not in {"write_file", "delete_file"}:
            label = f"Инструмент · {visible_line(name)}"
            self.line(self.TASK_ACTIONS.get(name, label) if brief else label)

    def tool_result(self, event: dict) -> None:
        name, content = event["name"], event["content"]
        if name == "write_file" and content.startswith("wrote "):
            self.line(f"Файл записан · {self.target}")
        elif name == "delete_file" and content.startswith(("removed ", "already absent:")):
            label = "Файл уже отсутствовал" if content.startswith("already absent:") else "Файл удалён"
            self.line(f"{label} · {self.target}")
        elif name in {"Bash", "bash"}:
            if self.is_test and passing_tests(content):
                self.line("Тесты прошли")
            elif self.is_test and re.search(r"^Ran 0 tests? in ", content, re.MULTILINE):
                self.line("Тесты не найдены")
            else:
                status = re.match(r"exit_code=(-?\d+)\n", content)
                if status:
                    label = "Успех тестов не подтверждён" if self.is_test else "Команда завершена"
                    self.line(f"{label} · код {status[1]}")

    def line(self, text: str, *, end="\n") -> None:
        minutes, seconds = divmod(int(monotonic() - self.started_at), 60)
        print(f"{emphasis(f'{minutes:02}:{seconds:02}', '2')}  {text}", end=end, flush=True)


def report_header(action, workspace: Path, settings, request: str | None = None) -> None:
    command = f"agi {action}"
    if request:
        preview = visible_line(request)
        if len(preview) > 100:
            preview = preview[:97] + "…"
        command += " " + shlex.quote(preview)
    cwd = Path.cwd()
    if workspace == cwd:
        path = "."
    elif workspace.is_relative_to(cwd):
        path = "./" + workspace.relative_to(cwd).as_posix()
    else:
        path = str(workspace)
    context = path
    if settings is not None:
        provider = {"openrouter": "OpenRouter", "openai": "OpenAI", "ollama": "Ollama"}.get(settings.provider, settings.provider)
        context += f" · {provider} · {settings.model}"
    print(emphasis(command, "1;36"))
    print(emphasis(visible_line(context), "2"))


def result_details(agent):
    files, tests = {}, None
    for item in agent.store.calls(agent.session_id):
        function = item["call"]["function"]
        name = function["name"]
        if name not in {"write_file", "delete_file", "Bash", "bash"} or not item["result"]:
            continue
        try:
            arguments = json.loads(function["arguments"])
        except ValueError:
            continue
        if not isinstance(arguments, dict):
            continue
        if name in {"write_file", "delete_file"} and item["state"] == "completed":
            path = arguments.get("path")
            if isinstance(path, str):
                files[path] = "записан" if name == "write_file" else "удалён"
        elif name in {"Bash", "bash"}:
            try:
                test_command(arguments.get("command"))
            except PlanError:
                continue
            tests = item["result"]["content"]
    if agent.journal:
        files = {}
        for path, entry in agent.journal.entries().items():
            before, after = entry["before"], entry["after"]
            if before != after:
                files[path] = "удалён" if after is None else "создан" if before is None else "обновлён"
    return files, tests


def report_answer(agent, answer: str, *, verbose: bool, chat: bool = False) -> None:
    if verbose:
        print("\n" + visible(answer))
        return
    files, tests = result_details(agent)
    passed = passing_tests(tests) if tests is not None else False
    count = re.search(r"^Ran ([1-9][0-9]*) tests? in ", tests, re.MULTILINE) if passed else None
    all_passed = count is not None and tests.rstrip().endswith("\nOK")
    if sys.stdout.isatty():
        width = min(96, shutil.get_terminal_size().columns)
        print("\n" + emphasis("─" * width, "2"))
    print(emphasis("Задача выполнена" if files and all_passed else "Результат", "1;36"))
    check = emphasis("✓", "32")
    for path, operation in list(files.items())[:5]:
        print(f"  {check}  {visible_line(path)} {operation}")
    if len(files) > 5:
        print(f"  …  Ещё файлов {len(files) - 5}")
    if all_passed:
        print(f"  {check}  Тесты прошли · {count[1]} из {count[1]}")
    elif passed:
        details = f" · запущено {count[1]}" if count else ""
        print(f"  {check}  unittest завершился успешно{details}")
    elif tests is not None and re.search(r"^Ran 0 tests? in ", tests, re.MULTILINE):
        print("  ·  Тесты не найдены")
    elif tests is not None:
        print(f"  {emphasis('!', '33')}  Успех тестов не подтверждён")
    elif files:
        print("  ·  Тесты не запускались")

    cost = agent.store.cost(agent.session_id)
    unknown = cost["unknown_requests"]
    known = unknown < cost["requests"]
    value = "$0.00" if amount(cost["known_usd"]) == 0 else usd(cost["known_usd"])
    if known and unknown:
        value += f" · без данных по {unknown} запросам"
    mark = check if known and not unknown else emphasis("·", "2")
    print(f"  {mark}  Расход API · {value if known else 'неизвестно'}")
    if files and agent.journal:
        command = "/changes"
        if not chat:
            arguments = ["agi", "changes", agent.session_id]
            if agent.store.path != Path(".agent/sessions.sqlite3").resolve():
                arguments.extend(("--state-db", str(agent.store.path)))
            command = shlex.join(arguments)
        print(f"\n{emphasis('▶ Изменения', '36')} · {visible_line(command)}")
    if answer.strip():
        print("\n" + visible(answer))


def confirm_plan(plan: dict, *, verbose=False) -> bool:
    print('\n' + emphasis('План готов', '1;36'))
    plan_text = visible if verbose else visible_line
    print("Цель · " + plan_text(plan['goal']))
    for number, step in enumerate(plan['steps'], 1):
        print(f"{number}. {plan_text(step['description'])}")
        if verbose:
            print(f"   Инструменты: {', '.join(step['tools'])}")
        if 'id' in step:
            verification = step['verification']
            descriptions = {'file_written': 'запись', 'file_read': 'чтение', 'file_absent': 'удаление',
                            'tests_passed': 'тесты'}
            target = verification.get('path', verification.get('command', ''))
            details = [step['id']]
            if step['depends_on']:
                details.append("после " + ", ".join(step['depends_on']))
            details.append(descriptions[verification['kind']] + " " + plan_text(target))
            print("   " + " · ".join(details))
    for key, title in (('risks', 'Риски'), ('assumptions', 'Допущения'), ('done_criteria', 'Критерии готовности')):
        items = plan[key]
        if verbose:
            print(title)
            for item in items:
                print("  - " + visible(item))
            if not items:
                print("  (не указаны)")
        elif items:
            print(title + " · " + " · ".join(visible_line(item) for item in items))
    if not sys.stdin.isatty():
        raise PlanApprovalDeferred('План сохранён; запустите resume в терминале для утверждения')
    try:
        while True:
            answer = input('Утвердить план и перейти к исполнению? [y/n; Enter — пауза]: ').strip().lower()
            if answer in {'y', 'yes', 'да', 'д'}:
                return True
            if answer in {'n', 'no', 'нет', 'н'}:
                return False
            if not answer:
                raise PlanApprovalDeferred('План ожидает решения; продолжение через resume')
            print('Введите y для утверждения или n для отклонения.')
    except EOFError:
        raise PlanApprovalDeferred('Ответ не получен; план сохранён без утверждения') from None


def choose_workspace(root: Path) -> Path | None:
    root = root.absolute()
    try:
        while True:
            try:
                folders = sorted((path for path in root.iterdir() if not path.is_symlink() and path.is_dir()),
                                 key=lambda path: (path.name.casefold(), path.name))
            except FileNotFoundError:
                folders = []
            choices = {str(number): path for number, path in enumerate(folders, 1)}
            print(emphasis("Рабочая папка · workspace", "1;36"))
            for number, path in choices.items():
                print(f"  {number}. {visible_line(path.name)}")
            if not choices:
                print("  Папок пока нет.")
            print("  Enter — создать папку с датой · n — создать папку со своим именем")
            choice = input("Выбор › ").strip().casefold()
            if not choice:
                root.mkdir(parents=True, exist_ok=True)
                today = date.today().isoformat()
                for number in count(1):
                    name = today if number == 1 else f"{today}-{number}"
                    folder = root / name
                    try:
                        folder.mkdir()
                    except FileExistsError:
                        continue
                    print(f"Создана папка · workspace/{name}")
                    return folder
            elif choice in {"n", "н"}:
                name = input("Имя новой папки (Enter — назад) › ")
                if not name:
                    continue
                if name in {".", ".."} or Path(name).name != name or "\x00" in name:
                    print("Введите одно имя папки без пути.")
                    continue
                root.mkdir(parents=True, exist_ok=True)
                folder = root / name
                try:
                    folder.mkdir()
                except FileExistsError:
                    print("Это имя уже занято. Выберите папку по номеру или введите другое имя.")
                    continue
                print(f"Создана папка · workspace/{visible_line(name)}")
                return folder
            elif choice in choices:
                folder = choices[choice]
                if not folder.is_symlink() and folder.is_dir():
                    return folder
                print("Папка больше недоступна. Выберите другую.")
            else:
                print("Введите номер из списка, n или нажмите Enter.")
    except (EOFError, KeyboardInterrupt):
        return None


def choose_session(store, identifier: str | None, *, include_completed=False, journal_only=False, eligible=None) -> str:
    if identifier is not None:
        return identifier
    if not sys.stdin.isatty():
        raise SessionError('Без интерактивного терминала укажите SESSION_ID явно')
    candidates = []
    for session in store.list_sessions():
        if not include_completed and session['status'] in {'completed', 'aborted', 'rolled_back', 'context_limit', 'budget_exhausted'}:
            continue
        if journal_only or eligible is not None:
            metadata = store.get(session['id']).metadata
            if journal_only and not metadata.get('journal_mode', False):
                continue
            if eligible is not None and not eligible(metadata):
                continue
        candidates.append(session)
        if len(candidates) == 20:
            break
    if not candidates:
        raise SessionError('Нет незавершённых сессий для продолжения')
    if len(candidates) == 1:
        print('Продолжаем сессию: ' + candidates[0]['id'])
        return candidates[0]['id']
    print('\nВыберите сессию для продолжения:')
    for number, item in enumerate(candidates, 1):
        print(f"{number}. {item['status']} | {visible(item['model'] or 'модель не указана')} | {visible(item['workspace'])}")
    try:
        while True:
            value = input('Номер сессии (Enter — отмена): ').strip()
            if not value:
                raise SessionError('Выбор сессии отменён')
            if value.isascii() and value.isdigit() and 1 <= int(value) <= len(candidates):
                return candidates[int(value) - 1]['id']
            print('Введите номер из списка.')
    except EOFError:
        raise SessionError('Выбор сессии не получен') from None


def confirm_conflict(runtime, item: dict, verify=None) -> str | None:
    if not sys.stdin.isatty():
        return None
    function = item['call']['function']
    print(f"\nНеизвестен результат вызова {item['key']}: {visible(function['name'])}")
    try:
        arguments = json.loads(function['arguments'])
        if not isinstance(arguments, dict):
            raise ValueError('arguments are not an object')
        print('Предложенное действие: ' + visible(function['arguments'])[:4000])
        if 'path' in arguments:
            path = runtime._safe_path(arguments['path'])
            if path.exists():
                current = runtime.read_file(arguments['path'])
                print('Текущее содержимое:\n' + visible(current[:4000]))
            else:
                current = ''
                print('Файл сейчас отсутствует.')
            if function['name'] == 'write_file' and isinstance(arguments.get('content'), str):
                diff = ''.join(difflib.unified_diff(current.splitlines(True), arguments['content'].splitlines(True),
                                                  fromfile='current', tofile='proposed'))
                print('Сравнение с предложенной записью:\n' + visible(diff[:4000] or '(содержимое совпадает)'))
    except (ToolError, OSError, ValueError, TypeError):
        print('Текущее состояние нельзя безопасно показать; проверьте workspace вручную.')
    supported = function['name'] in {'write_file', 'delete_file', 'read_file', 'list_files', 'Bash', 'bash', 'submit_plan', 'complete_task', 'fail_task'}
    print('1 — повторить поддержанный инструмент' if supported else 'Повтор этого инструмента не поддерживается.')
    print('2 — считать вызов неуспешным и продолжить\n3 — завершить с ошибкой')
    if function['name'] == 'write_file':
        print('4 — считать запись выполненной (требуется точное совпадение байтов)')
    print('0 — оставить сессию заблокированной')
    try:
        while True:
            choice = input('Ваш выбор: ').strip()
            if choice in {'0', ''}:
                return None
            if choice == '1' and supported:
                return 'retry'
            if choice == '2':
                return 'skip'
            if choice == '3':
                return 'abort'
            if choice == '4' and function['name'] == 'write_file':
                if verify is not None and verify(item) is True:
                    return 'applied'
                print('Текущее содержимое не подтверждает запись; выберите другое решение.')
                continue
            print('Введите номер доступного действия.')
    except EOFError:
        return None


def confirm_attribution(view: dict) -> str | None:
    print('\nИзменения, обнаруженные после сбоя:\n' + visible(view['diff']))
    print('1 — отнести все показанные изменения к этой сессии')
    print('2 — считать их внешними; не включать в откат сессии')
    print('0 — оставить вызов заблокированным')
    if not sys.stdin.isatty():
        return None
    try:
        while True:
            answer=input('Принадлежность изменений: ').strip()
            if answer=='1':return 'own'
            if answer=='2':return 'external'
            if answer in {'0',''}:return None
            print('Введите 1, 2 или 0.')
    except EOFError:
        return None
