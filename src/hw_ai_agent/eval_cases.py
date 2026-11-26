from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .filesystem import open_read_file
from .session import SessionState


TRACKER_TOOL = "mcp__tracker__list_tasks"
REFERENCE_DATE = "2026-11-25"
NOTE = "Повторить RISC-V и транзакции."
UNAVAILABLE = "Данные трекера недоступны."

TASKS = [
    {"id": "T1", "title": "Решить задачи по RISC-V", "status": "open", "due_date": "2026-11-24"},
    {"id": "T2", "title": "Сдать схему базы", "status": "closed", "due_date": "2026-11-20"},
    {"id": "T3", "title": "Подготовить лабораторную", "status": "in_progress", "due_date": "2026-11-27"},
    {"id": "T4", "title": "Прочитать главу о транзакциях", "status": "open", "due_date": REFERENCE_DATE},
    {"id": "T5", "title": "Проверить разбор задачи", "status": "blocked", "due_date": "2026-11-19"},
]
OVERDUE = [
    {"id": "T5", "title": "Проверить разбор задачи", "due_date": "2026-11-19", "days_overdue": 6},
    {"id": "T1", "title": "Решить задачи по RISC-V", "due_date": "2026-11-24", "days_overdue": 1},
]
NEEDS_DEADLINE = [
    {"id": "T6", "title": "Проверить расписание", "status": "todo"},
    {"id": "T7", "title": "Выбрать тему доклада", "status": "open", "due_date": "2026-02-30"},
    {"id": "T8", "title": "Сдать старую работу", "status": "done", "due_date": None},
]


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    output: str
    skills: tuple[str, ...]
    snapshot: list[dict] | None
    mcp: str = "ok"


CASES = (
    Case("open-tasks",
         "Прочитай задачи учебного трекера. Сохрани в open-tasks.json JSON-массив строк с ID всех "
         "незавершённых задач без дублей. Учитывай статусы open, todo, in_progress и blocked.",
         "open-tasks.json", ("open-task-export",), [*TASKS, TASKS[0]]),
    Case("deadlines",
         f"Собери отчёт по учебному трекеру на {REFERENCE_DATE}. Сохрани в deadlines.json JSON-объект с полями "
         "as_of (дата отчёта), overdue (массив объектов с id, title, due_date, days_overdue), "
         "missing_due_date (число незавершённых задач без корректного срока) и skipped_records (число "
         "некорректных записей). В overdue включай только незавершённые задачи со сроком строго раньше "
         "даты отчёта. Сортируй по due_date, затем по id.",
         "deadlines.json", ("deadline-report",), TASKS),
    Case("note",
         f"Запиши в note.txt ровно этот текст, без кавычек и переноса строки: {NOTE}",
         "note.txt", (), TASKS, "unused"),
    Case("missing-deadlines",
         f"Проверь незавершённые задачи учебного трекера на {REFERENCE_DATE}. Сохрани в deadlines.json "
         "JSON-объект с двумя полями: overdue (массив объектов с id и due_date, по возрастанию даты и id) "
         "и missing_due_date_ids (массив ID незавершённых задач с отсутствующим или некорректным "
         "сроком). Просрочены задачи со сроком строго раньше указанной даты. Не выдумывай даты.",
         "deadlines.json", ("deadline-report",), [*TASKS, *NEEDS_DEADLINE]),
    Case("mcp-failure",
         "Выгрузи ID незавершённых задач учебного трекера в open-tasks.json. Если трекер недоступен, "
         f"не создавай файл и ответь ровно одной фразой: {UNAVAILABLE}",
         "open-tasks.json", ("open-task-export",), None, "error"),
)


def read_output(workspace: Path, name: str) -> str:
    with open_read_file(workspace, workspace / name) as descriptor:
        with os.fdopen(os.dup(descriptor), "rb") as source:
            raw = source.read(64_001)
    if len(raw) > 64_000:
        raise ValueError("Файл результата превышает 64 КБ")
    return raw.decode("utf-8")


def equal_ids(value, expected: set[str]) -> bool:
    return (isinstance(value, list) and all(isinstance(item, str) for item in value)
            and len(value) == len(expected) and set(value) == expected)


def outcome(case: Case, workspace: Path, state: SessionState) -> tuple[bool, str]:
    if state.status != "completed":
        return False, "Запуск не завершился"
    if case.mcp == "error":
        passed = not any(workspace.iterdir()) and (state.final_answer or "").strip() == UNAVAILABLE
        return passed, "Нет файла, ошибка сообщена" if passed else "Ожидалось сообщение о сбое без записи файла"
    try:
        entries = {path.name for path in workspace.iterdir()}
        if entries != {case.output}:
            return False, f"Ожидался единственный файл {case.output}"
        text = read_output(workspace, case.output)
        if case.id == "note":
            passed = text == NOTE
        else:
            value = json.loads(text)
            if case.id == "open-tasks":
                passed = equal_ids(value, {"T1", "T3", "T4", "T5"})
            elif case.id == "deadlines":
                expected = {"as_of": REFERENCE_DATE, "overdue": OVERDUE, "missing_due_date": 0, "skipped_records": 0}
                passed = value == expected and all(type(item["days_overdue"]) is int for item in value["overdue"])
                if passed:
                    passed = all(type(value[key]) is int for key in ("missing_due_date", "skipped_records"))
            else:
                expected = [{key: task[key] for key in ("id", "due_date")} for task in OVERDUE]
                passed = (isinstance(value, dict) and set(value) == {"overdue", "missing_due_date_ids"}
                          and value["overdue"] == expected and equal_ids(value["missing_due_date_ids"], {"T6", "T7"}))
    except (OSError, ValueError, RecursionError):
        return False, f"Не удалось прочитать корректный {case.output}"
    return passed, "Результат совпал с эталоном" if passed else "Содержимое не совпало с эталоном"


def actions(case: Case, variant: str, events: list[dict], *, compare_revisions=False) -> tuple[bool | str, bool]:
    tool_calls = {e["data"]["name"] for e in events if e["type"] == "tool_call"}
    results = [e["data"] for e in events if e["type"] == "tool_result"]
    allowed = {"list_files", "read_file"}
    skills_enabled = variant == "B" or compare_revisions
    if skills_enabled:
        allowed.add("load_skill")
    if case.mcp != "error":
        allowed.add("write_file")
    if case.mcp != "unused":
        allowed.add(TRACKER_TOOL)
    successful = {item["name"] for item in results if not item.get("failed")}
    tracker_error = any(item["name"] == TRACKER_TOOL and item.get("failed") and "MCP_TOOL_ERROR" in item["content"]
                        for item in results)
    needed = {"write_file"} if case.mcp != "error" else set()
    if case.mcp == "ok":
        needed.add(TRACKER_TOOL)
    tools_pass = tool_calls <= allowed and needed <= successful and (case.mcp != "error" or tracker_error)
    skills = {e["data"]["name"] for e in events if e["type"] == "skill_loaded"}
    routing = skills == set(case.skills)
    return routing if skills_enabled else "n/a", tools_pass
