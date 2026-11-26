from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from .budget import TRUNCATED
from .conversation import Conversation, RunStats
from .mcp import mcp_fields
from .session import SessionError, SessionStore
from .terminal import visible_line
from .text import json_text


TOKEN_FIELDS = {
    "prompt_tokens": "вход",
    "completion_tokens": "выход",
    "total_tokens": "всего",
    "cached_tokens": "чтение кэша",
    "cache_write_tokens": "запись кэша",
}
READ_TOOLS = {"list_files", "read_file", "mcp__tracker__list_tasks"}
WORKSPACE_ACTIONS = {"Bash", "bash", "write_file", "delete_file"}
READ_BOUNDARIES = {"session_resumed", "journal_action_started", "recovery_resolution",
                   "rollback_started", "rollback_file_started", "rollback_file_completed", "rollback_completed"}
TRUNCATION = (TRUNCATED, "... list truncated ...", "... output truncated ...")
NONZERO_EXIT = re.compile(r"^exit_code=-?[1-9]\d*(?:\n|$)")
ERROR_CODE = re.compile(r"^ERROR:\s*([A-Z][A-Z_0-9]+):")


def tool_label(data: dict) -> str:
    name = data["name"]
    labels = mcp_fields(name)
    return f"{labels['server']}/{labels['mcp_tool']}" if labels else name


def call_key(data: dict) -> str | None:
    key = data.get("call") or data.get("tool_call_id")
    return key if isinstance(key, str) and key else None


def arguments_key(data: dict) -> str | None:
    raw = data.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(arguments, dict):
            return json_text(arguments, sort_keys=True)
    except (ValueError, RecursionError):
        pass
    return None


def findings_for_run(run_id: str, events: list[dict]) -> tuple[list[dict], list[str]]:
    calls = {}
    errors = defaultdict(list)
    reads = {}
    read_groups = []
    incomplete_pairs = 0
    incomplete_reads = 0
    truncated = 0
    for event in events:
        kind, data = event["type"], event["data"]
        if kind in READ_BOUNDARIES:
            reads.clear()
        if kind == "tool_call":
            key = call_key(data)
            if key is not None:
                if key in calls:
                    incomplete_pairs += 1
                calls[key] = event
            name = data["name"]
            if name in WORKSPACE_ACTIONS or (name not in READ_TOOLS and mcp_fields(name)):
                reads.clear()
        if kind != "tool_result":
            continue

        call = calls.pop(call_key(data), None)
        content = data.get("content")
        if call is None or not isinstance(content, str) or data.get("name") != call["data"].get("name"):
            incomplete_pairs += 1
            continue
        if content == "~" or content.endswith(TRUNCATION):
            truncated += 1
            reads.clear()
            continue

        request = call["data"]
        name, args = request["name"], arguments_key(request)
        scope = (request.get("phase"), request.get("task_id"))
        observation = (call["id"], event["id"], args)
        key = (name, args, scope)
        if data.get("failed") is True or (name in {"Bash", "bash"} and NONZERO_EXIT.match(content)):
            reads.pop(key, None)
            if content:
                errors[tool_label(request), content].append(observation)
        elif name in READ_TOOLS:
            if data.get("failed") is not False or args is None:
                reads.clear()
                incomplete_reads += 1
                continue
            if key not in reads or reads[key][0] != content:
                observations = []
                reads[key] = (content, observations)
                read_groups.append((tool_label(request), observations))
            reads[key][1].append(observation)

    findings = []
    for (tool, content), observations in errors.items():
        if len(observations) >= 2:
            inputs = len({args for _, _, args in observations if args is not None})
            error = ERROR_CODE.match(content)
            code = error[1] if error else None
            recommendation = ("Учтите отказ в ответе и уберите повторные вызовы отклонённого действия."
                              if code in {"POLICY_DENIED", "APPROVAL_REQUIRED"}
                              else "Откройте указанные события. Устраните причину ошибки перед повтором или сообщите о сбое.")
            findings.append({
                "kind": "repeated_tool_error", "run_id": run_id, "tool": tool,
                "count": len(observations), "known_argument_sets": inputs, "error_code": code,
                "event_ids": [ref for call, result, _ in observations for ref in (call, result)],
                "interpretation": "Инструмент несколько раз вернул одинаковую ошибку.",
                "recommendation": recommendation,
            })
    for tool, observations in read_groups:
        if len(observations) >= 2:
            findings.append({
                "kind": "repeated_unchanged_read", "run_id": run_id, "tool": tool,
                "count": len(observations), "known_argument_sets": 1,
                "event_ids": [ref for call, result, _ in observations for ref in (call, result)],
                "interpretation": "Аргументы и сохранённый результат чтения совпали. Между этими вызовами в журнале нет операций, которые могли изменить источник.",
                "recommendation": "Если повторная проверка не нужна для текущей задачи, используйте ранее полученный результат.",
            })

    limitations = []
    if not events:
        limitations.append("Для запуска нет событий.")
    if incomplete_pairs or calls:
        limitations.append(f"Вызовы и результаты не удалось сопоставить для {incomplete_pairs + len(calls)} записей.")
    if incomplete_reads:
        limitations.append(f"У чтений без статуса или аргументов нет данных для сравнения. Таких чтений {incomplete_reads}.")
    if truncated:
        limitations.append(f"Обрезанные результаты пропущены при сравнении. Таких записей {truncated}.")
    return findings, limitations


def stats_snapshot(stats: RunStats) -> dict:
    usage = {}
    for key in TOKEN_FIELDS:
        count = stats.known[key]
        usage[key] = {"tokens": stats.tokens[key] if count or stats.requests == 0 else None,
                      "reported_requests": count, "requests": stats.requests}
    return {"model_requests": stats.requests, "usage": usage, "tool_calls": dict(stats.tools),
            "skills_discovered": sorted(stats.discovered_skills), "skill_loads": dict(stats.loaded_skills),
            "mcp_calls": {f"{server}/{tool}": count for (server, tool), count in stats.mcp_calls.items()}}


def inspect_session(path: Path, identifier: str) -> dict:
    try:
        with SessionStore(path, read_only=True) as store:
            records = defaultdict(list)
            for event in store.event_records(identifier):
                records[event["run_id"]].append(event)
            conversation = store.db.execute("SELECT 1 FROM conversations WHERE id=?", (identifier,)).fetchone()
            states = [turn.state for turn in Conversation(store, identifier).turns()] if conversation else [store.get(identifier)]
    except sqlite3.Error:
        raise SessionError("Не удалось прочитать журнал этой базы сессий") from None

    runs, total = [], RunStats()
    for state in states:
        events = sorted(records[state.id], key=lambda event: int(event["id"].rsplit(":", 1)[1]))
        findings, limitations = findings_for_run(state.id, events)
        stats = RunStats()
        stats.include(state)
        total.include(state)
        if any(stats.known[key] < stats.requests for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            limitations.append("API вернул токены только для части обращений или не вернул их совсем.")
        runs.append({"run_id": state.id, "status": state.status, "provider": state.metadata.get("provider"),
                     "model": state.metadata.get("model"), "event_count": len(events), **stats_snapshot(stats),
                     "findings": findings, "limitations": limitations})
    return {"session_id": identifier, "runs": runs, "total": stats_snapshot(total),
            "doctor_usage": {"model_requests": 0, "prompt_tokens": 0, "completion_tokens": 0}}


def usage_label(entry: dict) -> str:
    if entry["tokens"] is None:
        return "unknown"
    if entry["reported_requests"] < entry["requests"]:
        return f"{entry['tokens']} (частично, {entry['reported_requests']}/{entry['requests']})"
    return str(entry["tokens"])


def frequencies(counts: dict) -> str:
    return ", ".join(f"{visible_line(name)} × {count}" for name, count in sorted(counts.items())) or "0"


def format_stats(stats: dict) -> list[str]:
    usage = " · ".join(f"{label} {usage_label(stats['usage'][key])}" for key, label in TOKEN_FIELDS.items())
    return [f"  Обращения к модели · {stats['model_requests']}", f"  Токены API · {usage}",
            f"  Скиллы · обнаружено {len(stats['skills_discovered'])} · загрузки {frequencies(stats['skill_loads'])}",
            f"  Инструменты · {frequencies(stats['tool_calls'])}", f"  MCP · {frequencies(stats['mcp_calls'])}"]


def render_report(report: dict) -> str:
    lines = [f"Доктор сессий · {report['session_id']}", "Расход доктора · 0 обращений к модели · 0 токенов"]
    if not report["runs"]:
        lines.append("В диалоге пока нет запусков.")
    for run in report["runs"]:
        lines += ["", f"Запуск {run['run_id']} · {visible_line(run['status'])} · событий {run['event_count']}",
                  *format_stats(run)]
        for finding in run["findings"]:
            label = visible_line(finding["tool"])
            if finding.get("error_code"):
                label += f" / {finding['error_code']}"
            lines += ["", f"  {label} · совпавших вызовов {finding['count']} · "
                      f"известных наборов аргументов {finding['known_argument_sets']}",
                      f"    {finding['interpretation']}", f"    {finding['recommendation']}",
                      "    События · " + ", ".join(finding["event_ids"])]
        if not run["findings"]:
            lines.append("  По двум правилам совпадений не найдено.")
        lines.extend(f"  Ограничение · {item}" for item in run["limitations"])
    if len(report["runs"]) > 1:
        lines += ["", "Итого по сессии", *format_stats(report["total"])]
    return "\n".join(lines)


def show_session_report(path: Path, identifier: str, *, as_json: bool = False) -> None:
    report = inspect_session(path, identifier)
    print(json_text(report, indent=2) if as_json else render_report(report))
