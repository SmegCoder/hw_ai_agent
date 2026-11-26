from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import certifi
import yaml

from .agent import AgentError, CodingAgent
from .budget import BudgetExceeded, BudgetLimits
from .client import ModelClientError, OpenAICompatibleClient
from .config import Settings
from .conversation import RunStats
from .eval_cases import CASES, TRACKER_TOOL, Case, actions, outcome
from .eval_report import render_summary, summarize, write_reports
from .mcp import MCPRegistry
from .policy import Policy
from .session import SessionError, SessionStore
from .skills import BUILTIN_SKILLS, SKILL_NAME, Skill, SkillCatalog
from .text import json_text
from .tools import ToolError, ToolRuntime


HYPOTHESIS = ("Скиллы повысят долю правильных результатов в сценариях отбора задач и дедлайнов. "
              "Дополнительные инструкции могут увеличить расход входных токенов.")
BUDGET = BudgetLimits(model_requests=8, tool_calls=20, side_effects=3, result_bytes=65_536)
PACKAGE = Path(__file__).resolve().parent


class EvalError(ValueError):
    pass


@dataclass(frozen=True)
class Options:
    trials: int = 3
    timeout: float = 180
    max_rounds: int = 8
    max_tokens: int = 2048

    def __post_init__(self):
        if not 1 <= self.trials <= 30 or not 1 <= self.max_rounds <= 30 or self.max_tokens < 1:
            raise EvalError("Допустимы 1–30 попыток, 1–30 обращений и положительный лимит токенов")
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 3600:
            raise EvalError("Таймаут попытки должен быть от 1 до 3600 секунд")

    def limits(self) -> BudgetLimits:
        return BudgetLimits(self.max_rounds, BUDGET.tool_calls, BUDGET.side_effects, BUDGET.result_bytes)


def atomic_write(path: Path, content: str) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_json(path: Path, data) -> None:
    atomic_write(path, json_text(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def run_lock(output: Path):
    descriptor = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise EvalError("В этой папке уже выполняется серия eval") from None
        yield descriptor
    finally:
        os.close(descriptor)


def evaluation_cases(comparison: dict | None):
    if comparison is None:
        return CASES
    name, focus = comparison["name"], comparison["focus"]
    if not SKILL_NAME.fullmatch(name) or set(focus) - {case.id for case in CASES}:
        raise EvalError("Некорректные параметры сравнения скилла")
    return tuple(replace(case, prompt=f"Сначала загрузи установленный скилл {name}. " + case.prompt,
                         skills=(name,)) if case.id in focus else case for case in CASES)


def version_directory(output: Path, variant: str) -> Path:
    return output / "versions" / variant


def verify_skill_versions(output: Path, comparison: dict) -> None:
    name = comparison["name"]
    for variant, version in comparison["revisions"].items():
        directory = version_directory(output, variant)
        if (set(path.name for path in directory.iterdir()) != {name}
                or Skill.read(directory, name).digest != version["sha256"]):
            raise EvalError("Снимок версии скилла не совпадает с манифестом")


def manifest_for(settings: Settings, options: Options, comparison: dict | None = None) -> dict:
    paths = sorted(PACKAGE.glob("*.py")) + sorted(BUILTIN_SKILLS.glob("*/SKILL.md"))
    versions = {str(path.relative_to(PACKAGE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    manifest = {"format": 1, "hypothesis": HYPOTHESIS, "provider": settings.provider, "model": settings.model,
            "endpoint": settings.base_url, "parameters": asdict(options), "budget": options.limits().as_dict(),
            "mode": "auto", "approvals": "deny", "variants": {"A": "skills-off", "B": "skills-on"},
            "mcp": {"server": "tracker", "allow_tools": ["list_tasks"], "timeout": 5},
            "cases": [{"id": case.id, "prompt": case.prompt, "skills": list(case.skills), "mcp": case.mcp,
                       "snapshot": hashlib.sha256(json_text(case.snapshot, sort_keys=True).encode()).hexdigest()}
                      for case in evaluation_cases(comparison)],
            "python": sys.version.split()[0], "dependencies": {"certifi": certifi.__version__, "PyYAML": yaml.__version__},
            "versions": versions}
    if comparison is not None:
        manifest.update(comparison=comparison, variants={"A": "active-skill", "B": "candidate-skill"},
                        hypothesis="Новая версия сохраняет качество по всем сценариям и укладывается в лимит токенов.")
    return manifest


def prepare_trial(directory: Path, case: Case) -> None:
    workspace = directory / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        raise EvalError(f"Рабочая папка попытки без сессии уже содержит файлы: {workspace}")
    data = directory / "source"
    data.mkdir(exist_ok=True)
    snapshot = data / "tasks.json"
    if case.snapshot is None:
        if snapshot.exists():
            raise EvalError(f"В сценарии сбоя снимок должен отсутствовать: {snapshot}")
    else:
        save_json(snapshot, {"tasks": case.snapshot})
    save_json(data / "mcp.json", {"servers": {"tracker": {
        "snapshot": "tasks.json", "allow_tools": ["list_tasks"], "timeout": 5,
    }}})
    save_json(data / "policy.json", {"directories": {"allow": ["."]},
                                    "defaults": {"read_file": "allow", "list_files": "allow", "load_skill": "allow"},
                                    "rules": [{"tool": TRACKER_TOOL, "action": "allow"},
                                              {"tool": "write_file", "path": case.output, "action": "allow"}]})


def execute_trial(directory: Path, case: Case, variant: str, settings: Settings, options: Options,
                  *, skill_dir: Path | None = None) -> None:
    workspace, source = directory / "workspace", directory / "source"
    client = OpenAICompatibleClient(settings, timeout=options.timeout, max_tokens=options.max_tokens)
    with SessionStore(directory / "session.sqlite3") as store:
        runtime = ToolRuntime(workspace, mode="auto", policy=Policy.load(source / "policy.json"),
                              skills=SkillCatalog(workspace, skill_dir) if variant == "B" or skill_dir else None,
                              mcp=MCPRegistry(source / "mcp.json", workspace),
                              trash_dir=directory / "trash")
        agent = CodingAgent(client, runtime, store=store, max_rounds=options.max_rounds, cost_estimate=False,
                            budget_limits=options.limits(), max_context_chars=120_000,
                            session_metadata={"kind": "run", "policy_path": str(source / "policy.json"),
                                              "policy_source": "file", "approvals": "deny", "max_tokens": options.max_tokens})
        try:
            agent.run(case.prompt)
        except (ModelClientError, AgentError, BudgetExceeded, SessionError, ToolError):
            if agent.session_id is None:
                raise
            # The normal agent loop has already persisted the failed run.


def launch_trial(manifest: Path, directory: Path, case: Case, variant: str, options: Options, lock_fd: int) -> str | None:
    args = [sys.executable, "-B", "-m", "hw_ai_agent.eval_runner", str(manifest), str(directory), case.id, variant]
    try:
        completed = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True, timeout=options.timeout, pass_fds=(lock_fd,))
    except subprocess.TimeoutExpired:
        return "timeout"
    except KeyboardInterrupt:
        return "interrupted"
    return None if completed.returncode == 0 else f"worker_exit_{completed.returncode}"


def has_run(directory: Path) -> bool:
    database = directory / "session.sqlite3"
    if not database.exists():
        return False
    with SessionStore(database) as store:
        return bool(store.list_sessions())


def collect_trial(output: Path, directory: Path, case: Case, variant: str, number: int,
                  stop: str | None, settings: Settings, *, compare_revisions=False) -> dict:
    database = directory / "session.sqlite3"
    with SessionStore(database) as store:
        if settings.provider != "ollama" and settings.api_key:
            store.log_credentials.add(settings.api_key)
        rows = store.list_sessions()
        if len(rows) != 1:
            raise EvalError(f"В попытке {case.id}/{variant}{number} ожидается одна сессия")
        run_id = rows[0]["id"]
        state = store.get(run_id)
        if state.status == "interrupted" and state.error in {"timeout", "interrupted"}:
            stop = stop or state.error
        active = state.status in {"ready", "running", "waiting_approval", "waiting_plan_approval"}
        saved_stop = state.status == "interrupted" and state.error == stop
        if active or (stop is not None and state.status != "completed" and not saved_stop):
            stop = stop or "interrupted"
            store.set_status(run_id, "interrupted", stop)
            state = store.get(run_id)
        records = list(store.event_records(run_id))
        error = state.error or next((r["data"].get("content") for r in reversed(records)
                                    if r["type"] == "tool_result" and r["data"].get("failed")), None)
        error = store.redacted_event({"event": "eval_error", "error": error})["error"]
    success, detail = outcome(case, directory / "workspace", state)
    routing, tools = actions(case, variant, records, compare_revisions=compare_revisions)
    stats = RunStats()
    stats.include(state)
    journal = directory / "events.jsonl"
    atomic_write(journal, "".join(json_text(record) + "\n" for record in records))
    return {"case_id": case.id, "variant": variant, "trial": number, "run_id": run_id,
            "status": "timeout" if state.status == "interrupted" and stop == "timeout" else state.status,
            "outcome_pass": success, "check": detail,
            "routing_pass_or_na": routing, "tools_pass": tools, "model_requests": stats.requests,
            "input_tokens": stats.tokens["prompt_tokens"] if stats.known["prompt_tokens"] else None,
            "output_tokens": stats.tokens["completion_tokens"] if stats.known["completion_tokens"] else None,
            "input_usage_requests": stats.known["prompt_tokens"], "output_usage_requests": stats.known["completion_tokens"],
            "skill_activations": dict(stats.loaded_skills), "tool_calls": dict(stats.tools),
            "mcp_calls": {f"{server}/{tool}": n for (server, tool), n in stats.mcp_calls.items()},
            "error": error, "events": journal.relative_to(output).as_posix(),
            "artifact": (directory / "workspace" / case.output).relative_to(output).as_posix()}


def completed_rows(output: Path, attempts: list[tuple[Case, str, int]]) -> list[dict]:
    rows = []
    for case, variant, number in attempts:
        path = output / "trials" / case.id / f"{variant}{number}" / "result.json"
        if path.is_file():
            result = read_json(path)
            if (result["case_id"], result["variant"], result["trial"]) != (case.id, variant, number):
                raise EvalError(f"Неверный ключ сохранённой попытки: {path}")
            rows.append(result)
    return rows


def rebuild_summary(output: Path) -> None:
    output = output.expanduser().resolve()
    with run_lock(output):
        manifest = read_json(output / "manifest.json")
        rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        planned = len(manifest["cases"]) * len(manifest["variants"]) * manifest["parameters"]["trials"]
        summary = summarize(rows, planned, manifest)
        atomic_write(output / "summary.md", render_summary(rows, manifest, summary))
    print(f"Сводка · {output / 'summary.md'}")


def run_evaluation(output: Path, settings: Settings, options: Options, *, comparison: dict | None = None) -> int:
    if settings.model in {"auto", "?"}:
        raise EvalError("Для eval укажите фиксированный ID модели")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    attempts = [(case, variant, number) for case in evaluation_cases(comparison)
                for number in range(1, options.trials + 1) for variant in ("A", "B")]
    manifest = manifest_for(settings, options, comparison)
    manifest_path = output / "manifest.json"
    with run_lock(output) as lock_fd:
        if comparison is not None:
            verify_skill_versions(output, comparison)
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise EvalError("У сохранённой серии другие модель, параметры или версия кода. Выберите новый --output")
        else:
            if any((output / name).exists() for name in ("trials", "results.jsonl", "summary.json", "summary.md")):
                raise EvalError("В папке есть данные без manifest.json. Выберите новый --output")
            save_json(manifest_path, manifest)
        rows = completed_rows(output, attempts)
        completed = {(row["case_id"], row["variant"], row["trial"]) for row in rows}
        write_reports(output, rows, len(attempts), manifest, atomic_write)
        for index, (case, variant, number) in enumerate(attempts, 1):
            if (case.id, variant, number) in completed:
                continue
            directory = output / "trials" / case.id / f"{variant}{number}"
            stop = None
            print(f"Eval {index}/{len(attempts)} · {case.id} · {variant}{number}", flush=True)
            if not has_run(directory):
                prepare_trial(directory, case)
                stop = launch_trial(manifest_path, directory, case, variant, options, lock_fd)
                if not has_run(directory):
                    if stop == "interrupted":
                        return 130
                    raise EvalError(f"Сессия не была создана для {case.id}/{variant}{number} ({stop or 'worker_exit'})")
            row = collect_trial(output, directory, case, variant, number, stop, settings,
                                compare_revisions=comparison is not None)
            save_json(directory / "result.json", row)
            rows = completed_rows(output, attempts)
            write_reports(output, rows, len(attempts), manifest, atomic_write)
            print(f"  {row['status']} · результат {'прошёл' if row['outcome_pass'] else 'не прошёл'}"
                  f" · run {row['run_id']}", flush=True)
            if stop == "interrupted":
                return 130
    print(f"Сохранено попыток · {len(rows)}/{len(attempts)}\nСводка · {output / 'summary.md'}")
    return 0


def worker(argv: list[str]) -> None:
    manifest_path, directory, case_id, variant = argv
    manifest = read_json(Path(manifest_path))
    settings = Settings.from_env(manifest["provider"], manifest["model"])
    options = Options(**manifest["parameters"])
    comparison = manifest.get("comparison")
    if manifest_for(settings, options, comparison) != manifest:
        raise EvalError("Изменилась конфигурация eval")
    case = next(case for case in evaluation_cases(comparison) if case.id == case_id)
    if comparison is None:
        execute_trial(Path(directory), case, variant, settings, options)
    else:
        output = Path(manifest_path).parent
        verify_skill_versions(output, comparison)
        execute_trial(Path(directory), case, variant, settings, options,
                      skill_dir=version_directory(output, variant))


if __name__ == "__main__":
    worker(sys.argv[1:])
