"""Verify the saved A/B evaluation without a model or provider credentials."""

import hashlib
import json
import tempfile
from pathlib import Path

from hw_ai_agent.conversation import RunStats
from hw_ai_agent.eval_cases import CASES, actions, outcome
from hw_ai_agent.eval_report import render_summary, summarize
from hw_ai_agent.session import SessionStore
from hw_ai_agent.text import json_text


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "study-tracker"
EVAL = EVIDENCE / "eval"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    manifest = json.loads((EVAL / "manifest.json").read_text())
    rows = read_rows(EVAL / "results.jsonl")
    cases = {case.id: case for case in CASES}
    expected = {(case, version, trial) for case in cases for version in ("A", "B") for trial in (1, 2, 3)}
    keys = {(row["case_id"], row["variant"], row["trial"]) for row in rows}
    require(keys == expected and len(rows) == len(expected) == 30, "Неполная серия или дубликаты попыток")
    ids = {row["run_id"] for row in rows}
    require(len(ids) == 30, "Повтор run_id")

    require([case["id"] for case in manifest["cases"]] == list(cases), "Состав сценариев")
    for card in manifest["cases"]:
        case = cases[card["id"]]
        digest = hashlib.sha256(json_text(case.snapshot, sort_keys=True).encode()).hexdigest()
        require((case.prompt, list(case.skills), case.mcp, digest) ==
                (card["prompt"], card["skills"], card["mcp"], card["snapshot"]), f"Сценарий {case.id}")
        inputs = EVAL / "inputs" / case.id
        snapshot = inputs / "tasks.json"
        saved = json.loads(snapshot.read_text())["tasks"] if snapshot.exists() else None
        require(saved == case.snapshot, f"Снимок {case.id}")

    inputs = EVAL / "inputs"
    hashes = json.loads((inputs / "hashes.json").read_text())
    actual = {path.relative_to(inputs).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in inputs.glob("*/*.json")}
    require(actual == hashes, "Контрольные суммы фактических входов")
    with SessionStore(EVIDENCE / "sessions.sqlite3", read_only=True) as db, tempfile.TemporaryDirectory() as empty:
        require({r["id"] for r in db.list_sessions()} == ids, "Неверный состав базы")
        for row in rows:
            case, variant = cases[row["case_id"]], row["variant"]
            trial = Path("trials") / case.id / f"{variant}{row['trial']}"
            journal = trial / "events.jsonl"
            workspace = EVAL / trial / "workspace"
            run = row["run_id"]
            require(row["events"] == str(journal) and row["artifact"] == str(trial / "workspace" / case.output), run)
            events = read_rows(EVAL / journal)
            require(events and events == list(db.event_records(run)), f"Журнал {run}")
            state = db.get(run)
            require(state.status == row["status"], f"Статус {run}")
            success, detail = outcome(case, workspace if workspace.exists() else Path(empty), state)
            routing, tools = actions(case, variant, events)
            stats = RunStats()
            stats.include(state)
            expected_fields = {
                "outcome_pass": success, "check": detail, "routing_pass_or_na": routing, "tools_pass": tools,
                "model_requests": stats.requests, "tool_calls": dict(stats.tools),
                "skill_activations": dict(stats.loaded_skills),
                "mcp_calls": {f"{s}/{t}": n for (s, t), n in stats.mcp_calls.items()},
                "error": state.error or next((e["data"]["content"] for e in reversed(events)
                                              if e["type"] == "tool_result" and e["data"].get("failed")), None),
            }
            for label, key in (("input", "prompt_tokens"), ("output", "completion_tokens")):
                expected_fields[label + "_tokens"] = stats.tokens[key] if stats.known[key] else None
                expected_fields[label + "_usage_requests"] = stats.known[key]
            for key, value in expected_fields.items():
                require(row[key] == value, f"{trial} · {key}")
    summary = summarize(rows, len(expected), manifest)
    require(summary == json.loads((EVAL / "summary.json").read_text()), "Итоги summary.json")
    require(render_summary(rows, manifest, summary) == (EVAL / "summary.md").read_text(), "Таблица summary.md")
    print("Сверены 5 сценариев, 30 журналов и строк результатов, файлы, выбор скиллов, вызовы и токены.")
    for variant, result in summary["by_variant"].items():
        print(f"{variant} · результат {result['outcome_pass']}/{result['attempts']} · инструменты {result['tools_pass']}/{result['attempts']}")


if __name__ == "__main__":
    verify()
