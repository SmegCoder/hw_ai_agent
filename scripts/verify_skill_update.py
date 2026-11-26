"""Recompute the published update decision using the saved versions and journal."""

import argparse
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

from hw_ai_agent.conversation import RunStats
from hw_ai_agent.eval_cases import actions, outcome
from hw_ai_agent.eval_report import render_summary, summarize
from hw_ai_agent.eval_runner import evaluation_cases, read_json, verify_skill_versions
from hw_ai_agent.skill_updates import decision_for

from verify_evals import read_rows, require


def verify(output: Path):
    output = output.resolve()
    manifest = read_json(output / "manifest.json")
    comparison = manifest["comparison"]
    require(manifest["variants"] == {"A": "active-skill", "B": "candidate-skill"}, "Состав вариантов")
    verify_skill_versions(output, comparison)
    rows = read_rows(output / "results.jsonl")
    cases = {case.id: case for case in evaluation_cases(comparison)}
    require([item["id"] for item in manifest["cases"]] == list(cases), "Состав сценариев")
    for item in manifest["cases"]:
        case = cases[item["id"]]
        require((item["prompt"], item["skills"], item["mcp"]) == (case.prompt, list(case.skills), case.mcp),
                f"Манифест сценария · {case.id}")
        snapshot = output / "inputs" / case.id / "tasks.json"
        tasks = read_json(snapshot)["tasks"] if snapshot.exists() else None
        require(tasks == case.snapshot, f"Снимок задач · {case.id}")
    keys = {(row["case_id"], row["variant"], row["trial"]) for row in rows}
    expected = {(case, side, trial) for case in cases for side in ("A", "B") for trial in (1, 2, 3)}
    require(len(rows) == 30 and keys == expected and len({row["run_id"] for row in rows}) == 30,
            "Ожидались 30 независимых попыток на пяти сценариях")
    with tempfile.TemporaryDirectory() as empty:
        for row in rows:
            case, side = cases[row["case_id"]], row["variant"]
            directory = output / "trials" / case.id / f"{side}{row['trial']}"
            require(row == read_json(directory / "result.json"), "Запись попытки расходится с results.jsonl")
            events = read_rows(directory / "events.jsonl")
            require(events and all(event["run_id"] == row["run_id"] for event in events), "Журнал попытки")
            workspace = directory / "workspace"
            final = next((item["data"]["content"] for item in reversed(events) if item["type"] == "final"), None)
            stats_events = [{"event": item["type"], **item["data"]} for item in events]
            state = SimpleNamespace(status=row["status"], final_answer=final, events=stats_events,
                                    rounds=sum(e["type"] == "model_request" for e in events))
            actual, detail = outcome(case, workspace if workspace.is_dir() else Path(empty), state)
            routing, tools = actions(case, side, events, compare_revisions=True)
            require((actual, detail, routing, tools) ==
                    (row["outcome_pass"], row["check"], row["routing_pass_or_na"], row["tools_pass"]),
                    f"Результат или инструменты · {row['run_id']}")

            stats = RunStats()
            stats.include(state)
            require((stats.requests, dict(stats.tools), dict(stats.loaded_skills),
                     {f"{s}/{t}": n for (s, t), n in stats.mcp_calls.items()}) ==
                    (row["model_requests"], row["tool_calls"], row["skill_activations"], row["mcp_calls"]),
                    f"Статистика · {row['run_id']}")
            for key, API_key in (("input", "prompt_tokens"), ("output", "completion_tokens")):
                require((stats.known[API_key], stats.tokens[API_key] if stats.known[API_key] else None) ==
                        (row[f"{key}_usage_requests"], row[f"{key}_tokens"]), f"Токены · {row['run_id']}")
            candidate_loads = {e["data"]["tool_call_id"] for e in events if e["type"] == "skill_loaded"
                               and e["data"]["name"] == comparison["name"]}
            for event in events:
                data = event["data"]
                if event["type"] == "tool_result" and data.get("tool_call_id") in candidate_loads:
                    digest = hashlib.sha256(data["content"].encode()).hexdigest()
                    require(digest == comparison["revisions"][side]["sha256"], f"Версия скилла · {row['run_id']}")

    summary = summarize(rows, len(expected), manifest)
    require(summary == read_json(output / "summary.json"), "Сводка JSON расходится с журналом")
    require(render_summary(rows, manifest, summary) == (output / "summary.md").read_text(),
            "Таблица Markdown расходится с журналом")
    actual = decision_for(summary, comparison)
    saved = read_json(output / "decision.json")
    require(all(saved[key] == value for key, value in actual.items()), "Решение не воспроизводится")
    print(f"Проверены 30 попыток и обе версии скилла. Решение · {actual['status']}")
    for reason in actual["reasons"]:
        print(f"  {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?",
                        default=Path(__file__).resolve().parents[1] / "docs" / "skill-update-eval")
    verify(parser.parse_args().output)
