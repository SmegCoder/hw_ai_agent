from __future__ import annotations

from collections import Counter
from pathlib import Path

from .text import json_text


def tally(rows: list[dict]) -> dict:
    attempts = len(rows)
    requests = sum(row["model_requests"] for row in rows)
    result = {"attempts": attempts, "outcome_pass": sum(row["outcome_pass"] for row in rows),
              "tools_pass": sum(row["tools_pass"] for row in rows), "model_requests": requests,
              "routing_errors": sum(row["routing_pass_or_na"] is False for row in rows),
              "errors": dict(Counter(row["error"] for row in rows if row["error"]))}
    for name in ("skill_activations", "tool_calls", "mcp_calls"):
        counts = Counter()
        for row in rows:
            counts.update(row[name])
        result[name] = dict(counts)
    for direction in ("input", "output"):
        key = direction + "_tokens"
        covered = sum(row[direction + "_usage_requests"] for row in rows)
        full = [row[key] for row in rows if row["model_requests"] > 0
                and row[direction + "_usage_requests"] == row["model_requests"]]
        result[key] = {"known_sum": sum(row[key] for row in rows if row[key] is not None) if covered else None,
                       "reported_requests": covered, "model_requests": requests,
                       "mean_complete_trial": round(sum(full) / len(full), 1) if full else None,
                       "complete_trials": len(full)}
    return result


def markdown_table(headers, rows, *, right=()) -> list[str]:
    cells = [tuple(str(value) for value in row) for row in (headers, *rows)]
    widths = [max(3, *(len(value) for value in column)) for column in zip(*cells, strict=True)]

    def line(row):
        return "| " + " | ".join(value.rjust(width) if i in right else value.ljust(width)
                                 for i, (value, width) in enumerate(zip(row, widths, strict=True))) + " |"

    rules = ["-" * (width - 1) + ":" if i in right else "-" * width for i, width in enumerate(widths)]
    return [line(cells[0]), line(rules), *(line(row) for row in cells[1:])]


def md_text(value) -> str:
    return " ".join(str(value).split()).replace("|", "\\|").replace(chr(96), "'")


def format_counts(counts: dict) -> str:
    return ", ".join(f"{md_text(key)} × {val}" for key, val in sorted(counts.items())) or "0"


def token_columns(data: dict) -> tuple:
    count = data["known_sum"] if data["known_sum"] is not None else "unknown"
    mean = data["mean_complete_trial"] if data["mean_complete_trial"] is not None else "—"
    return count, mean, f"{data['reported_requests']}/{data['model_requests']}"


def summarize(rows: list[dict], planned: int, manifest: dict) -> dict:
    totals = {variant: tally([r for r in rows if r["variant"] == variant]) for variant in ("A", "B")}
    by_case = {case["id"]: {variant: tally([r for r in rows if r["case_id"] == case["id"] and r["variant"] == variant])
                           for variant in ("A", "B")} for case in manifest["cases"]}
    return {"completed": len(rows), "planned": planned, "by_case": by_case, "by_variant": totals}


def render_summary(rows: list[dict], manifest: dict, summary: dict) -> str:
    planned, by_case = summary["planned"], summary["by_case"]
    groups = [*by_case.items(), ("Итого", summary["by_variant"])]
    results, tokens = [], []
    for case, variants in groups:
        for variant, data in variants.items():
            results.append((case, variant, f"{data['outcome_pass']}/{data['attempts']}",
                            f"{data['tools_pass']}/{data['attempts']}",
                            data["routing_errors"] if variant == "B" else "—", sum(data["tool_calls"].values()),
                            sum(data["mcp_calls"].values()), sum(data["errors"].values())))
            tokens.append((case, variant, *token_columns(data["input_tokens"]), *token_columns(data["output_tokens"])))
    lines = ["# Evals", "", f"Модель {md_text(manifest['provider'])}/{md_text(manifest['model'])}. "
             f"Сохранено попыток {len(rows)} из {planned}.", "", manifest["hypothesis"], "", "## Результаты", ""]
    lines += markdown_table(("Сценарий", "A/B", "Результат", "Вызовы", "Ошибки выбора", "Инстр.", "MCP", "Сбои"),
                            results, right=range(2, 8))
    lines += ["", "В столбце «Вызовы» указано число попыток, прошедших проверку инструментов. "
              "В столбце «Сбои» учтены и ожидаемые ошибки MCP.",
              "", "## Токены API", ""]
    lines += markdown_table(("Сценарий", "A/B", "Вход", "Ср. вход", "API вход", "Выход", "Ср. выход", "API выход"),
                            tokens, right=range(2, 8))
    lines += ["", "Колонки API показывают долю обращений, для которых провайдер вернул usage. Если usage известен "
              "частично, в сумме только известные значения. Средние рассчитаны по попыткам с полным usage."]
    errors = {error: f"E{number}" for number, error in enumerate(dict.fromkeys(row["error"] for row in rows if row["error"]), 1)}
    refs = {row["run_id"]: f"run-{number:03d}" for number, row in enumerate(rows, 1)}
    attempts = [(row["case_id"], f"[{row['variant']}{row['trial']}][{refs[row['run_id']]}]",
                 "да" if row["outcome_pass"] else "нет", "да" if row["tools_pass"] else "нет",
                 {True: "да", False: "нет"}.get(row["routing_pass_or_na"], "—"), format_counts(row["skill_activations"]),
                 sum(row["mcp_calls"].values()), errors.get(row["error"], "—")) for row in rows]
    lines += ["", "## Попытки", "", "Номер попытки открывает журнал.", ""]
    lines += markdown_table(("Сценарий", "Попытка", "Результат", "Вызовы", "Выбор", "Скиллы", "MCP", "Сбой"),
                            attempts, right=(6,))
    if errors:
        lines += ["", "## Ошибки", "", *(f"- {label} · {md_text(error)}" for error, label in errors.items())]
    if len(rows) == planned:
        selected = ["open-tasks", "deadlines", "missing-deadlines"]
        scores = {variant: sum(by_case[case][variant]["outcome_pass"] for case in selected) for variant in ("A", "B")}
        verdict = "подтвердилась" if scores["B"] > scores["A"] else "не подтвердилась"
        lines += ["", f"Гипотеза о росте качества {verdict} на этой серии. В трёх сценариях отбора задач "
                  f"и дедлайнов число успешных результатов составило {scores['A']} для A и {scores['B']} для B."]
    pairs = {}
    for row in rows:
        pairs.setdefault((row["case_id"], row["trial"]), {})[row["variant"]] = row
    matched = [pair for pair in pairs.values() if set(pair) == {"A", "B"}]
    if matched:
        pair = next((item for item in matched if item["A"]["outcome_pass"] != item["B"]["outcome_pass"]), matched[0])
        lines += ["", "## Пара A/B", ""]
        for variant, row in pair.items():
            lines.append(f"- [{row['case_id']} {variant}{row['trial']}][{refs[row['run_id']]}], проверка "
                         f"{'прошла' if row['outcome_pass'] else 'не прошла'}. "
                         f"{md_text(row['check'])}. Скиллы {format_counts(row['skill_activations'])}. "
                         f"Вызовы {format_counts(row['tool_calls'])}.")
    lines += ["", *(f"[{refs[row['run_id']]}]: {row['events']} \"{row['run_id']}\"" for row in rows)]
    return "\n".join(lines) + "\n"


def write_reports(output: Path, rows: list[dict], planned: int, manifest: dict, write) -> None:
    summary = summarize(rows, planned, manifest)
    write(output / "results.jsonl", "".join(json_text(row) + "\n" for row in rows))
    write(output / "summary.json", json_text(summary, indent=2, sort_keys=True) + "\n")
    write(output / "summary.md", render_summary(rows, manifest, summary))
