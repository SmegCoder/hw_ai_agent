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


def token_label(data: dict) -> str:
    coverage = f"{data['reported_requests']}/{data['model_requests']}"
    if data["known_sum"] is None:
        return f"unknown ({coverage})"
    value = str(data["known_sum"])
    if data["reported_requests"] < data["model_requests"]:
        value = f"{value} частично"
    mean = data["mean_complete_trial"]
    return f"{value} ({coverage})" if mean is None else f"{value} ({coverage}), среднее {mean}"


def md_text(value) -> str:
    return " ".join(str(value).split()).replace("|", "\\|").replace(chr(96), "'")


def format_counts(counts: dict) -> str:
    return ", ".join(f"{md_text(key)} × {val}" for key, val in sorted(counts.items())) or "0"


def summary_row(name: str, variant: str, data: dict) -> str:
    return (f"| {name} | {variant} | {data['outcome_pass']}/{data['attempts']} "
            f"| {data['tools_pass']}/{data['attempts']} | {data['routing_errors'] if variant == 'B' else 'n/a'} "
            f"| {sum(data['tool_calls'].values())} | {sum(data['mcp_calls'].values())} "
            f"| {token_label(data['input_tokens'])} | {token_label(data['output_tokens'])} "
            f"| {sum(data['errors'].values())} |")


def write_reports(output: Path, rows: list[dict], planned: int, manifest: dict, write) -> None:
    totals = {variant: tally([r for r in rows if r["variant"] == variant]) for variant in ("A", "B")}
    by_case = {case["id"]: {variant: tally([r for r in rows if r["case_id"] == case["id"] and r["variant"] == variant])
                           for variant in ("A", "B")} for case in manifest["cases"]}
    result = {"completed": len(rows), "planned": planned, "by_case": by_case, "by_variant": totals}
    write(output / "results.jsonl", "".join(json_text(row) + "\n" for row in rows))
    write(output / "summary.json", json_text(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Evals", "", f"Модель {md_text(manifest['provider'])}/{md_text(manifest['model'])}. "
             f"Сохранено попыток {len(rows)} из {planned}.", "", manifest["hypothesis"], "",
             "| Сценарий | Версия | Результат | Вызовы по критериям | Ошибки выбора | Инструменты | MCP "
             "| Входные токены API | Выходные токены API | Попытки с ошибками |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for case, data in by_case.items():
        lines.extend(summary_row(case, variant, data[variant]) for variant in ("A", "B"))
    lines.extend(summary_row("Итого", variant, totals[variant]) for variant in ("A", "B"))
    lines += ["", "Покрытие usage указано как число обращений с данными из общего числа обращений. "
              "Средние рассчитаны по попыткам с полным usage для данной метрики.", "",
              "## Попытки", "",
              "| Сценарий | Попытка | Run и журнал | Проверка результата | Выбор скиллов | Инструменты | Скиллы | MCP | Ошибка |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['variant']}{row['trial']} | [{row['run_id']}]({row['events']}) "
                     f"| {'да' if row['outcome_pass'] else 'нет'} | {row['routing_pass_or_na']} "
                     f"| {'да' if row['tools_pass'] else 'нет'} | {format_counts(row['skill_activations'])} "
                     f"| {format_counts(row['mcp_calls'])} | {md_text(row['error'] or '—')} |")
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
            lines.append(f"- [{row['case_id']} {variant}{row['trial']}]({row['events']}), проверка "
                         f"{'прошла' if row['outcome_pass'] else 'не прошла'}. "
                         f"{md_text(row['check'])}. Скиллы {format_counts(row['skill_activations'])}. "
                         f"Вызовы {format_counts(row['tool_calls'])}.")
    write(output / "summary.md", "\n".join(lines) + "\n")
