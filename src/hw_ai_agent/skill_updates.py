from __future__ import annotations

from pathlib import Path

from .config import Settings
from .eval_cases import CASES
from .eval_runner import Options, read_json, run_evaluation, run_lock, save_json
from .skill_market import MarketError, SkillMarket
from .skills import SKILL_NAME


TOKEN_GROWTH_PERCENT = 20
RULE = (
    "В каждой группе из трёх попыток число успешных результатов B не ниже A.",
    "Хотя бы одна задача решена в каждом варианте, а проверяемый скилл загружен в обеих версиях.",
    "Все 15 попыток кандидата проходят проверку допустимых и необходимых инструментов.",
    "Usage входа и выхода известен для всех обращений в обоих вариантах.",
    "Средний суммарный расход входных и выходных токенов на попытку вырос максимум на 20%.",
)


def decision_for(summary: dict, comparison: dict) -> dict:
    reasons = []
    by_case = summary["by_case"]
    cases = {case.id for case in CASES}
    if (summary["completed"] != 30 or summary["planned"] != 30 or set(by_case) != cases
            or any(side["attempts"] != 3 for case in by_case.values() for side in case.values())):
        raise MarketError("Для решения нужна полная серия · пять сценариев × три попытки × две версии")
    for case, variants in by_case.items():
        if variants["B"]["outcome_pass"] < variants["A"]["outcome_pass"]:
            reasons.append(f"{case} · результат снизился с {variants['A']['outcome_pass']}/3 до {variants['B']['outcome_pass']}/3")
    totals = summary["by_variant"]
    if any(totals[side]["outcome_pass"] == 0 for side in ("A", "B")):
        reasons.append("В каждом варианте нужна хотя бы одна успешная попытка")
    if totals["B"]["tools_pass"] != 15:
        reasons.append(f"Кандидат прошёл проверку инструментов в {totals['B']['tools_pass']}/15 попытках")
    name = comparison["name"]
    if any(totals[side]["skill_activations"].get(name, 0) == 0 for side in ("A", "B")):
        reasons.append("Проверяемый скилл должен быть загружен в обеих версиях")
    usage = {}
    for side, total in totals.items():
        directions = (total["input_tokens"], total["output_tokens"])
        if total["model_requests"] and all(entry["reported_requests"] == total["model_requests"]
                                           and entry["known_sum"] is not None for entry in directions):
            usage[side] = sum(entry["known_sum"] for entry in directions)
        else:
            usage[side] = None
    if None in usage.values():
        reasons.append("Неполный usage · соблюдение лимита токенов подтвердить нельзя")
    elif 100 * usage["B"] > (100 + TOKEN_GROWTH_PERCENT) * usage["A"]:
        reasons.append(f"Рост суммарного расхода токенов выше {TOKEN_GROWTH_PERCENT}% · {usage['A']} → {usage['B']}")
    return {"status": "rejected" if reasons else "accepted",
            "reasons": reasons or ["Качество, вызовы и расход удовлетворяют всем заранее заданным критериям"],
            "token_sums": usage, "metrics": summary, "comparison": comparison,
            "report": "summary.md"}


def evaluate_candidate(market: SkillMarket, name: str, settings: Settings, options: Options,
                       *, focus: tuple[str, ...] = (), output: Path | None = None) -> tuple[int, dict | None]:
    if options.trials != 3:
        raise MarketError("Для проверки обновления требуется по три попытки каждого сценария")
    if not SKILL_NAME.fullmatch(name) or len(name) > 64:
        raise MarketError("Некорректное имя скилла")
    parent = market.db_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with run_lock(parent, f".{market.db_path.stem}-{name}.lock"):
        return evaluate_locked(market, name, settings, options, focus, output)


def evaluate_locked(market: SkillMarket, name: str, settings: Settings, options: Options,
                    focus: tuple[str, ...], output: Path | None) -> tuple[int, dict | None]:
    candidate, old, new = market.candidate(name)
    output = (output or market.db_path.parent / "skill-update-evals" / candidate["id"]).expanduser().resolve()
    market.outside_workspace(output)
    comparison = {
        "candidate_id": candidate["id"], "name": name, "focus": list(focus), "rule": list(RULE),
        "revisions": {
            "A": {"version": candidate["old_version"], "sha256": old.digest},
            "B": {"version": candidate["new_version"], "sha256": new.digest},
        },
    }
    status = run_evaluation(output, settings, options, comparison=comparison, skill_versions={"A": old, "B": new})
    if status:
        return status, None
    decision = decision_for(read_json(output / "summary.json"), comparison)
    decision["output"] = str(output)
    market.check()
    decision = market.finish_candidate(candidate["id"], decision)
    save_json(output / "decision.json", decision)
    return 0, decision
