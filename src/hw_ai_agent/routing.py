"""Explainable local task routing; no paid classifier or mid-session model switch."""
from __future__ import annotations

import re
from decimal import Decimal

from .model_catalog import Model, comparison
from .pricing import PricingError, estimate_tokens


POOLS = {
    'openrouter': {
        'simple': ('deepseek/deepseek-v4.1-flash', 'openai/gpt-4.1-mini', 'qwen/qwen3-coder', 'deepseek/deepseek-v3.2'),
        'medium': ('qwen/qwen3-coder', 'deepseek/deepseek-v3.2', 'openai/gpt-4.1-mini'),
        'complex': ('anthropic/claude-sonnet-4.6', 'openai/gpt-4.1'),
    },
    'openai': {
        'simple': ('gpt-4.1-nano', 'gpt-4o-mini', 'gpt-4.1-mini'),
        'medium': ('gpt-4.1-mini', 'gpt-5-mini'),
        'complex': ('gpt-4.1', 'gpt-5'),
    },
}


def complexity(request: str, *, tasks=False, plan=False) -> dict:
    lowered = request.lower()
    reasons, score = [], 0
    for pattern, points, reason in (
        (r'архитектур|architect|миграци|migrat|concurren|race condition|гонк|безопасност|security|уязвим|vulnerab|deadlock', 4, 'архитектура, миграция, конкуренция или безопасность'),
        (r'нескольк.*файл|многофайл|multi.?file|зависимост|dependenc|рефактор|refactor|оптимиз|optimi', 2, 'изменения нескольких частей проекта или рефакторинг'),
        (r'исправ|fix|ошиб|bug|тест|test|реализ|implement|добав|add', 1, 'реализация, исправление или тестирование'),
    ):
        if re.search(pattern, lowered):
            score += points; reasons.append(reason)
    if len(request) > 1500:
        score += 2; reasons.append('длинное описание задачи')
    if tasks:
        score += 2; reasons.append('включены задачи с зависимостями')
    elif plan:
        score += 1; reasons.append('включено планирование')
    level = 'complex' if score >= 4 else 'medium' if score >= 2 else 'simple'
    return {'level': level, 'score': score, 'reasons': reasons or ['короткая задача без признаков сложных изменений'],
            'method': 'local heuristic v1; not a quality benchmark'}


def eligible(model: Model, needed: int, output: int):
    return model.tools and model.price is not None and (model.context is None or model.context >= needed) and (model.max_output is None or model.max_output >= output)


def choose_route(catalog, request: str, *, max_tokens=2048, tasks=False, plan=False, overrides=None):
    result = complexity(request, tasks=tasks, plan=plan)
    needed = estimate_tokens([{'role': 'user', 'content': request}], []) + max_tokens + 8192
    level = result['level']
    overrides = overrides or {}
    if overrides.get(level):
        pool = [overrides[level]]
    elif catalog.provider == 'ollama':
        # Installed tool-capable local models: parameter count is a policy proxy,
        # not a measured quality guarantee. Prefer larger models for hard tasks.
        candidates = [m for m in catalog.models if eligible(m, needed, max_tokens)]
        def size(model):
            match = re.search(r'(\d+(?:\.\d+)?)b', model.id.lower())
            return float(match[1]) if match else 0
        candidates.sort(key=lambda m: (size(m), m.id), reverse=level != 'simple')
        pool = [m.id for m in candidates]
    else:
        pool = list(POOLS.get(catalog.provider, {}).get(level, ()))
    candidates = [m for identifier in pool if (m := catalog.find(identifier)) and eligible(m, needed, max_tokens)]
    if not candidates:
        raise PricingError(f'Нет подходящей модели для уровня {level}; задайте AGENT_ROUTE_{level.upper()}_MODEL или явный --model')
    if level == 'simple' and catalog.provider != 'ollama' and not overrides.get(level):
        candidates.sort(key=lambda m: (m.price.calculate(1000, 1000) or Decimal(0), m.id))
    selected = candidates[0]
    result.update({'selected_model': selected.id, 'candidates': [m.id for m in candidates],
                   'required_context_estimate': needed, 'catalog_updated_at': catalog.updated_at,
                   'catalog_status': catalog.status})
    return selected, result


def shortlist(catalog, current=None, max_tokens=2048):
    models = [m for m in catalog.models if m.tools and (m.max_output is None or m.max_output >= max_tokens)]
    preferred = [current] if current else []
    for pool in POOLS.get(catalog.provider, {}).values():
        preferred.extend(pool)
    chosen = []
    for identifier in preferred:
        model = next((m for m in models if m.id == identifier), None)
        if model and model not in chosen:
            chosen.append(model)
    for model in sorted(models, key=lambda m: m.id):
        if model not in chosen:
            chosen.append(model)
        if len(chosen) >= 20:
            break
    return chosen[:20]


def pick_model(catalog, *, current=None, max_tokens=2048, interactive=True, read=input, show=print):
    if not interactive:
        raise PricingError("Выбор --model '?' требует терминала; укажите ID модели или --model auto")
    choices = shortlist(catalog, current, max_tokens)
    if not choices:
        raise PricingError('В каталоге нет моделей с подтверждённой поддержкой tools')
    show(f'Тарифы USD; обновлены {catalog.updated_at}; источник: {catalog.source}; состояние: {catalog.status}')
    if catalog.warning:
        show(catalog.warning)
    show(comparison(choices, numbered=True))
    while True:
        try:
            answer = read('Выберите номер модели (0 — отмена): ').strip()
        except EOFError:
            raise PricingError('Выбор модели отменён: ответ не получен') from None
        if answer in {'', '0'}:
            raise PricingError('Выбор модели отменён')
        if answer.isdecimal() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        show('Введите номер из списка или 0 для отмены.')
