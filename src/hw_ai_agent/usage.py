"""Russian CLI help; viewing it never loads settings or starts an agent."""
from __future__ import annotations

import argparse
import textwrap


COMMAND_HELP = {
    "models": "Сравнить модели по актуальным тарифам, контексту и цене одинакового сценария",
    "cost": "Стоимость сохранённой сессии; без ID — выбор в терминале",
    "cost-demo": "Автономное демо оценки стоимости, учёта и выбора модели по сложности",
    "help": "Краткая инструкция и флаги; help КОМАНДА — подробная справка команды",
    "set": "Сохранить настройки будущих запусков в .env, не изменяя API-ключи",
    "setup": "Создать .env с настройками провайдера и скрытым вводом ключа",
    "doctor": "Проверить настройки, подключение и системную песочницу",
    "run": "Выполнить вашу задачу в существующей рабочей папке",
    "plan": "Сначала составить план; выполнять только после вашего утверждения",
    "demo": "Настоящая модель исправляет расчёт скидки и запускает тесты",
    "policy-demo": "Автономная проверка разрешений, отказов и защиты файлов",
    "session-demo": "Автономная проверка продолжения без повторной записи",
    "workflow-demo": "Многофайловое исправление и очистка с настоящей моделью",
    "plan-demo": "Демо планирования с настоящей моделью; --offline — фиксированный сценарий",
    "budget-demo": "Автономная проверка четырёх бюджетов, продолжения и аварии",
    "tasks-demo": "Задачи с зависимостями и проверкой результата; поддерживает --offline",
    "rollback-demo": "Демо исправления с журналом для отката; поддерживает --offline",
    "sessions": "Список сохранённых сессий",
    "session": "Подробное состояние сессии по её ID",
    "resume": "Продолжить сессию; без ID — выбор в терминале",
    "changes": "Показать изменения сессии с журналом; без ID — выбор в терминале",
    "rollback": "Явно откатить изменения сессии с журналом; без ID — выбор в терминале",
    "resolve": "Явно разрешить неопределённый вызов по ID сессии и ключу вызова",
    "approve-plan": "Утвердить сохранённый план по ID; исполнение запускается через resume",
    "reject-plan": "Отклонить сохранённый план по ID и завершить сессию",
}

# The same descriptions are used in the guide and each command's --help.
FLAG_HELP = {
    "--provider": "Выберите одного из 3 провайдеров: openrouter, openai или ollama. Без флага — AGENT_PROVIDER из настроек, иначе openrouter; setup использует openrouter.",
    "--model": "ID модели; '?' — выбор с ценами в терминале, auto — автоматический роутинг. Без флага — модель из настроек или встроенное значение, если оно есть.",
    "--env-file": "Файл настроек и ключей (по умолчанию .env). Переменные окружения имеют приоритет.",
    "--workspace": "AGENT_WORKSPACE в .env для run/plan. Рабочая папка: для run/plan она должна существовать; для демо укажите новую папку. Без флага демо создаёт отдельную папку автоматически.",
    "--mode": "AGENT_MODE в .env; без настройки safe: подтверждать каждую запись и Bash; auto: выполнять их после проверок. Каждое удаление подтверждается в обоих режимах.",
    "--policy": "AGENT_POLICY в .env или путь к JSON-политике разрешений allow/ask/deny. Файл должен находиться вне рабочей папки. Явные запреты действуют в обоих режимах.",
    "--approvals": "AGENT_APPROVALS в .env (обычно ask). ask — запрашивать нужные подтверждения; deny — отклонять их. Это не автоматическое согласие; утверждение плана и решения конфликтов задаются отдельно.",
    "--plan": "AGENT_PLAN=true/false в .env. Включить планирование: до утверждения разрешено только чтение. У команды plan включено автоматически.",
    "--tasks": "AGENT_TASKS=true/false в .env. План с ID задач, зависимостями и проверяемыми результатами. Автоматически включает планирование.",
    "--journal": "AGENT_JOURNAL=true/false в .env. Сохранять исходные файлы и изменения для changes/rollback. Включается при старте; в rollback-demo включён всегда.",
    "--max-rounds": "AGENT_MAX_ROUNDS в .env. Общий лимит запросов к модели, включая планирование и resume. Без настройки 12; workflow-demo, plan-demo и rollback-demo — 20, tasks-demo — 30.",
    "--max-tokens": "AGENT_MAX_TOKENS в .env. Максимум выходных токенов одного ответа модели (обычно 2048, tasks-demo — 3072). Это не размер контекста и не общий бюджет.",
    "--max-tool-calls": "AGENT_MAX_TOOL_CALLS в .env. Общий лимит вызовов инструментов (по умолчанию 128).",
    "--max-side-effects": "AGENT_MAX_SIDE_EFFECTS в .env. Общий лимит попыток записи, удаления и Bash (по умолчанию 32). Значение 0 запрещает эти действия.",
    "--max-result-bytes": "AGENT_MAX_RESULT_BYTES в .env. Общий лимит результатов инструментов в байтах UTF-8 (по умолчанию 1000000). Большие результаты обрезаются с отметкой.",
    "--state-db": "AGENT_STATE_DB в .env. Путь к SQLite-базе сессий (по умолчанию .agent/sessions.sqlite3). Храните вне workspace; для продолжения укажите ту же базу.",
    "--stop-after-tools": "Пауза после N сохранённых результатов инструментов. Продолжение — resume; флаг паузы не наследуется.",
    "--stop-after-tasks": "Пауза после N завершённых и проверенных задач. Требуется режим задач; продолжение — resume.",
    "--offline": "Фиксированные ответы вместо настоящей модели, без API-ключа. Доступен в workflow-demo, plan-demo, tasks-demo и rollback-demo; инструменты настоящие.",
    "--crash-after-write": "Демо аварии: завершить процесс кодом 86 после записи, до сохранения результата. Через resume потребуется явное решение конфликта.",
    "--call": "Ключ неопределённого вызова для resolve, например 3:0. Возьмите из session ID.",
    "--resolution": "Решение для resolve: retry — повторить; skip — сохранить ошибку и продолжить; abort — завершить; applied — признать запись при точном совпадении байтов.",
    "--changes": "Для resolve с журналом: own — отнести показанные изменения к сессии; external — считать внешними и исключить из отката этого вызова.",
    "--no-plan": "Отключить AGENT_PLAN для run/demo или сохранить false через set. Команда plan всегда планирует.",
    "--no-tasks": "Отключить AGENT_TASKS для run/plan/demo или сохранить false через set. tasks-demo всегда использует задачи.",
    "--no-journal": "Отключить AGENT_JOURNAL для нового запуска или сохранить false через set. В rollback-demo журнал обязателен.",
    "--route": "AGENT_ROUTE=true/false в .env. Выбирать модель один раз по сложности задачи; явный --model ID имеет приоритет.",
    "--no-route": "Отключить AGENT_ROUTE для нового запуска или сохранить false через set. --model auto сам включает роутинг.",
    "--route-simple-model": "AGENT_ROUTE_SIMPLE_MODEL: модель для простых задач; должна поддерживать tools и иметь известный тариф.",
    "--route-medium-model": "AGENT_ROUTE_MEDIUM_MODEL: модель для задач средней сложности.",
    "--route-complex-model": "AGENT_ROUTE_COMPLEX_MODEL: модель для сложных задач.",
    "--refresh": "Заново получить тарифы для models; при сбое старый кэш явно помечается устаревшим.",
    "--search": "Фильтр models по части ID модели, например deepseek или gpt-4.1.",
    "--limit": "Сколько моделей показать в models (1..5000, по умолчанию 50).",
    "--all": "Показать весь каталог models после фильтров; имеет приоритет над --limit.",
    "--sort": "Сортировка models: name — по имени (по умолчанию), price — по цене сценария. Сначала модели с tools.",
    "--paid": "Только модели с ненулевым тарифом входных или выходных токенов. Не совмещается с --free.",
    "--free": "Только модели с нулевыми тарифами входных и выходных токенов. Неизвестные цены исключаются.",
    "--cost-estimate": "AGENT_COST_ESTIMATE=true/false в .env. Включить предварительную оценку перед каждым обращением; учёт фактического расхода работает всегда.",
    "--no-cost-estimate": "Отключить предварительную оценку только для этого запуска (также resume), либо сохранить false через set.",
    "--input-tokens": "Входные токены одинакового сценария сравнения models (по умолчанию 1000).",
    "--output-tokens": "Выходные токены сценария сравнения models (по умолчанию 1000).",
    "--json": "Вывести сравнение models в JSON: тарифы, источник, время и цена сценария.",
    "--help": "Показать справку. Флаги конкретной команды: hw-ai-agent КОМАНДА --help.",
}

POSITIONAL_HELP = {
    "request": "Ваша задача; удобно заключить текст в кавычки",
    "session_id": "ID сохранённой сессии; если аргумент необязателен, можно выбрать в терминале",
    "command": "Команда для подробной справки; без аргумента — общее руководство",
}


class HelpParser(argparse.ArgumentParser):
    """Attach shared Russian descriptions without changing argument behavior."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", argparse.RawTextHelpFormatter)
        super().__init__(*args, **kwargs)

    def add_argument(self, *names, **kwargs):
        for name in names:
            description = FLAG_HELP.get(name) or POSITIONAL_HELP.get(name)
            if description:
                kwargs["help"] = textwrap.fill(description, width=74)
                break
        return super().add_argument(*names, **kwargs)


def project_guide() -> str:
    flags = "\n".join(
        name + "\n" + textwrap.fill(description, width=86, initial_indent="    ", subsequent_indent="    ")
        for name, description in FLAG_HELP.items()
    )
    return """БЫСТРЫЙ СТАРТ
  Активируйте .venv и запускайте команды из корня проекта.
  Если .env ещё нет: hw-ai-agent setup --provider openrouter
  Проверка подключения: hw-ai-agent doctor --provider openrouter
  Настоящая модель из настроек: hw-ai-agent demo --mode safe

НАСТРОЙКИ
  hw-ai-agent set --provider ollama
  hw-ai-agent set --max-rounds 30 --max-tokens 4096 --mode safe
  hw-ai-agent set --provider ollama --model qwen3:4b-instruct
  set сохраняет изменения в .env; флаги обычного запуска действуют один раз.
  Приоритет: флаг > переменная окружения > .env > встроенное значение.
  AGENT_MAX_ROUNDS и остальные лимиты из .env общие для всех новых запусков.
  --env-file позволяет выбрать другой файл настроек.

ЦЕНЫ И ВЫБОР МОДЕЛИ
  hw-ai-agent models --provider openrouter --refresh
  hw-ai-agent models --paid --limit 100
  hw-ai-agent models --all
  hw-ai-agent run --workspace ./my-project --model '?' "Исправь ошибку"
  hw-ai-agent run --workspace ./my-project --model auto "Исправь ошибку"
  hw-ai-agent set --route
  hw-ai-agent cost
  hw-ai-agent run --workspace ./my-project --no-cost-estimate "Исправь ошибку"
  hw-ai-agent set --no-cost-estimate
  По умолчанию — 50 моделей по имени, таблица с колонками; --all показывает весь каталог.
  --cost-estimate / --no-cost-estimate управляют оценкой; AGENT_COST_ESTIMATE — настройка .env.
  До каждого обращения — приблизительная оценка; после — списание или расчёт по токенам.
  Итог задачи — сумма всей сессии. Неизвестные расходы не считаются нулевыми.
  OpenRouter: live API; OpenAI: официальная таблица Standard; Ollama: оплата API $0.
  Кэш тарифов — 1 час, время источника показывается; --refresh обновляет его.
  Роутинг — локальная эвристика; модель фиксируется на всю сессию, включая resume.

СВОЯ ЗАДАЧА
  hw-ai-agent run --workspace ./my-project "Исправь ошибку и проверь тестами"
  hw-ai-agent plan --tasks --journal --workspace ./my-project --max-rounds 30 "Исправь ошибку"
  Папка должна существовать. Bash поддерживается на macOS через sandbox-exec.
  План: y — утвердить, n — отклонить, Enter — сохранить на паузе.
  После паузы: hw-ai-agent resume
  С журналом: hw-ai-agent changes, затем при необходимости hw-ai-agent rollback

ДЕМО БЕЗ API
  hw-ai-agent policy-demo
  hw-ai-agent session-demo
  hw-ai-agent budget-demo
  hw-ai-agent cost-demo
  hw-ai-agent tasks-demo --offline --mode auto
  hw-ai-agent rollback-demo --offline --mode auto
  Без --offline демо с моделью используют настоящий API или локальную Ollama.

ФЛАГИ ЗАПУСКА
  Не все флаги доступны каждой команде: hw-ai-agent help КОМАНДА.
""" + flags + """

ПРОДОЛЖЕНИЕ
  В resume сохраняются исходные модель, режим и бюджеты: менять их нельзя.
  resume/changes/rollback без ID предлагают выбор только в интерактивном терминале.
  Бюджеты не сбрасываются; откат выполняется только по явной команде rollback.
  Полная инструкция и демо: README.md в корне проекта.
"""
