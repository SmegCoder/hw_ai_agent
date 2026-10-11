# Coding agent — первый блок ДЗ

Python 3.11+; единственная внешняя Python-зависимость — `certifi` для доверенных
HTTPS-сертификатов (проверка TLS остаётся включена). Для безопасного
исполнения Bash эта версия требует macOS с `/usr/bin/sandbox-exec`.
Другие ОС отклоняют Bash: небезопасного запасного режима нет.

## Установка после клонирования

Нужны Git и Python 3.11 или новее. Команда `python3` ниже должна запускать
именно эту версию Python. Полная демонстрация и системные проверки Bash
работают на macOS; поддержка других ОС для Bash пока не реализована.

```bash
git clone https://github.com/SmegCoder/hw_ai_agent.git
cd hw_ai_agent
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

В PyCharm откройте папку `hw_ai_agent` и выберите интерпретатор
`.venv/bin/python`. Если импорты `hw_ai_agent` подсвечиваются красным,
отметьте папку `src` через **Mark Directory as → Sources Root**.

## Быстрый запуск с настоящим DeepSeek

После установки выполните из корня проекта в терминале с активированной `.venv`:

```bash
hw-ai-agent setup --provider openrouter --model deepseek/deepseek-v3.2
hw-ai-agent doctor
hw-ai-agent demo --provider openrouter --model deepseek/deepseek-v3.2
```

`setup` просит API-ключ через скрытый ввод и создаёт локальный `.env` с правами
`0600`. Ключ не нужно отправлять в чат. `.env` исключён из Git. Если файл уже
есть, команда откажется его перезаписывать: отредактируйте его локально.
Переменные окружения имеют приоритет над `.env`.

`demo` **вызывает настоящую модель**. Для каждого запуска создаётся новая папка
`workspace/demo-<время>/`. Сначала программа убеждается, что исходные тесты
падают. Затем модель читает файлы, исправляет ошибку расчёта скидки и запускает
тесты через `Bash`. Программа отдельно повторяет проверку и сверяет SHA-256 теста.

Успех обозначается `DEMO PASS` и кодом завершения 0. Отчёт `demo-report.json`
содержит модель, вызовы инструментов, результаты до/после, diff и usage.
Утверждение модели «готово» само по себе не засчитывается. Для сдачи сохраните
отчёт своего успешного реального запуска.

Уже проведённая демонстрация DeepSeek завершилась успешно. Описание находится
в [результатах проверки](docs/DEMO_RESULTS.md), полный сохранённый отчёт —
в [docs/deepseek-demo-report.json](docs/deepseek-demo-report.json).
Эти два файла можно проверить после клонирования без API-ключа.

Ограничения платного запуска: до 12 обращений, до 2048 выходных токенов на
обращение и до 120 000 символов истории. Автоматических повторов сетевых
запросов нет. Это ограничивает объём работы, но не является точным долларовым
лимитом; отдельный денежный лимит задайте в кабинете OpenRouter.

## OpenAI

Дополнительный ключ OpenAI не требуется для OpenRouter. При желании использовать
OpenAI напрямую добавьте `OPENAI_API_KEY` и `OPENAI_MODEL` в локальный `.env`:

```bash
hw-ai-agent demo --provider openai --model <ID-модели-с-tool-calling>
```

Или сохраните отдельный файл конфигурации:

```bash
hw-ai-agent setup --provider openai --model <ID-модели> --env-file .env.openai
hw-ai-agent demo --provider openai --env-file .env.openai
```

## Локальная Ollama

Установите Ollama из официального приложения или через Homebrew:

```bash
brew install ollama
ollama serve
```

Оставьте сервер запущенным; в другом терминале:

```bash
ollama pull qwen3:4b-instruct
hw-ai-agent doctor --provider ollama --model qwen3:4b-instruct
hw-ai-agent demo --provider ollama --model qwen3:4b-instruct --max-tokens 2048
```

Агент обращается к `http://127.0.0.1:11434/v1/chat/completions`. Платный ключ
в Ollama не передаётся. Instruct-модель загружается отдельно и занимает около
2,5 ГБ на диске. Её успешная демонстрация в этом проекте пока не подтверждена.
Подключение само по себе не гарантирует результат: проверяйте `DEMO PASS`.

Ранее установленный `qwen3:4b` оказался Qwen3 4B Thinking 2507: его метаданные
разрешают только `thinking=true`. Наш клиент отправляет `reasoning_effort=none`,
поэтому попытка отключить размышления для этого варианта не подходит.
Локальные прогоны с ним были медленными и не подтвердили успешную демонстрацию.
В настройках по умолчанию и `.env.example` пока остаётся `qwen3:4b`, поэтому
для Instruct явно передавайте `--model qwen3:4b-instruct`, как в примере выше,
или задайте `OLLAMA_MODEL=qwen3:4b-instruct` в локальном `.env`.

`--max-tokens` ограничивает длину ответа, а не увеличивает контекст модели.
Увеличение до 20 000 может продлить генерацию. В исследованном локальном запуске
контекст составлял 4096 токенов; его размер настраивается отдельно в Ollama.
Сейчас `stream=false`, поэтому терминал показывает вызовы инструментов только
после получения полного ответа модели.

## Своя задача и workspace

```bash
hw-ai-agent run --workspace ./workspace "Прочитай файлы, исправь ошибку и запусти тесты"
```

Workspace — отдельная папка с файлами, над которыми работает модель.
По умолчанию используется `workspace/`, также поддерживается `AGENT_WORKSPACE`.
Папка должна существовать. Исходники агента лежат в `src/hw_ai_agent/`.

## Проверки

```bash
# Автономные проверки протокола, клиента, путей и управления процессами:
python -m unittest discover -s tests -v

# Обязательные проверки реальной системной защиты на Mac:
REQUIRE_SANDBOX_TESTS=1 python -m unittest discover -s tests -v
```

Без `REQUIRE_SANDBOX_TESTS=1` четыре проверки, требующие реальной песочницы,
помечаются skipped. Это не эквивалент полной проверке перед сдачей.
Последний полный локальный прогон: 27 тестов прошли без пропусков.
В `.github/workflows/tests.yml` настроен запуск полного набора после push
или pull request на macOS и Python 3.11/3.13/3.14; это конфигурация CI,
а не свидетельство уже успешного запуска на GitHub.
Ответы API в автономных тестах подставные; успешный платный или локальный запуск
подтверждается отдельно командой `demo`.

## Официальная документация подключений

- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenRouter tool calling](https://openrouter.ai/docs/guides/features/tool-calling)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Ollama: варианты Qwen3](https://ollama.com/library/qwen3/tags)
