from __future__ import annotations
import argparse
from dataclasses import replace, asdict
from decimal import Decimal
from functools import partial
import getpass
import json
import os
import math
import shlex
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
from urllib.request import urlopen
from .agent import AgentError, CodingAgent
from .budget import BudgetLimits, BudgetExceeded
from .conversation import Conversation, ConversationShell
from .client import ModelClientError, OpenAICompatibleClient
from .config import (ConfigurationError, PROVIDERS, Settings, load_env, LAUNCH_OPTIONS,
                     START_COMMANDS, apply_launch_defaults, validate_explicit_options,
                     initial_launch_values, option_value, set_options, load_launch_env)
from .usage import COMMAND_HELP, HelpParser, project_guide
from .pricing import PricingError, Price, usd, amount
from .model_catalog import ModelCatalog, comparison
from .routing import choose_route, pick_model
from .tools import ToolError, ToolRuntime
from .filesystem import open_read_file
from .policy import ApprovalDeferred, ApprovalRequest, Policy, PolicyError
from .session import SessionError, SessionStore
from .terminal import (visible, visible_line, emphasis, confirm_plan, choose_session, choose_workspace,
                       confirm_conflict, confirm_attribution, report_header, report_answer,
                       color_diff, read_confirmation, ProgressDisplay)
from .journal import ChangeJournal, JournalError
from .tasks import checklist
from .text import json_text
from .skills import SkillCatalog
from .skill_market import ANTHROPIC_MARKET, SkillMarket
from .skill_updates import evaluate_candidate
from .eval_cases import CASES
from .mcp import MCPRegistry
from .eval_runner import Options as EvalOptions, rebuild_summary, run_evaluation
from .session_doctor import show_session_report

def show_context_warning(event):
    projected, window = event["projected"], event["window"]
    counts = f"≈{projected:,} из {window:,}".replace(",", " ")
    print(f"Контекст близок к пределу · {counts} токенов с резервом ответа", file=sys.stderr, flush=True)


def show_model_retry(event):
    print(f"{event['error']} · повтор {event['retry']}/{event['limit']} через {event['delay']:g} с",
          file=sys.stderr, flush=True)


def show_event(event: dict) -> None:
    if event["event"] == "model_request":
        print(f"\nМодель: обращение {event['round']}", flush=True)
    elif event["event"] == "model_cost_estimate":
        print(f"Оценка запроса: вход ≈{event['input_tokens_estimate']} токенов, выход 0..{event['output_tokens_limit']}; "
              f"{usd(event['usd_input_estimate'])} .. {usd(event['usd_with_full_output_estimate'])}. Это приблизительный расчёт.", flush=True)
    elif event["event"] == "model_cost":
        label = {"reported": "Списано по данным провайдера", "calculated": "Рассчитано по фактическим токенам", "local": "Оплата локального API", "unknown": "Стоимость ответа"}[event['kind']]
        print(f"{label}: {usd(event['usd'])}", flush=True)
    elif event["event"] == "session_cost":
        cost = event['cost']
        print(f"Сессия: учтённая сумма {usd(cost['known_usd'])}; подтверждено провайдером {usd(cost['reported_usd'])}; "
              f"расчёт по токенам {usd(cost['calculated_usd'])}; неизвестных запросов {cost['unknown_requests']}.", flush=True)
    elif event["event"] == "context_warning":
        show_context_warning(event)
    elif event["event"] == "model_retry":
        show_model_retry(event)
    elif event["event"] == "mcp_error":
        print(f"MCP · {visible_line(event['server'])} · {visible_line(event['error'])}", file=sys.stderr, flush=True)
    elif event["event"] == "policy_decision":
        print(f"Policy: {event['tool']} → {event['action']} / {event['outcome']}", flush=True)
        if event.get("source") == "session_grant":
            print("Используется разрешение, уже выданное для этой сессии.", flush=True)
    elif event["event"] == "tasks":
        print("\nЗадачи:\n" + visible(checklist(event["tasks"])), flush=True)
    elif event["event"] == "budget":
        budget = event["budget"]
        labels = {"model_requests": "модель", "tool_calls": "инструменты", "side_effects": "опасные действия", "result_bytes": "результаты (байт)"}
        print("Бюджет: " + "; ".join(f"{labels[k]} {budget['spent'][k]}/{budget['limits'][k]}" for k in labels), flush=True)
    elif event["event"] in {"session_created", "session_resumed", "session_cached"}:
        print(f"Сессия: {event['session_id']} ({event['event']})", flush=True)

def confirm_action(request: ApprovalRequest, *, verbose=False) -> bool:
    if not sys.stdin.isatty():
        raise ApprovalDeferred("Подтверждение требует терминала. Продолжите через agi resume")

    scope = " до конца сессии" if request.scope == "session" else ""
    target = visible_line(request.target)
    if not verbose and request.tool == "Bash" and request.working_directory is not None:
        directory = f" (в {visible_line(request.working_directory)})" if request.working_directory != "." else ""
        question = f"{emphasis('Выполнить', '1;33')} {target}{directory}{scope}? [y/N] "
    else:
        if verbose:
            print(f"\nТребуется подтверждение: {request.tool} {target}")
            print(f"Причина: {visible(request.reason)}")
            print(f"SHA-256 запроса: {request.fingerprint}")
        else:
            label = {"write_file": "Запись", "delete_file": "Удаление",
                     "read_file": "Чтение", "list_files": "Список файлов"}.get(request.tool, request.tool)
            print(f"\n{emphasis(label, '1;33')} · {target}")
            if request.reason not in {f"{mode} mode requires approval for {request.tool}" for mode in ("safe", "auto")}:
                print(visible(request.reason))
        if request.preview:
            preview = color_diff(request.preview) if request.tool == "write_file" else visible(request.preview)
            print(preview.rstrip("\n"))
        question = f"Разрешить{scope}? [y/N] "
    try:
        return read_confirmation(question) is True
    except EOFError:
        raise ApprovalDeferred("Ответ не получен. Продолжите через agi resume") from None


def show_compact_event(event: dict, *, chat=False) -> None:
    kind = event["event"]
    if kind in {"session_created", "session_resumed", "session_cached"} and not chat:
        print(emphasis(f"Сессия · {event['session_id']}", "2"), flush=True)
    elif kind == "model_request" and event["round"] == 1:
        print("Агент работает…", flush=True)
    elif kind == "context_warning":
        show_context_warning(event)
    elif kind == "model_retry":
        show_model_retry(event)
    elif kind == "mcp_error":
        print(f"MCP · {visible_line(event['server'])} · {visible_line(event['error'])}", file=sys.stderr, flush=True)
    elif kind == "task_completed":
        print(f"[x] {visible(event['task_id'])}", flush=True)
    elif kind == "tool_result":
        if event.get("failed", event["content"].startswith("ERROR:")):
            print(f"{visible_line(event['name'])} · {visible_line(event['content'][:240])}", flush=True)


def show_market_event(event):
    kind = event["event"]
    if kind == "market_check":
        if event["status"] == "error":
            print(f"Маркет · {visible_line(event['error'])}", file=sys.stderr, flush=True)
        return
    name, version = visible_line(event["name"]), visible_line(event["new_version"])
    if kind == "market_new":
        print(f"Маркет · новый скилл {name} {version}", flush=True)
    elif kind == "market_update":
        old = visible_line(event["old_version"])
        status = {"updated": "установлено", "available": "доступно", "error": "ошибка",
                  "pending": "ожидает eval"}[event["status"]]
        detail = f" · {visible_line(event['error'])}" if event.get("error") else ""
        print(f"Маркет · {name} {old} → {version} · {status}{detail}", flush=True)


def market_config_argument(value: str) -> Path | str:
    return ANTHROPIC_MARKET if value == ANTHROPIC_MARKET else Path(value)


def market_command(args) -> int:
    market = SkillMarket(args.market_config or ANTHROPIC_MARKET, args.state_db, args.workspace)
    if args.market_action == "events":
        for line in market.events():
            print(line)
    elif args.market_action == "status":
        print(json_text(market.installed(), indent=2))
    elif args.market_action == "candidates":
        print(json_text(market.candidates(), indent=2))
    elif args.market_action == "search":
        source, events = market.check()
        if events[0]["status"] == "error":
            show_market_event(events[0])
            return 1
        offers = market.search(source, args.skill or "")
        print(f"Маркет · найдено {len(offers)}")
        for offer in offers:
            print(f"{offer.name} {offer.version} · {visible_line(offer.description)}")
    elif args.market_action == "evaluate":
        if not args.skill:
            raise ConfigurationError("Укажите имя скилла после market evaluate")
        load_env(args.env_file)
        settings = Settings.from_env(args.provider, args.model)
        _, events = market.check()
        for event in events:
            show_market_event(event)
        options = EvalOptions(timeout=args.timeout, max_rounds=args.max_rounds, max_tokens=args.max_tokens)
        status, decision = evaluate_candidate(market, args.skill, settings, options,
                                              focus=tuple(args.focus), output=args.output)
        if decision:
            print("Маркет · " + ("обновление принято" if decision["status"] == "accepted" else "обновление отклонено"))
            for reason in decision["reasons"]:
                print("  " + visible_line(reason))
            print(f"Решение · {Path(decision['output']) / 'decision.json'}")
        return status
    else:
        _, events = market.check()
        for event in events:
            show_market_event(event)
        if "revision" in events[0]:
            check = events[0]
            print(f"Anthropic · Git {check['revision'][:12]} · карточек {check['count']}"
                  f" · пропущено {len(check['skipped'])}")
        if events[0]["status"] == "unchanged":
            print("Маркет · каталог без изменений")
        if any(event.get("status") == "error" for event in events):
            return 1
    return 0


def event_listener(stop_after_tools=None, stop_after_tasks=None, *, verbose=False, chat=False, progress=None):
    def render(event):
        if event["event"].startswith("market_"):
            show_market_event(event)
            return
        if progress and progress.mode != "off" and progress.render(event):
            return
        if verbose and event["event"] not in {"tool_call", "tool_result"}:
            if progress:
                progress.finish_line()
            show_event(event)
        else:
            show_compact_event(event, chat=chat)

    if stop_after_tools is None and stop_after_tasks is None:
        return render
    completed, tasks_completed = 0, 0

    def listener(event):
        nonlocal completed, tasks_completed
        render(event)
        if event["event"] == "tool_result":
            completed += 1
            if stop_after_tools is not None and completed >= stop_after_tools:
                if progress:
                    progress.finish_line()
                print("Пауза после сохранённого результата инструмента.", flush=True)
                raise KeyboardInterrupt
        if event["event"] == "task_completed":
            tasks_completed += 1
            if stop_after_tasks is not None and tasks_completed >= stop_after_tasks:
                print("Пауза после сохранённого результата задачи.", flush=True)
                raise KeyboardInterrupt

    return listener


def load_policy(path: Path | None, workspace: Path) -> Policy:
    if path is None:
        return Policy.from_dict({"defaults": {"Bash": "ask"}})
    if path.resolve().is_relative_to(workspace.resolve()):
        raise PolicyError("Keep policy files outside the model workspace")
    return Policy.load(path)

def setup(provider: str, model: str | None, path: Path) -> None:
    if path.exists():
        raise ConfigurationError(f"{path} already exists; edit it locally to preserve existing keys")
    default_url, key_name, default_model = PROVIDERS[provider]
    chosen_model = model or default_model
    if not chosen_model:
        raise ConfigurationError("Specify the model ID with --model")
    values = {**initial_launch_values(), "AGENT_PROVIDER": provider, f"{provider.upper()}_MODEL": chosen_model,
              f"{provider.upper()}_BASE_URL": default_url}
    if key_name:
        if not sys.stdin.isatty():
            raise ConfigurationError("Run setup directly in your terminal for hidden key input")
        key = getpass.getpass(f"{key_name} (скрытый ввод): ")
        if not key or any(c in key for c in "\n\r\x00"):
            raise ConfigurationError("Invalid API key")
        values[key_name] = key
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("\n".join(f"{k}={v}" for k, v in values.items()) + "\n")
    print(f"Настройки сохранены в {path}; ключ не отображается.")

def doctor(settings: Settings) -> bool:
    print(f"Provider: {settings.provider}; model: {settings.model}")
    success = True
    with tempfile.TemporaryDirectory() as temporary:
        try:
            result = ToolRuntime(temporary).bash("pwd")
            success = result.startswith("exit_code=0\n")
            print("Bash sandbox: " + ("OK" if success else result))
        except ToolError as exc:
            success = False
            print(f"Bash sandbox: {exc}")
    if settings.provider == "ollama":
        try:
            with urlopen(settings.base_url.removesuffix("/v1") + "/api/tags", timeout=5) as response:
                names = [m["name"] for m in json.load(response)["models"]]
            print("Ollama models: " + ", ".join(names))
            if settings.model not in names:
                success = False
                print(f"Загрузите модель: ollama pull {settings.model}")
        except (OSError, ValueError, KeyError):
            success = False
            print("Ollama недоступна: установите приложение и запустите ollama serve")
    else:
        print("API key: configured (network request not made)")
    return success

def open_skills(workspace: Path, directory: Path | None) -> SkillCatalog:
    catalog = SkillCatalog(workspace.expanduser(), directory)
    for name, reason in catalog.errors:
        print(f"Скилл пропущен · {visible_line(name)} · {visible_line(reason)}", file=sys.stderr)
    return catalog


def saved_runtime(state, db_path: Path, approvals=None, on_event=show_event, *,
                  allow_policy_change=False, verbose=False, load_extensions=False):
    metadata = state.metadata
    workspace = Path(metadata["workspace"])
    if db_path.resolve().is_relative_to(workspace.resolve()):
        raise SessionError("Session database must be outside workspace")
    policy_path = metadata.get("policy_path")
    checkpoint_policy = Policy.restore(metadata["policy"], builtin=metadata.get("policy_source") == "default")
    if policy_path:
        policy = load_policy(Path(policy_path) if policy_path else None, workspace)
    else:
        policy = checkpoint_policy
    if not allow_policy_change and policy.as_dict() != checkpoint_policy.as_dict():
        raise SessionError("Policy changed since the checkpoint; continuation was refused")
    limits = metadata.get("runtime_limits", {})
    if set(limits) != {"command_timeout", "max_output_chars", "max_file_chars"}:
        raise SessionError("Invalid saved runtime limits")
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in limits.values()) or
        any(type(limits[k]) is not int for k in ("max_output_chars", "max_file_chars"))):
        raise SessionError("Invalid saved runtime limits")
    mode = approvals or metadata.get("approvals", "ask")
    options = metadata.get("skill_config")
    skills = None
    if load_extensions and options is not None:
        if not isinstance(options, dict) or set(options) != {"directory"}:
            raise SessionError("Не удалось прочитать настройки каталога скиллов")
        directory = options["directory"]
        if directory is not None and (not isinstance(directory, str) or not Path(directory).is_absolute()):
            raise SessionError("Некорректный путь сохранённого каталога скиллов")
        skills = open_skills(workspace, Path(directory) if directory is not None else None)
    market = None
    market_config = metadata.get("market_config")
    if load_extensions:
        project_skills = metadata.get("project_skills", True)
        if type(project_skills) is not bool or (market_config is not None and (
                skills is None or not project_skills or not isinstance(market_config, str)
                or (market_config != ANTHROPIC_MARKET and not Path(market_config).is_absolute()))):
            raise SessionError("Не удалось прочитать настройки маркета скиллов")
        if skills is not None and project_skills:
            config = market_config_argument(market_config) if market_config is not None else None
            market = SkillMarket(config, db_path, workspace)
    mcp = None
    mcp_config = metadata.get("mcp_config")
    if load_extensions and mcp_config is not None:
        if (not isinstance(mcp_config, dict) or not isinstance(mcp_config.get("path"), str)
                or not Path(mcp_config["path"]).is_absolute() or not isinstance(mcp_config.get("digest"), str)):
            raise SessionError("Не удалось прочитать настройки MCP")
        mcp = MCPRegistry(Path(mcp_config["path"]), workspace, expected_digest=mcp_config["digest"])
    return ToolRuntime(workspace, policy=policy, mode=metadata.get("execution_mode", "safe"),
                       skills=skills, market=market, mcp=mcp,
                       trash_dir=metadata.get("trash_dir", str(db_path.resolve().parent / "trash")),
                       approve=partial(confirm_action, verbose=verbose) if mode == "ask" else None,
                       on_policy_event=on_event, **limits)

def verify_applied(runtime, item):
    try:
        arguments = json.loads(item["call"]["function"]["arguments"])
        expected = arguments["content"].encode("utf-8")
        path = runtime._safe_path(arguments["path"])
        with open_read_file(runtime.root, path) as descriptor:
            if os.fstat(descriptor).st_size != len(expected):
                return False
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                return stream.read(len(expected) + 1) == expected
    except (ValueError, KeyError, TypeError, AttributeError, ToolError, OSError):
        return False

def can_resume(metadata: dict) -> bool:
    return metadata.get("kind", "run") == "run" and metadata.get("provider") in PROVIDERS


def session_command(args):
    with SessionStore(args.state_db) as store:
        if args.action == "events":
            for record in store.event_records(args.session_id):
                print(json_text(record))
            return 0
        if args.action == "sessions":
            print(json_text(store.list_sessions(), indent=2))
            return 0
        if args.action == "resume":
            args.session_id = choose_session(store, args.session_id, eligible=can_resume)
        if args.action in {"changes","rollback"}:
            args.session_id = choose_session(
                store, args.session_id, include_completed=True, journal_only=True,
                eligible=(lambda metadata: metadata.get("kind") != "chat-turn") if args.action == "rollback" else None,
            )
        if args.action == "cost":
            args.session_id = choose_session(store, args.session_id, include_completed=True)
        state = store.get(args.session_id)
        if state.metadata.get("kind") == "chat-turn" and args.action in {"resume", "rollback"}:
            row = store.db.execute("SELECT conversation_id FROM conversation_turns WHERE session_id=?",
                                   (state.id,)).fetchone()
            command = "/resume" if args.action == "resume" else "/rollback"
            if row is None:
                raise SessionError("Запрос не связан с диалогом")
            raise SessionError(f"Откройте диалог через agi chat --resume {row['conversation_id']} и введите {command}")
        if args.action == "cost":
            print(json_text({"id": state.id, "provider": state.metadata.get("provider"), "model": state.metadata.get("model"), "cost": store.cost(state.id)}, indent=2))
            return 0
        if args.action in {"changes","rollback"}:
            if not state.metadata.get("journal_mode",False):
                raise JournalError("This session was created without a change journal")
            runtime = saved_runtime(state, store.path, getattr(args, "approvals", None),
                                    allow_policy_change=args.action == "rollback",
                                    verbose=getattr(args, "verbose", False))
            journal=ChangeJournal(store,args.session_id,runtime)
            if args.action=="rollback":
                print(json_text(journal.rollback(), indent=2))
            else:
                with store.lock(args.session_id):
                    print(f"Сессия: {args.session_id}; журнал: {journal.head()['status']}")
                    print(color_diff(journal.diff()))
                    current,_=journal.capture();expected=journal.image(journal.head()['revision'])
                    if current!=expected:
                        view=journal.recovery_view({'key':'observed'})
                        print('Текущее дерево отличается от сохранённого:\n' + color_diff(view['diff']))
            return 0
        if args.action == "session":
            calls = []
            for item in store.calls(args.session_id, True):
                function = item["call"]["function"]
                try:
                    arguments = json.loads(function["arguments"])
                    target = arguments.get("path", arguments.get("command", ".")) if isinstance(arguments, dict) else "invalid arguments"
                except ValueError:
                    target = "invalid arguments"
                calls.append({"key": item["key"], "state": item["state"], "tool": function["name"], "target": target, "task_id": item.get("task_id"), "phase": item.get("phase")})
            print(json_text({"id": state.id, "status": state.status, "workspace": state.metadata["workspace"],
                "provider": state.metadata.get("provider"), "model": state.metadata.get("model"),
                "rounds": state.rounds, "max_rounds": state.max_rounds, "error": state.error,
                "mode": state.metadata.get("execution_mode", "safe"),
                "budget": store.budget(args.session_id), "cost": store.cost(args.session_id), "routing": state.metadata.get("routing"),
                "tasks": store.tasks(args.session_id), "plan": store.plan(args.session_id), "unfinished_calls": calls}, indent=2))
            return 0
        if args.action in {"approve-plan", "reject-plan"}:
            store.decide_plan(args.session_id, args.action == "approve-plan")
            print("Решение по плану сохранено." + (" Продолжение: agi resume" if args.action == "approve-plan" else " Сессия завершена."))
            return 0
        progress = ProgressDisplay(getattr(args, "progress", "off"))
        listener = event_listener(getattr(args, "stop_after_tools", None), getattr(args, "stop_after_tasks", None),
                                  verbose=getattr(args, "verbose", False),
                                  progress=progress)
        if args.action == "resolve":
            verifier = None
            journal = None
            if state.metadata.get("journal_mode"):
                runtime=saved_runtime(state,store.path,on_event=listener)
                journal=ChangeJournal(store,args.session_id,runtime)
                item=next((i for i in store.calls(args.session_id,True) if i['key']==args.call),None)
                if item is None:raise SessionError('Uncertain call key was not found')
                view=journal.recovery_view(item)
                journal.reviewed_tree=view['_current']
                print('Обнаруженные изменения:\n'+visible(view['diff']))
            if args.resolution == "applied":
                runtime = saved_runtime(state, store.path, on_event=listener)
                verifier = lambda item: verify_applied(runtime, item)
            store.resolve(args.session_id, args.call, args.resolution, verifier, journal=journal, changes=getattr(args,"changes",None))
            if args.resolution == "abort":
                print("Сессия завершена с ошибкой по решению пользователя. Продолжение запрещено.")
                return 1
            print("Решение сохранено. Продолжение: agi resume " + args.session_id)
            return 0
        if state.status == "budget_exhausted":
            raise SessionError(state.error or "BUDGET_EXHAUSTED: saved budget was exhausted")
        if state.status == "aborted":
            raise SessionError("This session was terminated by the operator and cannot be resumed")
        metadata = state.metadata
        if not can_resume(metadata):
            raise SessionError("Эту сессию можно просмотреть, но её запуск больше не поддерживается")
        runtime = saved_runtime(state, store.path, getattr(args, "approvals", None), listener,
                                verbose=args.verbose, load_extensions=True)
        load_env(args.env_file)
        settings = Settings.from_env(metadata["provider"], metadata["model"])
        client = new_client(settings, metadata["max_tokens"], args)
        client.routing = metadata.get("routing")
        agent = CodingAgent(client, runtime, store=store, on_event=listener,
                            **interactive_options(runtime, args))
        if not args.verbose:
            report_header("resume", runtime.root, settings)
        with progress:
            answer = agent.resume(args.session_id)
        report_answer(agent, answer, verbose=args.verbose)
        return 0

def catalog_for(args, provider):
    db = getattr(args, "state_db", None) or Path(os.getenv("AGENT_STATE_DB", ".agent/sessions.sqlite3"))
    cache = Path(db).resolve().parent
    workspace = getattr(args, "workspace", None)
    if workspace is not None and cache.is_relative_to(Path(workspace).resolve()):
        raise ConfigurationError("Каталог и база сессий должны храниться вне workspace")
    base_url = Settings.from_env("ollama").base_url if provider == "ollama" else None
    return ModelCatalog(provider, cache, base_url=base_url).load(refresh=getattr(args, "refresh", False))


def choose_settings(args):
    settings = Settings.from_env(args.provider, args.model or ("auto" if getattr(args, "route", False) else None))
    choice = settings.model
    route = getattr(args, "route", False) or choice == "auto"
    if args.model and args.model not in {"?", "auto"}:
        route = False  # An explicit model wins over a routing default in .env.
    if choice == "?":
        if not sys.stdin.isatty():
            raise PricingError("Выбор --model '?' требует терминала; укажите ID модели или --model auto")
        catalog = catalog_for(args, settings.provider)
        selected = pick_model(catalog, max_tokens=getattr(args, "max_tokens", 2048),
                              interactive=True, read=input, show=lambda text: print(visible(text)))
        args._catalog = catalog
        return replace(settings, model=selected.id)
    if route:
        catalog = catalog_for(args, settings.provider)
        request = " ".join(getattr(args, "request", ())) or "Проверь код"
        selected, result = choose_route(catalog, request, max_tokens=getattr(args, "max_tokens", 2048),
            tasks=getattr(args, "tasks", False), plan=getattr(args, "plan", False),
            overrides={level: getattr(args, f"route_{level}_model", None) for level in ("simple", "medium", "complex")})
        args._catalog, args._routing = catalog, result
        print(visible(f"Выбрана модель · {selected.id}"))
        if getattr(args, "verbose", False):
            print(visible("Роутинг · " + ", ".join(result['reasons'])))
        return replace(settings, model=selected.id)
    return settings


def configure_client(client, settings, args):
    if getattr(args, "_routing", None):
        client.routing = args._routing
    if not getattr(type(client), "billing_enabled", False):
        return client
    if settings.provider == "ollama":
        client.price = Price('0', '0', source=settings.base_url)
        if getattr(args, "verbose", False):
            print("Ollama: оплата локального API $0; расходы компьютера не учитываются.")
        return client
    # A custom compatibility server need not bill at the official provider rates.
    if settings.provider != "ollama" and settings.base_url != PROVIDERS[settings.provider][0]:
        if getattr(args, "verbose", False):
            print("Пользовательский API endpoint: тариф неизвестен; списание учитывается только из ответа провайдера.")
        return client
    try:
        catalog = getattr(args, "_catalog", None) or catalog_for(args, settings.provider)
        model = catalog.find(settings.model)
        client.price = model.price if model else None
        client.context_window = model.context if model else None
        if getattr(args, "verbose", False):
            print(visible(f"Тарифы: {catalog.source}; обновлены {catalog.updated_at}; {catalog.status}."))
            if catalog.warning:
                print(visible(catalog.warning))
            if client.price is None:
                print("Тариф выбранной модели неизвестен; фактическое списание будет учтено, если провайдер его вернёт.")
    except PricingError:
        if getattr(args, "verbose", False):
            print("Тарифы недоступны: оценка неизвестна; фактическое списание будет учтено из ответа, если доступно.")
    return client


def new_client(settings, max_tokens, args):
    timeout = {"ollama": 180, "openrouter": 150}.get(settings.provider, 90)
    client = OpenAICompatibleClient(settings, timeout=timeout, max_tokens=max_tokens)
    return configure_client(client, settings, args)


def interactive_options(runtime, args):
    return {
        "approve_plan": partial(confirm_plan, verbose=getattr(args, "verbose", False)),
        "cost_estimate": args.cost_estimate,
        "resolve_uncertain": lambda item: confirm_conflict(runtime, item, lambda i: verify_applied(runtime, i)),
        "verify_recovery": lambda item: verify_applied(runtime, item),
        "attribute_changes": confirm_attribution,
    }


def starting_runtime(args, listener):
    workspace = args.workspace or Path("workspace")
    if args.state_db.expanduser().resolve().is_relative_to(workspace.expanduser().resolve()):
        raise SessionError("База сессий должна находиться вне рабочей папки")
    market_config = getattr(args, "market_config", None)
    if market_config is not None and args.skills is False:
        raise ConfigurationError("Для --market-config нужен включённый каталог скиллов")
    demo_only = getattr(args, "demo_skills_only", False)
    if demo_only and (args.skills is False or args.skills_dir is not None or market_config is not None):
        raise ConfigurationError("--demo-skills-only несовместим с --no-skills, --skills-dir и --market-config")
    return ToolRuntime(workspace, policy=load_policy(args.policy, workspace), mode=args.mode,
                       skills=open_skills(workspace, args.skills_dir) if args.skills is not False else None,
                       market=SkillMarket(market_config or ANTHROPIC_MARKET, args.state_db, workspace)
                       if args.skills is not False and not demo_only else None,
                       mcp=MCPRegistry(args.mcp_config, workspace) if args.mcp_config is not None else None,
                       trash_dir=args.state_db.expanduser().resolve().parent / "trash",
                       approve=partial(confirm_action, verbose=args.verbose) if args.approvals == "ask" else None,
                       on_policy_event=listener)


def new_agent(args, store, runtime, listener, budget_limits, *, chat=False):
    settings = choose_settings(args)
    client = new_client(settings, args.max_tokens, args)
    metadata = {"kind": "chat-turn" if chat else "run",
                "policy_path": str(args.policy.resolve()) if args.policy else None,
                "policy_source": "file" if args.policy else "default", "approvals": args.approvals,
                "max_tokens": args.max_tokens}
    return CodingAgent(client, runtime, max_rounds=args.max_rounds,
                       max_context_chars=4_000_000 if chat else 120_000, on_event=listener,
                       store=store, session_metadata=metadata, plan_mode=args.plan, task_mode=args.tasks,
                       budget_limits=budget_limits, journal_mode=chat or args.journal,
                       **interactive_options(runtime, args))


def restore_chat_agent(args, store, listener, configuration):
    binding = configuration["binding"]
    args.workspace = Path(binding["workspace"])
    settings = Settings.from_env(binding["provider"], binding["model"])
    if settings.base_url != binding["base_url"]:
        raise ConfigurationError("Адрес модели изменился. Откройте диалог с прежним файлом настроек")
    if args.cost_estimate is None:
        args.cost_estimate = configuration["cost_estimate"]
    runtime = saved_runtime(SimpleNamespace(metadata={"skill_config": {"directory": None}, **binding}),
                            store.path, on_event=listener, verbose=args.verbose, load_extensions=True)
    client = new_client(settings, binding["max_tokens"], args)
    if getattr(client, "context_window", None) is None:
        client.context_window = configuration["context_window"]
    client.routing = binding.get("routing")
    limits = BudgetLimits.restore(configuration["budget"])
    metadata = {key: binding[key] for key in ("kind", "policy_path", "policy_source", "approvals", "max_tokens")}
    return CodingAgent(client, runtime, max_rounds=limits.model_requests, max_context_chars=4_000_000,
                       on_event=listener, store=store, session_metadata=metadata, plan_mode=binding["plan_mode"],
                       task_mode=binding["task_mode"], budget_limits=limits, journal_mode=True,
                       **interactive_options(runtime, args))


def chat_command(args, limits):
    if not sys.stdin.isatty():
        raise ConfigurationError("Для chat нужен интерактивный терминал")
    progress = ProgressDisplay(args.progress or "compact")
    listener = event_listener(verbose=args.verbose, chat=True, progress=progress)
    runtime = starting_runtime(args, listener) if args.resume_chat is None else None
    with SessionStore(args.state_db) as store:
        if args.resume_chat == "":
            args.resume_chat = Conversation.choose_open(store)
            if args.resume_chat is None:
                return 0
        if args.resume_chat is not None:
            conversation = Conversation(store, args.resume_chat)
        else:
            agent = new_agent(args, store, runtime, listener, limits, chat=True)
            conversation = Conversation.create(store, {
                "binding": agent.binding(), "budget": limits.as_dict(), "cost_estimate": args.cost_estimate,
                "context_window": getattr(agent.client, "context_window", None),
                "progress": progress.mode,
            })
        with store.lock(conversation.id):
            if args.resume_chat is not None:
                configuration = conversation.configuration()
                if args.progress is None:
                    saved = configuration.get("progress")
                    if saved is None:
                        saved = os.getenv("AGENT_PROGRESS", "compact")
                    progress.mode = option_value("progress", saved)
                agent = restore_chat_agent(args, store, listener, configuration)
            if agent.client.settings.provider == "openrouter" and not getattr(agent.client, "context_window", None):
                print("Размер окна модели недоступен · проверка по лимиту OpenRouter пропущена.", file=sys.stderr)
            resume_args = ["agi", "chat", "--resume", conversation.id,
                           "--env-file", str(args.env_file.resolve()), "--state-db", str(store.path)]
            if args.verbose:
                resume_args.append("--verbose")
            resume_command = shlex.join(resume_args)
            shell = ConversationShell(conversation, agent,
                                      restore_runtime=partial(saved_runtime, db_path=store.path, verbose=args.verbose),
                                      report_answer=partial(report_answer, verbose=args.verbose, chat=True),
                                      resume_command=resume_command,
                                      approve_action=partial(confirm_action, verbose=args.verbose),
                                      progress=progress)
            return shell.run(" ".join(getattr(args, "request", ())))


def show_models(args):
    if not 1 <= args.limit <= 5000 or min(args.input_tokens, args.output_tokens) < 0:
        raise ConfigurationError("--limit: 1..5000; --input-tokens и --output-tokens: минимум 0")
    load_env(args.env_file)
    provider = args.provider or os.getenv("AGENT_PROVIDER", "openrouter")
    if provider not in PROVIDERS:
        raise ConfigurationError("Неизвестный провайдер для сравнения моделей")
    catalog = catalog_for(args, provider)
    models = [m for m in catalog.models if args.search.lower() in m.id.lower()]
    if args.free:
        models = [m for m in models if m.price is not None and amount(m.price.input) == 0 and amount(m.price.output) == 0]
    elif args.paid:
        models = [m for m in models if m.price is not None and (amount(m.price.input) > 0 or amount(m.price.output) > 0)]
    def total(model):
        value = model.price.calculate(args.input_tokens, args.output_tokens) if model.price else None
        return value if value is not None else Decimal('Infinity')
    models.sort(key=(lambda m: (not m.tools, total(m), m.id)) if args.sort == "price" else (lambda m: (not m.tools, m.id)))
    matched = len(models)
    if not args.all:
        models = models[:args.limit]
    if args.json:
        print(json_text({'provider': provider, 'currency': 'USD', 'source': catalog.source,
            'updated_at': catalog.updated_at, 'status': catalog.status, 'warning': catalog.warning,
            'input_tokens': args.input_tokens, 'output_tokens': args.output_tokens,
            'matched': matched, 'shown': len(models), 'sort': args.sort,
            'models': [{**asdict(m), 'scenario_usd': str(total(m)) if total(m).is_finite() else None} for m in models]}, indent=2))
    else:
        print(visible(f"Тарифы: {catalog.source}; обновлены {catalog.updated_at}; {catalog.status}."))
        if catalog.warning:
            print(catalog.warning)
        print(f"Сценарий сравнения: {args.input_tokens} входных + {args.output_tokens} выходных токенов, без кэша.")
        print(comparison(models, args.input_tokens, args.output_tokens, numbered=True))
        print(f"Показано {len(models)} из {matched}; сортировка: {'по цене' if args.sort == 'price' else 'по имени'}. Полный список: models --all.")
        if provider == "openai":
            print("Официальные тарифы Standard; доступность модели для аккаунта проверяется при запуске.")
        if provider == "ollama":
            print("Локальная Ollama: оплата API равна нулю; расходы компьютера в расчёт не входят.")
    return 0


def add_budget_arguments(command):
    command.add_argument("--max-tool-calls", type=int, default=None, help="Total tool-call budget for all run phases")
    command.add_argument("--max-side-effects", type=int, default=None, help="Total write/delete/Bash attempts; zero permits reading only")
    command.add_argument("--max-result-bytes", type=int, default=None, help="Total UTF-8 bytes of tool results")


def add_progress_option(command):
    command.add_argument("--progress", nargs="?", const="compact", choices=LAUNCH_OPTIONS["progress"].choices)
    command.add_argument("--no-progress", dest="progress", action="store_const", const="off")


def progress_argv(argv):
    # An existing bare --progress may be followed by a positional prompt or session ID.
    result = list(argv)
    for index, token in enumerate(argv):
        if token == "--":
            break
        following = argv[index + 1] if index + 1 < len(argv) else None
        if token == "--progress" and following not in LAUNCH_OPTIONS["progress"].choices:
            result[index] = "--progress=compact"
    return result


def main() -> int:
    parser = HelpParser(prog="agi", usage="agi КОМАНДА [ПАРАМЕТРЫ]", description="Агент для работы с кодом",
                        epilog=project_guide())
    subparsers = parser.add_subparsers(dest="action", required=True)
    help_command = subparsers.add_parser("help", help=COMMAND_HELP["help"], description=COMMAND_HELP["help"])
    help_command.add_argument("command", nargs="?", choices=list(COMMAND_HELP))
    eval_command = subparsers.add_parser("eval", help=COMMAND_HELP["eval"], description=COMMAND_HELP["eval"])
    eval_actions = eval_command.add_subparsers(dest="eval_action", required=True, parser_class=argparse.ArgumentParser)
    eval_run = eval_actions.add_parser("run", help="Запустить или продолжить A/B-серию")
    eval_run.add_argument("--provider", choices=list(PROVIDERS), help="Провайдер модели, по умолчанию из настроек")
    eval_run.add_argument("--model", help="Фиксированный ID модели, по умолчанию из настроек")
    eval_run.add_argument("--env-file", type=Path, default=Path(".env"), help="Файл настроек и ключей (по умолчанию .env)")
    eval_run.add_argument("--output", type=Path, default=Path(".agent/evals/study-tracker"), help="Папка всей серии")
    eval_run.add_argument("--trials", type=int, default=3, help="Попыток на сценарий и версию (по умолчанию 3)")
    eval_run.add_argument("--timeout", type=float, default=180, help="Лимит всей попытки в секундах (по умолчанию 180)")
    eval_run.add_argument("--max-rounds", type=int, default=8, help="Лимит обращений к модели на попытку (по умолчанию 8)")
    eval_run.add_argument("--max-tokens", type=int, default=2048, help="Лимит выходных токенов на ответ (по умолчанию 2048)")
    eval_report = eval_actions.add_parser("report", help="Обновить summary.md по сохранённым результатам")
    eval_report.add_argument("--output", type=Path, default=Path(".agent/evals/study-tracker"), help="Папка сохранённой серии")
    market = subparsers.add_parser("market", help=COMMAND_HELP["market"])
    market.add_argument("market_action", choices=("check", "search", "status", "events", "candidates", "evaluate"))
    market.add_argument("skill", nargs="?", help="Запрос для search или имя скилла для evaluate")
    market.add_argument("--market-config", type=market_config_argument,
                        help="Путь к JSON-конфигурации или anthropic (по умолчанию)")
    market.add_argument("--workspace", type=Path, default=Path("workspace"))
    market.add_argument("--state-db", type=Path)
    market.add_argument("--env-file", type=Path, default=Path(".env"))
    market.add_argument("--provider", choices=list(PROVIDERS))
    market.add_argument("--model")
    market.add_argument("--output", type=Path, help="Папка сравнения версий")
    market.add_argument("--focus", action="append", default=[], choices=[case.id for case in CASES],
                        help="Сценарий, где обе версии проверяются с явной загрузкой скилла")
    market.add_argument("--timeout", type=float, default=180)
    market.add_argument("--max-rounds", type=int, default=8)
    market.add_argument("--max-tokens", type=int, default=2048)
    set_command = subparsers.add_parser("set", help=COMMAND_HELP["set"], description=COMMAND_HELP["set"])
    set_command.add_argument("--provider", choices=list(PROVIDERS))
    set_command.add_argument("--model")
    set_command.add_argument("--env-file", type=Path, default=Path(".env"))
    for name, option in LAUNCH_OPTIONS.items():
        if name == "progress":
            add_progress_option(set_command)
            continue
        flags = {"default": None}
        if option.kind is bool:
            flags["action"] = argparse.BooleanOptionalAction
        else:
            flags["type"] = option.kind
            if option.choices:
                flags["choices"] = option.choices
        set_command.add_argument("--" + name.replace("_", "-"), **flags)
    for action in ("sessions", "session", "events", "resume", "resolve", "approve-plan", "reject-plan", "changes", "rollback", "cost"):
        command = subparsers.add_parser(action, help=COMMAND_HELP[action], description=COMMAND_HELP[action])
        command.add_argument("--state-db", type=Path, default=None)
        if action != "sessions":
            command.add_argument("session_id", nargs="?" if action in {"resume","changes","rollback","cost"} else None)
        command.add_argument("--env-file", type=Path, default=Path(".env"))
        if action == "resume":
            command.add_argument("--verbose", action="store_true")
            add_progress_option(command)
            command.add_argument("--cost-estimate", action=argparse.BooleanOptionalAction, default=None)
            command.add_argument("--approvals", choices=["ask", "deny"], default=None)
            command.add_argument("--stop-after-tools", type=int, default=None)
            command.add_argument("--stop-after-tasks", type=int, default=None)
        if action == "rollback":
            command.add_argument("--approvals", choices=["ask", "deny"], default=None)
            command.add_argument("--verbose", action="store_true")
        if action == "resolve":
            command.add_argument("--changes",choices=["own","external"],help="Explicit attribution of all observed changes after a crash")
            command.add_argument("--call", required=True, help="Uncertain call key, for example 1:0")
            command.add_argument("--resolution", choices=["applied", "retry", "skip", "abort"], required=True)
    for action in ("setup", "doctor", "run", "plan", "chat"):
        command = subparsers.add_parser(action, help=COMMAND_HELP[action], description=COMMAND_HELP[action])
        command.add_argument("--provider", choices=list(PROVIDERS))
        command.add_argument("--model")
        command.add_argument("--env-file", type=Path, default=Path(".env"))
        if action == "doctor":
            doctor_actions = command.add_subparsers(dest="doctor_action", parser_class=argparse.ArgumentParser)
            session_report = doctor_actions.add_parser("session", help="Разобрать сохранённый диалог или запуск")
            session_report.add_argument("session_id", help="ID диалога или запуска")
            session_report.add_argument("--state-db", type=Path, help="Путь к базе SQLite")
            session_report.add_argument("--env-file", type=Path, default=argparse.SUPPRESS, help="Файл настроек")
            session_report.add_argument("--json", action="store_true", help="Вывести отчёт в JSON")
        if action in START_COMMANDS:
            command.add_argument("--verbose", action="store_true")
            add_progress_option(command)
            command.add_argument("--skills", action=argparse.BooleanOptionalAction, default=None,
                                 help="Включить каталог скиллов (по умолчанию включён)")
            command.add_argument("--skills-dir", type=Path, help="Папка дополнительных SKILL.md вне рабочей папки")
            command.add_argument("--demo-skills-only", action="store_true", default=None,
                                 help="Включить только три встроенных демо-скилла")
            command.add_argument("--market-config", type=market_config_argument,
                                 help="Путь к JSON-конфигурации вне рабочей папки или anthropic (по умолчанию)")
            command.add_argument("--mcp-config", type=Path, help="JSON-конфигурация локальных MCP-серверов вне рабочей папки")
            if action == "chat":
                command.add_argument("--resume", dest="resume_chat", metavar="ID", nargs="?", const="")
            else:
                command.add_argument("--journal", action=argparse.BooleanOptionalAction, default=None)
            add_budget_arguments(command)
            command.add_argument("--tasks", action=argparse.BooleanOptionalAction, default=None, help="Plan with dependency tasks and runtime verification")
            if action != "chat":
                command.add_argument("--stop-after-tasks", type=int)
                command.add_argument("--stop-after-tools", type=int)
            command.add_argument("--plan", action="store_true" if action == "plan" else argparse.BooleanOptionalAction, default=None, help="Read-only planning before explicit approval")
            command.add_argument("--mode", choices=["safe", "auto"], default=None)
            command.add_argument("--state-db", type=Path, default=None)
            command.add_argument("--workspace", type=Path, default=None)
            command.add_argument("--max-rounds", type=int, default=None)
            command.add_argument("--max-tokens", type=int, default=None)
            command.add_argument("--policy", type=Path, help="Trusted JSON policy outside workspace")
            command.add_argument("--approvals", choices=["ask", "deny"], default=None,
                                 help="Prompt for ask decisions, or refuse them without prompting")
            command.add_argument("request", nargs="*" if action == "chat" else "+")
    for name in START_COMMANDS:
        command = subparsers.choices[name]
        command.add_argument("--route", action=argparse.BooleanOptionalAction, default=None)
        command.add_argument("--cost-estimate", action=argparse.BooleanOptionalAction, default=None)
        for level in ("simple", "medium", "complex"):
            command.add_argument(f"--route-{level}-model")
    models_command = subparsers.add_parser("models", help=COMMAND_HELP["models"], description=COMMAND_HELP["models"])
    models_command.add_argument("--provider", choices=list(PROVIDERS))
    models_command.add_argument("--env-file", type=Path, default=Path(".env"))
    models_command.add_argument("--state-db", type=Path, default=None)
    models_command.add_argument("--refresh", action="store_true")
    models_command.add_argument("--search", default="")
    models_command.add_argument("--limit", type=int, default=50)
    models_command.add_argument("--all", action="store_true")
    models_command.add_argument("--sort", choices=["name", "price"], default="name")
    price_filter = models_command.add_mutually_exclusive_group()
    price_filter.add_argument("--paid", action="store_true")
    price_filter.add_argument("--free", action="store_true")
    models_command.add_argument("--input-tokens", type=int, default=1000)
    models_command.add_argument("--output-tokens", type=int, default=1000)
    models_command.add_argument("--json", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args(progress_argv(sys.argv[1:]))
    if args.action == "help":
        target = subparsers.choices[args.command] if args.command else parser
        target.print_help()
        return 0
    try:
        validate_explicit_options(args)
        if args.action == "doctor" and args.doctor_action == "session":
            if args.state_db is None:
                load_launch_env(args.env_file)
                args.state_db = option_value("state_db", os.getenv("AGENT_STATE_DB", ".agent/sessions.sqlite3"))
            show_session_report(args.state_db, args.session_id, as_json=args.json)
            return 0
        if args.action == "market":
            if args.state_db is None:
                load_launch_env(args.env_file)
                args.state_db = option_value("state_db", os.getenv("AGENT_STATE_DB", ".agent/sessions.sqlite3"))
            return market_command(args)
        if args.action == "eval":
            if args.eval_action == "report":
                rebuild_summary(args.output)
                return 0
            load_env(args.env_file)
            settings = Settings.from_env(args.provider, args.model)
            status = run_evaluation(args.output, settings, EvalOptions(args.trials, args.timeout, args.max_rounds, args.max_tokens))
            if status == 130:
                print("Серия приостановлена. Повторите ту же команду eval run.", file=sys.stderr)
            return status
        if args.action == "set":
            updates = set_options(args)
            print(f"Настройки сохранены в {args.env_file}: " + ", ".join(updates))
            print("Применяются к новым запускам; начатые сессии сохраняют прежние параметры.")
            return 0
        if args.action == "resume" and args.session_id == "chat":
            if args.stop_after_tools is not None or args.stop_after_tasks is not None:
                raise ConfigurationError("Флаги --stop-after-tools и --stop-after-tasks относятся к отдельному запуску")
            args.resume_chat = ""
        if getattr(args, "resume_chat", None) is not None:
            options = set(LAUNCH_OPTIONS) - {"state_db", "cost_estimate", "progress"}
            if any(getattr(args, name, None) is not None
                   for name in options | {"provider", "model", "skills", "skills_dir", "demo_skills_only",
                                          "mcp_config", "market_config"}):
                raise ConfigurationError("Для продолжения диалога используются сохранённые настройки запуска")
            load_env(args.env_file)
            if args.state_db is None:
                args.state_db = option_value("state_db", os.getenv("AGENT_STATE_DB", ".agent/sessions.sqlite3"))
            return chat_command(args, None)
        select_workspace = args.action in START_COMMANDS and args.workspace is None and sys.stdin.isatty()
        if args.action in START_COMMANDS:
            load_env(args.env_file)
            apply_launch_defaults(args)
        elif hasattr(args, "state_db") and args.state_db is None:
            load_env(args.env_file)
            args.state_db = option_value("state_db", os.getenv("AGENT_STATE_DB", ".agent/sessions.sqlite3"))
        if args.action == "resume":
            load_launch_env(args.env_file)
            if args.cost_estimate is None:
                args.cost_estimate = option_value("cost_estimate", os.getenv("AGENT_COST_ESTIMATE", "true"))
            if args.progress is None:
                args.progress = option_value("progress", os.getenv("AGENT_PROGRESS", "compact"))
        if args.action == "models":
            return show_models(args)
        budget_limits = None
        if args.action in START_COMMANDS:
            budget_limits = BudgetLimits(model_requests=args.max_rounds, tool_calls=args.max_tool_calls,
                                         side_effects=args.max_side_effects, result_bytes=args.max_result_bytes)
        if getattr(args, "stop_after_tools", None) is not None and args.stop_after_tools < 1:
            raise ConfigurationError("--stop-after-tools must be positive")
        if getattr(args, "stop_after_tasks", None) is not None and args.stop_after_tasks < 1:
            raise ConfigurationError("--stop-after-tasks must be positive")
        if args.action in {"sessions", "session", "events", "resume", "resolve", "approve-plan", "reject-plan", "changes", "rollback", "cost"}:
            return session_command(args)
        if args.action == "setup":
            setup(args.provider or "openrouter", args.model, args.env_file)
            return 0
        if args.action == "doctor":
            load_env(args.env_file)
            return 0 if doctor(choose_settings(args)) else 1
        if select_workspace:
            args.workspace = choose_workspace(Path("workspace"))
            if args.workspace is None:
                print("\nВыбор рабочей папки отменён.", file=sys.stderr)
                return 130
        if args.action == "chat":
            return chat_command(args, budget_limits)
        progress = ProgressDisplay(args.progress)
        listener = event_listener(args.stop_after_tools, args.stop_after_tasks, verbose=args.verbose, progress=progress)
        runtime = starting_runtime(args, listener)
        with SessionStore(args.state_db) as store:
            agent = new_agent(args, store, runtime, listener, budget_limits)
            request = " ".join(args.request)
            if not args.verbose:
                report_header(args.action, runtime.root, getattr(agent.client, "settings", None), request)
            with progress:
                answer = agent.run(request)
            report_answer(agent, answer, verbose=args.verbose)
        return 0
    except ApprovalDeferred as exc:
        print(f"\nОжидается подтверждение: {exc}", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        message = ("Серия приостановлена. Повторите ту же команду eval run." if args.action == "eval"
                   else "Работа остановлена. Продолжите через agi resume и выберите сессию в терминале.")
        print("\n" + message, file=sys.stderr)
        return 130
    except (ConfigurationError, PricingError, ModelClientError, AgentError, JournalError, BudgetExceeded, ToolError, PolicyError, SessionError, OSError, ValueError) as exc:
        print(f"Ошибка: {visible(exc)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
