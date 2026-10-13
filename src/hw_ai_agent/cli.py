from __future__ import annotations
import argparse
from dataclasses import replace, asdict
from decimal import Decimal
import getpass
import json
import os
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from .agent import AgentError, CodingAgent
from .budget import BudgetLimits, BudgetExceeded
from .client import ModelClientError, OpenAICompatibleClient
from .config import (ConfigurationError, PROVIDERS, Settings, load_env, LAUNCH_OPTIONS,
                     START_COMMANDS, apply_launch_defaults, validate_explicit_options,
                     initial_launch_values, option_value, set_options, load_launch_env)
from .usage import COMMAND_HELP, HelpParser, project_guide
from .pricing import PricingError, Price, usd, amount
from .model_catalog import ModelCatalog, comparison
from .routing import choose_route, pick_model
from .cost_demo import run_cost_demo
from .demo import run_demo, verify_demo
from .tools import ToolError, ToolRuntime
from .filesystem import open_read_file
from .policy import ApprovalDeferred, ApprovalRequest, Policy, PolicyError
from .policy_demo import run_policy_demo
from .session import SessionError, SessionStore, UncertainActionError
from .session_demo import run_session_demo
from .workflow_demo import WorkflowClient, run_workflow_demo, verify_workflow
from .planning import PlanApprovalDeferred
from .terminal import visible, confirm_plan, choose_session, confirm_conflict, confirm_attribution
from .plan_demo import PlanDemoClient, run_plan_demo
from .budget_demo import run_budget_demo
from .journal import ChangeJournal, JournalError
from .tasks import checklist
from .tasks_demo import TasksDemoClient, run_tasks_demo, verify_tasks_demo

def show_event(event: dict) -> None:
    if event["event"] == "model_request":
        print(f"\nМодель: обращение {event['round']}", flush=True)
    elif event["event"] == "model_cost_estimate":
        print(f"Оценка запроса: вход ≈{event['input_tokens_estimate']} токенов, выход 0..{event['output_tokens_limit']}; "
              f"{usd(event['usd_input_estimate'])} .. {usd(event['usd_with_full_output_estimate'])}. Это приблизительный расчёт.", flush=True)
    elif event["event"] == "model_cost":
        label = {"reported": "Списано по данным провайдера", "calculated": "Рассчитано по фактическим токенам", "local": "Оплата API (локально/offline)", "unknown": "Стоимость ответа"}[event['kind']]
        print(f"{label}: {usd(event['usd'])}", flush=True)
    elif event["event"] == "session_cost":
        cost = event['cost']
        print(f"Сессия: учтённая сумма {usd(cost['known_usd'])}; подтверждено провайдером {usd(cost['reported_usd'])}; "
              f"расчёт по токенам {usd(cost['calculated_usd'])}; неизвестных запросов {cost['unknown_requests']}.", flush=True)
    elif event["event"] == "tool_result":
        if event['name'] == 'submit_plan' and not event['content'].startswith('ERROR:'):
            print("План сохранён.", flush=True)
        else:
            print(f"\n[{event['name']}]\n{visible(event['content'])}", flush=True)
    elif event["event"] == "tool_call":
        if event['name'] == 'submit_plan':
            print("Модель подготовила план.", flush=True)
        else:
            prefix = f"Задача {event['task_id']}: " if event.get('task_id') else ''
            print(prefix + f"Вызов {event['name']}: {visible(event['arguments'])}", flush=True)
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

def confirm_action(request: ApprovalRequest) -> bool:
    if not sys.stdin.isatty():
        raise ApprovalDeferred("Подтверждение требует интерактивного терминала; продолжите эту сессию через resume")

    print(f"\nТребуется подтверждение: {request.tool} {visible(request.target)}")
    print(f"Причина: {visible(request.reason)}")
    print(f"SHA-256 запроса: {request.fingerprint}")
    if request.preview:
        print(visible(request.preview))
    try:
        question = (f"Разрешить {request.tool} до конца этой сессии в пределах policy и песочницы? [y/N]: "
                    if request.scope == "session" else "Разрешить только этот вызов? [y/N]: ")
        return input(question).strip().lower() in {"y", "yes", "д", "да"}
    except EOFError:
        raise ApprovalDeferred("Ответ не получен; продолжите эту сессию через resume") from None

def event_listener(stop_after_tools=None, stop_after_tasks=None):
    if stop_after_tools is None and stop_after_tasks is None:
        return show_event
    completed, tasks_completed = 0, 0
    def listener(event):
        nonlocal completed, tasks_completed
        show_event(event)
        if event["event"] == "tool_result":
            completed += 1
            if stop_after_tools is not None and completed >= stop_after_tools:
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

def doctor(settings: Settings, *, check_sandbox=True) -> bool:
    print(f"Provider: {settings.provider}; model: {settings.model}")
    success = True
    if check_sandbox:
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

def report_demo(report):
    print("\nИзменения:\n" + report["diff"])
    print("Независимая проверка:\n" + report["verification"])
    print(f"Тесты неизменны: {report['tests_unchanged']}")
    print(f"Модель сама запустила успешные тесты: {report['model_ran_passing_tests']}")
    print(f"Отчёт: {Path(report['workspace']) / 'demo-report.json'}")
    if report["error"]:
        print(f"Ошибка модели: {report['error']}")
    print("DEMO PASS" if report["success"] else "DEMO FAIL")
    return 0 if report["success"] else 1

def report_workflow(report):
    print("\nНезависимая проверка:\n" + report["verification"])
    print("Запросы разрешений: " + json.dumps(report["permission_requests"], ensure_ascii=False))
    print("Попытки записи: " + json.dumps(report["write_attempts"], ensure_ascii=False))
    print(f"Функциональные тесты: {report['functional_success']}; оптимизация: {report['cleanup_complete']}")
    print("Убраны неиспользуемые файлы: " + json.dumps(report["removed_optional_files"], ensure_ascii=False))
    if report.get("error"):
        print("Ошибка модели: " + report["error"])
    print(f"Отчёт: {Path(report['workspace']) / 'workflow-report.json'}")
    print("WORKFLOW DEMO PASS" if report["success"] else "WORKFLOW DEMO FAIL")
    return 0 if report["success"] else 1

def report_tasks(report):
    print("\nЗадачи:\n" + visible(report["checklist"]))
    print("Независимая проверка:\n" + report["verification"])
    print(f"Тесты неизменны: {report['tests_unchanged']}; вызовы связаны с задачами: {report['task_calls_scoped']}")
    if report["error"]:
        print("Ошибка: " + report["error"])
    print(f"Отчёт: {Path(report['workspace']) / 'tasks-report.json'}")
    print("TASKS DEMO PASS" if report["success"] else "TASKS DEMO FAIL")
    return 0 if report["success"] else 1


def saved_runtime(state, db_path: Path, approvals=None, on_event=show_event, *, allow_policy_change=False):
    metadata = state.metadata
    workspace = Path(metadata["workspace"])
    if db_path.resolve().is_relative_to(workspace.resolve()):
        raise SessionError("Session database must be outside workspace")
    policy_path = metadata.get("policy_path")
    if policy_path:
        policy = load_policy(Path(policy_path) if policy_path else None, workspace)
    else:
        policy = Policy.restore(metadata["policy"])
    if not allow_policy_change and policy.as_dict() != Policy.restore(metadata["policy"]).as_dict():
        raise SessionError("Policy changed since the checkpoint; continuation was refused")
    limits = metadata.get("runtime_limits", {})
    if set(limits) != {"command_timeout", "max_output_chars", "max_file_chars"}:
        raise SessionError("Invalid saved runtime limits")
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in limits.values()) or
        any(type(limits[k]) is not int for k in ("max_output_chars", "max_file_chars"))):
        raise SessionError("Invalid saved runtime limits")
    mode = approvals or metadata.get("approvals", "ask")
    return ToolRuntime(workspace, policy=policy, mode=metadata.get("execution_mode", "safe"),
                       trash_dir=metadata.get("trash_dir", str(db_path.resolve().parent / "trash")),
                       approve=confirm_action if mode == "ask" else None,
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

def session_command(args):
    with SessionStore(args.state_db) as store:
        if args.action == "sessions":
            print(json.dumps(store.list_sessions(), ensure_ascii=False, indent=2))
            return 0
        if args.action == "resume":
            args.session_id = choose_session(store, args.session_id)
        if args.action in {"changes","rollback"}:
            args.session_id=choose_session(store,args.session_id,include_completed=True,journal_only=True)
        if args.action == "cost":
            args.session_id = choose_session(store, args.session_id, include_completed=True)
        state = store.get(args.session_id)
        if args.action == "cost":
            print(json.dumps({"id": state.id, "provider": state.metadata.get("provider"), "model": state.metadata.get("model"), "cost": store.cost(state.id)}, ensure_ascii=False, indent=2))
            return 0
        if args.action in {"changes","rollback"}:
            if not state.metadata.get("journal_mode",False):
                raise JournalError("This session was created without a change journal")
            runtime=saved_runtime(state,store.path,getattr(args,"approvals",None),allow_policy_change=args.action=="rollback")
            journal=ChangeJournal(store,args.session_id,runtime)
            if args.action=="rollback":
                print(json.dumps(journal.rollback(),ensure_ascii=False,indent=2))
            else:
                with store.lock(args.session_id):
                    print(f"Сессия: {args.session_id}; журнал: {journal.head()['status']}")
                    print(visible(journal.diff()))
                    current,_=journal.capture();expected=journal.image(journal.head()['revision'])
                    if current!=expected:
                        view=journal.recovery_view({'key':'observed'})
                        print('Текущее дерево отличается от сохранённого:\n'+visible(view['diff']))
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
            print(json.dumps({"id": state.id, "status": state.status, "workspace": state.metadata["workspace"],
                "provider": state.metadata.get("provider"), "model": state.metadata.get("model"),
                "rounds": state.rounds, "max_rounds": state.max_rounds, "error": state.error,
                "mode": state.metadata.get("execution_mode", "safe"),
                "budget": store.budget(args.session_id), "cost": store.cost(args.session_id), "routing": state.metadata.get("routing"),
                "tasks": store.tasks(args.session_id), "plan": store.plan(args.session_id), "unfinished_calls": calls}, ensure_ascii=False, indent=2))
            return 0
        if args.action in {"approve-plan", "reject-plan"}:
            store.decide_plan(args.session_id, args.action == "approve-plan")
            print("Решение по плану сохранено." + (" Продолжение: hw-ai-agent resume" if args.action == "approve-plan" else " Сессия завершена."))
            return 0
        listener = event_listener(getattr(args, "stop_after_tools", None), getattr(args, "stop_after_tasks", None))
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
            print("Решение сохранено. Продолжение: hw-ai-agent resume " + args.session_id)
            return 0
        if state.status == "budget_exhausted":
            raise SessionError(state.error or "BUDGET_EXHAUSTED: saved budget was exhausted")
        if state.status == "aborted":
            raise SessionError("This session was terminated by the operator and cannot be resumed")
        runtime = saved_runtime(state, store.path, getattr(args, "approvals", None), listener)
        metadata = state.metadata
        interactive = {"approve_plan": confirm_plan, "cost_estimate": getattr(args, "cost_estimate", True),
                       "resolve_uncertain": lambda item: confirm_conflict(runtime, item, lambda i: verify_applied(runtime, i)),
                       "verify_recovery": lambda item: verify_applied(runtime, item), "attribute_changes": confirm_attribution}
        if metadata.get("kind") == "tasks-demo" and metadata.get("provider") not in PROVIDERS:
            if not metadata["demo"].get("offline"):
                raise SessionError("Real task demo cannot resume with a scripted client")
            client = TasksDemoClient()
            client.max_tokens = metadata.get("max_tokens", 3072)
            agent = CodingAgent(client, runtime, store=store, on_event=listener, **interactive)
            answer = agent.resume(args.session_id)
            return report_tasks(verify_tasks_demo(client, runtime, store, args.session_id, answer, metadata))
        if metadata.get("kind") == "plan-demo" and metadata.get("provider") not in PROVIDERS:
            if not metadata["demo"].get("offline", metadata.get("provider") == "scripted"):
                raise SessionError("Real-model plan session cannot resume with a scripted client")
            client = PlanDemoClient()
            client.max_tokens = metadata.get("max_tokens", 2048)
            agent = CodingAgent(client, runtime, store=store, on_event=listener, **interactive)
            answer = agent.resume(args.session_id)
            demo = metadata["demo"]
            return report_demo(verify_demo(client, runtime.root, runtime, demo["baseline"], demo["test_hash"],
                                          answer, agent.events, session_id=args.session_id))
        if metadata.get("kind") == "workflow" and metadata.get("provider") not in PROVIDERS:
            agent = CodingAgent(WorkflowClient(), runtime, store=store, on_event=listener, **interactive)
            agent.resume(args.session_id)
            return report_workflow(verify_workflow(agent, runtime, metadata))
        load_env(args.env_file)
        if metadata.get("provider") not in PROVIDERS:
            raise SessionError("This offline session cannot use a real model through the CLI")
        settings = Settings.from_env(metadata["provider"], metadata["model"])
        client = OpenAICompatibleClient(settings, timeout=180 if settings.provider == "ollama" else 90,
                                        max_tokens=metadata["max_tokens"])
        client = configure_client(client, settings, args)
        client.routing = metadata.get("routing")
        agent = CodingAgent(client, runtime, store=store, on_event=listener, **interactive)
        answer = agent.resume(args.session_id)
        if metadata.get("kind") == "tasks-demo":
            return report_tasks(verify_tasks_demo(client, runtime, store, args.session_id, answer, metadata))
        if metadata.get("kind") == "workflow":
            return report_workflow(verify_workflow(agent, runtime, metadata))
        if metadata.get("kind") in {"demo", "plan-demo"}:
            demo = metadata["demo"]
            report = verify_demo(client, runtime.root, runtime, demo["baseline"], demo["test_hash"],
                                 answer, agent.events, session_id=args.session_id)
            return report_demo(report)
        print(answer)
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
        requests = {"demo": "Исправь расчёт скидки и проверь тестами", "plan-demo": "Составь план и исправь расчёт скидки",
                    "rollback-demo": "Составь план и исправь расчёт скидки", "workflow-demo": "Оптимизируй многофайловый проект и исправь ошибки",
                    "tasks-demo": "Исправь расчёт налога и чека; минимум три задачи с зависимостями"}
        request = " ".join(getattr(args, "request", [])) or requests.get(args.action, "Проверь код")
        selected, result = choose_route(catalog, request, max_tokens=getattr(args, "max_tokens", 2048),
            tasks=getattr(args, "tasks", False) or args.action == "tasks-demo",
            plan=getattr(args, "plan", False) or args.action in {"plan-demo", "rollback-demo"},
            overrides={level: getattr(args, f"route_{level}_model", None) for level in ("simple", "medium", "complex")})
        args._catalog, args._routing = catalog, result
        print(visible(f"Роутинг: {result['level']} → {selected.id}; причины: " + "; ".join(result['reasons'])))
        print("Это локальная эвристика; выбор сохраняется до конца сессии.")
        return replace(settings, model=selected.id)
    return settings


def configure_client(client, settings, args):
    if getattr(args, "_routing", None):
        client.routing = args._routing
    if not getattr(type(client), "billing_enabled", False):
        return client
    if settings.provider == "ollama":
        client.price = Price('0', '0', source=settings.base_url)
        print("Ollama: оплата локального API $0; расходы компьютера не учитываются.")
        return client
    # A custom compatibility server need not bill at the official provider rates.
    if settings.provider != "ollama" and settings.base_url != PROVIDERS[settings.provider][0]:
        print("Пользовательский API endpoint: тариф неизвестен; списание учитывается только из ответа провайдера.")
        return client
    try:
        catalog = getattr(args, "_catalog", None) or catalog_for(args, settings.provider)
        model = catalog.find(settings.model)
        client.price = model.price if model else None
        print(visible(f"Тарифы: {catalog.source}; обновлены {catalog.updated_at}; {catalog.status}."))
        if catalog.warning:
            print(catalog.warning)
        if client.price is None:
            print("Тариф выбранной модели неизвестен; фактическое списание будет учтено, если провайдер его вернёт.")
    except PricingError:
        print("Тарифы недоступны: оценка неизвестна; фактическое списание будет учтено из ответа, если доступно.")
    return client


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
        print(json.dumps({'provider': provider, 'currency': 'USD', 'source': catalog.source,
            'updated_at': catalog.updated_at, 'status': catalog.status, 'warning': catalog.warning,
            'input_tokens': args.input_tokens, 'output_tokens': args.output_tokens,
            'matched': matched, 'shown': len(models), 'sort': args.sort,
            'models': [{**asdict(m), 'scenario_usd': str(total(m)) if total(m).is_finite() else None} for m in models]}, ensure_ascii=False, indent=2))
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


def main() -> int:
    parser = HelpParser(description="Coding agent: исправление файлов, планирование и безопасное продолжение сессий",
                        epilog=project_guide())
    subparsers = parser.add_subparsers(dest="action", required=True)
    help_command = subparsers.add_parser("help", help=COMMAND_HELP["help"], description=COMMAND_HELP["help"])
    help_command.add_argument("command", nargs="?", choices=list(COMMAND_HELP))
    subparsers.add_parser("cost-demo", help=COMMAND_HELP["cost-demo"], description=COMMAND_HELP["cost-demo"])
    subparsers.add_parser("budget-demo", help=COMMAND_HELP["budget-demo"], description=COMMAND_HELP["budget-demo"])
    subparsers.add_parser("policy-demo", help=COMMAND_HELP["policy-demo"], description=COMMAND_HELP["policy-demo"])
    subparsers.add_parser("session-demo", help=COMMAND_HELP["session-demo"], description=COMMAND_HELP["session-demo"])
    workflow = subparsers.add_parser("workflow-demo", help=COMMAND_HELP["workflow-demo"], description=COMMAND_HELP["workflow-demo"])
    workflow.add_argument("--offline", action="store_true", help="Use the small deterministic recovery scenario without a model")
    workflow.add_argument("--provider", choices=list(PROVIDERS))
    workflow.add_argument("--model")
    workflow.add_argument("--env-file", type=Path, default=Path(".env"))
    workflow.add_argument("--journal", action=argparse.BooleanOptionalAction, default=None)
    add_budget_arguments(workflow)
    workflow.add_argument("--max-rounds", type=int, default=None)
    workflow.add_argument("--max-tokens", type=int, default=None)
    workflow.add_argument("--mode", choices=["safe", "auto"], default=None)
    workflow.add_argument("--approvals", choices=["ask", "deny"], default=None)
    workflow.add_argument("--state-db", type=Path, default=None)
    workflow.add_argument("--workspace", type=Path)
    workflow.add_argument("--stop-after-tools", type=int)
    workflow.add_argument("--crash-after-write", action="store_true",
                          help="Exit with code 86 after the first write, before its result checkpoint")
    for demo_name in ("plan-demo","rollback-demo"):
        plan_demo = subparsers.add_parser(demo_name, help=COMMAND_HELP[demo_name], description=COMMAND_HELP[demo_name])
        plan_demo.add_argument("--offline", action="store_true", help="Use scripted planning without a model or API key")
        plan_demo.add_argument("--provider", choices=list(PROVIDERS))
        plan_demo.add_argument("--model")
        plan_demo.add_argument("--env-file", type=Path, default=Path(".env"))
        plan_demo.add_argument("--journal", action="store_true" if demo_name == "rollback-demo" else argparse.BooleanOptionalAction, default=None)
        add_budget_arguments(plan_demo)
        plan_demo.add_argument("--max-rounds", type=int, default=None)
        plan_demo.add_argument("--max-tokens", type=int, default=None)
        plan_demo.add_argument("--approvals", choices=["ask", "deny"], default=None)
        plan_demo.add_argument("--policy", type=Path)
        plan_demo.add_argument("--stop-after-tools", type=int)
        plan_demo.add_argument("--mode", choices=["safe", "auto"], default=None)
        plan_demo.add_argument("--state-db", type=Path, default=None)
        plan_demo.add_argument("--workspace", type=Path)
        plan_demo.add_argument("--crash-after-write", action="store_true")
    tasks_demo = subparsers.add_parser("tasks-demo", help=COMMAND_HELP["tasks-demo"], description=COMMAND_HELP["tasks-demo"])
    tasks_demo.add_argument("--offline", action="store_true")
    tasks_demo.add_argument("--provider", choices=list(PROVIDERS))
    tasks_demo.add_argument("--model")
    tasks_demo.add_argument("--env-file", type=Path, default=Path(".env"))
    tasks_demo.add_argument("--journal", action=argparse.BooleanOptionalAction, default=None)
    add_budget_arguments(tasks_demo)
    tasks_demo.add_argument("--max-rounds", type=int, default=None)
    tasks_demo.add_argument("--max-tokens", type=int, default=None)
    tasks_demo.add_argument("--mode", choices=["safe", "auto"], default=None)
    tasks_demo.add_argument("--approvals", choices=["ask", "deny"], default=None)
    tasks_demo.add_argument("--policy", type=Path)
    tasks_demo.add_argument("--workspace", type=Path)
    tasks_demo.add_argument("--state-db", type=Path, default=None)
    tasks_demo.add_argument("--stop-after-tools", type=int)
    tasks_demo.add_argument("--stop-after-tasks", type=int)
    tasks_demo.add_argument("--crash-after-write", action="store_true")
    set_command = subparsers.add_parser("set", help=COMMAND_HELP["set"], description=COMMAND_HELP["set"])
    set_command.add_argument("--provider", choices=list(PROVIDERS))
    set_command.add_argument("--model")
    set_command.add_argument("--env-file", type=Path, default=Path(".env"))
    for name, option in LAUNCH_OPTIONS.items():
        flags = {"default": None}
        if option.kind is bool:
            flags["action"] = argparse.BooleanOptionalAction
        else:
            flags["type"] = option.kind
            if option.choices:
                flags["choices"] = option.choices
        set_command.add_argument("--" + name.replace("_", "-"), **flags)
    for action in ("sessions", "session", "resume", "resolve", "approve-plan", "reject-plan", "changes", "rollback", "cost"):
        command = subparsers.add_parser(action, help=COMMAND_HELP[action], description=COMMAND_HELP[action])
        command.add_argument("--state-db", type=Path, default=None)
        if action != "sessions":
            command.add_argument("session_id", nargs="?" if action in {"resume","changes","rollback","cost"} else None)
        command.add_argument("--env-file", type=Path, default=Path(".env"))
        if action == "resume":
            command.add_argument("--cost-estimate", action=argparse.BooleanOptionalAction, default=None)
            command.add_argument("--approvals", choices=["ask", "deny"], default=None)
            command.add_argument("--stop-after-tools", type=int, default=None)
            command.add_argument("--stop-after-tasks", type=int, default=None)
        if action=="rollback":
            command.add_argument("--approvals",choices=["ask","deny"],default=None)
        if action == "resolve":
            command.add_argument("--changes",choices=["own","external"],help="Explicit attribution of all observed changes after a crash")
            command.add_argument("--call", required=True, help="Uncertain call key, for example 1:0")
            command.add_argument("--resolution", choices=["applied", "retry", "skip", "abort"], required=True)
    for action in ("setup", "doctor", "demo", "run", "plan"):
        command = subparsers.add_parser(action, help=COMMAND_HELP[action], description=COMMAND_HELP[action])
        command.add_argument("--provider", choices=list(PROVIDERS))
        command.add_argument("--model")
        command.add_argument("--env-file", type=Path, default=Path(".env"))
        if action in {"demo", "run", "plan"}:
            command.add_argument("--journal", action=argparse.BooleanOptionalAction, default=None, help="Track file changes and enable safe rollback")
            add_budget_arguments(command)
            command.add_argument("--tasks", action=argparse.BooleanOptionalAction, default=None, help="Plan with dependency tasks and runtime verification")
            command.add_argument("--stop-after-tasks", type=int)
            command.add_argument("--plan", action="store_true" if action == "plan" else argparse.BooleanOptionalAction, default=None, help="Read-only planning before explicit approval")
            command.add_argument("--mode", choices=["safe", "auto"], default=None)
            command.add_argument("--state-db", type=Path, default=None)
            command.add_argument("--stop-after-tools", type=int, default=None,
                                 help="Pause after N checkpointed tool results; resume does not inherit this flag")
            command.add_argument("--workspace", type=Path, default=None)
            command.add_argument("--max-rounds", type=int, default=None)
            command.add_argument("--max-tokens", type=int, default=None)
            command.add_argument("--policy", type=Path, help="Trusted JSON policy outside workspace")
            command.add_argument("--approvals", choices=["ask", "deny"], default=None,
                                 help="Prompt for ask decisions, or refuse them without prompting")
        if action in {"run", "plan"}:
            command.add_argument("request", nargs="+")
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
    args = parser.parse_args()
    if args.action == "help":
        target = subparsers.choices[args.command] if args.command else parser
        target.print_help()
        return 0
    try:
        validate_explicit_options(args)
        if args.action == "set":
            updates = set_options(args)
            print(f"Настройки сохранены в {args.env_file}: " + ", ".join(updates))
            print("Применяются к новым запускам; начатые сессии сохраняют прежние параметры.")
            return 0
        if args.action in START_COMMANDS:
            if getattr(args, "offline", False):
                load_launch_env(args.env_file)
            else:
                load_env(args.env_file)
            apply_launch_defaults(args)
        elif hasattr(args, "state_db") and args.state_db is None:
            load_env(args.env_file)
            args.state_db = option_value("state_db", os.getenv("AGENT_STATE_DB", ".agent/sessions.sqlite3"))
        if args.action == "resume":
            load_launch_env(args.env_file)
            if args.cost_estimate is None:
                args.cost_estimate = option_value("cost_estimate", os.getenv("AGENT_COST_ESTIMATE", "true"))
        if args.action == "models":
            return show_models(args)
        budget_limits = None
        if args.action in {"run", "plan", "demo", "plan-demo", "workflow-demo", "tasks-demo", "rollback-demo"}:
            budget_limits = BudgetLimits(model_requests=args.max_rounds, tool_calls=args.max_tool_calls,
                                         side_effects=args.max_side_effects, result_bytes=args.max_result_bytes)
        if getattr(args, "stop_after_tools", None) is not None and args.stop_after_tools < 1:
            raise ConfigurationError("--stop-after-tools must be positive")
        if getattr(args, "stop_after_tasks", None) is not None and args.stop_after_tasks < 1:
            raise ConfigurationError("--stop-after-tasks must be positive")
        if args.action in {"sessions", "session", "resume", "resolve", "approve-plan", "reject-plan", "changes", "rollback", "cost"}:
            return session_command(args)
        if args.action == "cost-demo":
            print("Автономное демо: тарифы и ответы тестовые, реальных списаний нет.")
            report = run_cost_demo(show_event)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("COST DEMO PASS" if report["success"] else "COST DEMO FAIL")
            return 0 if report["success"] else 1
        if args.action == "budget-demo":
            report = run_budget_demo(show_event)
            print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
            print("BUDGET DEMO PASS" if report["success"] else "BUDGET DEMO FAIL")
            return 0 if report["success"] else 1
        if args.action == "tasks-demo":
            if args.max_tokens < 1:
                raise ConfigurationError("Limits must be positive")
            client = None
            if not args.offline:
                settings = choose_settings(args)
                if not doctor(settings, check_sandbox=False):
                    return 1
                client = OpenAICompatibleClient(settings, timeout=180 if settings.provider == "ollama" else 90, max_tokens=args.max_tokens)
                client = configure_client(client, settings, args)
            workspace = (args.workspace or Path("workspace") / ("tasks-demo-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).resolve()
            metadata = {"policy_path": str(args.policy.resolve()) if args.policy else None,
                        "policy_source": "file" if args.policy else "default", "max_tokens": args.max_tokens}
            print(f"Demo workspace: {workspace}", flush=True)
            return report_tasks(run_tasks_demo(workspace, args.state_db, client=client, offline=args.offline,
                mode=args.mode, approve=confirm_action if args.approvals == "ask" else None,
                approve_plan=confirm_plan, on_event=event_listener(args.stop_after_tools, args.stop_after_tasks),
                max_rounds=args.max_rounds, max_tokens=args.max_tokens, budget_limits=budget_limits,
                policy=load_policy(args.policy, workspace), session_metadata=metadata, approvals=args.approvals,
                crash_after_write=args.crash_after_write, journal_mode=args.journal, cost_estimate=args.cost_estimate))
        if args.action in {"plan-demo","rollback-demo"}:
            if args.max_rounds < 1 or args.max_tokens < 1:
                raise ConfigurationError("Limits must be positive")
            client = None
            if not args.offline:
                settings = choose_settings(args)
                if not doctor(settings, check_sandbox=False):
                    return 1
                client = OpenAICompatibleClient(settings, timeout=180 if settings.provider == "ollama" else 90,
                                                max_tokens=args.max_tokens)
                client = configure_client(client, settings, args)
            workspace = (args.workspace or Path("workspace") / (args.action + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).resolve()
            policy = load_policy(args.policy, workspace)
            metadata = {"policy_path": str(args.policy.resolve()) if args.policy else None,
                        "policy_source": "file" if args.policy else "default", "max_tokens": args.max_tokens}
            print(f"Demo workspace: {workspace}", flush=True)
            return report_demo(run_plan_demo(workspace, args.state_db, mode=args.mode,
                                            approve=confirm_action if args.approvals == "ask" else None,
                                            approve_plan=confirm_plan, on_event=event_listener(args.stop_after_tools),
                                            crash_after_write=args.crash_after_write, client=client, offline=args.offline,
                                            max_rounds=args.max_rounds, approvals=args.approvals, policy=policy,
                                            session_metadata=metadata, budget_limits=budget_limits, journal_mode=args.journal, cost_estimate=args.cost_estimate))
        if args.action == "session-demo":
            report = run_session_demo(show_event)
            print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
            print("SESSION DEMO PASS" if report["success"] else "SESSION DEMO FAIL")
            return 0 if report["success"] else 1
        if args.action == "policy-demo":
            report = run_policy_demo(show_event)
            print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
            print("POLICY DEMO PASS" if report["success"] else "POLICY DEMO FAIL")
            return 0 if report["success"] else 1
        if args.action == "workflow-demo":
            if args.max_rounds < 1 or args.max_tokens < 1:
                raise ConfigurationError("Limits must be positive")
            client = None
            if not args.offline:
                settings = choose_settings(args)
                if not doctor(settings):
                    return 1
                client = OpenAICompatibleClient(settings, timeout=180 if settings.provider == "ollama" else 90,
                                                max_tokens=args.max_tokens)
                client = configure_client(client, settings, args)
            workspace = (args.workspace or Path("workspace") / ("workflow-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).resolve()
            print(f"Demo workspace: {workspace}", flush=True)
            report = run_workflow_demo(workspace, args.state_db, mode=args.mode,
                approve=confirm_action if args.approvals == "ask" else None, approvals=args.approvals,
                on_event=event_listener(args.stop_after_tools), crash_after_write=args.crash_after_write,
                client=client, max_rounds=args.max_rounds, budget_limits=budget_limits, journal_mode=args.journal, cost_estimate=args.cost_estimate)
            return report_workflow(report)
        if args.action == "setup":
            setup(args.provider or "openrouter", args.model, args.env_file)
            return 0
        if args.action == "doctor":
            load_env(args.env_file)
        settings = choose_settings(args)
        if args.action == "doctor":
            return 0 if doctor(settings) else 1
        if args.max_rounds < 1 or args.max_tokens < 1:
            raise ConfigurationError("Limits must be positive")
        listener = event_listener(args.stop_after_tools, args.stop_after_tasks)
        client = OpenAICompatibleClient(settings, timeout=180 if settings.provider == "ollama" else 90,
                                        max_tokens=args.max_tokens)
        client = configure_client(client, settings, args)
        if args.action == "demo":
            # This preflight precedes every potentially billable model request.
            if not doctor(settings, check_sandbox=not args.plan):
                return 1
            workspace = args.workspace or Path("workspace") / ("demo-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            workspace = workspace.resolve()
            policy = load_policy(args.policy, workspace)
            print(f"Demo workspace: {workspace}")
            metadata = {"policy_path": str(args.policy.resolve()) if args.policy else None,
                        "policy_source": "file" if args.policy else "default", "approvals": args.approvals,
                        "max_tokens": args.max_tokens}
            with SessionStore(args.state_db) as store:
                if store.path.is_relative_to(workspace):
                    raise SessionError("Session database must be outside workspace")
                report = run_demo(client, workspace, args.max_rounds, listener,
                    policy=policy, approve=confirm_action if args.approvals == "ask" else None,
                    store=store, session_metadata=metadata, mode=args.mode,
                    trash_dir=args.state_db.resolve().parent / "trash", plan_mode=args.plan, task_mode=args.tasks, approve_plan=confirm_plan, budget_limits=budget_limits, journal_mode=args.journal, cost_estimate=args.cost_estimate)
            return report_demo(report)
        workspace = args.workspace or Path(os.getenv("AGENT_WORKSPACE", "workspace"))
        runtime = ToolRuntime(workspace, policy=load_policy(args.policy, workspace), mode=args.mode,
                              trash_dir=args.state_db.resolve().parent / "trash",
                              approve=confirm_action if args.approvals == "ask" else None,
                              on_policy_event=listener)
        metadata = {"kind": "run", "policy_path": str(args.policy.resolve()) if args.policy else None,
                    "policy_source": "file" if args.policy else "default", "approvals": args.approvals,
                    "max_tokens": args.max_tokens}
        with SessionStore(args.state_db) as store:
            agent = CodingAgent(client, runtime, max_rounds=args.max_rounds, on_event=listener,
                                store=store, session_metadata=metadata, plan_mode=args.plan, task_mode=args.tasks, approve_plan=confirm_plan, budget_limits=budget_limits, journal_mode=args.journal, cost_estimate=args.cost_estimate)
            print(agent.run(" ".join(args.request)))
        return 0
    except ApprovalDeferred as exc:
        print(f"\nОжидается подтверждение: {exc}", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print("\nРабота остановлена. Продолжите через hw-ai-agent resume и выберите сессию в терминале.", file=sys.stderr)
        return 130
    except (ConfigurationError, PricingError, ModelClientError, AgentError, JournalError, BudgetExceeded, ToolError, PolicyError, SessionError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
