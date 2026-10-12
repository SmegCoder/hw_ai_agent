from __future__ import annotations
import argparse
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
from .client import ModelClientError, OpenAICompatibleClient
from .config import ConfigurationError, PROVIDERS, Settings, load_env
from .demo import run_demo, verify_demo
from .tools import ToolError, ToolRuntime
from .policy import ApprovalRequest, Policy, PolicyError
from .policy_demo import run_policy_demo
from .session import SessionError, SessionStore
from .session_demo import run_session_demo
from .workflow_demo import WorkflowClient, run_workflow_demo, verify_workflow

def show_event(event: dict) -> None:
    if event["event"] == "model_request":
        print(f"\nМодель: обращение {event['round']}", flush=True)
    elif event["event"] == "tool_result":
        print(f"\n[{event['name']}]\n{event['content']}", flush=True)
    elif event["event"] == "tool_call":
        print(f"Вызов {event['name']}: {event['arguments']}", flush=True)
    elif event["event"] == "policy_decision":
        print(f"Policy: {event['tool']} → {event['action']} / {event['outcome']}", flush=True)
        if event.get("source") == "session_grant":
            print("Используется разрешение, уже выданное для этой сессии.", flush=True)
    elif event["event"] in {"session_created", "session_resumed", "session_cached"}:
        print(f"Сессия: {event['session_id']} ({event['event']})", flush=True)

def confirm_action(request: ApprovalRequest) -> bool:
    if not sys.stdin.isatty():
        return False

    # Escape terminal control characters in model-provided filenames and diffs.
    def visible(value: str) -> str:
        return "".join(c if c.isprintable() or c in "\n\t" else repr(c)[1:-1] for c in value)
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
        return False

def event_listener(stop_after_tools=None):
    if stop_after_tools is None:
        return show_event
    completed = 0
    def listener(event):
        nonlocal completed
        show_event(event)
        if event["event"] == "tool_result":
            completed += 1
            if completed >= stop_after_tools:
                print("Пауза после сохранённого результата инструмента.", flush=True)
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
    values = {"AGENT_PROVIDER": provider, f"{provider.upper()}_MODEL": chosen_model,
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
    print(f"Отчёт: {Path(report['workspace']) / 'workflow-report.json'}")
    print("WORKFLOW DEMO PASS" if report["success"] else "WORKFLOW DEMO FAIL")
    return 0 if report["success"] else 1

def saved_runtime(state, db_path: Path, approvals=None, on_event=show_event):
    metadata = state.metadata
    workspace = Path(metadata["workspace"])
    if db_path.resolve().is_relative_to(workspace.resolve()):
        raise SessionError("Session database must be outside workspace")
    policy_path = metadata.get("policy_path")
    if policy_path:
        policy = load_policy(Path(policy_path) if policy_path else None, workspace)
    else:
        policy = Policy.from_dict(metadata["policy"])
    if policy.as_dict() != metadata["policy"]:
        raise SessionError("Policy changed since the checkpoint; continuation was refused")
    limits = metadata.get("runtime_limits", {})
    if set(limits) != {"command_timeout", "max_output_chars", "max_file_chars"}:
        raise SessionError("Invalid saved runtime limits")
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in limits.values()) or
        any(type(limits[k]) is not int for k in ("max_output_chars", "max_file_chars"))):
        raise SessionError("Invalid saved runtime limits")
    mode = approvals or metadata.get("approvals", "ask")
    return ToolRuntime(workspace, policy=policy, mode=metadata.get("execution_mode", "safe"),
                       approve=confirm_action if mode == "ask" else None,
                       on_policy_event=on_event, **limits)

def verify_applied(runtime, item):
    try:
        arguments = json.loads(item["call"]["function"]["arguments"])
        expected = arguments["content"].encode("utf-8")
        path = runtime._safe_path(arguments["path"])
        return path.is_file() and path.stat().st_size == len(expected) and path.read_bytes() == expected
    except (ValueError, KeyError, TypeError, AttributeError, ToolError, OSError):
        return False

def session_command(args):
    with SessionStore(args.state_db) as store:
        if args.action == "sessions":
            print(json.dumps(store.list_sessions(), ensure_ascii=False, indent=2))
            return 0
        state = store.get(args.session_id)
        if args.action == "session":
            calls = []
            for item in store.calls(args.session_id, True):
                function = item["call"]["function"]
                try:
                    arguments = json.loads(function["arguments"])
                    target = arguments.get("path", arguments.get("command", ".")) if isinstance(arguments, dict) else "invalid arguments"
                except ValueError:
                    target = "invalid arguments"
                calls.append({"key": item["key"], "state": item["state"], "tool": function["name"], "target": target})
            print(json.dumps({"id": state.id, "status": state.status, "workspace": state.metadata["workspace"],
                "provider": state.metadata.get("provider"), "model": state.metadata.get("model"),
                "rounds": state.rounds, "max_rounds": state.max_rounds, "error": state.error,
                "mode": state.metadata.get("execution_mode", "safe"),
                "unfinished_calls": calls}, ensure_ascii=False, indent=2))
            return 0
        listener = event_listener(getattr(args, "stop_after_tools", None))
        runtime = saved_runtime(state, store.path, getattr(args, "approvals", None), listener)
        if args.action == "resolve":
            store.resolve(args.session_id, args.call, args.resolution, lambda item: verify_applied(runtime, item))
            if args.resolution == "abort":
                print("Сессия завершена с ошибкой по решению пользователя. Продолжение запрещено.")
                return 1
            print("Решение сохранено. Продолжение: hw-ai-agent resume " + args.session_id)
            return 0
        metadata = state.metadata
        if metadata.get("kind") == "workflow":
            agent = CodingAgent(WorkflowClient(), runtime, store=store, on_event=listener)
            agent.resume(args.session_id)
            return report_workflow(verify_workflow(agent, runtime, metadata))
        load_env(args.env_file)
        if metadata.get("provider") not in PROVIDERS:
            raise SessionError("This offline session cannot use a real model through the CLI")
        settings = Settings.from_env(metadata["provider"], metadata["model"])
        client = OpenAICompatibleClient(settings, timeout=180 if settings.provider == "ollama" else 90,
                                        max_tokens=metadata["max_tokens"])
        agent = CodingAgent(client, runtime, store=store, on_event=listener)
        answer = agent.resume(args.session_id)
        if metadata.get("kind") == "demo":
            demo = metadata["demo"]
            report = verify_demo(client, runtime.root, runtime, demo["baseline"], demo["test_hash"],
                                 answer, agent.events, session_id=args.session_id)
            return report_demo(report)
        print(answer)
        return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="Coding agent: OpenRouter / OpenAI / Ollama")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("policy-demo", help="Check allow/ask/deny offline, without an API key")
    subparsers.add_parser("session-demo", help="Interrupt and resume a scripted session without an API key")
    workflow = subparsers.add_parser("workflow-demo", help="Multi-file permissions and recovery demo, without an API key")
    workflow.add_argument("--mode", choices=["safe", "auto"], default="safe")
    workflow.add_argument("--approvals", choices=["ask", "deny"], default="ask")
    workflow.add_argument("--state-db", type=Path, default=Path(".agent/sessions.sqlite3"))
    workflow.add_argument("--workspace", type=Path)
    workflow.add_argument("--stop-after-tools", type=int)
    workflow.add_argument("--crash-after-write", action="store_true",
                          help="Exit with code 86 after the first write, before its result checkpoint")
    for action in ("sessions", "session", "resume", "resolve"):
        command = subparsers.add_parser(action)
        command.add_argument("--state-db", type=Path, default=Path(".agent/sessions.sqlite3"))
        if action != "sessions":
            command.add_argument("session_id")
        if action == "resume":
            command.add_argument("--env-file", type=Path, default=Path(".env"))
            command.add_argument("--approvals", choices=["ask", "deny"], default=None)
            command.add_argument("--stop-after-tools", type=int, default=None)
        if action == "resolve":
            command.add_argument("--call", required=True, help="Uncertain call key, for example 1:0")
            command.add_argument("--resolution", choices=["applied", "retry", "skip", "abort"], required=True)
    for action in ("setup", "doctor", "demo", "run"):
        command = subparsers.add_parser(action)
        command.add_argument("--provider", choices=list(PROVIDERS))
        command.add_argument("--model")
        command.add_argument("--env-file", type=Path, default=Path(".env"))
        if action in {"demo", "run"}:
            command.add_argument("--mode", choices=["safe", "auto"], default="safe")
            command.add_argument("--state-db", type=Path, default=Path(".agent/sessions.sqlite3"))
            command.add_argument("--stop-after-tools", type=int, default=None,
                                 help="Pause after N checkpointed tool results; resume does not inherit this flag")
            command.add_argument("--workspace", type=Path, default=None)
            command.add_argument("--max-rounds", type=int, default=12)
            command.add_argument("--max-tokens", type=int, default=2048)
            command.add_argument("--policy", type=Path, help="Trusted JSON policy outside workspace")
            command.add_argument("--approvals", choices=["ask", "deny"], default="ask",
                                 help="Prompt for ask decisions, or refuse them without prompting")
        if action == "run":
            command.add_argument("request", nargs="+")
    args = parser.parse_args()
    try:
        if getattr(args, "stop_after_tools", None) is not None and args.stop_after_tools < 1:
            raise ConfigurationError("--stop-after-tools must be positive")
        if args.action in {"sessions", "session", "resume", "resolve"}:
            return session_command(args)
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
            workspace = (args.workspace or Path("workspace") / ("workflow-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).resolve()
            print(f"Demo workspace: {workspace}", flush=True)
            report = run_workflow_demo(workspace, args.state_db, mode=args.mode,
                approve=confirm_action if args.approvals == "ask" else None, approvals=args.approvals,
                on_event=event_listener(args.stop_after_tools), crash_after_write=args.crash_after_write)
            return report_workflow(report)
        if args.action == "setup":
            setup(args.provider or "openrouter", args.model, args.env_file)
            return 0
        load_env(args.env_file)
        settings = Settings.from_env(args.provider, args.model)
        if args.action == "doctor":
            return 0 if doctor(settings) else 1
        if args.max_rounds < 1 or args.max_tokens < 1:
            raise ConfigurationError("Limits must be positive")
        listener = event_listener(args.stop_after_tools)
        client = OpenAICompatibleClient(settings, timeout=180 if settings.provider == "ollama" else 90,
                                        max_tokens=args.max_tokens)
        if args.action == "demo":
            # This preflight precedes every potentially billable model request.
            if not doctor(settings):
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
                    store=store, session_metadata=metadata, mode=args.mode)
            return report_demo(report)
        workspace = args.workspace or Path(os.getenv("AGENT_WORKSPACE", "workspace"))
        runtime = ToolRuntime(workspace, policy=load_policy(args.policy, workspace), mode=args.mode,
                              approve=confirm_action if args.approvals == "ask" else None,
                              on_policy_event=listener)
        preflight = runtime.bash("pwd")
        if not preflight.startswith("exit_code=0\n"):
            raise ToolError("Bash preflight failed")
        metadata = {"kind": "run", "policy_path": str(args.policy.resolve()) if args.policy else None,
                    "policy_source": "file" if args.policy else "default", "approvals": args.approvals,
                    "max_tokens": args.max_tokens}
        with SessionStore(args.state_db) as store:
            agent = CodingAgent(client, runtime, max_rounds=args.max_rounds, on_event=listener,
                                store=store, session_metadata=metadata)
            print(agent.run(" ".join(args.request)))
        return 0
    except KeyboardInterrupt:
        print("\nРабота остановлена. Найдите ID через hw-ai-agent sessions и продолжите через resume.", file=sys.stderr)
        return 130
    except (ConfigurationError, ModelClientError, AgentError, ToolError, PolicyError, SessionError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
