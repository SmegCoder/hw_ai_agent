from __future__ import annotations
import argparse
import getpass
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from .agent import AgentError, CodingAgent
from .client import ModelClientError, OpenAICompatibleClient
from .config import ConfigurationError, PROVIDERS, Settings, load_env
from .demo import run_demo
from .tools import ToolError, ToolRuntime
from .policy import ApprovalRequest, Policy, PolicyError
from .policy_demo import run_policy_demo

def show_event(event: dict) -> None:
    if event["event"] == "model_request":
        print(f"\nМодель: обращение {event['round']}", flush=True)
    elif event["event"] == "tool_result":
        print(f"\n[{event['name']}]\n{event['content']}", flush=True)
    elif event["event"] == "tool_call":
        print(f"Вызов {event['name']}: {event['arguments']}", flush=True)
    elif event["event"] == "policy_decision":
        print(f"Policy: {event['tool']} → {event['action']} / {event['outcome']}", flush=True)

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
        return input("Разрешить только этот вызов? [y/N]: ").strip().lower() in {"y", "yes", "д", "да"}
    except EOFError:
        return False

def load_policy(path: Path | None, workspace: Path) -> Policy:
    if path is None:
        return Policy()
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

def main() -> int:
    parser = argparse.ArgumentParser(description="Coding agent: OpenRouter / OpenAI / Ollama")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("policy-demo", help="Check allow/ask/deny offline, without an API key")
    for action in ("setup", "doctor", "demo", "run"):
        command = subparsers.add_parser(action)
        command.add_argument("--provider", choices=list(PROVIDERS))
        command.add_argument("--model")
        command.add_argument("--env-file", type=Path, default=Path(".env"))
        if action in {"demo", "run"}:
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
        if args.action == "policy-demo":
            report = run_policy_demo(show_event)
            print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
            print("POLICY DEMO PASS" if report["success"] else "POLICY DEMO FAIL")
            return 0 if report["success"] else 1
        if args.action == "setup":
            setup(args.provider or "openrouter", args.model, args.env_file)
            return 0
        load_env(args.env_file)
        settings = Settings.from_env(args.provider, args.model)
        if args.action == "doctor":
            return 0 if doctor(settings) else 1
        if args.max_rounds < 1 or args.max_tokens < 1:
            raise ConfigurationError("Limits must be positive")
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
            report = run_demo(client, workspace, args.max_rounds, show_event,
                              policy=policy, approve=confirm_action if args.approvals == "ask" else None)
            print("\nИзменения:\n" + report["diff"])
            print("Независимая проверка:\n" + report["verification"])
            print(f"Тесты неизменны: {report['tests_unchanged']}")
            print(f"Модель сама запустила успешные тесты: {report['model_ran_passing_tests']}")
            print(f"Отчёт: {workspace / 'demo-report.json'}")
            if report["error"]:
                print(f"Ошибка модели: {report['error']}")
            print("DEMO PASS" if report["success"] else "DEMO FAIL")
            return 0 if report["success"] else 1
        workspace = args.workspace or Path(os.getenv("AGENT_WORKSPACE", "workspace"))
        runtime = ToolRuntime(workspace, policy=load_policy(args.policy, workspace),
                              approve=confirm_action if args.approvals == "ask" else None,
                              on_policy_event=show_event)
        preflight = runtime.bash("pwd")
        if not preflight.startswith("exit_code=0\n"):
            raise ToolError("Bash preflight failed")
        agent = CodingAgent(client, runtime, max_rounds=args.max_rounds, on_event=show_event)
        print(agent.run(" ".join(args.request)))
        return 0
    except (ConfigurationError, ModelClientError, AgentError, ToolError, PolicyError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
