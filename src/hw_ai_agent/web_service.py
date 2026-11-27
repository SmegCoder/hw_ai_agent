"""Local web controls over the existing conversation, market and evaluation services."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from . import cli
from .budget import BudgetLimits
from .config import PROVIDERS, Settings, apply_launch_defaults
from .conversation import Conversation, RunStats
from .eval_cases import CASES
from .eval_runner import Options, read_json, run_evaluation, save_json
from .filesystem import create_chat_folder
from .mcp import MCPRegistry
from .session import SessionStore
from .session_doctor import stats_snapshot
from .skill_market import SkillMarket
from .skill_updates import evaluate_candidate
from .skills import SKILL_NAME, SkillCatalog


CHAT_MODELS = {
    "openrouter": [
        ("nvidia/nemotron-3-ultra-550b-a55b:free", "NVIDIA Nemotron 3 Ultra (Free)"),
        ("baidu/ernie-5.5-flash-preview:free", "ERNIE 5.5 Flash Preview (Free)"),
        ("openrouter/auto-code", "OpenRouter Auto Code"),
        ("openrouter/auto", "OpenRouter Auto"),
        ("openai/gpt-6.2-sol", "GPT-6.2 Sol"),
        ("anthropic/claude-fable-5.2", "Claude Fable 5.2"),
        ("google/gemini-4-neon", "Gemini 4 Neon"),
        ("openai/gpt-6-luna", "GPT-6 Luna"),
        ("deepseek/deepseek-v4.1-flash-1017", "DeepSeek V4.1 Flash 1017"),
        ("anthropic/claude-sonnet-5.5", "Claude Sonnet 5.5"),
    ],
    "openai": [("gpt-4.1-mini", "GPT-4.1 mini")],
    "ollama": [("qwen3:4b", "Qwen3 4B"), ("qwen3:4b-instruct", "Qwen3 4B Instruct")],
}


class Job:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.answer = None
        self.error = ""
        self.running = True

    def ask(self, content):
        with self.condition:
            self.pending = {"id": uuid.uuid4().hex, **content}
            self.answer = None
            self.condition.wait_for(lambda: self.answer is not None, timeout=600)
            allowed = self.answer is True
            self.pending = None
            return allowed

    def decide(self, identifier, allowed):
        with self.condition:
            if (not self.pending or self.pending["id"] != identifier or self.answer is not None
                    or type(allowed) is not bool):
                raise ValueError("Этот запрос подтверждения уже закрыт")
            self.answer = allowed
            self.condition.notify_all()

    def state(self):
        with self.condition:
            return {"running": self.running, "approval": self.pending, "error": self.error}


class WebService:
    def __init__(self, db, workspace, policy=None):
        self.db = Path(db).expanduser().resolve()
        self.workspace = Path(workspace).expanduser().resolve()
        if self.db.parent.is_relative_to(self.workspace) or not self.workspace.is_dir():
            raise ValueError("Укажите существующую рабочую папку и каталог состояния вне неё")
        self.policy = policy
        self.home = self.db.parent / "web"
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        with SessionStore(self.db):
            pass
        self.lock = threading.RLock()
        self.jobs = {}
        self.prefs_file = self.home / "settings.json"
        self.prefs = read_json(self.prefs_file) if self.prefs_file.exists() else {
            "provider": os.getenv("AGENT_PROVIDER", "openrouter"), "model": "",
            "mcp": "mcp/tracker.json" if Path("mcp/tracker.json").is_file() else "",
            "source": "", "sources": {}, "disabled": [], "demo": False,
        }

    def scrub(self, value):
        for _, key, _ in PROVIDERS.values():
            secret = os.getenv(key) if key else None
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value

    def chat_options(self):
        models = CHAT_MODELS.copy()
        if not os.getenv("OPENAI_API_KEY", "").strip():
            del models["openai"]
        with self.lock:
            provider, model = self.prefs["provider"], self.prefs["model"]
        if provider == "openai" and provider not in models:
            provider = "openrouter" if os.getenv("OPENROUTER_API_KEY", "").strip() else "ollama"
            model = ""
        default_model = model or os.getenv(f"{provider.upper()}_MODEL") or PROVIDERS[provider][2]
        return {"models": models, "provider": provider, "model": model, "default_model": default_model}

    def model_choice(self, provider, model, *, allow_default=False):
        if provider not in self.chat_options()["models"]:
            raise ValueError("Провайдер недоступен. Для OpenAI требуется OPENAI_API_KEY.")
        if not isinstance(model, str):
            raise ValueError("Проверьте ID модели")
        model = model.strip()
        if not (allow_default and not model) and not re.fullmatch(r"[\w./:+@-]{1,180}", model):
            raise ValueError("Проверьте ID модели")
        return {"provider": provider, "model": model}

    def save_model(self, data):
        with self.lock:
            prefs = {**self.prefs, **self.model_choice(data["provider"], data["model"], allow_default=True)}
            save_json(self.prefs_file, prefs)
            self.prefs = prefs
        return self.chat_options()

    def folder_path(self, relative):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Выберите папку внутри workspace")
        folder = self.workspace
        for part in Path(relative).parts:
            folder = folder / part
            if part.startswith(".") or folder.is_symlink() or not folder.is_dir():
                raise ValueError("Папка недоступна")
        return folder

    def folders(self, relative):
        folder = self.folder_path(relative)
        return {"root": str(self.workspace), "path": str(folder.relative_to(self.workspace)),
                "folders": sorted((p.name for p in folder.iterdir()
                                   if not p.name.startswith(".") and not p.is_symlink() and p.is_dir()),
                                  key=str.casefold)}

    def save_settings(self, data):
        mcp, source = data["mcp"].strip(), data["source"].strip()
        if mcp:
            MCPRegistry(Path(mcp), self.workspace)
        if source and source not in {"anthropic", "demo"}:
            path = Path(source).expanduser().resolve()
            if not path.is_file() or path.is_relative_to(self.workspace):
                raise ValueError("Укажите JSON-каталог маркета вне workspace")
            source = str(path)
        directory = data.get("skills_dir", "").strip()
        if directory and not Path(directory).expanduser().is_dir():
            raise ValueError("Папка дополнительных скиллов не найдена")
        if directory:
            directory = str(Path(directory).expanduser().resolve())
            SkillCatalog(self.workspace, Path(directory))
        disabled = data["disabled"]
        if not isinstance(disabled, list) or not all(isinstance(n, str) and SKILL_NAME.fullmatch(n) for n in disabled):
            raise ValueError("Некорректный список скиллов")
        if any(type(data[k]) is not bool for k in ("demo", "auto_update", "check_on_start")):
            raise ValueError("Некорректный переключатель")
        with self.lock:
            sources = dict(self.prefs["sources"])
            sources[source] = {k: data[k] for k in ("check_on_start", "auto_update")}
            model = self.model_choice(data["provider"], data["model"], allow_default=True) if "provider" in data else {}
            prefs = dict(self.prefs, **model)
            prefs.update(mcp=mcp, source=source, sources=sources,
                         disabled=sorted(set(disabled)), demo=data["demo"], skills_dir=directory)
            self.market(prefs)
            save_json(self.prefs_file, prefs)
            self.prefs = prefs
        return self.settings()

    def market(self, prefs=None):
        p = prefs if prefs is not None else self.prefs
        source = p["source"]
        path = None
        if source:
            path = self.home / ("market-" + hashlib.sha256(source.encode()).hexdigest()[:20] + ".json")
            config = {**p["sources"][source], "require_eval": True}
            catalog = self.demo_catalog() if source == "demo" else source
            config["marketplace" if source == "anthropic" else "catalog"] = str(catalog)
            if not path.exists() or read_json(path) != config:
                save_json(path, config)
        return SkillMarket(path, self.db, self.workspace)

    def demo_catalog(self, publish=None):
        example = Path("examples/skill-update")
        if not example.is_dir():
            raise ValueError("Запустите agi web из корня репозитория для учебного каталога")
        with self.lock:
            folder = self.home / "demo-market"
            path = folder / "catalog.json"
            if not path.exists():
                shutil.copytree(example / "packages", folder / "packages", dirs_exist_ok=True)
            if publish or not path.exists():
                version = publish or "1.0.0"
                if version not in {"1.0.0", "1.1.0"}:
                    raise ValueError("Неизвестная публикация")
                save_json(path, read_json(example / f"catalog-{version}.json"))
            return path

    def settings(self):
        with self.lock:
            p = self.prefs.copy()
            market = self.market(p)
            available = market.installed()
            builtin = SkillCatalog(self.workspace, Path(p["skills_dir"]) if p.get("skills_dir") else None)
            return {**p, "workspace": str(self.workspace), "policy": str(self.policy or "default"),
                    "skills": [dict(name=s.name, description=s.description, origin="Встроенный" if s.name in builtin.builtin_names else "Из папки")
                               for s in builtin.skills.values()] +
                              [dict(name=s["name"], description=s["version"], origin="Установлен") for s in available],
                    "offers": [asdict(o) for o in market.search(market.config().source, "")] if market.path else [],
                    "events": market.recent_events(40), "cases": [case.id for case in CASES],
                    "candidates": market.candidates()}

    def args(self, p):
        args = SimpleNamespace(action="chat", workspace=self.workspace, state_db=self.db, policy=self.policy, mode=None,
            provider=p["provider"], model=p["model"] or None, skills=True,
            skills_dir=Path(p["skills_dir"]) if p.get("skills_dir") and not p["demo"] else None,
            demo_skills_only=p["demo"], market_config=None if p["demo"] else self.market(p).path,
            mcp_config=Path(p["mcp"]) if p["mcp"] else None, approvals=None, verbose=False,
            max_rounds=None, max_tokens=None, max_tool_calls=None, max_side_effects=None, max_result_bytes=None,
            plan=None, tasks=None, cost_estimate=None)
        apply_launch_defaults(args)
        return args

    def new_chat(self, *, provider=None, model=None, folder=None):
        with self.lock:
            p = self.prefs.copy()
            options = self.chat_options()
        choice = (self.model_choice(provider, model) if provider is not None or model is not None
                  else {"provider": options["provider"], "model": options["model"]})
        p.update(choice)
        args = self.args(p)
        identifier = None
        if folder is None:
            identifier, args.workspace = create_chat_folder(self.workspace)
        else:
            args.workspace = self.folder_path(folder)
        try:
            return self.create_chat(args, p, identifier)
        except Exception:
            if folder is None:
                try:
                    args.workspace.rmdir()
                except OSError:
                    pass
            raise

    def create_chat(self, args, p, identifier):
        runtime = cli.starting_runtime(args, None)
        runtime.disable_skills(p["disabled"])
        limits = BudgetLimits(args.max_rounds, args.max_tool_calls, args.max_side_effects, args.max_result_bytes)
        with SessionStore(self.db) as store:
            agent = cli.new_agent(args, store, runtime, None, limits, chat=True)
            chat = Conversation.create(store, {"binding": agent.binding(), "budget": limits.as_dict(),
                "cost_estimate": args.cost_estimate, "context_window": getattr(agent.client, "context_window", None)},
                identifier=identifier)
            return chat.id

    def sessions(self):
        with SessionStore(self.db, read_only=True) as store:
            rows = store.db.execute("""SELECT c.id, c.ended, (SELECT prompt FROM conversation_turns
                WHERE conversation_id=c.id ORDER BY number LIMIT 1) AS title FROM conversations c
                ORDER BY c.rowid DESC""")
            return [dict(r) for r in rows]

    def job_state(self, key):
        with self.lock:
            job = self.jobs.get(key)
            return job.state() if job else {"running": False, "approval": None, "error": ""}

    def start(self, key, action):
        with self.lock:
            if self.job_state(key)["running"]:
                raise ValueError("Операция уже выполняется")
            job = self.jobs[key] = Job()

        def work():
            try:
                action(job)
            except Exception as exc:
                with job.condition:
                    job.error = self.scrub(str(exc)) or type(exc).__name__
            finally:
                with job.condition:
                    job.running = False
                    job.pending = None
        threading.Thread(target=work, daemon=True).start()

    def send(self, identifier, prompt, resume=False):
        SessionStore.validate_id(identifier)
        prompt = self.scrub(prompt.strip())
        if not resume and not 0 < len(prompt) <= 32_000:
            raise ValueError("Введите сообщение длиной до 32 000 символов")
        def turn(job):
            with SessionStore(self.db) as store, store.lock(identifier):
                store.log_credentials.update(os.environ[key] for _, key, _ in PROVIDERS.values() if os.getenv(key))
                chat = Conversation(store, identifier)
                config = chat.configuration()
                args = SimpleNamespace(verbose=False, cost_estimate=None, state_db=self.db)
                agent = cli.restore_chat_agent(args, store, None, config)
                if config["binding"].get("approvals", "ask") == "ask":
                    agent.runtime.approve = lambda request: job.ask(asdict(request))
                agent.approve_plan = lambda plan: job.ask({"tool": "plan", "target": "План действий",
                                                            "preview": json.dumps(plan, ensure_ascii=False, indent=2)})
                agent.resolve_uncertain = agent.attribute_changes = None
                turns = chat.turns()
                pending = chat.pending(turns)
                if pending and resume and pending.journal_status != "rollback_pending":
                    agent.resume(pending.state.id)
                elif pending:
                    raise ValueError("Остался незавершённый запрос. Продолжите его перед новым сообщением")
                elif resume:
                    raise ValueError("Нет незавершённого запроса")
                else:
                    agent.run(prompt, history=chat.history(turns), conversation_id=identifier)
        self.start(identifier, turn)

    def chat(self, identifier):
        with SessionStore(self.db, read_only=True) as store:
            chat = Conversation(store, identifier)
            row = store.db.execute("SELECT configuration, ended FROM conversations WHERE id=?", (identifier,)).fetchone()
            if not row:
                raise ValueError("Нет такого диалога")
            turns = chat.turns()
            total = RunStats()
            messages = []
            for turn in turns:
                total.include(turn.state)
                run = RunStats()
                run.include(turn.state)
                messages.append({"prompt": turn.prompt, "answer": turn.state.final_answer, "status": turn.state.status,
                                 "id": turn.state.id, "stats": stats_snapshot(run)})
            binding = json.loads(row["configuration"])["binding"]
            return {"turns": messages, "events": list(store.event_records(identifier)), "stats": stats_snapshot(total),
                    "model": binding["model"], "workspace": binding.get("workspace"), "ended": bool(row["ended"]), "pending": bool(chat.pending(turns)),
                    **self.job_state(identifier)}

    def catalog_action(self, action, name=""):
        market = self.market()
        if action == "rollback":
            return market.rollback(name)
        if not market.path:
            raise ValueError("Сначала выберите источник маркета")
        source, events = market.check()
        if any(e.get("status") == "error" for e in events):
            raise ValueError(next(e["error"] for e in events if e.get("status") == "error"))
        if action == "install":
            offers = {o.name: o for o in market.search(source, "")}
            if name not in offers or any(s["name"] == name for s in self.settings()["skills"]):
                raise ValueError("Скилл недоступен для установки")
            return market.install(source, offers[name], uuid.uuid4().hex)
        return events

    def mcp_tools(self):
        if not self.prefs["mcp"]:
            return {"tools": [], "events": []}
        registry = MCPRegistry(Path(self.prefs["mcp"]), self.workspace)
        return {"tools": [tool.definition() for tool in registry.tools.values()], "events": registry.events}

    def reports(self):
        result = {"archive-ab": Path("docs/study-tracker/eval"), "archive-update": Path("docs/skill-update-eval")}
        for prefix, folder in (("ab", self.home / "evals"), ("update", self.db.parent / "skill-update-evals")):
            if folder.is_dir():
                for path in sorted(folder.iterdir()):
                    if path.is_dir():
                        result[f"{prefix}/{path.name}"] = path
        return {k: v for k, v in result.items() if (v / "manifest.json").is_file()}

    def report(self, key, run=None):
        path = self.reports().get(key)
        if path is None:
            raise ValueError("Серия не найдена")
        rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()] if (path / "results.jsonl").exists() else []
        if run:
            row = next((r for r in rows if r["run_id"] == run), None)
            if not row:
                raise ValueError("Попытка не найдена")
            event_file = (path / row["events"]).resolve()
            if not event_file.is_relative_to(path.resolve()):
                raise ValueError("Неверный путь журнала")
            return [json.loads(line) for line in event_file.read_text().splitlines()]
        return {"rows": rows, "summary": read_json(path / "summary.json") if (path / "summary.json").exists() else {},
                "decision": read_json(path / "decision.json") if (path / "decision.json").exists() else None}

    def evaluate(self, candidate, focus=(), series=None):
        prefs = self.prefs.copy()
        settings = Settings.from_env(prefs["provider"], prefs["model"] or None)
        market = self.market(prefs)
        if not isinstance(focus, (tuple, list)) or set(focus) - {case.id for case in CASES}:
            raise ValueError("Неверный список сценариев")
        if series:
            if series not in self.reports() or not series.startswith("ab/"):
                raise ValueError("Эта серия недоступна для продолжения")
            output = self.reports()[series]
        else:
            output = self.home / "evals" / uuid.uuid4().hex[:12]
        def work(job):
            if candidate:
                status, _ = evaluate_candidate(market, candidate, settings, Options(),
                                               focus=tuple(focus))
            else:
                status = run_evaluation(output, settings, Options())
            if status:
                raise ValueError("Серия прервана. Повторите запуск для продолжения")
        self.start("eval", work)
        return None if candidate else "ab/" + output.name
