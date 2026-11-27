"""Loopback-only HTTP transport for agi web."""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .config import load_env
from .session_doctor import inspect_session
from .web_service import WebService


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, status, content, mime="application/json; charset=utf-8"):
        if not isinstance(content, bytes):
            content = self.server.app.scrub(json.dumps(content, ensure_ascii=False)).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(content)

    def allowed(self):
        port = self.server.server_port
        host = self.headers.get("Host", "")
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"} and self.headers.get("Origin", "http://" + host) == "http://" + host

    def do_GET(self):
        self.handle_web(False)

    def do_POST(self):
        self.handle_web(True)

    def handle_web(self, post):
        if not self.allowed():
            self.respond(403, {"error": "Недопустимый источник запроса"})
            return
        try:
            parts = urlsplit(self.path)
            if post:
                if self.headers.get("X-Agi-UI") != "1" or self.headers.get_content_type() != "application/json":
                    self.respond(403, {"error": "Требуется локальное окно agi"})
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 262144:
                    raise ValueError("Недопустимый размер запроса")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Ожидается объект")
                result = self.mutate(parts.path, body)
            elif parts.path in {"/", "/app.js", "/style.css"}:
                name = parts.path[1:] or "index.html"
                mime = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[name]
                self.respond(200, (Path(__file__).with_name("web_assets") / name).read_bytes(), mime + "; charset=utf-8")
                return
            else:
                query = {k: v[0] for k, v in parse_qs(parts.query).items()}
                result = self.get_data(parts.path, query)
            self.respond(200, {"data": result})
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as exc:
            self.respond(400, {"error": str(exc) or type(exc).__name__})

    def get_data(self, path, query):
        app = self.server.app
        if path == "/api/sessions":
            return app.sessions()
        if path == "/api/chat":
            return app.chat(query["id"])
        if path == "/api/settings":
            return app.settings()
        if path == "/api/doctor":
            return inspect_session(app.db, query["id"])
        if path == "/api/mcp":
            return app.mcp_tools()
        if path == "/api/reports":
            return {"names": list(app.reports()), **app.job_state("eval")}
        if path == "/api/report":
            return app.report(query["key"], query.get("run"))
        raise ValueError("Страница не найдена")

    def mutate(self, path, body):
        app = self.server.app
        if path == "/api/new":
            return app.new_chat()
        if path == "/api/send":
            return app.send(body["id"], body["prompt"], body.get("resume") is True)
        if path == "/api/decision":
            job = app.jobs.get(body["id"])
            if job is None:
                raise ValueError("Запуск завершён")
            return job.decide(body["request"], body["allow"])
        if path == "/api/settings":
            return app.save_settings(body)
        if path in {"/api/check", "/api/install", "/api/rollback"}:
            return app.catalog_action(path.rsplit("/", 1)[1], body.get("name", ""))
        if path == "/api/publish-demo":
            if app.prefs["source"] != "demo":
                raise ValueError("Выберите учебный источник")
            app.demo_catalog(body["version"])
            return True
        if path == "/api/eval":
            return app.evaluate(body.get("candidate"), body.get("focus", []), body.get("series"))
        raise ValueError("Действие не найдено")


def serve(args):
    load_env(args.env_file)
    db = args.state_db or os.getenv("AGENT_STATE_DB", ".agent/sessions.sqlite3")
    workspace = args.workspace or os.getenv("AGENT_WORKSPACE", "workspace")
    policy = args.policy or os.getenv("AGENT_POLICY") or ("policies/tracker.json" if Path("policies/tracker.json").exists() else None)
    app = WebService(db, workspace, Path(policy) if policy else None)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.app = app
        print(f"agi · http://127.0.0.1:{server.server_port} · Ctrl+C для выхода", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Сервер остановлен", file=sys.stderr)
    return 0
