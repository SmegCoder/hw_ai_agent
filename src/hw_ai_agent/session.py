from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from .planning import validate_plan, PlanError


class SessionError(RuntimeError):
    pass


class SessionBusyError(SessionError):
    pass


class UncertainActionError(SessionError):
    pass


class PlanRejectedError(SessionError):
    pass


@dataclass(frozen=True)
class SessionState:
    id: str
    metadata: dict
    messages: list
    events: list
    status: str
    rounds: int
    max_rounds: int
    max_context_chars: int
    final_answer: str | None
    error: str | None


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def encoded(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class SessionStore:
    def __init__(self, path: str | Path):
        raw_path = Path(path).expanduser().absolute()
        if raw_path.is_symlink():
            raise SessionError("Session database must not be a symlink")
        self.path = raw_path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SessionError("Session database must be a regular file without hard links")
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        try:
            self.db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            self.db.row_factory = sqlite3.Row
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise SessionError("Unsupported session database version")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, metadata TEXT NOT NULL, messages TEXT NOT NULL,
                    events TEXT NOT NULL, status TEXT NOT NULL, rounds INTEGER NOT NULL,
                    max_rounds INTEGER NOT NULL, max_context_chars INTEGER NOT NULL,
                    final_answer TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calls (
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    batch INTEGER NOT NULL, position INTEGER NOT NULL,
                    call TEXT NOT NULL, state TEXT NOT NULL, result TEXT,
                    PRIMARY KEY (session_id, batch, position)
                );
                CREATE TABLE IF NOT EXISTS plans (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id),
                    state TEXT NOT NULL, data TEXT, updated_at TEXT NOT NULL
                );
                PRAGMA user_version=2;
            ''')
        except (sqlite3.Error, SessionError):
            if hasattr(self, "db"):
                self.db.close()
            raise SessionError("Cannot initialize session database") from None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    @contextmanager
    def transaction(self):
        try:
            self.db.execute("BEGIN IMMEDIATE")
            yield
            self.db.execute("COMMIT")
        except BaseException as exc:
            try:
                if self.db.in_transaction:
                    self.db.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if isinstance(exc, sqlite3.Error):
                raise SessionError("Session checkpoint could not be saved") from None
            raise

    @staticmethod
    def validate_id(session_id: str):
        if not isinstance(session_id, str) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise SessionError("Invalid session ID")

    @contextmanager
    def lock(self, session_id: str):
        self.validate_id(session_id)
        lock_path = self.path.with_name(f"{self.path.name}.{session_id}.lock")
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SessionBusyError("This session is already running in another process") from None
            yield
        finally:
            os.close(fd)

    def create(self, metadata: dict, messages: list, max_rounds: int, max_context_chars: int) -> str:
        workspace = Path(metadata["workspace"]).resolve()
        if self.path.is_relative_to(workspace):
            raise SessionError("Session database must be outside the model workspace")
        if type(max_rounds) is not int or max_rounds < 1 or type(max_context_chars) is not int or max_context_chars < 1:
            raise SessionError("Invalid session limits")
        session_id = uuid.uuid4().hex
        now = stamp()
        with self.transaction():
            self.db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (session_id, encoded(metadata), encoded(messages), "[]", "ready", 0,
                 max_rounds, max_context_chars, None, None, now, now))
            if metadata.get("plan_mode"):
                self.db.execute("INSERT INTO plans VALUES (?, 'planning', NULL, ?)", (session_id, now))
        return session_id

    def get(self, session_id: str) -> SessionState:
        self.validate_id(session_id)
        try:
            row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise SessionError("Session not found")
            metadata, messages, events = (json.loads(row[k]) for k in ("metadata", "messages", "events"))
            if not isinstance(metadata, dict) or not isinstance(metadata.get("workspace"), str):
                raise SessionError("Invalid saved session metadata")
            if not isinstance(messages, list) or not isinstance(events, list):
                raise SessionError("Invalid saved session history")
            if not messages or any(not isinstance(m, dict) or m.get("role") not in
                {"system", "user", "assistant", "tool"} for m in messages):
                raise SessionError("Invalid saved messages")
            if row["status"] not in {"ready", "running", "interrupted", "waiting_approval", "waiting_plan_approval", "failed", "blocked", "completed", "aborted"}:
                raise SessionError("Invalid session status")
            if row["status"] == "completed" and not isinstance(row["final_answer"], str):
                raise SessionError("Completed session has no final answer")
            if not 0 <= row["rounds"] <= row["max_rounds"] or row["max_rounds"] < 1 or row["max_context_chars"] < 1:
                raise SessionError("Invalid saved session limits")
            return SessionState(session_id, metadata, messages, events, row["status"], row["rounds"],
                row["max_rounds"], row["max_context_chars"], row["final_answer"], row["error"])
        except (sqlite3.Error, ValueError, TypeError, KeyError):
            raise SessionError("Cannot read valid session state") from None

    def list_sessions(self) -> list[dict]:
        ids = self.db.execute("SELECT id FROM sessions ORDER BY created_at DESC").fetchall()
        return [{"id": s.id, "status": s.status, "rounds": s.rounds,
                 "workspace": s.metadata["workspace"], "model": s.metadata.get("model")}
                for row in ids for s in [self.get(row["id"])]]

    def set_status(self, session_id: str, status: str, error: str | None = None):
        with self.transaction():
            self.db.execute("UPDATE sessions SET status=?,error=?,updated_at=? WHERE id=?",
                            (status, error, stamp(), session_id))

    def plan(self, session_id: str) -> dict | None:
        self.validate_id(session_id)
        row = self.db.execute("SELECT * FROM plans WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            if self.get(session_id).metadata.get("plan_mode"):
                raise SessionError("Planning session has no saved planning state")
            return None
        if self.get(session_id).metadata.get("plan_mode") is not True:
            raise SessionError("Saved plan does not match session planning mode")
        try:
            data = validate_plan(json.loads(row["data"])) if row["data"] is not None else None
            if row["state"] not in {"planning", "awaiting_approval", "approved", "rejected"}:
                raise SessionError("Invalid plan state")
            if (row["state"] == "planning") != (data is None):
                raise SessionError("Invalid saved plan")
            return {"state": row["state"], "data": data}
        except (ValueError, TypeError):
            raise SessionError("Invalid saved plan") from None

    def message(self, session_id: str, message: dict):
        with self.transaction():
            state = self.get(session_id)
            self.db.execute("UPDATE sessions SET messages=?,updated_at=? WHERE id=?",
                            (encoded([*state.messages, message]), stamp(), session_id))

    def decide_plan(self, session_id: str, approved: bool):
        with self.lock(session_id):
            self._decide_plan(session_id, approved)

    def _decide_plan(self, session_id: str, approved: bool):
        if type(approved) is not bool:
            raise SessionError("Plan decision must be explicitly true or false")
        with self.transaction():
            saved = self.plan(session_id)
            if self.get(session_id).status == "aborted" or not saved or saved["state"] != "awaiting_approval":
                raise SessionError("This plan is not waiting for a decision")
            if self.calls(session_id, True):
                raise SessionError("Finish or resolve pending planning calls before deciding the plan")
            decision = "approved" if approved else "rejected"
            self.db.execute("UPDATE plans SET state=?,updated_at=? WHERE session_id=?", (decision, stamp(), session_id))
            self.db.execute("UPDATE sessions SET status=?,error=?,updated_at=? WHERE id=?",
                            ("interrupted" if approved else "aborted", None if approved else "PlanRejected",
                             stamp(), session_id))
            message = {"role": "user", "content":
                "The operator approved this plan. Execute it now, following runtime permissions.\n" + encoded(saved["data"])
                if approved else "The operator rejected the plan. Do not execute it."}
            state = self.get(session_id)
            self.db.execute("UPDATE sessions SET messages=? WHERE id=?", (encoded([*state.messages, message]), session_id))
            self._event(session_id, {"event": "plan_decision", "decision": decision})

    def upgrade_approval_contract(self, session_id: str):
        with self.transaction():
            state = self.get(session_id)
            metadata = {**state.metadata, "approval_contract": 2}
            self.db.execute("UPDATE sessions SET metadata=? WHERE id=?", (encoded(metadata), session_id))
            self._event(session_id, {"event": "approval_contract_upgrade", "version": 2})

    def _event(self, session_id: str, event: dict):
        row = self.db.execute("SELECT events FROM sessions WHERE id=?", (session_id,)).fetchone()
        events = json.loads(row[0])
        events.append(event)
        self.db.execute("UPDATE sessions SET events=?,updated_at=? WHERE id=?", (encoded(events), stamp(), session_id))

    def event(self, session_id: str, event: dict):
        with self.transaction():
            self._event(session_id, event)

    def start_model(self, session_id: str) -> int:
        with self.transaction():
            state = self.get(session_id)
            if state.rounds >= state.max_rounds:
                raise SessionError("Saved model round limit has been reached")
            number = state.rounds + 1
            self.db.execute("UPDATE sessions SET rounds=?,updated_at=? WHERE id=?", (number, stamp(), session_id))
            self._event(session_id, {"event": "model_request", "round": number})
        return number

    def assistant(self, session_id: str, message: dict, batch: int):
        with self.transaction():
            state = self.get(session_id)
            self.db.execute("UPDATE sessions SET messages=?,updated_at=? WHERE id=?",
                (encoded([*state.messages, message]), stamp(), session_id))
            for position, call in enumerate(message.get("tool_calls") or []):
                self.db.execute("INSERT INTO calls VALUES (?,?,?,?,?,NULL)",
                    (session_id, batch, position, encoded(call), "pending"))

    def calls(self, session_id: str, unfinished_only=False) -> list[dict]:
        self.validate_id(session_id)
        clause = " AND state IN ('pending','awaiting_approval','running','uncertain')" if unfinished_only else ""
        rows = self.db.execute("SELECT * FROM calls WHERE session_id=?" + clause + " ORDER BY batch,position", (session_id,)).fetchall()
        try:
            items = [{"key": f"{r['batch']}:{r['position']}", "batch": r["batch"], "position": r["position"],
                     "call": json.loads(r["call"]), "state": r["state"],
                     "result": json.loads(r["result"]) if r["result"] else None} for r in rows]
            for item in items:
                call = item["call"]
                if (not isinstance(call, dict) or not isinstance(call.get("id"), str) or
                    not isinstance(call.get("function"), dict) or
                    not isinstance(call["function"].get("name"), str) or
                    not isinstance(call["function"].get("arguments"), str) or
                    item["state"] not in {"pending", "awaiting_approval", "running", "uncertain", "completed", "failed"}):
                    raise SessionError("Invalid saved tool call")
            return items
        except (ValueError, TypeError):
            raise SessionError("Invalid saved tool calls") from None

    def start_call(self, session_id: str, item: dict):
        with self.transaction():
            cursor = self.db.execute("UPDATE calls SET state='running' WHERE session_id=? AND batch=? AND position=? AND state IN ('pending','awaiting_approval')",
                (session_id, item["batch"], item["position"]))
            if cursor.rowcount != 1:
                raise SessionError("Tool call is not pending")
            function = item["call"]["function"]
            attempt = 1 + sum(e.get("event") == "tool_call" and e.get("call") == item["key"]
                              for e in self.get(session_id).events)
            self._event(session_id, {"event": "tool_call", "call": item["key"], "attempt": attempt,
                                    "tool_call_id": item["call"]["id"],
                                    "name": function["name"], "arguments": function["arguments"]})

    def wait_for_approval(self, session_id: str, item: dict):
        with self.transaction():
            self.db.execute("UPDATE calls SET state='awaiting_approval' WHERE session_id=? AND batch=? AND position=? AND state='running'",
                            (session_id, item["batch"], item["position"]))

    def end_approval(self, session_id: str, item: dict):
        with self.transaction():
            self.db.execute("UPDATE calls SET state='running' WHERE session_id=? AND batch=? AND position=? AND state='awaiting_approval'",
                            (session_id, item["batch"], item["position"]))

    def finish_call(self, session_id: str, item: dict, result: dict):
        with self.transaction():
            state = self.get(session_id)
            cursor = self.db.execute("UPDATE calls SET state=?,result=? WHERE session_id=? AND batch=? AND position=? AND state IN ('running','uncertain')",
                ("failed" if result["content"].startswith("ERROR:") else "completed", encoded(result),
                 session_id, item["batch"], item["position"]))
            if cursor.rowcount != 1:
                raise SessionError("Cannot checkpoint this tool result")
            self.db.execute("UPDATE sessions SET messages=?,updated_at=? WHERE id=?",
                            (encoded([*state.messages, result]), stamp(), session_id))
            self._event(session_id, {"event": "tool_result", **result})
            if item["call"]["function"]["name"] == "submit_plan" and not result["content"].startswith("ERROR:"):
                plan = validate_plan(json.loads(item["call"]["function"]["arguments"]))
                saved = self.plan(session_id)
                if saved and saved["data"] == plan:
                    return
                if not saved or saved["state"] != "planning":
                    raise SessionError("Cannot replace a saved plan")
                self.db.execute("UPDATE plans SET state='awaiting_approval',data=?,updated_at=? WHERE session_id=?",
                                (encoded(plan), stamp(), session_id))
                self.db.execute("UPDATE sessions SET status='waiting_plan_approval' WHERE id=?", (session_id,))
                self._event(session_id, {"event": "plan_saved", "plan": plan})

    def complete(self, session_id: str, answer: str):
        with self.transaction():
            self.db.execute("UPDATE sessions SET status='completed',final_answer=?,error=NULL,updated_at=? WHERE id=?", (answer, stamp(), session_id))
            self._event(session_id, {"event": "final", "content": answer})

    def recover(self, session_id: str):
        with self.transaction():
            for item in self.calls(session_id, True):
                if item["state"] != "running":
                    continue
                new_state = "uncertain"
                self.db.execute("UPDATE calls SET state=? WHERE session_id=? AND batch=? AND position=?",
                                (new_state, session_id, item["batch"], item["position"]))
                self._event(session_id, {"event": "recovery", "call": item["key"], "state": new_state})
        uncertain = [c for c in self.calls(session_id, True) if c["state"] == "uncertain"]
        if uncertain:
            self.set_status(session_id, "blocked", "UncertainActionError")
            raise UncertainActionError(f"Call {uncertain[0]['key']} has an unknown result; inspect the workspace and use resolve before resuming")

    def resolve(self, session_id: str, key: str, resolution: str, verify_applied=None):
        with self.lock(session_id):
            self._resolve(session_id, key, resolution, verify_applied)

    def _resolve(self, session_id: str, key: str, resolution: str, verify_applied):
        if self.get(session_id).status == "aborted":
            raise SessionError("This session was terminated by the operator")
        calls = self.calls(session_id, True)
        if not calls or calls[0]["key"] != key or calls[0]["state"] not in {"running", "uncertain"}:
            raise SessionError("Resolve only the first uncertain call")
        item = calls[0]
        if resolution == "retry":
            if item["call"]["function"]["name"] not in {"write_file", "delete_file", "read_file", "list_files", "Bash", "bash", "submit_plan"}:
                raise SessionError("This instrument has no supported safe retry contract")
            with self.transaction():
                self.db.execute("UPDATE calls SET state='pending' WHERE session_id=? AND batch=? AND position=?",
                    (session_id, item["batch"], item["position"]))
                self._event(session_id, {"event": "recovery_resolution", "call": key, "resolution": resolution})
        elif resolution == "abort":
            with self.transaction():
                self.db.execute("UPDATE sessions SET status='aborted',error='RecoveryAborted',updated_at=? WHERE id=?",
                                (stamp(), session_id))
                self._event(session_id, {"event": "recovery_resolution", "call": key, "resolution": resolution})
            return
        elif resolution in {"applied", "skip"}:
            function = item["call"]["function"]
            if resolution == "applied" and (function["name"] != "write_file" or
                    verify_applied is None or verify_applied(item) is not True):
                raise SessionError("The current file does not verify the proposed write; inspect it before resolving")
            content = ("RECOVERY_CONFIRMED: operator verified the write; check tests before finishing" if resolution == "applied"
                       else "ERROR: RECOVERY_SKIPPED: operator declined repeating the uncertain action; inspect the workspace")
            result = {"role": "tool", "tool_call_id": item["call"]["id"], "name": function["name"], "content": content}
            self.finish_call(session_id, item, result)
            self.event(session_id, {"event": "recovery_resolution", "call": key, "resolution": resolution})
        else:
            raise SessionError("Unknown recovery resolution")
        self.set_status(session_id, "interrupted")
