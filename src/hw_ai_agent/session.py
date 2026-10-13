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
from .budget import BudgetLimits, BudgetExceeded, DIMENSIONS, snapshot, debit, batch_cost, truncate_result


class SessionError(RuntimeError):
    pass


class SessionBudgetExceeded(BudgetExceeded, SessionError):
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
            if version not in (0, 1, 2, 3):
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
                CREATE TABLE IF NOT EXISTS budgets (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id),
                    limits TEXT NOT NULL, spent TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS call_budgets (
                    session_id TEXT NOT NULL, batch INTEGER NOT NULL, position INTEGER NOT NULL,
                    paid INTEGER NOT NULL, output_quota INTEGER, forfeited INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(session_id,batch,position),
                    FOREIGN KEY(session_id,batch,position) REFERENCES calls(session_id,batch,position)
                );
            ''')
            if version < 3:
                self._migrate_budgets()
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

    def _migrate_budgets(self):
        with self.transaction():
            for row in self.db.execute("SELECT id FROM sessions").fetchall():
                state = self.get(row["id"])
                if self.db.execute("SELECT 1 FROM budgets WHERE session_id=?", (state.id,)).fetchone():
                    continue  # Existing counters are never reset, even by a legacy version marker.
                calls = self.calls(state.id)
                attempts = [e for e in state.events if e.get("event") == "tool_call"]
                spent = {"model_requests": state.rounds, "tool_calls": len(attempts),
                         "side_effects": sum(e.get("name") in {"write_file", "delete_file", "Bash", "bash"} for e in attempts),
                         "result_bytes": sum(len(m.get("content", "").encode("utf-8")) for m in state.messages if m.get("role") == "tool")}
                defaults = BudgetLimits(model_requests=state.max_rounds).as_dict()
                limits = {k: max(defaults[k], spent[k]) for k in DIMENSIONS}
                self.db.execute("INSERT INTO budgets VALUES (?,?,?)", (state.id, encoded(limits), encoded(spent)))
                for item in calls:
                    paid = int(any(e.get("call") == item["key"] for e in attempts))
                    self.db.execute("INSERT INTO call_budgets VALUES (?,?,?,?,NULL,0)",
                                    (state.id, item["batch"], item["position"], paid))
                metadata = {**state.metadata, "budget_contract": 1}
                self.db.execute("UPDATE sessions SET metadata=? WHERE id=?", (encoded(metadata), state.id))
                self._event(state.id, {"event": "budget_migrated", "budget": snapshot(limits, spent)})
            self.db.execute("PRAGMA user_version=3")

    def budget(self, session_id: str) -> dict:
        state = self.get(session_id)
        row = self.db.execute("SELECT * FROM budgets WHERE session_id=?", (session_id,)).fetchone()
        try:
            if row is None:
                raise ValueError("Missing budget")
            budget = snapshot(json.loads(row["limits"]), json.loads(row["spent"]))
            if budget["spent"]["model_requests"] != state.rounds or budget["limits"]["model_requests"] != state.max_rounds:
                raise ValueError("Model counter differs")
            paid = self.db.execute("SELECT COALESCE(SUM(paid),0) FROM call_budgets WHERE session_id=?", (session_id,)).fetchone()[0]
            held = self.db.execute("SELECT COALESCE(SUM(output_quota),0) FROM call_budgets WHERE session_id=?", (session_id,)).fetchone()[0]
            delivered = sum(len(m.get("content", "").encode("utf-8")) for m in state.messages if m.get("role") == "tool")
            if paid > budget["spent"]["tool_calls"] or held + delivered > budget["spent"]["result_bytes"]:
                raise ValueError("Counters do not cover saved calls/results")
            return budget
        except (ValueError, TypeError, KeyError):
            raise SessionError("Invalid saved budget; continuation refused") from None

    def _charge(self, session_id: str, **cost):
        try:
            budget = debit(self.budget(session_id), **cost)
        except BudgetExceeded as exc:
            raise SessionBudgetExceeded(exc.dimension, exc.requested, exc.remaining) from None
        self.db.execute("UPDATE budgets SET spent=? WHERE session_id=?", (encoded(budget["spent"]), session_id))
        self._event(session_id, {"event": "budget_debit", "cost": cost, "spent": budget["spent"]})
        return budget

    def _allocation(self, session_id, item):
        row = self.db.execute("SELECT * FROM call_budgets WHERE session_id=? AND batch=? AND position=?",
                              (session_id, item["batch"], item["position"])).fetchone()
        if (row is None or row["paid"] not in (0, 1) or row["forfeited"] not in (0, 1)
                or row["output_quota"] is not None and (type(row["output_quota"]) is not int or row["output_quota"] < 1)):
            raise SessionError("Invalid saved call budget")
        if row["output_quota"] is not None and row["output_quota"] > self.budget(session_id)["spent"]["result_bytes"]:
            raise SessionError("Saved output reservation exceeds charged budget")
        return row

    def reserve_pending(self, session_id: str):
        # Covers legacy pending batches and explicit retries before any call starts.
        with self.transaction():
            pending = self.calls(session_id, True)
            unpaid = [i for i in pending if not self._allocation(session_id, i)["paid"]]
            unreserved = sum(self._allocation(session_id, i)["output_quota"] is None for i in pending)
            available = self.budget(session_id)["remaining"]["result_bytes"]
            if unreserved > available:
                raise SessionBudgetExceeded("result_bytes", unreserved, available)
            if unpaid:
                cached = {i["call"]["id"] for i in self.calls(session_id) if i["result"] is not None}
                self._charge(session_id, **batch_cost([i["call"] for i in unpaid], cached))
                for item in unpaid:
                    self.db.execute("UPDATE call_budgets SET paid=1 WHERE session_id=? AND batch=? AND position=?",
                                    (session_id, item["batch"], item["position"]))

    def create(self, metadata: dict, messages: list, max_rounds: int, max_context_chars: int,
               budgets: BudgetLimits | None = None) -> str:
        workspace = Path(metadata["workspace"]).resolve()
        if self.path.is_relative_to(workspace):
            raise SessionError("Session database must be outside the model workspace")
        if type(max_rounds) is not int or max_rounds < 1 or type(max_context_chars) is not int or max_context_chars < 1:
            raise SessionError("Invalid session limits")
        budgets = budgets or BudgetLimits(model_requests=max_rounds)
        if budgets.model_requests != max_rounds:
            raise SessionError("Model budget differs from max_rounds")
        metadata = {**metadata, "budget_contract": 1}
        session_id = uuid.uuid4().hex
        now = stamp()
        with self.transaction():
            self.db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (session_id, encoded(metadata), encoded(messages), "[]", "ready", 0,
                 max_rounds, max_context_chars, None, None, now, now))
            self.db.execute("INSERT INTO budgets VALUES (?,?,?)", (session_id, encoded(budgets.as_dict()), encoded(dict.fromkeys(DIMENSIONS, 0))))
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
            if row["status"] not in {"ready", "running", "interrupted", "waiting_approval", "waiting_plan_approval", "failed", "blocked", "completed", "aborted", "budget_exhausted"}:
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
            self._charge(session_id, model_requests=1)
            number = state.rounds + 1
            self.db.execute("UPDATE sessions SET rounds=?,updated_at=? WHERE id=?", (number, stamp(), session_id))
            self._event(session_id, {"event": "model_request", "round": number})
        return number

    def assistant(self, session_id: str, message: dict, batch: int):
        failure = None
        with self.transaction():
            state = self.get(session_id)
            calls = message.get("tool_calls") or []
            cached = {i["call"]["id"] for i in self.calls(session_id) if i["result"] is not None}
            try:
                available = self.budget(session_id)["remaining"]["result_bytes"]
                if len(calls) > available:
                    raise SessionBudgetExceeded("result_bytes", len(calls), available)
                self._charge(session_id, **batch_cost(calls, cached))
            except SessionBudgetExceeded as exc:
                failure = exc
                self._event(session_id, {"event": "batch_rejected", "assistant": message, "reason": str(exc)})
                self.db.execute("UPDATE sessions SET status='budget_exhausted',error=?,updated_at=? WHERE id=?",
                                (str(exc), stamp(), session_id))
            else:
                self.db.execute("UPDATE sessions SET messages=?,updated_at=? WHERE id=?",
                    (encoded([*state.messages, message]), stamp(), session_id))
                for position, call in enumerate(calls):
                    self.db.execute("INSERT INTO calls VALUES (?,?,?,?,?,NULL)",
                        (session_id, batch, position, encoded(call), "pending"))
                    self.db.execute("INSERT INTO call_budgets VALUES (?,?,?,?,NULL,0)",
                                    (session_id, batch, position, 1))
        if failure:
            raise failure

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

    def start_call(self, session_id: str, item: dict, output_ceiling: int = 1024) -> int:
        if type(output_ceiling) is not int or output_ceiling < 1:
            raise SessionError("Invalid output ceiling")
        with self.transaction():
            allocation = self._allocation(session_id, item)
            if not allocation["paid"]:
                raise SessionError("Tool batch budget has not been reserved")
            quota = allocation["output_quota"]
            if quota is None:
                future = sum(i["key"] != item["key"] and self._allocation(session_id, i)["output_quota"] is None
                             for i in self.calls(session_id, True))
                available = self.budget(session_id)["remaining"]["result_bytes"]
                if available <= future:
                    raise SessionBudgetExceeded("result_bytes", future + 1, available)
                quota = min(output_ceiling, available - future)
                self._charge(session_id, result_bytes=quota)
                self.db.execute("UPDATE call_budgets SET output_quota=? WHERE session_id=? AND batch=? AND position=?",
                                (quota, session_id, item["batch"], item["position"]))
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
        return quota

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
            allocation = self._allocation(session_id, item)
            quota = allocation["output_quota"]
            failed = result["content"].startswith("ERROR:")
            if quota is None:
                # Legacy uncertain calls have no reservation from the old runtime.
                quota = max(1, len(result["content"].encode("utf-8")))
                self._charge(session_id, result_bytes=quota)
            result = {**result, "content": truncate_result(result["content"], quota)}
            if not allocation["forfeited"] and allocation["output_quota"] is not None:
                budget = self.budget(session_id)
                spent = {**budget["spent"], "result_bytes": budget["spent"]["result_bytes"] - quota + len(result["content"].encode("utf-8"))}
                self.db.execute("UPDATE budgets SET spent=? WHERE session_id=?", (encoded(spent), session_id))
            cursor = self.db.execute("UPDATE calls SET state=?,result=? WHERE session_id=? AND batch=? AND position=? AND state IN ('running','uncertain')",
                ("failed" if failed else "completed", encoded(result),
                 session_id, item["batch"], item["position"]))
            if cursor.rowcount != 1:
                raise SessionError("Cannot checkpoint this tool result")
            self.db.execute("UPDATE call_budgets SET output_quota=NULL WHERE session_id=? AND batch=? AND position=?",
                            (session_id, item["batch"], item["position"]))
            self.db.execute("UPDATE sessions SET messages=?,updated_at=? WHERE id=?",
                            (encoded([*state.messages, result]), stamp(), session_id))
            self._event(session_id, {"event": "tool_result", **result})
            self._event(session_id, {"event": "budget_settlement", "call": item["key"], "quota": quota,
                                     "result_bytes": len(result["content"].encode("utf-8")),
                                     "forfeited": bool(allocation["forfeited"])})
            if item["call"]["function"]["name"] == "submit_plan" and not failed:
                plan = validate_plan(json.loads(item["call"]["function"]["arguments"]))
                saved = self.plan(session_id)
                if saved and saved["data"] == plan:
                    return result
                if not saved or saved["state"] != "planning":
                    raise SessionError("Cannot replace a saved plan")
                self.db.execute("UPDATE plans SET state='awaiting_approval',data=?,updated_at=? WHERE session_id=?",
                                (encoded(plan), stamp(), session_id))
                self.db.execute("UPDATE sessions SET status='waiting_plan_approval' WHERE id=?", (session_id,))
                self._event(session_id, {"event": "plan_saved", "plan": plan})
        return result

    def complete(self, session_id: str, answer: str):
        with self.transaction():
            self.db.execute("UPDATE sessions SET status='completed',final_answer=?,error=NULL,updated_at=? WHERE id=?", (answer, stamp(), session_id))
            self._event(session_id, {"event": "final", "content": answer})

    def recover(self, session_id: str):
        with self.transaction():
            # Reservations surviving interruption are never released afterwards.
            self.db.execute("UPDATE call_budgets SET forfeited=1 WHERE session_id=? AND output_quota IS NOT NULL AND (batch,position) IN (SELECT batch,position FROM calls WHERE session_id=? AND state IN ('running','uncertain','awaiting_approval'))", (session_id, session_id))
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
                self.db.execute("UPDATE call_budgets SET paid=0,output_quota=NULL,forfeited=0 WHERE session_id=? AND batch=? AND position=?",
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
            with self.transaction():
                self.db.execute("UPDATE call_budgets SET forfeited=1 WHERE session_id=? AND batch=? AND position=?",
                                (session_id, item["batch"], item["position"]))
            self.finish_call(session_id, item, result)
            self.event(session_id, {"event": "recovery_resolution", "call": key, "resolution": resolution})
        else:
            raise SessionError("Unknown recovery resolution")
        self.set_status(session_id, "interrupted")
