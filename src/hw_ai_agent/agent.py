from __future__ import annotations

import copy
import json
import math
import uuid
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from .client import ModelClient
from .tools import ToolError, ToolRuntime
from .policy import ApprovalDeferred, Policy
from .session import SessionError, SessionStore, UncertainActionError, PlanRejectedError
from .budget import BudgetLimits, BudgetExceeded, MemoryBudget, output_ceiling
from .journal import ChangeJournal, JournalError, JournalConflict
from .tasks import TASK_INSTRUCTIONS, TASK_PLAN_SCHEMA, TASK_SCHEMAS, validate_task_plan, verify_task, TaskError
from .planning import PLAN_SCHEMA, PLAN_INSTRUCTIONS, validate_plan, PlanError, PlanApprovalDeferred


DEFAULT_SYSTEM_PROMPT = """You are a careful coding agent.
Use the provided tools to inspect and modify only the workspace.
Run the relevant tests after changes.
Tests are read-only. Never modify, rename or delete them.
You may create, edit or remove workspace files when justified by the task.
For testing, call Bash with python -m unittest discover -v.
Bash accepts ONLY python -m unittest, pwd, ls, cat and sleep.
Do not use python -c, run scripts, or create scratch check scripts.
Use path='.' for list_files; never use an empty path.
After relevant tests pass, immediately provide your final summary without more tool calls.
When a tool reports an error, correct the request or explain the limitation.
Runtime permissions are authoritative. If the operator refuses an action,
do not retry it or find a workaround; explain the refusal in your final answer.
"""


class AgentError(RuntimeError):
    """Raised when the agent loop cannot finish safely."""


class AgentBudgetExceeded(BudgetExceeded, AgentError, SessionError):
    pass


class CodingAgent:
    def __init__(
        self,
        client: ModelClient,
        runtime: ToolRuntime,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_rounds: int = 12,
        max_context_chars: int = 120_000,
        budget_limits: BudgetLimits | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        store: SessionStore | None = None,
        session_metadata: dict | None = None,
        plan_mode: bool = False,
        task_mode: bool = False,
        journal_mode: bool = False,
        attribute_changes: Callable[[dict], str | None] | None = None,
        approve_plan: Callable[[dict], bool] | None = None,
        resolve_uncertain: Callable[[dict], str | None] | None = None,
        verify_recovery: Callable[[dict], bool] | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if type(journal_mode) is not bool:
            raise ValueError("journal_mode must be a boolean")
        if type(task_mode) is not bool:
            raise ValueError("task_mode must be a boolean")
        if type(plan_mode) is not bool:
            raise ValueError("plan_mode must be a boolean")
        self.client = client
        self.runtime = runtime
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.max_context_chars = max_context_chars
        self.budget_limits = budget_limits or BudgetLimits(model_requests=max_rounds)
        if self.budget_limits.model_requests != max_rounds:
            raise ValueError("Model budget must equal max_rounds")
        self._memory_budget = MemoryBudget(self.budget_limits)
        self.on_event = on_event
        self.events: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.store = store
        self.session_metadata = dict(session_metadata or {})
        self.plan_mode = plan_mode or task_mode
        self.task_mode = task_mode
        self._pending_task_outcome = None
        self.journal_mode, self.journal = journal_mode, None
        self.attribute_changes = attribute_changes
        self._journal_failure = None
        self.approve_plan = approve_plan
        self.resolve_uncertain = resolve_uncertain
        self.verify_recovery = verify_recovery
        self.session_id: str | None = None
        self._call_records: dict[str, tuple[tuple, dict | None]] = {}
        self._failed_call_ids: set[str] = set()
        self._original_policy_callback = getattr(runtime, "_original_agent_policy_callback", runtime.on_policy_event)
        runtime._original_agent_policy_callback = self._original_policy_callback
        runtime.on_policy_event = self._policy_event

    def run(self, request: str) -> str:
        if not request.strip():
            raise ValueError("request must not be empty")
        if (self.plan_mode or self.journal_mode) and not self.store:
            raise SessionError("Plan Mode requires a session store")
        self.runtime.phase = "planning" if self.plan_mode else "execution"
        self.events = []
        self._memory_budget = MemoryBudget(self.budget_limits)
        self._call_records = {}
        self._failed_call_ids = set()
        self.runtime.session_permissions = set()
        self.messages = [
            {"role": "system", "content": self.system_prompt + (PLAN_INSTRUCTIONS if self.plan_mode else "") + (TASK_INSTRUCTIONS if self.task_mode else "")},
            {"role": "user", "content": request},
        ]
        if self.store:
            self.session_id = self.store.create(self._binding(), self.messages, self.max_rounds, self.max_context_chars, self.budget_limits)
            return self._stored_run(self.session_id, "session_created")
        return self._loop(0)

    def _binding(self) -> dict:
        allowed_metadata = {"kind", "policy_path", "policy_source", "approvals", "max_tokens", "demo"}
        if set(self.session_metadata) - allowed_metadata:
            raise SessionError("Unsupported session metadata fields")
        metadata = {**self.session_metadata, "workspace": str(self.runtime.root),
                    "execution_mode": self.runtime.mode,
                    "approval_contract": 2,
                    "plan_mode": self.plan_mode,
                    "task_mode": self.task_mode,
                    "journal_mode": self.journal_mode,
                    "trash_dir": str(self.runtime.trash_dir),
                    "policy": self.runtime.policy.as_dict(),
                    "runtime_limits": {name: getattr(self.runtime, name) for name in
                        ("command_timeout", "max_output_chars", "max_file_chars")}}
        settings = getattr(self.client, "settings", None)
        if settings is not None:
            # Explicit whitelist: API credentials and client internals never enter metadata.
            metadata.update({name: getattr(settings, name) for name in ("provider", "model", "base_url")})
        metadata.setdefault("max_tokens", getattr(self.client, "max_tokens", 2048))
        return metadata

    def _check_binding(self, metadata: dict):
        current = self._binding()
        for key in ("workspace", "policy", "runtime_limits", "provider", "model", "base_url", "max_tokens"):
            expected = Policy.restore(metadata[key]).as_dict() if key == "policy" else metadata.get(key)
            if expected != current.get(key):
                raise SessionError(f"Session {key} differs from current configuration; continuation was refused")
        if metadata.get("execution_mode", "safe") != current["execution_mode"]:
            raise SessionError("Session execution mode differs from current configuration")
        if metadata.get("trash_dir", current["trash_dir"]) != current["trash_dir"]:
            raise SessionError("Session recovery directory differs from current configuration")

    def resume(self, session_id: str) -> str:
        if not self.store:
            raise SessionError("Resume requires a session store")
        return self._stored_run(session_id, "session_resumed")

    def _stored_run(self, session_id: str, event: str) -> str:
        assert self.store is not None
        with self.store.lock(session_id):
            state = self.store.get(session_id)
            if state.status in {"aborted", "rolled_back"}:
                raise SessionError("This session was terminated by the operator and cannot be resumed")
            self.store.budget(session_id)  # Validate saved counters before model or tool access.
            if state.status == "budget_exhausted":
                raise SessionError(state.error or "BUDGET_EXHAUSTED: saved budget was exhausted")
            self._check_binding(state.metadata)
            self.plan_mode = state.metadata.get("plan_mode", False)
            if type(self.plan_mode) is not bool:
                raise SessionError("Invalid saved planning mode")
            self.task_mode = state.metadata.get("task_mode", False)
            self.store.tasks(session_id)
            saved_plan = self.store.plan(session_id)
            self.runtime.phase = "planning" if saved_plan and saved_plan["state"] != "approved" else "execution"
            self.session_id = session_id
            self.journal_mode = state.metadata.get("journal_mode", False)
            if type(self.journal_mode) is not bool:
                raise SessionError("Invalid saved journal mode")
            self.journal = ChangeJournal(self.store,session_id,self.runtime) if self.journal_mode else None
            self.runtime.journal = self.journal
            if self.journal:
                head = self.journal.start()
                if head['status'] != 'active':
                    raise JournalError("Finish rollback before attempting to resume the agent")
            self.messages, self.events = copy.deepcopy(state.messages), copy.deepcopy(state.events)
            self._call_records = {}
            self._failed_call_ids = set()
            for item in self.store.calls(session_id):
                self._remember_call(item["call"], item["result"], failed=item["state"] == "failed" if item["result"] is not None else None)
            self.runtime.session_permissions = set()
            self.max_rounds, self.max_context_chars = state.max_rounds, state.max_context_chars
            self.budget_limits = BudgetLimits.restore(self.store.budget(session_id)["limits"])
            if state.status == "completed":
                if self.on_event:
                    self.on_event({"event": "session_cached", "session_id": session_id})
                return state.final_answer or ""
            try:
                while True:
                    try:
                        self.store.recover(session_id)
                        break
                    except UncertainActionError:
                        item = self.store.calls(session_id, True)[0]
                        resolution = self.resolve_uncertain(copy.deepcopy(item)) if self.resolve_uncertain else None
                        if resolution is None:
                            raise
                        ownership = None
                        if self.journal:
                            view = self.journal.recovery_view(item)
                            self.journal.reviewed_tree = view['_current']
                            if view['changed']:
                                ownership = self.attribute_changes({k:v for k,v in view.items() if not k.startswith('_')}) if self.attribute_changes else None
                                if ownership not in {'own','external'}:
                                    raise UncertainActionError("Workspace changes require explicit attribution before recovery")
                        self.store._resolve(session_id, item["key"], resolution, self.verify_recovery, journal=self.journal, changes=ownership)
                        if resolution == "abort":
                            raise SessionError("This session was terminated by the operator")
                        state = self.store.get(session_id)
                        self.messages = copy.deepcopy(state.messages)
                        self._call_records = {}
                        self._failed_call_ids = set()
                        for saved in self.store.calls(session_id):
                            self._remember_call(saved["call"], saved["result"], failed=saved["state"] == "failed" if saved["result"] is not None else None)
                if state.metadata.get("approval_contract") != 2:
                    self.store.upgrade_approval_contract(session_id)
                self.events = copy.deepcopy(self.store.get(session_id).events)
                self.store.set_status(session_id, "running")
                self._emit({"event": event, "session_id": session_id})
                self._emit_budget()
                # A final assistant message may have been committed just before process death.
                last = self.messages[-1]
                if (self.runtime.phase != "planning" and last.get("role") == "assistant"
                        and not last.get("tool_calls") and last.get("content")
                        and (not self.task_mode or all(t["state"] == "completed" for t in self.store.tasks(session_id)))):
                    answer = last["content"]
                    self.store.complete(session_id, answer)
                    self._emit({"event": "final", "content": answer}, persist=False)
                    return answer
                return self._loop(state.rounds)
            except BaseException as exc:
                status = ("journal_conflict" if isinstance(exc, JournalError) else
                          "budget_exhausted" if isinstance(exc, BudgetExceeded) else
                          "blocked" if isinstance(exc, UncertainActionError) else
                          "waiting_plan_approval" if isinstance(exc, PlanApprovalDeferred) else
                          "aborted" if isinstance(exc, PlanRejectedError) or self.store.get(session_id).status == "aborted" else
                          "waiting_approval" if isinstance(exc, ApprovalDeferred) else
                          "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed")
                try:
                    self.store.set_status(session_id, status, str(exc) if isinstance(exc, (BudgetExceeded, JournalError)) else type(exc).__name__)
                    if isinstance(exc, BudgetExceeded):
                        self._emit_budget()
                except SessionError:
                    pass
                raise

    def _loop(self, rounds: int) -> str:
        if self.store and self.session_id:
            self.store.reserve_pending(self.session_id)
            for item in self.store.calls(self.session_id, True):
                self._registered_call(item)
            self._review_plan()
        while rounds < self.max_rounds:
            if len(json.dumps(self.messages, ensure_ascii=False)) > self.max_context_chars:
                raise AgentError("Conversation exceeds the context size limit")
            if self.store and self.session_id:
                rounds = self.store.start_model(self.session_id)
            else:
                self._memory_budget.charge(model_requests=1)
                rounds += 1
            self._emit({"event": "model_request", "round": rounds}, persist=False)
            self._emit_budget()
            schemas = self.runtime.tool_schemas
            if self.runtime.phase == "planning":
                schemas = [*schemas, copy.deepcopy(TASK_PLAN_SCHEMA if self.task_mode else PLAN_SCHEMA)]
            elif self.task_mode:
                task = self.store.current_task(self.session_id)
                schemas = ([schema for schema in schemas if schema["function"]["name"] in task["tools"]] + copy.deepcopy(TASK_SCHEMAS)) if task else []
            response = self.client.complete(self.messages, schemas)
            if self.store and getattr(self.client, "usage", []):
                last_usage = self.client.usage[-1]
                usage = {key: last_usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
                         if type(last_usage.get(key)) in (int, float) and math.isfinite(last_usage[key])}
                self._emit({"event": "model_usage", "usage": usage})
            assistant = self._assistant_message(response)
            tool_calls = assistant.get("tool_calls") or []
            for call in tool_calls:
                self._remember_call(call)
            if not tool_calls and (not isinstance(assistant.get("content"), str) or not assistant["content"].strip()):
                raise AgentError("Model returned neither tool calls nor a final answer")
            if self.store and self.session_id:
                self.store.assistant(self.session_id, assistant, rounds)
            else:
                cached = {key for key, (_, result) in self._call_records.items() if result is not None}
                self._memory_budget.batch(tool_calls, cached)
            self.messages.append(assistant)
            if not tool_calls:
                content = assistant.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise AgentError("Model returned neither tool calls nor a final answer")
                if self.runtime.phase == "planning":
                    message = {"role": "user", "content": "Submit the plan with submit_plan; a text answer is not an approved plan."}
                    self.store.message(self.session_id, message)
                    self.messages.append(message)
                    continue
                if self.task_mode and any(t["state"] != "completed" for t in self.store.tasks(self.session_id)):
                    message = {"role": "user", "content": "TASKS_NOT_COMPLETE: finish CURRENT_TASK with verified complete_task; your final text is not evidence."}
                    self.store.message(self.session_id, message)
                    self.messages.append(message)
                    continue
                if self.store and self.session_id:
                    self.store.complete(self.session_id, content)
                self._emit({"event": "final", "content": content}, persist=False)
                return content

            if self.store and self.session_id:
                for item in self.store.calls(self.session_id, True):
                    self._registered_call(item)
                self._review_plan()
                continue
            for index, call in enumerate(tool_calls):
                quota = self._memory_budget.reserve_output(output_ceiling(call["function"]["name"], self.runtime), len(tool_calls) - index - 1)
                self._emit({"event": "tool_call", "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"]})
                with self.runtime.output_budget(quota):
                    tool_result = self._execute_tool_call(call)
                tool_result = {**tool_result, "content": self._memory_budget.finish_output(tool_result["content"], quota)}
                self._remember_call(call, tool_result)
                self.messages.append(tool_result)
                self._emit({"event": "tool_result", **tool_result})
                self._emit_budget()

        raise AgentBudgetExceeded("model_requests", 1, 0)

    def _review_plan(self):
        if not self.store or not self.session_id:
            return
        saved = self.store.plan(self.session_id)
        if not saved:
            return
        if saved["state"] == "awaiting_approval":
            if self.task_mode:
                self._emit({"event": "tasks", "tasks": self.store.tasks(self.session_id)}, persist=False)
            self._emit({"event": "plan_waiting", "plan": saved["data"]})
            if self.approve_plan is None:
                raise PlanApprovalDeferred("План сохранён и ожидает явного утверждения")
            try:
                decision = self.approve_plan(copy.deepcopy(saved["data"]))
            except PlanApprovalDeferred:
                raise
            except Exception:
                raise PlanApprovalDeferred("Ответ на план не получен; решение не сохранено") from None
            if type(decision) is not bool:
                raise PlanApprovalDeferred("План требует явного ответа: утвердить или отклонить")
            self.store._decide_plan(self.session_id, decision)
            state = self.store.get(self.session_id)
            self.messages, self.events = copy.deepcopy(state.messages), copy.deepcopy(state.events)
            if not decision:
                raise PlanRejectedError("План отклонён. Сессия завершена без исполнения плана.")
        saved = self.store.plan(self.session_id)
        self.runtime.phase = "execution" if saved["state"] == "approved" else "planning"
        if self.runtime.phase == "execution":
            self.store.set_status(self.session_id, "running")
            if self.task_mode:
                self._advance_task()

    def _advance_task(self):
        task = self.store.activate_task(self.session_id)
        tasks = self.store.tasks(self.session_id)
        self.messages = copy.deepcopy(self.store.get(self.session_id).messages)
        self._emit({"event": "tasks", "tasks": tasks}, persist=False)
        if task is None:
            if any(t["state"] != "completed" for t in tasks):
                raise AgentError("TASKS_FAILED: task failure blocks unfinished dependencies")
            if not self.messages[-1].get("content", "").startswith("ALL_TASKS_COMPLETED"):
                message = {"role": "user", "content": "ALL_TASKS_COMPLETED: return your final summary without more tool calls."}
                self.store.message(self.session_id, message)
                self.messages.append(message)

    @staticmethod
    def _call_signature(call: dict) -> tuple:
        function = call["function"]
        try:
            arguments = json.dumps(json.loads(function["arguments"]), sort_keys=True, ensure_ascii=False)
        except (ValueError, TypeError):
            arguments = function["arguments"]
        return function["name"], arguments

    def _remember_call(self, call: dict, result: dict | None = None, *, failed: bool | None = None):
        signature = self._call_signature(call)
        previous = self._call_records.get(call["id"])
        if previous is not None and previous[0] != signature:
            raise AgentError("PROTOCOL_ERROR: repeated tool_call_id has different name or arguments")
        self._call_records[call["id"]] = (signature, result or (previous[1] if previous else None))
        if failed is not None:
            if failed:
                self._failed_call_ids.add(call["id"])
            else:
                self._failed_call_ids.discard(call["id"])

    def _registered_call(self, item: dict):
        assert self.store is not None and self.session_id is not None
        call = item["call"]
        quota = self.store.start_call(self.session_id, item, output_ceiling(call["function"]["name"], self.runtime))
        self._active_call = item
        self._pending_task_outcome = None
        self._journal_failure = None
        if self.journal:self.journal.active_call = item
        self._emit({"event": "tool_call", "name": call["function"]["name"],
                    "task_id": item.get("task_id"), "phase": item.get("phase"),
                    "arguments": call["function"]["arguments"]}, persist=False)
        try:
            with self.runtime.output_budget(quota):
                result = self._execute_tool_call(call)
        finally:
            self._active_call = None
            if self.journal:self.journal.active_call = None
        outcome = self._pending_task_outcome
        self._pending_task_outcome = None
        if outcome:
            result = self.store.finish_call(self.session_id, item, result, call["id"] in self._failed_call_ids, task_outcome=outcome)
        else:
            result = self.store.finish_call(self.session_id, item, result, call["id"] in self._failed_call_ids)
        self._remember_call(call, result)
        self.messages.append(result)
        self._emit({"event": "tool_result", "task_id": item.get("task_id"), "phase": item.get("phase"), **result}, persist=False)
        self._emit_budget()
        if self._journal_failure:
            raise self._journal_failure
        if outcome:
            self._emit({"event": "tasks", "tasks": self.store.tasks(self.session_id)}, persist=False)
            self._emit({"event": "task_" + outcome["state"], "task_id": outcome["task_id"]}, persist=False)

    @property
    def budget(self) -> dict:
        return self.store.budget(self.session_id) if self.store and self.session_id else copy.deepcopy(self._memory_budget.state)

    def _emit_budget(self):
        self._emit({"event": "budget", "budget": self.budget}, persist=False)

    @staticmethod
    def _assistant_message(response: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(response, Mapping):
            raise AgentError("model returned an invalid assistant message")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": response.get("content"),
        }
        tool_calls = response.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise AgentError("model returned invalid tool_calls")
            if len(tool_calls) > 16:
                raise AgentError("Too many tool calls in one model response")
            normalized = []
            seen_ids = set()
            for call in tool_calls:
                if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
                    raise AgentError("Malformed tool call from model")
                function = call["function"]
                if not isinstance(function.get("name"), str) or not function["name"]:
                    raise AgentError("Tool call is missing its function name")
                call_id = call.get("id") or "call_" + uuid.uuid4().hex
                if not isinstance(call_id, str) or call_id in seen_ids:
                    raise AgentError("Invalid or duplicate tool call ID")
                seen_ids.add(call_id)
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                normalized.append({"id": call_id, "type": "function", "function": {
                    "name": function["name"], "arguments": arguments,
                }})
            message["tool_calls"] = normalized
        # Some reasoning providers require these fields on replay.
        for field in ("reasoning_content", "reasoning_details"):
            if field in response:
                message[field] = copy.deepcopy(response[field])
        return message

    def _emit(self, event: dict[str, Any], *, persist=True) -> None:
        self.events.append(copy.deepcopy(event))
        if persist and self.store and self.session_id:
            self.store.event(self.session_id, event)
        if self.on_event:
            self.on_event(event)

    def _policy_event(self, event: dict[str, Any]):
        if (event.get("outcome") == "waiting" and self.store and self.session_id
                and getattr(self, "_active_call", None) is not None):
            self.store.wait_for_approval(self.session_id, self._active_call)
        elif (event.get("outcome") in {"approved", "refused"} and self.store and self.session_id
                and getattr(self, "_active_call", None) is not None):
            self.store.end_approval(self.session_id, self._active_call)
        self._emit(event)
        callback = self._original_policy_callback
        if callback is not None and callback is not self.on_event:
            callback(event)

    def _execute_tool_call(self, call: Any) -> dict[str, Any]:
        if isinstance(call, Mapping) and isinstance(call.get("id"), str):
            previous = self._call_records.get(call["id"])
            if previous is not None:
                if previous[0] != self._call_signature(call):
                    raise AgentError("PROTOCOL_ERROR: repeated tool_call_id has different name or arguments")
                if previous[1] is not None:
                    return copy.deepcopy(previous[1])
        call_id = "unknown"
        name = "unknown"
        failed = False
        try:
            if not isinstance(call, Mapping):
                raise ToolError("tool call must be an object")
            call_id = str(call.get("id") or "unknown")
            function = call.get("function")
            if not isinstance(function, Mapping):
                raise ToolError("tool call has no function object")
            name = str(function.get("name") or "unknown")
            raw_arguments = function.get("arguments", "{}")
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            if not isinstance(arguments, Mapping):
                raise ToolError("tool arguments must be a JSON object")
            if name == "submit_plan":
                if self.runtime.phase != "planning" or not self.store or not self.session_id:
                    raise ToolError("submit_plan is available only during planning")
                if self.store.plan(self.session_id)["state"] != "planning":
                    raise ToolError("PLAN_ALREADY_SAVED: wait for the operator decision")
                (validate_task_plan if self.task_mode else validate_plan)(dict(arguments))
                content = "PLAN_SAVED: runtime will request explicit operator approval before execution"
            elif name in {"complete_task", "fail_task"}:
                content = self._execute_task_control(name, dict(arguments))
            else:
                if self.task_mode and self.runtime.phase == "execution":
                    task = self.store.current_task(self.session_id)
                    normalized = "Bash" if name == "bash" else name
                    if not task or normalized not in task["tools"]:
                        raise TaskError("TASK_TOOL_DENIED: tool is not allowed for CURRENT_TASK")
                content = self.runtime.execute(name, arguments)
        except JournalConflict as exc:
            self._journal_failure = exc
            failed = True
            content = f"ERROR: {exc}"
        except (ToolError, PlanError, json.JSONDecodeError, OSError) as exc:
            failed = True
            content = f"ERROR: {exc}"

        result = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }
        if call_id != "unknown":
            self._remember_call(call, result, failed=failed)
        return result


    def _execute_task_control(self, name: str, arguments: dict) -> str:
        if not self.task_mode or self.runtime.phase != "execution" or not self.store or not self.session_id:
            raise TaskError("Task controls require approved task mode")
        keys = {"task_id"} if name == "complete_task" else {"task_id", "reason"}
        if set(arguments) != keys or any(not isinstance(v, str) or not v.strip() or len(v) > 4000 for v in arguments.values()):
            raise TaskError("Invalid task control arguments")
        task = self.store.current_task(self.session_id)
        if not task or arguments["task_id"] != task["id"]:
            raise TaskError("TASK_NOT_CURRENT: runtime chooses the next eligible task")
        item = getattr(self, "_active_call", None)
        pending = self.store.calls(self.session_id, True)
        if not item or pending[-1]["key"] != item["key"]:
            raise TaskError("TASK_BOUNDARY: complete_task/fail_task must be the last call in its batch")
        if name == "complete_task":
            # This read checks recorded execution, not a model-written claim.
            try:
                with self.runtime.output_budget(None):
                    proof = verify_task(task, self.store.calls(self.session_id), self.runtime)
            except (OSError, UnicodeError):
                raise TaskError("TASK_NOT_VERIFIED: cannot safely inspect the result") from None
            self._pending_task_outcome = {"task_id": task["id"], "state": "completed", "proof": proof}
            return "TASK_COMPLETED: " + json.dumps(proof, ensure_ascii=False)
        self._pending_task_outcome = {"task_id": task["id"], "state": "failed", "error": arguments["reason"]}
        return "TASK_FAILED: " + arguments["reason"]
