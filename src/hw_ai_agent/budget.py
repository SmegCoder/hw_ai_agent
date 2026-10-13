from __future__ import annotations

from dataclasses import asdict, dataclass


DIMENSIONS = ("model_requests", "tool_calls", "side_effects", "result_bytes")
SIDE_EFFECT_TOOLS = frozenset({"write_file", "delete_file", "Bash", "bash"})
TRUNCATED = "\n[TRUNCATED: result budget]"


class BudgetExceeded(RuntimeError):
    def __init__(self, dimension: str, requested: int, remaining: int):
        self.dimension, self.requested, self.remaining = dimension, requested, remaining
        super().__init__(f"BUDGET_EXHAUSTED: {dimension} limit exceeded; needs {requested}, remaining {remaining}")


@dataclass(frozen=True)
class BudgetLimits:
    model_requests: int = 12
    tool_calls: int = 128
    side_effects: int = 32
    result_bytes: int = 1_000_000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0 or (name == "model_requests" and value == 0):
                raise ValueError(f"{name} must be a non-negative integer; model_requests must be positive")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def restore(cls, value: dict) -> BudgetLimits:
        if not isinstance(value, dict) or set(value) != set(DIMENSIONS):
            raise ValueError("Invalid budget limits")
        return cls(**value)


def snapshot(limits: dict, spent: dict) -> dict:
    BudgetLimits.restore(limits)
    if not isinstance(spent, dict) or set(spent) != set(DIMENSIONS):
        raise ValueError("Invalid spent budget")
    if any(type(spent[k]) is not int or not 0 <= spent[k] <= limits[k] for k in DIMENSIONS):
        raise ValueError("Invalid spent budget")
    return {"limits": dict(limits), "spent": dict(spent),
            "remaining": {k: limits[k] - spent[k] for k in DIMENSIONS}}


def debit(budget: dict, **cost: int) -> dict:
    # Check every dimension before changing any of them.
    for name, value in cost.items():
        if name not in DIMENSIONS or type(value) is not int or value < 0:
            raise ValueError("Invalid budget debit")
        if value > budget["remaining"][name]:
            raise BudgetExceeded(name, value, budget["remaining"][name])
    return snapshot(budget["limits"], {k: budget["spent"][k] + cost.get(k, 0) for k in DIMENSIONS})


def batch_cost(calls: list[dict], cached_ids=()) -> dict:
    return {"tool_calls": len(calls), "side_effects": sum(
        c["function"]["name"] in SIDE_EFFECT_TOOLS and c["id"] not in cached_ids for c in calls)}


def truncate_result(content: str, quota: int) -> str:
    if type(quota) is not int or quota < 1:
        raise ValueError("Output quota must be positive")
    data = content.encode("utf-8")
    if len(data) <= quota:
        return content
    marker = TRUNCATED if quota >= len(TRUNCATED.encode("utf-8")) else "~"
    prefix = data[:max(0, quota - len(marker.encode("utf-8")))].decode("utf-8", errors="ignore")
    return prefix + marker


def output_ceiling(name: str, runtime) -> int:
    if name == "read_file":
        return runtime.max_file_chars * 4 + 1024
    if name in {"Bash", "bash", "list_files"}:
        return runtime.max_output_chars * 4 + 1024
    return 1024


class MemoryBudget:
    """The same limits for SDK runs without persistence; CLI runs use SQLite."""
    def __init__(self, limits: BudgetLimits):
        self.state = snapshot(limits.as_dict(), dict.fromkeys(DIMENSIONS, 0))

    def charge(self, **cost):
        self.state = debit(self.state, **cost)

    def batch(self, calls, cached_ids):
        if len(calls) > self.state["remaining"]["result_bytes"]:
            raise BudgetExceeded("result_bytes", len(calls), self.state["remaining"]["result_bytes"])
        self.charge(**batch_cost(calls, cached_ids))

    def reserve_output(self, ceiling: int, pending: int = 0) -> int:
        available = self.state["remaining"]["result_bytes"] - pending
        if available < 1:
            raise BudgetExceeded("result_bytes", 1 + pending, self.state["remaining"]["result_bytes"])
        quota = min(ceiling, available)
        self.charge(result_bytes=quota)
        return quota

    def finish_output(self, content: str, quota: int) -> str:
        result = truncate_result(content, quota)
        spent = {**self.state["spent"], "result_bytes": self.state["spent"]["result_bytes"] - quota + len(result.encode("utf-8"))}
        self.state = snapshot(self.state["limits"], spent)
        return result
