from __future__ import annotations

import tempfile
from pathlib import Path

from .policy import Policy
from .tools import ToolError, ToolRuntime


def run_policy_demo(on_event=None) -> dict:
    checks = {}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "order.py"
        source.write_text("original\n", encoding="utf-8")
        test = root / "test_order.py"
        test.write_text("# protected\n", encoding="utf-8")
        (root / ".env").write_text("EXAMPLE=not-a-real-key\n", encoding="utf-8")
        approvals = iter([False, True])
        runtime = ToolRuntime(root, approve=lambda request: next(approvals), on_policy_event=on_event)
        checks["Чтение разрешено"] = runtime.execute("read_file", {"path": "order.py"}) == "original\n"
        def denied(name, arguments, expected):
            try:
                runtime.execute(name, arguments)
            except ToolError as exc:
                return expected in str(exc)
            return False
        checks["Отказ оператора сохраняет файл"] = (
            denied("write_file", {"path": "order.py", "content": "changed\n"}, "APPROVAL_REQUIRED")
            and source.read_text() == "original\n"
        )
        runtime.execute("write_file", {"path": "order.py", "content": "approved\n"})
        checks["Подтверждённая запись выполняется"] = source.read_text() == "approved\n"
        checks["Тесты защищены"] = denied("write_file", {"path": "test_order.py", "content": "changed"}, "read-only")
        checks["Секреты защищены"] = denied("read_file", {"path": ".env"}, "protected")
        checks["Выход из workspace запрещён"] = denied("read_file", {"path": "../outside"}, "escapes")
        runtime.policy = Policy.from_dict({"rules": [{"tool": "write_file", "action": "deny"}]})
        checks["Явный deny блокирует запись"] = denied("write_file", {"path": "order.py", "content": "blocked"}, "POLICY_DENIED")
        runtime.policy = Policy()
        runtime.approve = None
        checks["Без подтверждения запись запрещена"] = denied("write_file", {"path": "order.py", "content": "blocked"}, "APPROVAL_REQUIRED")
        checks["Контрольный тест не изменён"] = test.read_text() == "# protected\n"
        events = runtime.policy_events
    return {"success": all(checks.values()), "checks": checks, "policy_events": events}
