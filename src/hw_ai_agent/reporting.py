from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def public_report(report: dict, workspace: Path) -> dict:
    workspace_paths = {str(workspace.absolute()), str(workspace.resolve())}
    replacements = [(path, "workspace/demo") for path in workspace_paths]
    home = str(Path.home())
    if home != Path(home).anchor:
        replacements.append((home, "<home>"))
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)

    def clean(value: Any) -> Any:
        if isinstance(value, str):
            for source, target in replacements:
                value = re.sub(re.escape(source) + r'''(?=$|[/\\\s"'<>:,)\]])''',
                               lambda match: target, value)
            value = re.sub(r'''(?<![\w:/])/(?:Users|home)/[^/\s"'<>]+''', "<home>", value)
            value = re.sub(r'''(?<![\w:/\\])[A-Za-z]:\\{1,2}Users\\{1,2}[^\\/\s"'<>]+''',
                           "<home>", value)
            return re.sub(r'''(?<![\w:/])/(?:private/)?var/folders/[^/\s]+/[^/\s]+/(?:T|C|0)(?=/|$)''',
                          "<temp>", value)
        if isinstance(value, dict):
            return {clean(key): clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    result = clean(report)
    result["workspace"] = "workspace/demo"
    return result
