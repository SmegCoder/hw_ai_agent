from __future__ import annotations

import json
import re


_SURROGATE = re.compile(r"[\ud800-\udfff]")


def has_surrogates(value: str) -> bool:
    return _SURROGATE.search(value) is not None


def escape_surrogates(value: str) -> str:
    """Keep valid Unicode and make isolated surrogate code points visible."""
    return _SURROGATE.sub(lambda match: f"\\u{ord(match[0]):04x}", value)


def json_text(value, *, sort_keys=False, indent=None, allow_nan=False) -> str:
    """Serialize JSON to UTF-8 text, preserving data through JSON escapes."""
    return escape_surrogates(json.dumps(value, ensure_ascii=False, sort_keys=sort_keys,
                                       indent=indent, allow_nan=allow_nan))


def model_text(value):
    """Make JSON strings readable by providers that reject isolated surrogates."""
    if isinstance(value, str):
        return escape_surrogates(value)
    if isinstance(value, dict):
        return {model_text(key): model_text(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [model_text(item) for item in value]
    return value
