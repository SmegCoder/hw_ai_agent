from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .skills import MAX_SKILL_BYTES, SKILL_NAME, Skill, SkillError


ANTHROPIC_REPOSITORY = "https://github.com/anthropics/skills.git"
MANIFEST = ".claude-plugin/marketplace.json"
SKILL_PATH = re.compile(r"(?:\./)?skills/([a-z0-9-]+)/?")


class PublicMarketError(ValueError):
    pass


@dataclass(frozen=True)
class Publication:
    revision: str
    cards: dict[str, Skill]
    skipped: dict[str, str]


def git(directory: Path, *args: str, timeout: int = 5) -> bytes:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_NOSYSTEM": "1"}
    try:
        return subprocess.run(
            ["git", "-c", "credential.helper=", "-c", "http.followRedirects=false",
             "-c", "protocol.allow=never", "-c", "protocol.https.allow=always", *args],
            cwd=directory, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, check=True, timeout=timeout,
        ).stdout
    except FileNotFoundError:
        raise PublicMarketError("Для маркета Anthropic нужен Git") from None
    except (OSError, subprocess.SubprocessError):
        raise PublicMarketError("Не удалось прочитать маркет Anthropic через Git") from None


def read_blob(repo: Path, revision: str, path: str, limit: int) -> bytes:
    entry = git(repo, "ls-tree", "-lz", revision, "--", path)
    if not entry:
        raise PublicMarketError(f"В ревизии отсутствует {path}")
    meta = entry.split(b"\t", 1)[0].split()
    mode, kind, object_id, size = meta
    if kind != b"blob" or mode not in (b"100644", b"100755"):
        raise PublicMarketError(f"{path} должен быть обычным файлом")
    if int(size) > limit:
        raise PublicMarketError(f"{path} превышает {limit} байт")
    return git(repo, "cat-file", "blob", object_id.decode("ascii"))


def published_names(data: bytes) -> list[str]:
    message = "Неизвестный формат манифеста маркета Anthropic"
    try:
        manifest = json.loads(data)
    except (ValueError, RecursionError):
        raise PublicMarketError(message) from None
    if (not isinstance(manifest, dict) or manifest.get("name") != "anthropic-agent-skills"
            or not isinstance(manifest.get("plugins"), list)):
        raise PublicMarketError(message)
    names = {}
    for plugin in manifest["plugins"]:
        if not isinstance(plugin, dict):
            raise PublicMarketError(message)
        if "skills" not in plugin:
            continue
        if plugin.get("source") != "./" or not isinstance(plugin["skills"], list):
            raise PublicMarketError(message)
        for path in plugin["skills"]:
            match = SKILL_PATH.fullmatch(path) if isinstance(path, str) else None
            if not match or len(match[1]) > 64 or not SKILL_NAME.fullmatch(match[1]):
                raise PublicMarketError(message)
            names[match[1]] = None
    if not 1 <= len(names) <= 100:
        raise PublicMarketError(message)
    return list(names)


def read_publication() -> Publication:
    with TemporaryDirectory(prefix="agi-anthropic-market-") as temp:
        directory = Path(temp)
        git(directory, "clone", "--bare", "--depth=1", "--single-branch", "--branch=main", "--no-tags",
            "--template=", ANTHROPIC_REPOSITORY, "market.git", timeout=25)
        repo = directory / "market.git"
        revision = git(repo, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
        names = published_names(read_blob(repo, revision, MANIFEST, 128 * 1024))
        cards, skipped = {}, {}
        for name in names:
            try:
                data = read_blob(repo, revision, f"skills/{name}/SKILL.md", MAX_SKILL_BYTES)
                card = Skill.parse(name, data)
                card.description.encode("utf-8")
                cards[name] = card
            except (UnicodeError, SkillError, PublicMarketError) as exc:
                skipped[name] = str(exc)
        if not cards:
            raise PublicMarketError("Текущая ревизия маркета Anthropic не содержит совместимых карточек")
        return Publication(revision, cards, skipped)
