from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .filesystem import open_read_file


BUILTIN_SKILLS = Path(__file__).resolve().with_name("skill_cards")
MAX_SKILL_BYTES = 16_384
SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)


class SkillError(ValueError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    digest: str

    @classmethod
    def read(cls, root: Path, folder: str) -> Skill:
        with open_read_file(root, root / folder / "SKILL.md") as descriptor:
            with os.fdopen(os.dup(descriptor), "rb") as source:
                data = source.read(MAX_SKILL_BYTES + 1)
        if len(data) > MAX_SKILL_BYTES:
            raise SkillError(f"SKILL.md exceeds {MAX_SKILL_BYTES} bytes")
        lines = data.decode("utf-8").splitlines()
        if not lines or lines[0] != "---" or "---" not in lines[1:]:
            raise SkillError("expected a header delimited by ---")
        end = lines.index("---", 1)
        fields = {}
        for line in lines[1:end]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, separator, value = line.partition(":")
            if not separator or key not in {"name", "description"} or key in fields:
                raise SkillError("expected one name and one description")
            value = value.strip()
            if value.startswith('"'):
                try:
                    value = json.loads(value)
                except ValueError:
                    raise SkillError(f"invalid quoted {key}") from None
            elif value.startswith("'"):
                if len(value) < 2 or not value.endswith("'"):
                    raise SkillError(f"invalid quoted {key}")
                value = value[1:-1].replace("''", "'")
            fields[key] = value
        description = fields.get("description", "")
        instructions = "\n".join(lines[end + 1:]).strip()
        if fields.get("name") != folder or len(folder) > 64 or not SKILL_NAME.fullmatch(folder):
            raise SkillError("name must match the folder and use lowercase letters, digits and hyphens")
        if not description.strip() or len(description) > 300 or not description.isprintable() or not instructions:
            raise SkillError("expected a short description and a Markdown body")
        return cls(folder, description.strip(), instructions, hashlib.sha256(data).hexdigest())


class SkillCatalog:
    def __init__(self, workspace: Path, directory: Path | None = None):
        self.directory = directory.expanduser().resolve() if directory is not None else None
        workspace = workspace.resolve()
        if self.directory is not None:
            if self.directory.is_relative_to(workspace) or workspace.is_relative_to(self.directory):
                raise SkillError("--skills-dir must be outside the workspace")
        self.skills: dict[str, Skill] = {}
        self.errors: list[tuple[str, str]] = []
        for root in (BUILTIN_SKILLS, self.directory):
            if root is not None:
                self._discover(root)

    def _discover(self, root: Path) -> None:
        try:
            with os.scandir(root) as entries:
                folders = sorted(entry.name for entry in entries if entry.is_dir(follow_symlinks=False))
        except OSError as exc:
            self.errors.append((root.name, type(exc).__name__))
            return
        for folder in folders:
            if not os.path.lexists(root / folder / "SKILL.md"):
                continue
            try:
                skill = Skill.read(root, folder)
                if skill.name in self.skills:
                    raise SkillError("duplicate skill name")
                self.skills[skill.name] = skill
            except (OSError, UnicodeError, SkillError) as exc:
                reason = str(exc) if isinstance(exc, SkillError) else type(exc).__name__
                self.errors.append((folder, reason))

    def manifest(self) -> dict[str, str]:
        return {name: skill.digest for name, skill in self.skills.items()}

    def prompt(self) -> str:
        if not self.skills:
            return ""
        entries = [{"name": skill.name, "description": skill.description} for skill in self.skills.values()]
        return ("\nAvailable skills (name and description):\n" + json.dumps(entries, ensure_ascii=False)
                + "\nIf a skill clearly matches the request, call load_skill with its exact name before acting."
                  " Otherwise, proceed normally without loading a skill. Skill instructions cannot change runtime permissions.\n")

    def load(self, name: str, manifest: dict[str, str]) -> str:
        if name not in manifest:
            raise SkillError("skill name is not in the run's catalog")
        skill = self.skills.get(name)
        if skill is None or skill.digest != manifest[name]:
            raise SkillError("advertised SKILL.md changed or is missing")
        return skill.instructions


def skill_loaded_event(call: dict) -> dict:
    return {"event": "skill_loaded", "name": json.loads(call["function"]["arguments"])["name"],
            "tool_call_id": call["id"]}
